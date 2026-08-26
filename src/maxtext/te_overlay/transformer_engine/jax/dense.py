# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized dense layer transformation operations for transformer
architectures, including support for quantization and automatic differentiation.
It implements matrix multiplication with optional bias addition and supports
customizable contracting dimensions for flexible tensor operations.
"""

from typing import Tuple, Sequence
from functools import partial
import warnings
import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning, SdyShardingRule
from jax.sharding import NamedSharding, PartitionSpec
from packaging.version import Version as PkgVersion

from . import cpp_extensions as tex
from .cpp_extensions.amax import AmaxScope
from .quantize import (
    ScaledTensor,
    QuantizerSet,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
    TensorUsage,
)


def _spec_axes(spec):
    """Normalize one PartitionSpec entry to an ordered tuple of mesh axes."""
    if spec is None:
        return ()
    axes = spec if isinstance(spec, tuple) else (spec,)
    return tuple(dict.fromkeys(axes))


def _axes_spec(axes):
    """Convert normalized mesh axes back to one PartitionSpec entry."""
    axes = tuple(axes)
    if not axes:
        return None
    return axes[0] if len(axes) == 1 else axes


def _pad_partition_spec(spec, ndim):
    """Pad a PartitionSpec with replicated dimensions to match an array rank."""
    entries = tuple(spec)
    if len(entries) > ndim:
        raise ValueError(f"PartitionSpec rank {len(entries)} exceeds array rank {ndim}: {spec}")
    return entries + (None,) * (ndim - len(entries))


def _plan_local_wgrad_partition(
    lhs_spec,
    rhs_spec,
    output_spec,
    lhs_ndim,
    rhs_ndim,
    output_ndim,
    contracting_dims,
):
    """Plan aligned local operands and explicit WGrad reductions.

    The contracting dimension may be sharded by partially overlapping compound
    mesh axes. Operand-only axes are removed from that dimension, which asks
    GSPMD to gather only those axes before entering the local implementation.
    Axes shared by both operands remain local contraction shards and are reduced
    explicitly after the local dot. If a shared axis shards the kernel, the
    reduction is a reduce-scatter into that kernel dimension; otherwise it is
    an all-reduce.
    """
    lhs_cdims, rhs_cdims = contracting_dims
    if len(lhs_cdims) != 1 or len(rhs_cdims) != 1:
        raise NotImplementedError(
            "Explicit local JAX WGrad currently requires exactly one flattened "
            f"contracting dimension, got {contracting_dims}."
        )

    lhs_cdim, rhs_cdim = lhs_cdims[0], rhs_cdims[0]
    lhs_specs = list(_pad_partition_spec(lhs_spec, lhs_ndim))
    rhs_specs = list(_pad_partition_spec(rhs_spec, rhs_ndim))
    output_specs = _pad_partition_spec(output_spec, output_ndim)
    lhs_non_cdims = tuple(dim for dim in range(lhs_ndim) if dim != lhs_cdim)
    rhs_non_cdims = tuple(dim for dim in range(rhs_ndim) if dim != rhs_cdim)
    output_sources = tuple(("lhs", dim) for dim in lhs_non_cdims) + tuple(
        ("rhs", dim) for dim in rhs_non_cdims
    )
    if len(output_sources) != output_ndim:
        raise ValueError(
            f"WGrad output rank {output_ndim} does not match dot output rank "
            f"{len(output_sources)}."
        )

    lhs_axes = _spec_axes(lhs_specs[lhs_cdim])
    rhs_axis_set = set(_spec_axes(rhs_specs[rhs_cdim]))
    shared_axes = tuple(axis for axis in lhs_axes if axis in rhs_axis_set)
    shared_spec = _axes_spec(shared_axes)

    # Both local operands must contain the same slice of the contracting
    # dimension. Removing operand-only axes materializes only the minimum
    # alignment gather; shared axes stay sharded for the local partial dot.
    lhs_specs[lhs_cdim] = shared_spec
    rhs_specs[rhs_cdim] = shared_spec

    # A non-reduction output axis must be carried by the corresponding local
    # dot operand. Reduction axes are intentionally absent before the dot and
    # are introduced into the result by psum_scatter below.
    shared_axis_set = set(shared_axes)
    for output_dim, (operand, operand_dim) in enumerate(output_sources):
        output_axes = _spec_axes(output_specs[output_dim])
        local_output_axes = tuple(axis for axis in output_axes if axis not in shared_axis_set)
        reduction_output_axes = tuple(axis for axis in output_axes if axis in shared_axis_set)
        if output_axes != local_output_axes + reduction_output_axes:
            raise NotImplementedError(
                "Explicit local JAX WGrad requires carrier mesh axes to precede "
                "reduce-scatter axes within a compound kernel dimension, got "
                f"{output_specs[output_dim]!r} on output dimension {output_dim}."
            )
        if operand == "lhs":
            lhs_specs[operand_dim] = _axes_spec(local_output_axes)
        else:
            rhs_specs[operand_dim] = _axes_spec(local_output_axes)

    reduction_plan = []
    scattered_axes = set()
    for output_dim, spec in enumerate(output_specs):
        for mesh_axis in _spec_axes(spec):
            if mesh_axis not in shared_axis_set:
                continue
            if mesh_axis in scattered_axes:
                raise ValueError(
                    f"Kernel PartitionSpec uses mesh axis {mesh_axis!r} on multiple dimensions: "
                    f"{output_specs}"
                )
            reduction_plan.append((mesh_axis, output_dim))
            scattered_axes.add(mesh_axis)

    for mesh_axis in shared_axes:
        if mesh_axis not in scattered_axes:
            reduction_plan.append((mesh_axis, None))

    return tuple(lhs_specs), tuple(rhs_specs), output_specs, tuple(reduction_plan)


def _jax_bf16_wgrad_impl(lhs, rhs, contracting_dims, output_spec):
    """Reference/global implementation used outside a partitioned mesh."""
    del output_spec
    return jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=(contracting_dims, ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )


_jax_bf16_wgrad = custom_partitioning(
    _jax_bf16_wgrad_impl,
    static_argnums=(2, 3),
)


def _jax_bf16_wgrad_shardy_rule(
    contracting_dims,
    output_spec,
    mesh,
    operand_types,
    result_types,
):
    """Keep operand and result propagation independent; partition() owns communication."""
    del contracting_dims, output_spec, mesh
    lhs_type, rhs_type = operand_types
    result_type = result_types[0] if isinstance(result_types, (tuple, list)) else result_types
    lhs_mapping = tuple(f"Wgrad_lhs_{dim}" for dim in range(len(lhs_type.shape)))
    rhs_mapping = tuple(f"Wgrad_rhs_{dim}" for dim in range(len(rhs_type.shape)))
    output_mapping = tuple(f"Wgrad_out_{dim}" for dim in range(len(result_type.shape)))
    return SdyShardingRule(
        operand_mappings=(lhs_mapping, rhs_mapping),
        result_mappings=(output_mapping,),
    )


def _jax_bf16_wgrad_infer_sharding(
    contracting_dims,
    output_spec,
    mesh,
    arg_infos,
    result_infos,
):
    """Use the parameter layout as the WGrad output layout."""
    del contracting_dims, arg_infos, result_infos
    return NamedSharding(mesh, PartitionSpec(*output_spec))


def _jax_bf16_wgrad_partition(
    contracting_dims,
    output_spec,
    mesh,
    arg_infos,
    result_infos,
):
    """Lower WGrad to aligned local dot plus explicit reduction collectives."""
    lhs_info, rhs_info = arg_infos
    result_info = (
        result_infos[0] if isinstance(result_infos, (tuple, list)) else result_infos
    )
    lhs_specs, rhs_specs, output_specs, reduction_plan = _plan_local_wgrad_partition(
        lhs_info.sharding.spec,
        rhs_info.sharding.spec,
        output_spec,
        len(lhs_info.shape),
        len(rhs_info.shape),
        len(result_info.shape),
        contracting_dims,
    )

    lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_specs))
    rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_specs))
    output_sharding = NamedSharding(mesh, PartitionSpec(*output_specs))

    def _sharded_impl(lhs, rhs):
        output = _jax_bf16_wgrad_impl(lhs, rhs, contracting_dims, output_spec)
        for mesh_axis, scatter_dimension in reduction_plan:
            if scatter_dimension is None:
                output = jax.lax.psum(output, mesh_axis)
            else:
                output = jax.lax.psum_scatter(
                    output,
                    mesh_axis,
                    scatter_dimension=scatter_dimension,
                    tiled=True,
                )
        return output

    return mesh, _sharded_impl, output_sharding, (lhs_sharding, rhs_sharding)


_wgrad_partition_kwargs = {
    "partition": _jax_bf16_wgrad_partition,
    "sharding_rule": _jax_bf16_wgrad_shardy_rule,
}
if PkgVersion(jax.__version__) <= PkgVersion("0.9.1"):
    _wgrad_partition_kwargs["infer_sharding_from_operands"] = (
        _jax_bf16_wgrad_infer_sharding
    )
_jax_bf16_wgrad.def_partition(**_wgrad_partition_kwargs)


def _flatten_wgrad_contracting_dims(lhs, rhs, lhs_cdims, rhs_cdims):
    """Flatten contiguous leading token dimensions for the local partitioner."""
    expected_lhs = tuple(range(len(lhs_cdims)))
    expected_rhs = tuple(range(len(rhs_cdims)))
    if tuple(lhs_cdims) != expected_lhs or tuple(rhs_cdims) != expected_rhs:
        raise NotImplementedError(
            "Explicit local JAX WGrad requires leading contracting dimensions, got "
            f"{(lhs_cdims, rhs_cdims)}."
        )
    if len(lhs_cdims) != len(rhs_cdims):
        raise ValueError(
            "WGrad operands must have the same number of token dimensions, got "
            f"{(lhs_cdims, rhs_cdims)}."
        )
    if len(lhs_cdims) == 1:
        return lhs, rhs, ((0,), (0,))

    lhs = lhs.reshape((-1, *lhs.shape[len(lhs_cdims) :]))
    rhs = rhs.reshape((-1, *rhs.shape[len(rhs_cdims) :]))
    return lhs, rhs, ((0,), (0,))


def _logical_to_partition_spec(logical_axes):
    """Resolve MaxText logical kernel axes while the axis-rules context is active."""
    if not logical_axes:
        raise ValueError(
            "te_wgrad_backend=jax_bf16 requires kernel logical axes so its explicit "
            "local WGrad partitioner can place the reduced gradient."
        )

    import flax  # pylint: disable=import-outside-toplevel

    if not flax.linen.get_logical_axis_rules():
        raise ValueError(
            "te_wgrad_backend=jax_bf16 requires an active Flax logical_axis_rules context."
        )
    return flax.linen.logical_to_mesh_axes(logical_axes)


def _all_gather_kernel(kernel, mesh_axis, axis_idx):
    assert mesh_axis is not None
    assert 0 < axis_idx < len(kernel.shape)

    # TODO(Ming Hunag): Add a condition branch for with/without shmap.
    kernel_shape = kernel.shape
    kernel_whole_shape = (*kernel_shape[:axis_idx], -1, *kernel_shape[axis_idx + 1 :])
    global_kernel = jax.lax.all_gather(kernel, mesh_axis, axis=axis_idx)
    global_kernel = global_kernel.reshape(*kernel_whole_shape)
    return global_kernel


def _psum_scatter_kernel(kernel, scattered_kernel_shape, mesh_axis, axis_idx):
    assert mesh_axis is not None
    assert 0 < axis_idx < len(scattered_kernel_shape)

    # TODO(Ming Hunag): Add a condition branch for with/without shmap.
    kernel = kernel.reshape(
        *scattered_kernel_shape[:axis_idx],
        -1,
        scattered_kernel_shape[axis_idx],
        *scattered_kernel_shape[axis_idx + 1 :],
    )
    kernel = jax.lax.psum_scatter(kernel, mesh_axis, scatter_dimension=axis_idx)
    kernel = kernel.reshape(scattered_kernel_shape)
    return kernel


def dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    transpose_batch_sequence: bool = False,
    input_axes: Tuple[str, ...] = None,
    kernel_axes: Tuple[str, ...] = None,
    output_axes: Tuple[str, ...] = None,
    collective_op_set: tex.CollectiveOpSet = tex.noop_collective_op_set,
    quantizer_set: QuantizerSet = noop_quantizer_set,
    wgrad_backend: str = "te",
):
    """Perform dense layer transformation with optional quantization.

    This function implements matrix multiplication with optional bias addition,
    supporting quantization and custom contracting dimensions. It's optimized
    for transformer architectures and supports automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix for the dense layer transformation
        bias: Optional bias tensor to add after the transformation
        contracting_dims: Tuple of sequences specifying which dimensions to contract
        transpose_batch_sequence: Transpose the batch and sequence dimensions of the input tensor.
        input_axes: Logical axes for sharding the activation input
        kernel_axes: Logical axes for sharding the weight matrix
        output_axes: Logical axes for sharding the output
        collective_op_set: A set of CollectiveOp objects for forward and backward passes.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types
        wgrad_backend: ``"te"`` for the standard TE WGrad, or ``"jax_bf16"`` to
            compute only WGrad with JAX BF16 dot_general.

    Returns:
        Transformed output tensor
    """
    if wgrad_backend not in ("te", "jax_bf16"):
        raise ValueError(
            f"Unsupported Dense WGrad backend {wgrad_backend!r}; expected 'te' or 'jax_bf16'."
        )

    if transpose_batch_sequence:
        warnings.warn("transpose_batch_sequence is not well tested, use with caution!")

    if collective_op_set != tex.noop_collective_op_set and not output_axes:
        warnings.warn(
            "Collective GEMM with Shardy propagation may produce an incorrect sharding pattern"
            " for the output. Set `output_axes` to apply the correct sharding constraint.",
            UserWarning,
        )

    if quantizer_set == noop_quantizer_set:
        input_dtype = x.dtype
        kernel = kernel.astype(input_dtype)

    output = _dense(
        x,
        kernel,
        bias,
        contracting_dims,
        transpose_batch_sequence,
        input_axes,
        kernel_axes,
        output_axes,
        collective_op_set,
        wgrad_backend,
        quantizer_set,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def _dense(
    x,
    kernel,
    bias,
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    wgrad_backend,
    quantizer_set,  # need to be a diff_arg for DelayedScaling state management
):
    """Internal implementation of dense layer transformation with custom VJP.

    This function implements the core dense layer transformation logic with support
    for custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix
        bias: Optional bias tensor
        contracting_dims: Contracting dimensions specification
        transpose_batch_sequence: Transpose the batch and sequence dimensions of the input tensor.
        input_axes: Logical axes for sharding the activation input
        output_axes: Logical axes for sharding the output_axes
        kernel_axes: Logical axes for sharding the weight matrix
        collective_op_set: A set of CollectiveOp objects for forward and backward passes.
        wgrad_backend: Backend used only for WGrad.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    output, _ = _dense_fwd_rule(
        x,
        kernel,
        bias,
        contracting_dims,
        transpose_batch_sequence,
        input_axes,
        kernel_axes,
        output_axes,
        collective_op_set,
        wgrad_backend,
        quantizer_set,
    )
    return output


def _dense_fwd_rule(
    x,
    kernel,
    bias,
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    wgrad_backend,
    quantizer_set,
):
    """Forward pass rule for dense layer transformation.

    Returns:
        Tuple of (output, context) for backward pass
    """
    x_contracting_dims, k_contracting_dims = map(
        tex.sanitize_dims, (x.ndim, kernel.ndim), contracting_dims
    )

    # Check supported input layout
    x_is_transposed = x.ndim - 1 not in x_contracting_dims
    k_is_transposed = kernel.ndim - 1 in k_contracting_dims
    assert (
        not x_is_transposed and not k_is_transposed
    ), "Dense layer only supports `NN` layout inputs, i.e. non-transposed X and Kernel."

    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape)

    casted_x = tex.quantize(
        x,
        flatten_axis=flatten_axis_x,
        quantizer=quantizer_set.x,
        amax_scope=AmaxScope.TPSP,
        transpose_batch_sequence=transpose_batch_sequence,
    )
    casted_x = with_sharding_constraint_by_logical_axes(casted_x, input_axes)

    casted_kernel = tex.quantize(
        kernel,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.kernel,
        amax_scope=AmaxScope.FSDP,
    )
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

    # GEMM NN
    output = tex.gemm(
        casted_x.get_tensor(usage=TensorUsage.LHS),
        casted_kernel.get_tensor(usage=TensorUsage.RHS),
        bias=bias,
        contracting_dims=(x_contracting_dims, k_contracting_dims),
        transpose_batch_sequence=transpose_batch_sequence,
        collective_op=collective_op_set.forward,
    )
    output = with_sharding_constraint_by_logical_axes(output, output_axes)

    has_bias = bias is not None
    wgrad_x = (
        x
        if wgrad_backend == "jax_bf16"
        else casted_x.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(quantizer_set.x)
    )
    ctx = (
        wgrad_x,
        casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(quantizer_set.kernel),
        jnp.zeros((), dtype=kernel.dtype),
        x.shape,
        kernel.shape,
        quantizer_set,
        flatten_axis_k,
        has_bias,
    )
    return output, ctx


def _dense_bwd_rule(
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    wgrad_backend,
    ctx,
    grad,
):
    """Backward pass rule for dense layer transformation.

    Returns:
        Tuple of gradients with respect to inputs
    """
    (
        wgrad_x,
        casted_kernel_rhs,
        kernel_dtype_ref,
        x_shape,
        kernel_shape,
        quantizer_set,
        flatten_axis_k,
        has_bias,
    ) = ctx
    grad = with_sharding_constraint_by_logical_axes(grad, output_axes)

    fwd_x_contracting_dims, fwd_k_contracting_dims = map(
        tex.sanitize_dims, (wgrad_x.ndim, casted_kernel_rhs.ndim), contracting_dims
    )

    casted_grad, dbias = tex.quantize_dbias(
        grad,
        is_dbias=has_bias,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.dgrad,
        amax_scope=AmaxScope.TPSP,
        transpose_batch_sequence=transpose_batch_sequence,
    )

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_contracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_contracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )

    dgrad = tex.gemm(
        casted_grad.get_tensor(usage=TensorUsage.LHS),
        casted_kernel_rhs,
        contracting_dims=(g_contracting_dim, k_contracting_dim),
        transpose_batch_sequence=transpose_batch_sequence,
        collective_op=collective_op_set.backward,
    )

    # GEMM TN
    # x_non_contracting_dims
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    if wgrad_backend == "jax_bf16":
        kernel_pspec = _logical_to_partition_spec(kernel_axes)
        wgrad_lhs, wgrad_rhs, wgrad_contracting_dims = _flatten_wgrad_contracting_dims(
            wgrad_x.astype(jnp.bfloat16),
            grad.astype(jnp.bfloat16),
            x_contracting_dim,
            g_contracting_dim,
        )
        wgrad = _jax_bf16_wgrad(
            wgrad_lhs,
            wgrad_rhs,
            wgrad_contracting_dims,
            kernel_pspec,
        )
    else:
        wgrad = tex.gemm(
            wgrad_x,
            casted_grad.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=(x_contracting_dim, g_contracting_dim),
            transpose_batch_sequence=transpose_batch_sequence,
            preferred_element_type=jnp.float32,
        )

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, input_axes)
    # The JAX backend already returns the explicit parameter shard. Keep the
    # logical constraint as a consistency check before casting to kernel dtype.
    wgrad = with_sharding_constraint_by_logical_axes(wgrad, kernel_axes)
    wgrad = wgrad.astype(kernel_dtype_ref.dtype)

    return dgrad, wgrad, dbias, quantizer_set


_dense.defvjp(_dense_fwd_rule, _dense_bwd_rule)


def grouped_dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    group_sizes: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (1,)),
    bias: jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype = None,
    group_offset: jnp.array = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
    kernel_fsdp_info: Tuple[str, int] = (None, -1),
):
    """
    Perform grouped dense (linear) layer transformation with optional quantization.

    Args:
        x: Input tensor of shape (M, K)
        kernel: Weight matrix of shape (G, K, N)
        group_sizes: 1D array of shape (G,) specifying the size of each group
        contracting_dims: Tuple of sequences specifying which dimensions to contract
                          (currently only supports ((1,), (1,)))
        bias: Bias tensor of shape (G, N)
        precision: JAX precision for the GEMM operation
        preferred_element_type: Preferred data type for the output tensor
        group_offset: 1D array containing offsets for each group (not yet implemented)
        quantizer_set: Set of quantizers for FP8 quantization of the input and output
        kernel_fsdp_info: A tuple containing FSDP-related information for a weight matrix
                          represented in the format (str, int). The first element is the
                          FSDP mesh axis, and the second element is the dimension along
                          which the weight is sharded.

    Returns:
        A jnp.ndarray containing the result of the grouped linear operation
    """
    output = _grouped_dense(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
        quantizer_set,
        kernel_fsdp_info,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 5, 6, 7, 9))
def _grouped_dense(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
    kernel_fsdp_info,
):
    output, _ = _grouped_dense_fwd_rule(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
        quantizer_set,
        kernel_fsdp_info,
    )
    return output


def _grouped_dense_fwd_rule(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
    kernel_fsdp_info,
):
    use_bias = bias is not None

    kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx = kernel_fsdp_info
    kernel_fsdp_enabled = kernel_fsdp_mesh_axis is not None
    assert not kernel_fsdp_enabled, "FSDP sharding for grouped_dense is not supported yet."
    del kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx, kernel_fsdp_info, kernel_fsdp_enabled

    x_contracting_dims, k_contracting_dims = contracting_dims
    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape) + 1  # +1 for G axis

    casted_x = tex.grouped_quantize(
        x,
        quantizer_set.x,
        group_sizes,
        flatten_axis=flatten_axis_x,
    )

    casted_kernel = tex.grouped_quantize(kernel, quantizer_set.kernel, flatten_axis=flatten_axis_k)
    contracting_dims = (x_contracting_dims, k_contracting_dims)

    # For x_contracting_dims == (1,) and k_contracting_dims == (1,), we should have
    # rowwise_casted_x.original_shape == (M, K)
    # colwise_casted_kernel.original_shape == (G, N, K)
    grouped_gemm_x = casted_x.get_tensor(usage=TensorUsage.LHS)
    # Checkpoint the rowwise inputs so that te_grouped_quantize_ffi can be DCE'd in the
    # backward-scan remat block.  Without this, JAX would re-run the quantize kernel to
    # obtain grouped_gemm_x / grouped_gemm_kernel for the forward-GEMM recomputation even
    # though the colwise residuals (ctx_x / ctx_kernel) are already saved.  With both
    # orientations checkpointed, all outputs of the custom-call become dead in the remat trace.
    grouped_gemm_x = (
        grouped_gemm_x.checkpoint(quantizer_set.x)
        if isinstance(grouped_gemm_x, ScaledTensor)
        else grouped_gemm_x
    )
    ctx_x = casted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
    ctx_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS)

    grouped_gemm_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS)
    grouped_gemm_kernel = (
        grouped_gemm_kernel.checkpoint(quantizer_set.kernel)
        if isinstance(grouped_gemm_kernel, ScaledTensor)
        else grouped_gemm_kernel
    )
    output = tex.grouped_gemm(
        grouped_gemm_x,
        grouped_gemm_kernel,
        contracting_dims=contracting_dims,
        bias=bias,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    ctx = (
        group_sizes,
        ctx_x.checkpoint(quantizer_set.x) if isinstance(ctx_x, ScaledTensor) else ctx_x,
        (
            ctx_kernel.checkpoint(quantizer_set.kernel)
            if isinstance(ctx_kernel, ScaledTensor)
            else ctx_kernel
        ),
        x.shape,
        kernel.shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
    )
    return output, ctx


def _grouped_dense_bwd_rule(
    contracting_dims, precision, preferred_element_type, group_offset, kernel_fsdp_info, ctx, grad
):
    kernel_fsdp_mesh_axis, _ = kernel_fsdp_info
    kernel_fsdp_enabled = kernel_fsdp_mesh_axis is not None
    assert not kernel_fsdp_enabled, "FSDP sharding for grouped_dense is not supported yet."

    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims

    (
        group_sizes,
        ctx_x,
        ctx_kernel,
        x_shape,
        kernel_shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
    ) = ctx

    # The 1 in range is for excluding the group dimension (shall we use the hardcoded results below?)
    # g_contracting_dim = (1, )
    # k_contracting_dim = (2, )
    g_contracting_dim = tuple(
        range(1 + grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    k_contracting_dim = tuple(
        dim for dim in range(1, len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )

    casted_grad = tex.grouped_quantize(
        grad, quantizer_set.dgrad, group_sizes, flatten_axis=flatten_axis_k
    )

    dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)
    dgrad_grad = casted_grad.get_tensor(usage=TensorUsage.LHS)
    dgrad_kernel_T = ctx_kernel

    # g_contracting_dim = (0, )
    # x_contracting_dim = (0, )
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )
    wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)

    wgrad_x_T = ctx_x
    wgrad_grad = casted_grad.get_tensor(usage=TensorUsage.RHS)
    dgrad = tex.grouped_gemm(
        dgrad_grad,
        dgrad_kernel_T,
        contracting_dims=dgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    wgrad = tex.grouped_gemm(
        wgrad_x_T,
        wgrad_grad,
        contracting_dims=wgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    group_sizes_grad = None
    dbias = tex.grouped_dbias(grad, group_sizes) if use_bias else None

    return dgrad, wgrad, group_sizes_grad, dbias, quantizer_set


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
