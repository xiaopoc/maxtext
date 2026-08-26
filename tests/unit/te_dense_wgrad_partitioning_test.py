# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for explicit local JAX Dense WGrad partitioning."""

import importlib
import unittest

from jax.sharding import PartitionSpec


dense = importlib.import_module("transformer_engine.jax.dense")


class TeDenseWgradPartitioningTest(unittest.TestCase):

  def test_partial_compound_contracting_axes_gather_only_operand_difference(self):
    lhs_spec, rhs_spec, output_spec, reduction_plan = dense._plan_local_wgrad_partition(
        PartitionSpec(("fsdp", "tensor", "expert"), None),
        PartitionSpec(("fsdp", "expert"), "tensor"),
        PartitionSpec(("fsdp", "expert"), "tensor"),
        lhs_ndim=2,
        rhs_ndim=2,
        output_ndim=2,
        contracting_dims=((0,), (0,)),
    )

    self.assertEqual(lhs_spec, (("fsdp", "expert"), None))
    self.assertEqual(rhs_spec, (("fsdp", "expert"), "tensor"))
    self.assertEqual(output_spec, (("fsdp", "expert"), "tensor"))
    self.assertEqual(reduction_plan, (("fsdp", 0), ("expert", 0)))

  def test_shared_contracting_axis_not_in_kernel_uses_all_reduce(self):
    lhs_spec, rhs_spec, output_spec, reduction_plan = dense._plan_local_wgrad_partition(
        PartitionSpec("data", None),
        PartitionSpec("data", "tensor"),
        PartitionSpec(None, "tensor"),
        lhs_ndim=2,
        rhs_ndim=2,
        output_ndim=2,
        contracting_dims=((0,), (0,)),
    )

    self.assertEqual(lhs_spec, ("data", None))
    self.assertEqual(rhs_spec, ("data", "tensor"))
    self.assertEqual(output_spec, (None, "tensor"))
    self.assertEqual(reduction_plan, (("data", None),))

  def test_reduce_scatter_follows_compound_kernel_axis_order(self):
    _, _, _, reduction_plan = dense._plan_local_wgrad_partition(
        PartitionSpec(("fsdp", "expert"), None),
        PartitionSpec(("fsdp", "expert"), "tensor"),
        PartitionSpec(("expert", "fsdp"), "tensor"),
        lhs_ndim=2,
        rhs_ndim=2,
        output_ndim=2,
        contracting_dims=((0,), (0,)),
    )

    self.assertEqual(reduction_plan, (("expert", 0), ("fsdp", 0)))

  def test_non_reduction_output_axes_are_carried_by_dot_operands(self):
    lhs_spec, rhs_spec, _, reduction_plan = dense._plan_local_wgrad_partition(
        PartitionSpec(("fsdp", "expert"), "tensor"),
        PartitionSpec(("fsdp", "expert"), None),
        PartitionSpec(None, "tensor"),
        lhs_ndim=2,
        rhs_ndim=2,
        output_ndim=2,
        contracting_dims=((0,), (0,)),
    )

    self.assertEqual(lhs_spec, (("fsdp", "expert"), None))
    self.assertEqual(rhs_spec, (("fsdp", "expert"), "tensor"))
    self.assertEqual(reduction_plan, (("fsdp", None), ("expert", None)))

  def test_mixed_compound_dimension_rejects_scatter_before_carrier(self):
    with self.assertRaisesRegex(NotImplementedError, "carrier mesh axes to precede"):
      dense._plan_local_wgrad_partition(
          PartitionSpec(("fsdp", "expert"), None),
          PartitionSpec(("fsdp", "expert"), "tensor"),
          PartitionSpec(("fsdp", "tensor"), None),
          lhs_ndim=2,
          rhs_ndim=2,
          output_ndim=2,
          contracting_dims=((0,), (0,)),
      )


if __name__ == "__main__":
  unittest.main()
