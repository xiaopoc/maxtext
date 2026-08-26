# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for TE GEMM contracting PartitionSpec handling."""

import importlib
import inspect
import unittest


gemm = importlib.import_module("transformer_engine.jax.cpp_extensions.gemm")


class TeGemmPartitioningTest(unittest.TestCase):

  def test_wgrad_reduction_flag_reaches_default_shardy_callback(self):
    parameters = inspect.signature(gemm.GemmPrimitive.shardy_sharding_rule).parameters
    self.assertIn("infer_contracting_reduction_axes", parameters)
    self.assertIn(15, gemm.GemmPrimitive.impl_static_args)

  def test_scalar_contracting_spec(self):
    self.assertEqual(gemm._shared_contracting_spec(("tensor",), ("tensor",)), "tensor")

  def test_compound_contracting_spec(self):
    self.assertEqual(
        gemm._shared_contracting_spec(
            (("expert", "tensor", "fsdp"),),
            (("expert", "tensor", "fsdp"),),
        ),
        ("expert", "tensor", "fsdp"),
    )

  def test_partial_compound_intersection(self):
    self.assertEqual(
        gemm._shared_contracting_spec(
            (("tensor", "expert", "fsdp"),),
            (("tensor", "fsdp"),),
        ),
        ("tensor", "fsdp"),
    )

  def test_scalar_compound_intersection(self):
    self.assertEqual(
        gemm._shared_contracting_spec((("expert", "tensor"),), ("tensor",)),
        "tensor",
    )

  def test_unmatched_contracting_specs(self):
    self.assertIsNone(
        gemm._shared_contracting_spec((("expert", "tensor"),), ("fsdp",))
    )

  def test_multiple_sharded_contracting_dimensions_remain_unsupported(self):
    with self.assertRaisesRegex(RuntimeError, "Multiple reduce dimension"):
      gemm._shared_contracting_spec(
          (("tensor",), ("expert",)),
          (("tensor",), ("expert",)),
      )

  def test_intersection_preserves_operand_axis_order(self):
    self.assertEqual(
        gemm._intersect_spec_axes(
            ("expert", "tensor", "fsdp"),
            ("fsdp", "tensor"),
        ),
        ("tensor", "fsdp"),
    )

  def test_duplicate_mesh_axes_are_reduced_once(self):
    self.assertEqual(
        gemm._shared_contracting_spec(
            (("tensor", "tensor", "fsdp"),),
            (("fsdp", "tensor", "tensor"),),
        ),
        ("tensor", "fsdp"),
    )

  def test_wgrad_reduction_axes_survive_contracting_dimension_reordering(self):
    self.assertEqual(
        gemm._infer_contracting_reduction_spec(
            (("expert", "fsdp"), None),
            (None, ("expert", "fsdp")),
            ("tensor",),
            (None,),
        ),
        ("expert", "fsdp"),
    )

  def test_wgrad_reduction_excludes_output_sharding_axis(self):
    self.assertEqual(
        gemm._infer_contracting_reduction_spec(
            (("expert", "fsdp", "tensor"),),
            (("expert", "fsdp", "tensor"),),
            ("tensor",),
            (None,),
        ),
        ("expert", "fsdp"),
    )

  def test_wgrad_reduction_ignores_axes_not_shared_by_both_operands(self):
    self.assertEqual(
        gemm._infer_contracting_reduction_spec(
            (("expert", "fsdp"),),
            (("expert", "tensor"),),
            (None,),
            ("tensor",),
        ),
        "expert",
    )


if __name__ == "__main__":
  unittest.main()
