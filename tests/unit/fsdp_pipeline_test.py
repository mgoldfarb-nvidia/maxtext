# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for explicit scanned-layer FSDP pipelining."""

import os
import subprocess
import sys

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.layers import fsdp_pipeline


_NUM_LAYERS = 5


class FsdpPipelineTest(absltest.TestCase):

  def test_layer_pipeline_matches_dense_reference(self):
    env = os.environ.copy()
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    env["MAXTEXT_FSDP_PIPELINE_TEST_CHILD"] = "1"

    result = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    self.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    self.assertIn("FSDP_PIPELINE_CHECKS_PASSED", result.stdout)

  def test_build_gather_plan_removes_only_fsdp_axis(self):
    specs = {
        "row": P("fsdp", None),
        "combined": P(None, ("tensor", "fsdp")),
        "replicated": P(None),
    }

    plan = fsdp_pipeline.build_fsdp_gather_plan(specs)

    self.assertEqual(plan["row"].gather_dimension, 0)
    self.assertEqual(plan["row"].output_spec, P(None, None))
    self.assertEqual(plan["combined"].gather_dimension, 1)
    self.assertEqual(plan["combined"].output_spec, P(None, "tensor"))
    self.assertIsNone(plan["replicated"].gather_dimension)
    self.assertEqual(plan["replicated"].output_spec, specs["replicated"])

  def test_build_gather_plan_preserves_tree(self):
    specs = ({"a": P("fsdp")}, (P(), P(None, "tensor")))

    plan = fsdp_pipeline.build_fsdp_gather_plan(specs)

    self.assertEqual(
        jax.tree.structure(
            plan,
            is_leaf=lambda value: isinstance(value, fsdp_pipeline.FsdpGatherLeaf),
        ),
        jax.tree.structure(specs, is_leaf=lambda value: isinstance(value, P)),
    )

  def test_build_gather_plan_rejects_axis_on_multiple_dimensions(self):
    with self.assertRaisesRegex(ValueError, "at most one tensor dimension"):
      fsdp_pipeline.build_fsdp_gather_plan(P("fsdp", "fsdp"))


def _run_layer_pipeline_checks():
  """Checks pipeline semantics and optimized collective placement."""
  mesh = Mesh(np.asarray(jax.devices()), ("fsdp",), axis_types=(AxisType.Explicit,))
  rules = (("embed", "fsdp"),)
  weights = jnp.arange(_NUM_LAYERS * 4 * 4, dtype=jnp.float32).reshape(_NUM_LAYERS, 4, 4) / 50
  weights += jnp.eye(4)[None]
  inputs = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4) / 10
  sharded_weights = jax.device_put(weights, NamedSharding(mesh, P(None, "fsdp", None)))
  sharded_inputs = jax.device_put(inputs, NamedSharding(mesh, P("fsdp", None)))

  def make_params(value):
    kernel = nnx.Param(value).replace(
        sharding=("layers", "embed"),
        **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 0},
    )
    return nnx.State({"kernel": kernel})

  def layer_fn(carry, layer_vars):
    params, state = layer_vars
    updated_state = nnx.State({"counter": state["counter"].replace(value=state["counter"].get_value() + 1)})
    return carry @ params["kernel"].get_value(), updated_state

  def pipeline(value, layer_inputs):
    output, updated_state = fsdp_pipeline.apply_layer_pipeline(
        layer_fn,
        layer_inputs,
        make_params(value),
        nnx.State({"counter": nnx.BatchStat(jnp.arange(_NUM_LAYERS, dtype=jnp.int32))}),
        length=_NUM_LAYERS,
        mesh=mesh,
        logical_axis_rules=rules,
    )
    return output, updated_state

  def reference(value, layer_inputs):
    for layer_index in range(_NUM_LAYERS):
      layer_inputs = layer_inputs @ value[layer_index]
    return layer_inputs

  with jax.set_mesh(mesh):
    actual, updated_state = pipeline(sharded_weights, sharded_inputs)
    expected = reference(weights, inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_array_equal(
        updated_state["counter"].get_value(),
        jnp.arange(1, _NUM_LAYERS + 1, dtype=jnp.int32),
    )

    actual_grad = jax.grad(lambda value: jnp.sum(pipeline(value, sharded_inputs)[0]))(sharded_weights)
    expected_grad = jax.grad(lambda value: jnp.sum(reference(value, inputs)))(weights)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5)

    lowered_pipeline = jax.jit(lambda value, layer_inputs: pipeline(value, layer_inputs)[0]).lower(
        sharded_weights, sharded_inputs
    )
    stablehlo = lowered_pipeline.as_text()
    assert '_scheduling_group_id = "60"' in stablehlo, stablehlo
    assert '_scheduling_group_id = "61"' in stablehlo, stablehlo
    stablehlo_all_gathers = [line for line in stablehlo.splitlines() if '"stablehlo.all_gather"' in line]
    assert any('_scheduling_group_id = "60"' in line for line in stablehlo_all_gathers), stablehlo_all_gathers
    assert any('_scheduling_group_id = "61"' in line for line in stablehlo_all_gathers), stablehlo_all_gathers
    assert any('_scheduling_group_id = "60"' not in line for line in stablehlo_all_gathers), stablehlo_all_gathers
    assert "scheduling_group =" not in stablehlo, stablehlo
    assert "optimization_barrier" not in stablehlo, stablehlo
    optimized_hlo = lowered_pipeline.compile().as_text()
    all_gathers = [line for line in optimized_hlo.splitlines() if " all-gather(" in line]
    assert any("/while/body/" in line for line in all_gathers), all_gathers
    assert any("/while/body/" not in line for line in all_gathers), all_gathers

    lowered_gradient = jax.jit(jax.grad(lambda value: jnp.sum(pipeline(value, sharded_inputs)[0]))).lower(sharded_weights)
    gradient_stablehlo = lowered_gradient.as_text()
    assert "scheduling_group =" not in gradient_stablehlo, gradient_stablehlo
    assert "optimization_barrier" not in gradient_stablehlo, gradient_stablehlo
    gradient_hlo = lowered_gradient.compile().as_text()
    backward_body = [line for line in gradient_hlo.splitlines() if "/transpose(jvp())/while/body/" in line]
    assert any(" all-gather(" in line for line in backward_body), backward_body
    assert any(" all-reduce(" in line or " reduce-scatter(" in line for line in backward_body), backward_body

  print("FSDP_PIPELINE_CHECKS_PASSED")


if __name__ == "__main__":
  if os.environ.get("MAXTEXT_FSDP_PIPELINE_TEST_CHILD") == "1":
    _run_layer_pipeline_checks()
  else:
    absltest.main()
