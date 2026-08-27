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

"""Explicit layer-level FSDP pipelining for scanned NNX decoders."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import sharding


_FSDP_AXIS = "fsdp"
_FORWARD_PREFETCH_GROUP_BASE = 60


@dataclass(frozen=True)
class FsdpGatherLeaf:
  """Collective plan for one parameter leaf."""

  input_spec: P
  output_spec: P
  gather_dimension: int | None


def _partition_axes(partition: Any) -> tuple[str, ...]:
  if partition is None or partition == P.UNCONSTRAINED:
    return ()
  if isinstance(partition, str):
    return (partition,)
  if isinstance(partition, tuple):
    return partition
  raise ValueError(f"Unsupported partition entry {partition!r}.")


def _without_axis(partition: Any, axis_name: str) -> Any:
  remaining = tuple(axis for axis in _partition_axes(partition) if axis != axis_name)
  if not remaining:
    return None
  if len(remaining) == 1:
    return remaining[0]
  return remaining


def _build_gather_leaf(input_spec: P, axis_name: str) -> FsdpGatherLeaf:
  """Builds the gather plan for one parameter leaf."""
  gather_dimensions = [
      dimension for dimension, partition in enumerate(input_spec) if axis_name in _partition_axes(partition)
  ]
  if not gather_dimensions:
    return FsdpGatherLeaf(input_spec, input_spec, None)
  if len(gather_dimensions) != 1:
    raise ValueError(f"Mesh axis {axis_name!r} must shard at most one tensor dimension; got {input_spec}.")

  gather_dimension = gather_dimensions[0]
  output_partitions = list(input_spec)
  output_partitions[gather_dimension] = _without_axis(output_partitions[gather_dimension], axis_name)
  output_spec = P(
      *output_partitions,
      unreduced=input_spec.unreduced,
      reduced=input_spec.reduced,
  )
  return FsdpGatherLeaf(input_spec, output_spec, gather_dimension)


def build_fsdp_gather_plan(physical_specs: Any, axis_name: str = _FSDP_AXIS) -> Any:
  """Builds an immutable per-leaf all-gather plan from physical shardings."""
  return jax.tree.map(
      lambda spec: _build_gather_leaf(spec, axis_name),
      physical_specs,
      is_leaf=lambda value: isinstance(value, P),
  )


def _logical_spec(variable: nnx.Variable) -> P:
  """Returns a parameter's logical partition specification."""
  metadata = variable.get_metadata()
  logical = metadata.get("out_sharding")
  if logical is None:
    logical = metadata.get("sharding_names")
  if logical is None:
    logical = metadata.get("sharding")
  if logical is None:
    return P()
  if isinstance(logical, P):
    return logical
  if isinstance(logical, str):
    return P(logical)
  if isinstance(logical, (tuple, list)):
    return P(*logical)
  if hasattr(logical, "spec"):
    return logical.spec
  raise ValueError(f"Unsupported NNX parameter sharding metadata {logical!r}.")


def _parameter_specs(params: Any, mesh: jax.sharding.Mesh, logical_axis_rules: Any) -> Any:
  logical_specs = jax.tree.map(
      _logical_spec,
      params,
      is_leaf=lambda value: isinstance(value, nnx.Variable),
  )
  return jax.tree.map(
      lambda spec: sharding.logical_to_mesh_axes(spec, mesh, rules=logical_axis_rules),
      logical_specs,
      is_leaf=lambda value: isinstance(value, P),
  )


def _parameter_values(params: Any) -> Any:
  return jax.tree.map(
      lambda variable: variable.get_value(),
      params,
      is_leaf=lambda value: isinstance(value, nnx.Variable),
  )


def _replace_parameter_values(params: Any, values: Any) -> Any:
  return jax.tree.map(
      lambda variable, value: variable.replace(value=value),
      params,
      values,
      is_leaf=lambda value: isinstance(value, nnx.Variable),
  )


def _all_gather_params(params: Any, mesh: jax.sharding.Mesh, logical_axis_rules: Any) -> Any:
  """Gathers one layer's FSDP-sharded parameter values."""
  physical_specs = _parameter_specs(params, mesh, logical_axis_rules)
  plan = build_fsdp_gather_plan(physical_specs)
  plan_leaves = jax.tree.leaves(plan, is_leaf=lambda value: isinstance(value, FsdpGatherLeaf))
  if not any(leaf.gather_dimension is not None for leaf in plan_leaves):
    raise ValueError("fsdp_schedule='layer_pipeline' found no parameters sharded over the 'fsdp' mesh axis.")

  def gather_leaf(value, leaf):
    if leaf.gather_dimension is None:
      return value
    return jax.lax.all_gather(
        value,
        axis_name=_FSDP_AXIS,
        axis=leaf.gather_dimension,
        tiled=True,
        to="invarying",
    )

  input_specs = jax.tree.map(
      lambda leaf: leaf.input_spec,
      plan,
      is_leaf=lambda value: isinstance(value, FsdpGatherLeaf),
  )
  output_specs = jax.tree.map(
      lambda leaf: leaf.output_spec,
      plan,
      is_leaf=lambda value: isinstance(value, FsdpGatherLeaf),
  )

  def gather_values(values):
    return jax.tree.map(
        gather_leaf,
        values,
        plan,
        is_leaf=lambda value: isinstance(value, FsdpGatherLeaf),
    )

  gathered_values = jax.shard_map(
      gather_values,
      mesh=mesh,
      in_specs=(input_specs,),
      out_specs=output_specs,
      check_vma=True,
  )(_parameter_values(params))
  return _replace_parameter_values(params, gathered_values)


def _reduce_scatter_param_values(grads: Any, params: Any, mesh: jax.sharding.Mesh, logical_axis_rules: Any) -> Any:
  """Reshards one layer's gathered parameter gradients."""
  physical_specs = _parameter_specs(params, mesh, logical_axis_rules)
  plan = build_fsdp_gather_plan(physical_specs)
  grad_values = _parameter_values(grads)

  def reshard_leaf(value, leaf):
    if leaf.gather_dimension is None:
      return value
    return jax.reshard(value, NamedSharding(mesh, leaf.input_spec))

  return jax.tree.map(
      reshard_leaf,
      grad_values,
      plan,
      is_leaf=lambda value: isinstance(value, FsdpGatherLeaf),
  )


def _layer_slice(stacked: Any, layer_index: Any) -> Any:
  current = jax.tree.map(
      lambda value: jax.lax.dynamic_index_in_dim(value, layer_index, axis=0, keepdims=False),
      stacked,
  )
  return maxtext_utils_nnx.nnx_remove_scan_axis(current, "layers")


def _stack_last(prefix: Any, last: Any) -> Any:
  return jax.tree.map(
      lambda prefix_value, last_value: jnp.concatenate(
          (prefix_value, jnp.expand_dims(last_value, 0)),
          axis=0,
      ),
      prefix,
      last,
  )


def _stack_sequence(values: list[Any]) -> Any:
  return jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)


def _insert_layer(stacked: Any, current: Any, layer_index: Any) -> Any:
  return jax.tree.map(
      lambda stacked_value, current_value: jax.lax.dynamic_update_index_in_dim(
          stacked_value,
          current_value,
          layer_index,
          axis=0,
      ),
      stacked,
      current,
  )


def _empty_layer_stack(example: Any, length: int) -> Any:
  return jax.tree.map(
      lambda value: jnp.zeros((length, *value.shape), dtype=value.dtype),
      example,
  )


def _forward_layer_pipeline(
    layer_fn: Callable[[Any, tuple[Any, Any]], tuple[Any, Any]],
    initial_carry: Any,
    params: Any,
    state: Any,
    *,
    length: int,
    mesh: jax.sharding.Mesh,
    logical_axis_rules: Any,
) -> tuple[Any, Any, Any]:
  """Runs the forward one-layer-ahead pipeline."""
  first_params = _all_gather_params(_layer_slice(params, 0), mesh, logical_axis_rules)

  current_carry = initial_carry
  current_params = first_params
  prefix_states = []
  layer_inputs = _empty_layer_stack(initial_carry, length)
  for layer_index in range(length - 1):
    next_sharded_params = _layer_slice(params, layer_index + 1)
    current_state = _layer_slice(state, layer_index)
    with xla_metadata.set_xla_metadata(
        _scheduling_group_id=_FORWARD_PREFETCH_GROUP_BASE + layer_index,
        keep_original_sequence_order_in_group=True,
    ):
      next_params = _all_gather_params(next_sharded_params, mesh, logical_axis_rules)
      next_carry, updated_state = layer_fn(current_carry, (current_params, current_state))
    prefix_states.append(updated_state)
    layer_inputs = _insert_layer(layer_inputs, current_carry, layer_index)
    current_carry = next_carry
    current_params = next_params

  last_carry, last_state = layer_fn(current_carry, (current_params, _layer_slice(state, length - 1)))
  layer_inputs = _insert_layer(layer_inputs, current_carry, length - 1)
  return (
      last_carry,
      _stack_last(_stack_sequence(prefix_states), last_state),
      layer_inputs,
  )


def _backward_layer_pipeline(
    layer_fn: Callable[[Any, tuple[Any, Any]], tuple[Any, Any]],
    params: Any,
    state: Any,
    layer_inputs: Any,
    output_cotangent: Any,
    state_cotangent: Any,
    *,
    length: int,
    mesh: jax.sharding.Mesh,
    logical_axis_rules: Any,
) -> tuple[Any, Any]:
  """Runs the reverse pipeline with on-demand parameter gathers."""

  def layer_vjp(layer_index, current_params, carry_cotangent):
    current_input = _layer_slice(layer_inputs, layer_index)
    current_state = _layer_slice(state, layer_index)

    def apply_layer(layer_input, layer_params, layer_state):
      return layer_fn(layer_input, (layer_params, layer_state))

    _, pullback = jax.vjp(apply_layer, current_input, current_params, current_state)
    input_cotangent, params_cotangent, _ = pullback((carry_cotangent, _layer_slice(state_cotangent, layer_index)))
    return input_cotangent, params_cotangent

  def prefetch_and_compute(layer_index, current_params, carry_cotangent, next_sharded_params):
    next_params = _all_gather_params(next_sharded_params, mesh, logical_axis_rules)
    carry_cotangent, pending_grad = layer_vjp(layer_index, current_params, carry_cotangent)
    return carry_cotangent, next_params, pending_grad

  last_index = length - 1
  last_params = _all_gather_params(_layer_slice(params, last_index), mesh, logical_axis_rules)
  carry_cotangent, current_params, pending_grad = prefetch_and_compute(
      last_index,
      last_params,
      output_cotangent,
      _layer_slice(params, last_index - 1),
  )
  all_grad_values = jax.tree.map(jnp.zeros_like, _parameter_values(params))

  def scan_body(pipeline_carry, layer_index):
    carry_cotangent, current_params, pending_grad, all_grad_values = pipeline_carry
    reduced_grad = _reduce_scatter_param_values(
        pending_grad,
        _layer_slice(params, layer_index + 1),
        mesh,
        logical_axis_rules,
    )
    all_grad_values = _insert_layer(all_grad_values, reduced_grad, layer_index + 1)
    carry_cotangent, next_params, pending_grad = prefetch_and_compute(
        layer_index,
        current_params,
        carry_cotangent,
        _layer_slice(params, layer_index - 1),
    )
    return (carry_cotangent, next_params, pending_grad, all_grad_values), None

  layer_indices = jnp.arange(length - 2, 0, -1, dtype=jnp.int32)
  (carry_cotangent, first_params, pending_grad, all_grad_values), _ = jax.lax.scan(
      scan_body,
      (carry_cotangent, current_params, pending_grad, all_grad_values),
      layer_indices,
  )

  reduced_grad = _reduce_scatter_param_values(
      pending_grad,
      _layer_slice(params, 1),
      mesh,
      logical_axis_rules,
  )
  all_grad_values = _insert_layer(all_grad_values, reduced_grad, 1)
  carry_cotangent, first_grad = layer_vjp(0, first_params, carry_cotangent)
  reduced_first_grad = _reduce_scatter_param_values(
      first_grad,
      _layer_slice(params, 0),
      mesh,
      logical_axis_rules,
  )
  all_grad_values = _insert_layer(all_grad_values, reduced_first_grad, 0)
  return carry_cotangent, _replace_parameter_values(params, all_grad_values)


def apply_layer_pipeline(
    layer_fn: Callable[[Any, tuple[Any, Any]], tuple[Any, Any]],
    initial_carry: Any,
    params: Any,
    state: Any,
    *,
    length: int,
    mesh: jax.sharding.Mesh,
    logical_axis_rules: Any,
) -> tuple[Any, Any]:
  """Runs a scanned layer stack with one-layer-ahead FSDP weight prefetching.

  The collective is outside ``layer_fn`` so a checkpointed layer computation
  does not rematerialize communication. The custom backward pass explicitly
  gathers weights and reduces their gradients.
  """
  if length < 2:
    raise ValueError("fsdp_schedule='layer_pipeline' requires at least two scanned layers.")

  @jax.custom_vjp
  def run(layer_input, stacked_params, stacked_state):
    output, updated_state, _ = _forward_layer_pipeline(
        layer_fn,
        layer_input,
        stacked_params,
        stacked_state,
        length=length,
        mesh=mesh,
        logical_axis_rules=logical_axis_rules,
    )
    return output, updated_state

  def run_fwd(layer_input, stacked_params, stacked_state):
    output, updated_state, layer_inputs = _forward_layer_pipeline(
        layer_fn,
        layer_input,
        stacked_params,
        stacked_state,
        length=length,
        mesh=mesh,
        logical_axis_rules=logical_axis_rules,
    )
    return (output, updated_state), (stacked_params, stacked_state, layer_inputs)

  def run_bwd(residuals, cotangents):
    stacked_params, stacked_state, layer_inputs = residuals
    output_cotangent, state_cotangent = cotangents
    input_cotangent, params_cotangent = _backward_layer_pipeline(
        layer_fn,
        stacked_params,
        stacked_state,
        layer_inputs,
        output_cotangent,
        state_cotangent,
        length=length,
        mesh=mesh,
        logical_axis_rules=logical_axis_rules,
    )
    return input_cotangent, params_cotangent, None

  run.defvjp(run_fwd, run_bwd)
  return run(initial_carry, params, state)
