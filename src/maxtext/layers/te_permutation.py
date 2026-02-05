# Copyright 2023-2025 Google LLC
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

"""TransformerEngine Permutation Integration for MaxText MoE.

This module provides wrapper functions for TransformerEngine's token dispatch
and combine operations used in Mixture of Experts (MoE) models.

The integration provides:
- token_dispatch: Scatter tokens to their designated experts
- token_combine: Gather tokens back from experts
- sort_chunks_by_index: Reorder chunks for expert parallelism
- Efficient extraction of tokens_per_expert for ragged_all_to_all

Key Design:
- TE primitives use mask-based routing_map, so we convert MaxText's
  index-based top_k_indices to a binary routing mask.
- tokens_per_expert is computed efficiently inside the TE kernels,
  avoiding double iteration over the routing map.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

# Import TransformerEngine permutation primitives
try:
  from transformer_engine.jax.permutation import (
      token_dispatch,
      token_combine,
      sort_chunks_by_index,
  )
  TE_PERMUTATION_AVAILABLE = True
except ImportError:
  TE_PERMUTATION_AVAILABLE = False
  token_dispatch = None
  token_combine = None
  sort_chunks_by_index = None


def check_te_permutation_available():
  """Check if TransformerEngine permutation is available."""
  if not TE_PERMUTATION_AVAILABLE:
    raise ImportError(
        "TransformerEngine permutation is not available. "
        "Please install TransformerEngine with JAX support: "
        "pip install transformer-engine[jax]"
    )


def create_routing_map_from_indices(
    top_k_indices: jnp.ndarray,
    num_experts: int,
) -> jnp.ndarray:
  """Convert top-k expert indices to a binary routing mask.

  The TE permutation primitives expect a routing_map (binary mask) of shape
  [num_tokens, num_experts] where routing_map[i, j] = 1 means token i is
  routed to expert j.

  Args:
    top_k_indices: Expert indices of shape [batch, seq, num_experts_per_tok]
                   or [num_tokens, num_experts_per_tok].
    num_experts: Total number of experts.

  Returns:
    routing_map: Binary mask of shape [num_tokens, num_experts].
  """
  # Flatten to 2D if needed: [num_tokens, num_experts_per_tok]
  original_shape = top_k_indices.shape
  if top_k_indices.ndim == 3:
    num_tokens = original_shape[0] * original_shape[1]
    top_k_indices = top_k_indices.reshape(num_tokens, -1)

  # Create one-hot encoding for each selected expert
  # Shape: [num_tokens, num_experts_per_tok, num_experts]
  one_hot = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)

  # Sum across the top-k dimension to get binary mask
  # Shape: [num_tokens, num_experts]
  routing_map = jnp.sum(one_hot, axis=1)

  return routing_map


def create_dense_probs_from_topk(
    top_k_weights: jnp.ndarray,
    top_k_indices: jnp.ndarray,
    num_experts: int,
) -> jnp.ndarray:
  """Convert top-k weights to dense probability tensor for TE.

  TE's token_dispatch expects probs of shape [num_tokens, num_experts] where
  probs[i, j] = routing probability for token i to expert j (0 if not routed).

  Args:
    top_k_weights: Top-k routing weights of shape [batch, seq, num_experts_per_tok]
                   or [num_tokens, num_experts_per_tok]. These are the softmax
                   probabilities for the selected experts.
    top_k_indices: Expert indices of shape [batch, seq, num_experts_per_tok]
                   or [num_tokens, num_experts_per_tok].
    num_experts: Total number of experts.

  Returns:
    dense_probs: Dense probability tensor of shape [num_tokens, num_experts].
                 Non-selected experts have probability 0.
  """
  # Flatten to 2D if needed
  original_shape = top_k_indices.shape
  if top_k_indices.ndim == 3:
    num_tokens = original_shape[0] * original_shape[1]
    top_k_indices = top_k_indices.reshape(num_tokens, -1)
    top_k_weights = top_k_weights.reshape(num_tokens, -1)
  else:
    num_tokens = top_k_indices.shape[0]

  num_experts_per_tok = top_k_indices.shape[-1]

  # Create one-hot encoding for each selected expert
  # Shape: [num_tokens, num_experts_per_tok, num_experts]
  one_hot = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=top_k_weights.dtype)

  # Scale one-hot by weights: multiply each expert's one-hot by its weight
  # top_k_weights shape: [num_tokens, num_experts_per_tok]
  # Expand for broadcasting: [num_tokens, num_experts_per_tok, 1]
  weights_expanded = top_k_weights[:, :, None]

  # Element-wise multiply: [num_tokens, num_experts_per_tok, num_experts]
  weighted_one_hot = one_hot * weights_expanded

  # Sum across top-k dimension to get dense probs
  # Shape: [num_tokens, num_experts]
  dense_probs = jnp.sum(weighted_one_hot, axis=1)

  return dense_probs


def te_token_dispatch(
    inputs: jnp.ndarray,
    routing_map: jnp.ndarray,
    num_out_tokens: int,
    probs: Optional[jnp.ndarray] = None,
    align_size: Optional[int] = None,
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
]:
  """Dispatch tokens to experts using TransformerEngine kernels.

  This is a wrapper around TE's token_dispatch that also returns
  tokens_per_expert computed efficiently inside the kernel (no double iteration).

  Args:
    inputs: Input tensor of shape [num_tokens, hidden].
    routing_map: Binary routing mask of shape [num_tokens, num_experts].
    num_out_tokens: Number of output tokens (typically num_tokens * top_k).
    probs: Optional routing probabilities for weighted dispatch.
    align_size: Optional alignment for padding (for efficient GEMM).

  Returns:
    output: Permuted tokens of shape [num_out_tokens, hidden].
    permuted_probs: Permuted probabilities (or None).
    row_id_map: Mapping for token_combine.
    pad_offsets: Padding offsets if align_size provided (or None).
    tokens_per_expert: Token counts per expert [num_experts].
                       Without padding: actual counts.
                       With padding: aligned counts.
  """
  check_te_permutation_available()

  # Call TE token_dispatch
  # tokens_per_expert is always returned (actual counts without padding,
  # aligned counts with padding)
  result = token_dispatch(
      inputs,
      routing_map,
      num_out_tokens=num_out_tokens,
      probs=probs,
      align_size=align_size,
  )

  # Defensive unpacking with validation
  if len(result) != 5:
    raise ValueError(
        f"Expected 5 return values from TE token_dispatch, got {len(result)}. "
        f"Types: {[type(r).__name__ for r in result]}"
    )

  output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = result

  # Validate tokens_per_expert is not None (should always be returned by TE >= latest)
  if tokens_per_expert is None:
    raise ValueError(
        "TE token_dispatch returned None for tokens_per_expert. "
        "Please ensure you have the latest TransformerEngine with the fix that "
        "always returns tokens_per_expert. Try: pip install -e /path/to/TransformerEngine"
    )

  return output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert


def te_token_combine(
    inputs: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray] = None,
    pad_offsets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Combine tokens from experts back to original positions.

  Args:
    inputs: Expert outputs of shape [num_out_tokens, hidden].
    row_id_map: Row ID map from te_token_dispatch.
    merging_probs: Optional weights for combining expert outputs.
    pad_offsets: Padding offsets from te_token_dispatch (if used).

  Returns:
    output: Combined tensor of shape [num_tokens, hidden].
  """
  check_te_permutation_available()

  return token_combine(
      inputs,
      row_id_map,
      merging_probs=merging_probs,
      pad_offsets=pad_offsets,
  )


def te_sort_chunks_by_expert(
    inputs: jnp.ndarray,
    tokens_per_chunk: jnp.ndarray,
    sorted_chunk_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sort chunks of tokens according to expert indices.

  This is used after all-to-all to reorder tokens so that all tokens
  for a given local expert are contiguous.

  Args:
    inputs: Input tensor of shape [num_tokens, hidden].
    tokens_per_chunk: Token counts per chunk [num_chunks].
    sorted_chunk_indices: Permutation of chunk indices [num_chunks].

  Returns:
    output: Sorted tensor.
    row_id_map: Map for reversing the sort.
  """
  check_te_permutation_available()

  return sort_chunks_by_index(inputs, tokens_per_chunk, sorted_chunk_indices)


def compute_ragged_all_to_all_params(
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: int,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute ragged_all_to_all parameters from all shards' token counts.

  This is called after an all_gather of tokens_per_expert, so we have
  visibility into what every shard needs to send/receive.

  Args:
    all_shards_tokens_per_expert: Token counts from all shards.
                                  Shape [num_expert_shards, num_experts].
    shard_id: Current shard index.
    num_expert_shards: Number of expert-parallel shards.

  Returns:
    input_offsets: Shape [num_expert_shards].
    send_sizes: Shape [num_expert_shards].
    output_offsets: Shape [num_expert_shards].
    recv_sizes: Shape [num_expert_shards].
  """
  num_experts = all_shards_tokens_per_expert.shape[1]
  local_expert_size = num_experts // num_expert_shards

  # Get this shard's token counts
  # Use dynamic_slice since shard_id may be a traced value (from jax.lax.axis_index)
  local_tokens_per_expert = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(shard_id, 0),
      slice_sizes=(1, num_experts)
  ).squeeze(0)

  # Reshape to [num_expert_shards, local_expert_size]
  local_reshaped = local_tokens_per_expert.reshape(num_expert_shards, local_expert_size)

  # send_sizes[i] = tokens this shard sends to shard i
  send_sizes = jnp.sum(local_reshaped, axis=1)

  # input_offsets: cumulative send sizes
  input_offsets = jnp.concatenate([
      jnp.array([0], dtype=send_sizes.dtype),
      jnp.cumsum(send_sizes)[:-1]
  ])

  # recv_sizes[i] = tokens shard i sends to this shard
  # We need tokens that shard i has for our local experts
  # Our local experts are [shard_id * local_expert_size : (shard_id+1) * local_expert_size]
  local_expert_start = shard_id * local_expert_size

  # Use dynamic_slice since shard_id may be a traced value (from jax.lax.axis_index)
  # Extract columns [local_expert_start : local_expert_start + local_expert_size]
  local_expert_columns = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(0, local_expert_start),
      slice_sizes=(all_shards_tokens_per_expert.shape[0], local_expert_size)
  )

  # For each source shard, sum tokens destined for our experts
  recv_sizes = jnp.sum(local_expert_columns, axis=1)

  # output_offsets: where to place received tokens from each shard
  output_offsets = jnp.concatenate([
      jnp.array([0], dtype=recv_sizes.dtype),
      jnp.cumsum(recv_sizes)[:-1]
  ])

  return input_offsets, send_sizes, output_offsets, recv_sizes
