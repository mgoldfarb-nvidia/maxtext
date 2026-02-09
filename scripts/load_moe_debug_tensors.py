#!/usr/bin/env python3
"""
Script to load and compare MoE debug tensors dumped using te_inspect_array.

This script helps analyze tensors dumped from moe.py when debugging
EP (Expert Parallelism) and TE/MT permutation differences.

Usage:
    # Auto-calculate shape from config
    python load_moe_debug_tensors.py --dir /path/to/run/dir \
        --config src/MaxText/configs/models/mixtral-8x7b.yml \
        --batch-size 2 --seq-len 4096 --tensor-type output
    
    # Or manually specify shape
    python load_moe_debug_tensors.py --dir /path/to/run/dir --shape 8192,4096 --dtype bfloat16
    
    # Compare tensors from two different runs
    python load_moe_debug_tensors.py --dir1 /path/to/te_run --dir2 /path/to/mt_run \
        --config src/MaxText/configs/models/mixtral-8x7b.yml \
        --batch-size 2 --seq-len 4096 --tensor-type output

Prerequisites:
    1. Cherry-pick Jeremy's PR to TransformerEngine:
       cd /path/to/TransformerEngine
       git fetch origin pull/2651/head:pr-2651
       git cherry-pick pr-2651
       
    2. Set MOE_DEBUG_DUMP_TENSOR in moe.py to the tensor you want to dump
    3. Run your training - tensors will be saved to my_tensor_gpuX.bin
"""

import argparse
import os
import glob
import numpy as np

# Try to import yaml for config parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not available, --config option won't work")

# Try to import JAX for proper dtype handling
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX not available, using numpy with manual dtype mapping")

# Try to import TE's load function
try:
    from transformer_engine.jax.inspect import load_array_dump
    HAS_TE_INSPECT = True
except ImportError:
    HAS_TE_INSPECT = False
    print("Warning: transformer_engine.jax.inspect not available, using numpy fallback")


DTYPE_MAP = {
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': 'bfloat16',  # Special handling needed
    'int32': np.int32,
    'int64': np.int64,
}

# Tensor types and their shape calculations
# Shape depends on where in the MoE pipeline the tensor was captured
TENSOR_TYPES = {
    # Output tensors: (batch * seq, emb_dim)
    'output': ['after_te_unpermute_output', 'after_mt_unpermute_output'],
    # Permuted tensors: (batch * seq * num_experts_per_tok, emb_dim)
    'permuted': [
        'after_te_permute_x', 'after_mt_permute_x',
        'after_ragged_all_to_all_fwd_x',
        'after_te_local_permute_x', 'after_mt_local_permute_x',
        'after_gmm_intermediate_output',
        'after_te_local_unpermute', 'after_mt_local_unpermute',
        'after_te_ragged_all_to_all_rev', 'after_mt_ragged_all_to_all_rev',
    ],
}


def load_config(config_path: str) -> dict:
    """Load a MaxText YAML config file."""
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required to load config files. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_shape_from_config(
    config: dict,
    batch_size: int,
    seq_len: int,
    tensor_type: str,
    ep: int = 1,
) -> tuple:
    """Calculate tensor shape based on config and tensor type.
    
    Args:
        config: MaxText config dict
        batch_size: Batch size used in training
        seq_len: Sequence length (max_target_length)
        tensor_type: 'output' or 'permuted'
        ep: Expert parallelism (affects local tensor sizes)
    
    Returns:
        Tuple representing the tensor shape
    """
    emb_dim = config.get('base_emb_dim', 4096)
    num_experts_per_tok = config.get('num_experts_per_tok', 2)
    
    if tensor_type == 'output':
        # Output shape: (batch * seq, emb_dim)
        first_dim = batch_size * seq_len
    elif tensor_type == 'permuted':
        # Permuted shape: (batch * seq * num_experts_per_tok, emb_dim)
        # Note: With EP > 1, this is divided across shards
        first_dim = batch_size * seq_len * num_experts_per_tok
        if ep > 1:
            # Each shard sees a portion of the tokens
            # This is approximate - actual size depends on routing
            print(f"  Note: With EP={ep}, actual shape may vary based on token routing")
    else:
        raise ValueError(f"Unknown tensor_type: {tensor_type}. Must be 'output' or 'permuted'")
    
    return (first_dim, emb_dim)


def parse_shape(shape_str: str) -> tuple:
    """Parse shape string like '8192,4096' into tuple (8192, 4096)."""
    return tuple(int(x.strip()) for x in shape_str.split(','))


def load_tensor_numpy(filepath: str, shape: tuple, dtype_str: str) -> np.ndarray:
    """Load tensor using numpy (fallback when TE inspect not available)."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if dtype_str == 'bfloat16':
        # bfloat16 needs special handling - read as uint16 then view
        arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        print(f"  Note: bfloat16 loaded as uint16 (raw bits), use JAX for proper handling")
        return arr
    else:
        dtype = DTYPE_MAP.get(dtype_str, np.float32)
        return np.frombuffer(data, dtype=dtype).reshape(shape)


def load_tensor(filepath: str, shape: tuple, dtype_str: str):
    """Load tensor from binary file."""
    if HAS_TE_INSPECT and HAS_JAX:
        dtype = getattr(jnp, dtype_str)
        return load_array_dump(filepath, shape, dtype)
    else:
        return load_tensor_numpy(filepath, shape, dtype_str)


def find_gpu_files(directory: str, pattern: str = "my_tensor_gpu*.bin") -> list:
    """Find all GPU tensor dump files in a directory."""
    search_path = os.path.join(directory, pattern)
    files = sorted(glob.glob(search_path))
    return files


def compute_stats(arr) -> dict:
    """Compute statistics for an array."""
    arr_f32 = arr.astype(np.float32) if hasattr(arr, 'astype') else arr
    return {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'sum': float(np.sum(arr_f32)),
        'mean': float(np.mean(arr_f32)),
        'std': float(np.std(arr_f32)),
        'min': float(np.min(arr_f32)),
        'max': float(np.max(arr_f32)),
        'has_nan': bool(np.any(np.isnan(arr_f32))),
        'has_inf': bool(np.any(np.isinf(arr_f32))),
    }


def print_stats(name: str, stats: dict):
    """Print tensor statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {stats['shape']}, Dtype: {stats['dtype']}")
    print(f"  Sum: {stats['sum']:.6f}, Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    if stats['has_nan'] or stats['has_inf']:
        print(f"  WARNING: has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")


def compare_tensors(arr1, arr2, name1: str = "Tensor1", name2: str = "Tensor2"):
    """Compare two tensors and print differences."""
    arr1_f32 = arr1.astype(np.float32)
    arr2_f32 = arr2.astype(np.float32)
    
    diff = arr1_f32 - arr2_f32
    abs_diff = np.abs(diff)
    
    print(f"\n=== Comparison: {name1} vs {name2} ===")
    print(f"  Max absolute diff: {np.max(abs_diff):.6e}")
    print(f"  Mean absolute diff: {np.mean(abs_diff):.6e}")
    print(f"  Sum of diffs: {np.sum(diff):.6e}")
    
    # Relative difference (avoid divide by zero)
    denom = np.maximum(np.abs(arr1_f32), np.abs(arr2_f32))
    denom = np.where(denom == 0, 1.0, denom)
    rel_diff = abs_diff / denom
    print(f"  Max relative diff: {np.max(rel_diff):.6e}")
    print(f"  Mean relative diff: {np.mean(rel_diff):.6e}")
    
    # Check if tensors are close
    atol = 1e-5
    rtol = 1e-3
    close = np.allclose(arr1_f32, arr2_f32, atol=atol, rtol=rtol)
    print(f"  Are close (atol={atol}, rtol={rtol}): {close}")
    
    return diff


def main():
    parser = argparse.ArgumentParser(
        description="Load and analyze MoE debug tensors dumped using te_inspect_array",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-calculate shape from MaxText config (recommended)
  python load_moe_debug_tensors.py --dir ./run1 \\
      --config src/MaxText/configs/models/mixtral-8x7b.yml \\
      --batch-size 2 --seq-len 4096 --tensor-type output
  
  # For permuted tensors (after permute, before unpermute)
  python load_moe_debug_tensors.py --dir ./run1 \\
      --config src/MaxText/configs/models/mixtral-8x7b.yml \\
      --batch-size 2 --seq-len 4096 --tensor-type permuted
  
  # Manual shape specification
  python load_moe_debug_tensors.py --dir ./run1 --shape 8192,4096 --dtype bfloat16
  
  # Compare tensors between TE and MT runs
  python load_moe_debug_tensors.py --dir1 ./te_run --dir2 ./mt_run \\
      --config src/MaxText/configs/models/mixtral-8x7b.yml \\
      --batch-size 2 --seq-len 4096 --tensor-type output

Tensor types:
  - 'output': after_te_unpermute_output, after_mt_unpermute_output
              Shape: (batch * seq, emb_dim)
  
  - 'permuted': after_te_permute_x, after_mt_permute_x, after_gmm_intermediate_output, etc.
                Shape: (batch * seq * num_experts_per_tok, emb_dim)

Available tensor names to dump in moe.py (set MOE_DEBUG_DUMP_TENSOR):
  - after_te_permute_x, after_mt_permute_x
  - after_ragged_all_to_all_fwd_x
  - after_te_local_permute_x, after_mt_local_permute_x
  - after_gmm_intermediate_output
  - after_te_local_unpermute, after_mt_local_unpermute
  - after_te_ragged_all_to_all_rev, after_mt_ragged_all_to_all_rev
  - after_te_unpermute_output, after_mt_unpermute_output
        """
    )
    
    # Input sources
    parser.add_argument('--dir', type=str, help='Directory containing tensor dump files')
    parser.add_argument('--dir1', type=str, help='First directory for comparison')
    parser.add_argument('--dir2', type=str, help='Second directory for comparison')
    parser.add_argument('--file1', type=str, help='First file for direct comparison')
    parser.add_argument('--file2', type=str, help='Second file for direct comparison')
    
    # Shape specification - either manual or via config
    shape_group = parser.add_argument_group('Shape specification (use either --shape OR --config)')
    shape_group.add_argument('--shape', type=str,
                        help='Manual tensor shape as comma-separated values (e.g., "8192,4096")')
    shape_group.add_argument('--config', type=str,
                        help='Path to MaxText model config YAML file')
    shape_group.add_argument('--batch-size', type=int, default=2,
                        help='Batch size used in training (default: 2)')
    shape_group.add_argument('--seq-len', type=int, default=4096,
                        help='Sequence length / max_target_length (default: 4096)')
    shape_group.add_argument('--tensor-type', type=str, default='output',
                        choices=['output', 'permuted'],
                        help='Type of tensor: "output" or "permuted" (default: output)')
    shape_group.add_argument('--ep', type=int, default=1,
                        help='Expert parallelism (for shape calculation notes, default: 1)')
    
    # Other options
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16', 'int32', 'int64'],
                        help='Tensor dtype (default: bfloat16)')
    parser.add_argument('--pattern', type=str, default='my_tensor_gpu*.bin',
                        help='Glob pattern for finding tensor files (default: my_tensor_gpu*.bin)')
    
    args = parser.parse_args()
    
    # Determine shape
    if args.shape:
        shape = parse_shape(args.shape)
        print(f"Using manual shape: {shape}")
    elif args.config:
        if not HAS_YAML:
            print("Error: PyYAML is required for --config. Install with: pip install pyyaml")
            print("Or use --shape to specify the shape manually.")
            return
        config = load_config(args.config)
        shape = calculate_shape_from_config(
            config, 
            args.batch_size, 
            args.seq_len, 
            args.tensor_type,
            args.ep
        )
        print(f"Calculated shape from config:")
        print(f"  Config: {args.config}")
        print(f"  emb_dim: {config.get('base_emb_dim', 4096)}")
        print(f"  num_experts_per_tok: {config.get('num_experts_per_tok', 2)}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  seq_len: {args.seq_len}")
        print(f"  tensor_type: {args.tensor_type}")
        print(f"  -> Shape: {shape}")
    else:
        print("Error: Must specify either --shape or --config")
        parser.print_help()
        return
    
    print(f"Tensor shape: {shape}")
    print(f"Tensor dtype: {args.dtype}")
    print(f"TE inspect available: {HAS_TE_INSPECT}")
    print(f"JAX available: {HAS_JAX}")
    
    # Mode 1: Load from single directory
    if args.dir:
        files = find_gpu_files(args.dir, args.pattern)
        if not files:
            print(f"No files found matching {args.pattern} in {args.dir}")
            return
        
        print(f"\nFound {len(files)} GPU files in {args.dir}")
        tensors = {}
        for f in files:
            gpu_id = os.path.basename(f).replace('my_tensor_gpu', '').replace('.bin', '')
            print(f"\nLoading {f}...")
            arr = load_tensor(f, shape, args.dtype)
            tensors[gpu_id] = arr
            print_stats(f"GPU {gpu_id}", compute_stats(arr))
        
        # Compare GPUs if multiple
        if len(tensors) > 1:
            gpu_ids = sorted(tensors.keys())
            for i, gid1 in enumerate(gpu_ids):
                for gid2 in gpu_ids[i+1:]:
                    compare_tensors(tensors[gid1], tensors[gid2],
                                    f"GPU {gid1}", f"GPU {gid2}")
    
    # Mode 2: Compare two directories
    elif args.dir1 and args.dir2:
        files1 = find_gpu_files(args.dir1, args.pattern)
        files2 = find_gpu_files(args.dir2, args.pattern)
        
        print(f"\nFound {len(files1)} files in {args.dir1}")
        print(f"Found {len(files2)} files in {args.dir2}")
        
        # Match files by GPU ID
        for f1 in files1:
            basename = os.path.basename(f1)
            f2 = os.path.join(args.dir2, basename)
            if os.path.exists(f2):
                print(f"\n{'='*60}")
                print(f"Comparing {basename}")
                arr1 = load_tensor(f1, shape, args.dtype)
                arr2 = load_tensor(f2, shape, args.dtype)
                print_stats(f"Dir1 ({args.dir1})", compute_stats(arr1))
                print_stats(f"Dir2 ({args.dir2})", compute_stats(arr2))
                compare_tensors(arr1, arr2, "Dir1", "Dir2")
    
    # Mode 3: Compare two specific files
    elif args.file1 and args.file2:
        print(f"\nComparing:")
        print(f"  File1: {args.file1}")
        print(f"  File2: {args.file2}")
        
        arr1 = load_tensor(args.file1, shape, args.dtype)
        arr2 = load_tensor(args.file2, shape, args.dtype)
        
        print_stats("File1", compute_stats(arr1))
        print_stats("File2", compute_stats(arr2))
        compare_tensors(arr1, arr2, "File1", "File2")
    
    else:
        parser.print_help()
        print("\nError: Must specify --dir, --dir1/--dir2, or --file1/--file2")


if __name__ == "__main__":
    main()
