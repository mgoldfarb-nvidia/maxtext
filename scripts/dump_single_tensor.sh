#!/bin/bash
# =============================================================================
# dump_single_tensor.sh - Quick single tensor dump for debugging
#
# Usage: ./dump_single_tensor.sh <tensor_name> <mode> [--no-scp]
#
# Examples:
#   ./dump_single_tensor.sh after_te_permute_x te_pad128
#   ./dump_single_tensor.sh te_all_shards_tokens_per_expert te_nopad
#   ./dump_single_tensor.sh after_mt_permute_x mt
#   ./dump_single_tensor.sh after_ragged_all_to_all_fwd_x te_pad128 --no-scp
#
# Modes:
#   te_pad128 - TE permutation with 128-byte padding
#   te_nopad  - TE permutation without padding
#   mt        - MaxText (MT) permutation
#
# Available tensor names:
#   Tensor data:
#     after_te_permute_x, after_mt_permute_x
#     after_ragged_all_to_all_fwd_x
#     after_te_local_permute_x, after_mt_local_permute_x
#     after_gmm_intermediate_output
#     after_te_local_unpermute, after_mt_local_unpermute
#     after_te_ragged_all_to_all_rev, after_mt_ragged_all_to_all_rev
#     after_te_unpermute_output, after_mt_unpermute_output
#
#   Ragged all-to-all parameters (TE mode only):
#     te_send_sizes, te_recv_sizes, te_input_offsets, te_output_offsets
#     te_all_shards_tokens_per_expert
#
# =============================================================================

set -e

# Configuration
MAXTEXT_DIR="/mnt/tdophung/ptyche-lustre-home/maxtext"
MOE_PY="${MAXTEXT_DIR}/src/MaxText/layers/moe.py"
OUTPUT_BASE_DIR="${MAXTEXT_DIR}/tensor_dumps"
LOCAL_TARGET="tdophung@10.110.69.56:~/Repos/maxtext/data"

EP=2
FSDP=2
STEPS=1

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <tensor_name> <mode> [--no-scp]"
    echo ""
    echo "Modes: te_pad128, te_nopad, mt"
    echo ""
    echo "Run '$0 --help' for more details"
    exit 1
fi

TENSOR_NAME="$1"
MODE="$2"
DO_SCP=true

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    head -40 "$0" | tail -35
    exit 0
fi

if [ "$3" = "--no-scp" ]; then
    DO_SCP=false
fi

# Set mode parameters
case "$MODE" in
    te_pad128)
        USE_TE_PERM="true"
        ALIGN_SIZE="128"
        GLOBAL_MODE="te"
        LOCAL_MODE="te"
        ;;
    te_nopad)
        USE_TE_PERM="true"
        ALIGN_SIZE="0"
        GLOBAL_MODE="te"
        LOCAL_MODE="te"
        ;;
    mt)
        USE_TE_PERM="false"
        ALIGN_SIZE="0"
        GLOBAL_MODE="mt"
        LOCAL_MODE="mt"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'"
        echo "Valid modes: te_pad128, te_nopad, mt"
        exit 1
        ;;
esac

echo "=============================================="
echo "Single Tensor Dump"
echo "=============================================="
echo "Tensor: ${TENSOR_NAME}"
echo "Mode: ${MODE}"
echo "  use_te_permutation=${USE_TE_PERM}"
echo "  te_permutation_align_size=${ALIGN_SIZE}"
echo "  global_mode=${GLOBAL_MODE}, local_mode=${LOCAL_MODE}"
echo "=============================================="

# Update moe.py
echo ""
echo "Updating moe.py..."
sed -i "s/^MOE_DEBUG_DUMP_TENSOR = .*/MOE_DEBUG_DUMP_TENSOR = \"${TENSOR_NAME}\"/" "$MOE_PY"
sed -i "s/^MOE_DEBUG_GLOBAL_PERM_MODE = .*/MOE_DEBUG_GLOBAL_PERM_MODE = \"${GLOBAL_MODE}\"/" "$MOE_PY"
sed -i "s/^MOE_DEBUG_LOCAL_PERM_MODE = .*/MOE_DEBUG_LOCAL_PERM_MODE = \"${LOCAL_MODE}\"/" "$MOE_PY"

# Cleanup function
cleanup() {
    echo ""
    echo "Resetting moe.py..."
    sed -i 's/^MOE_DEBUG_DUMP_TENSOR = .*/MOE_DEBUG_DUMP_TENSOR = None/' "$MOE_PY"
    sed -i 's/^MOE_DEBUG_GLOBAL_PERM_MODE = .*/MOE_DEBUG_GLOBAL_PERM_MODE = None/' "$MOE_PY"
    sed -i 's/^MOE_DEBUG_LOCAL_PERM_MODE = .*/MOE_DEBUG_LOCAL_PERM_MODE = None/' "$MOE_PY"
}
trap cleanup EXIT

# Create output directory
TARGET_DIR="${OUTPUT_BASE_DIR}/${MODE}/${TENSOR_NAME}"
mkdir -p "$TARGET_DIR"

# Run the test
echo ""
echo "Running test..."
cd "$MAXTEXT_DIR"

ADDITIONAL_ARGS="max_target_length=4096 \
sparse_matmul=true \
megablox=false \
capacity_factor=1.0 \
logits_dot_in_fp32=false \
use_ring_of_experts=false \
ici_expert_parallelism=${EP} \
ici_data_parallelism=1 \
use_te_permutation=${USE_TE_PERM} \
te_permutation_align_size=${ALIGN_SIZE} \
max_segments_per_seq=32 \
ici_fsdp_parallelism=${FSDP}"

TOTAL_GPUS=$((EP * FSDP))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1)))

MAXTEXT_DIR=${MAXTEXT_DIR} test-maxtext.sh -b 2 --model-name=mixtral-8x7b \
    --attn-type=cudnn_flash_te --remat-policy=minimal_with_context \
    --steps=${STEPS} --data-parallel=1 --fsdp=${FSDP} \
    --tensor-parallel=1 --tensor-sequence-parallel=1 \
    -a "$ADDITIONAL_ARGS"

# Move tensor files
echo ""
echo "Collecting tensor files..."
cd "$MAXTEXT_DIR"
if ls my_tensor_gpu*.bin 1>/dev/null 2>&1; then
    mv my_tensor_gpu*.bin "$TARGET_DIR/"
    echo "Saved to: ${TARGET_DIR}"
    ls -la "${TARGET_DIR}/"
else
    echo "WARNING: No tensor files found!"
    exit 1
fi

# Optionally send to local
if [ "$DO_SCP" = true ]; then
    echo ""
    echo "Creating archive and sending to local..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_NAME="${MODE}_${TENSOR_NAME}_${TIMESTAMP}.tar.gz"
    
    cd "$OUTPUT_BASE_DIR"
    tar -czvf "$ARCHIVE_NAME" "${MODE}/${TENSOR_NAME}/"
    
    scp "$ARCHIVE_NAME" "${LOCAL_TARGET}/"
    echo "Sent: ${ARCHIVE_NAME}"
fi

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
