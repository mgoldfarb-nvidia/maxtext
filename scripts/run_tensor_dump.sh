#!/bin/bash
# =============================================================================
# run_tensor_dump.sh - Automated tensor dumping for MoE debugging
#
# This script loops through different configurations and tensor names,
# dumps tensors, organizes them into directories, and sends them to a local machine.
#
# Usage: ./run_tensor_dump.sh [--dry-run] [--scenario SCENARIO] [--tensor TENSOR]
#
# Options:
#   --dry-run     Print what would be done without executing
#   --scenario    Run only a specific scenario (te_pad128, te_nopad, mt)
#   --tensor      Dump only a specific tensor name
#
# =============================================================================

set -e

# Configuration
MAXTEXT_DIR="/mnt/tdophung/ptyche-lustre-home/maxtext"
MOE_PY="${MAXTEXT_DIR}/src/MaxText/layers/moe.py"
OUTPUT_BASE_DIR="${MAXTEXT_DIR}/tensor_dumps"
LOCAL_TARGET="tdophung@10.110.69.56:~/Repos/maxtext/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="tensor_dump_${TIMESTAMP}.tar.gz"

# Test configuration
EP=2
FSDP=2
STEPS=1  # Only need 1 step for tensor dumps

# Parse arguments
DRY_RUN=false
ONLY_SCENARIO=""
ONLY_TENSOR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --scenario)
            ONLY_SCENARIO="$2"
            shift 2
            ;;
        --tensor)
            ONLY_TENSOR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Define scenarios: name|use_te_permutation|te_permutation_align_size|global_mode|local_mode
# Note: global_mode and local_mode should match (te/te or mt/mt) for valid runs
SCENARIOS=(
    "te_pad128|true|128|te|te"
    "te_nopad|true|0|te|te"
    "mt|false|0|mt|mt"
)

# Define tensor names to dump for each mode type
# TE mode tensors
TE_TENSORS=(
    "after_te_permute_x"
    "after_ragged_all_to_all_fwd_x"
    "after_te_local_permute_x"
    "after_gmm_intermediate_output"
    "after_te_local_unpermute"
    "after_te_ragged_all_to_all_rev"
    "after_te_unpermute_output"
)

# MT mode tensors
MT_TENSORS=(
    "after_mt_permute_x"
    "after_ragged_all_to_all_fwd_x"
    "after_mt_local_permute_x"
    "after_gmm_intermediate_output"
    "after_mt_local_unpermute"
    "after_mt_ragged_all_to_all_rev"
    "after_mt_unpermute_output"
)

# Ragged all-to-all parameters (only for TE mode)
RAGGED_PARAMS=(
    "te_send_sizes"
    "te_recv_sizes"
    "te_input_offsets"
    "te_output_offsets"
    "te_all_shards_tokens_per_expert"
)

# Function to update MOE_DEBUG_DUMP_TENSOR in moe.py
update_dump_tensor() {
    local tensor_name="$1"
    echo "  Setting MOE_DEBUG_DUMP_TENSOR = \"${tensor_name}\""
    if [ "$DRY_RUN" = false ]; then
        sed -i "s/^MOE_DEBUG_DUMP_TENSOR = .*/MOE_DEBUG_DUMP_TENSOR = \"${tensor_name}\"/" "$MOE_PY"
    fi
}

# Function to update perm mode variables in moe.py
update_perm_modes() {
    local global_mode="$1"
    local local_mode="$2"
    echo "  Setting MOE_DEBUG_GLOBAL_PERM_MODE = \"${global_mode}\", MOE_DEBUG_LOCAL_PERM_MODE = \"${local_mode}\""
    if [ "$DRY_RUN" = false ]; then
        sed -i "s/^MOE_DEBUG_GLOBAL_PERM_MODE = .*/MOE_DEBUG_GLOBAL_PERM_MODE = \"${global_mode}\"/" "$MOE_PY"
        sed -i "s/^MOE_DEBUG_LOCAL_PERM_MODE = .*/MOE_DEBUG_LOCAL_PERM_MODE = \"${local_mode}\"/" "$MOE_PY"
    fi
}

# Function to reset moe.py to default (no debugging)
reset_moe_py() {
    echo "Resetting moe.py debug settings to defaults..."
    if [ "$DRY_RUN" = false ]; then
        sed -i 's/^MOE_DEBUG_DUMP_TENSOR = .*/MOE_DEBUG_DUMP_TENSOR = None/' "$MOE_PY"
        sed -i 's/^MOE_DEBUG_GLOBAL_PERM_MODE = .*/MOE_DEBUG_GLOBAL_PERM_MODE = None/' "$MOE_PY"
        sed -i 's/^MOE_DEBUG_LOCAL_PERM_MODE = .*/MOE_DEBUG_LOCAL_PERM_MODE = None/' "$MOE_PY"
    fi
}

# Function to run the test and collect tensors
run_test() {
    local use_te_perm="$1"
    local align_size="$2"
    
    echo "  Running test with use_te_permutation=${use_te_perm}, te_permutation_align_size=${align_size}"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would run test_perm.sh"
        return 0
    fi
    
    # Create a temporary test script with the right settings
    local TMP_SCRIPT=$(mktemp)
    cat > "$TMP_SCRIPT" << EOF
#!/bin/bash
EP=${EP}
FSDP=${FSDP}

ADDITIONAL_ARGS="\\
max_target_length=4096 \\
sparse_matmul=true \\
megablox=false \\
capacity_factor=1.0 \\
logits_dot_in_fp32=false \\
use_ring_of_experts=false \\
ici_expert_parallelism=\$EP \\
ici_data_parallelism=1 \\
use_te_permutation=${use_te_perm} \\
te_permutation_align_size=${align_size} \\
max_segments_per_seq=32 \\
ici_fsdp_parallelism=\$FSDP
"

TOTAL_GPUS=\$((EP * FSDP))
export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$((TOTAL_GPUS - 1)))

MAXTEXT_DIR=${MAXTEXT_DIR} test-maxtext.sh -b 2 --model-name=mixtral-8x7b --attn-type=cudnn_flash_te --remat-policy=minimal_with_context --steps=${STEPS} --data-parallel=1 --fsdp=\$FSDP --tensor-parallel=1 --tensor-sequence-parallel=1 -a "\$ADDITIONAL_ARGS"
EOF
    chmod +x "$TMP_SCRIPT"
    
    # Run the test
    cd "$MAXTEXT_DIR"
    bash "$TMP_SCRIPT" 2>&1 | tee "${OUTPUT_BASE_DIR}/run_log.txt"
    local exit_code=${PIPESTATUS[0]}
    
    rm -f "$TMP_SCRIPT"
    return $exit_code
}

# Function to collect and move tensor files
collect_tensors() {
    local scenario_name="$1"
    local tensor_name="$2"
    
    local target_dir="${OUTPUT_BASE_DIR}/${scenario_name}/${tensor_name}"
    echo "  Collecting tensors to ${target_dir}"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would move my_tensor_gpu*.bin to ${target_dir}"
        return 0
    fi
    
    mkdir -p "$target_dir"
    
    # Move tensor files (they're created in the current working directory)
    cd "$MAXTEXT_DIR"
    if ls my_tensor_gpu*.bin 1>/dev/null 2>&1; then
        mv my_tensor_gpu*.bin "$target_dir/"
        echo "  Moved $(ls ${target_dir}/my_tensor_gpu*.bin | wc -l) tensor files"
    else
        echo "  WARNING: No tensor files found!"
    fi
}

# Main script
echo "=============================================="
echo "MoE Tensor Dump Script"
echo "=============================================="
echo "MAXTEXT_DIR: ${MAXTEXT_DIR}"
echo "OUTPUT_BASE_DIR: ${OUTPUT_BASE_DIR}"
echo "TIMESTAMP: ${TIMESTAMP}"
echo "DRY_RUN: ${DRY_RUN}"
if [ -n "$ONLY_SCENARIO" ]; then
    echo "ONLY_SCENARIO: ${ONLY_SCENARIO}"
fi
if [ -n "$ONLY_TENSOR" ]; then
    echo "ONLY_TENSOR: ${ONLY_TENSOR}"
fi
echo "=============================================="

# Create output directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$OUTPUT_BASE_DIR"
fi

# Trap to reset moe.py on exit
trap reset_moe_py EXIT

# Loop through scenarios
for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r scenario_name use_te_perm align_size global_mode local_mode <<< "$scenario"
    
    # Skip if --scenario specified and doesn't match
    if [ -n "$ONLY_SCENARIO" ] && [ "$ONLY_SCENARIO" != "$scenario_name" ]; then
        continue
    fi
    
    echo ""
    echo "====== Scenario: ${scenario_name} ======"
    echo "  use_te_permutation=${use_te_perm}"
    echo "  te_permutation_align_size=${align_size}"
    echo "  global_mode=${global_mode}, local_mode=${local_mode}"
    
    # Set perm modes
    update_perm_modes "$global_mode" "$local_mode"
    
    # Determine which tensors to dump based on mode
    if [ "$global_mode" = "te" ]; then
        TENSORS=("${TE_TENSORS[@]}" "${RAGGED_PARAMS[@]}")
    else
        TENSORS=("${MT_TENSORS[@]}")
    fi
    
    # Loop through tensors
    for tensor_name in "${TENSORS[@]}"; do
        # Skip if --tensor specified and doesn't match
        if [ -n "$ONLY_TENSOR" ] && [ "$ONLY_TENSOR" != "$tensor_name" ]; then
            continue
        fi
        
        echo ""
        echo "  ------ Tensor: ${tensor_name} ------"
        
        # Update MOE_DEBUG_DUMP_TENSOR
        update_dump_tensor "$tensor_name"
        
        # Run the test
        if run_test "$use_te_perm" "$align_size"; then
            # Collect tensors
            collect_tensors "$scenario_name" "$tensor_name"
        else
            echo "  ERROR: Test failed for ${scenario_name}/${tensor_name}"
        fi
    done
done

# Reset moe.py
reset_moe_py

# Create archive
echo ""
echo "====== Creating Archive ======"
if [ "$DRY_RUN" = false ]; then
    cd "$OUTPUT_BASE_DIR"
    tar -czvf "${ARCHIVE_NAME}" */
    echo "Created archive: ${OUTPUT_BASE_DIR}/${ARCHIVE_NAME}"
    
    # Send to local machine
    echo ""
    echo "====== Sending to Local Machine ======"
    echo "Sending ${ARCHIVE_NAME} to ${LOCAL_TARGET}"
    scp "${ARCHIVE_NAME}" "${LOCAL_TARGET}/"
    echo "Done!"
else
    echo "[DRY-RUN] Would create ${ARCHIVE_NAME} and scp to ${LOCAL_TARGET}"
fi

echo ""
echo "=============================================="
echo "Tensor dump complete!"
echo "=============================================="
