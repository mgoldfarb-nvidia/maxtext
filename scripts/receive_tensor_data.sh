#!/bin/bash
# =============================================================================
# receive_tensor_data.sh - Extract tensor dump archives on local machine
#
# This script extracts tensor dump archives received from the remote machine
# and organizes them into the appropriate directory structure.
#
# Usage: ./receive_tensor_data.sh [archive_file.tar.gz] [--list]
#
# If no archive file is specified, it will look for the most recent 
# tensor_dump_*.tar.gz in ~/Repos/maxtext/data/
#
# Options:
#   --list    List the contents of the archive without extracting
#
# =============================================================================

set -e

# Configuration
DATA_DIR="${HOME}/Repos/maxtext/data"
EXTRACT_DIR="${DATA_DIR}/extracted"

# Parse arguments
ARCHIVE_FILE=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --list)
            LIST_ONLY=true
            shift
            ;;
        *.tar.gz|*.tgz)
            ARCHIVE_FILE="$1"
            shift
            ;;
        *)
            echo "Unknown option or file: $1"
            exit 1
            ;;
    esac
done

# Find the archive file if not specified
if [ -z "$ARCHIVE_FILE" ]; then
    echo "No archive file specified, looking for most recent in ${DATA_DIR}..."
    ARCHIVE_FILE=$(ls -t "${DATA_DIR}"/tensor_dump_*.tar.gz 2>/dev/null | head -1)
    
    if [ -z "$ARCHIVE_FILE" ]; then
        echo "ERROR: No tensor_dump_*.tar.gz files found in ${DATA_DIR}"
        echo ""
        echo "Available files:"
        ls -la "${DATA_DIR}"/*.tar.gz 2>/dev/null || echo "  (none)"
        exit 1
    fi
fi

# Check if archive exists
if [ ! -f "$ARCHIVE_FILE" ]; then
    # Try prepending DATA_DIR
    if [ -f "${DATA_DIR}/${ARCHIVE_FILE}" ]; then
        ARCHIVE_FILE="${DATA_DIR}/${ARCHIVE_FILE}"
    else
        echo "ERROR: Archive file not found: $ARCHIVE_FILE"
        exit 1
    fi
fi

echo "=============================================="
echo "Tensor Data Extractor"
echo "=============================================="
echo "Archive: ${ARCHIVE_FILE}"
echo "Extract to: ${EXTRACT_DIR}"
echo ""

# List mode
if [ "$LIST_ONLY" = true ]; then
    echo "Contents of archive:"
    echo "----------------------------------------------"
    tar -tzvf "$ARCHIVE_FILE"
    exit 0
fi

# Create extract directory
mkdir -p "$EXTRACT_DIR"

# Extract the archive
echo "Extracting..."
tar -xzvf "$ARCHIVE_FILE" -C "$EXTRACT_DIR"

echo ""
echo "=============================================="
echo "Extraction complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "----------------------------------------------"
find "$EXTRACT_DIR" -type d | head -30
echo ""
echo "File counts by directory:"
echo "----------------------------------------------"
for dir in "$EXTRACT_DIR"/*/; do
    if [ -d "$dir" ]; then
        scenario=$(basename "$dir")
        count=$(find "$dir" -name "*.bin" | wc -l)
        echo "  ${scenario}: ${count} tensor files"
    fi
done

echo ""
echo "To analyze tensors, use:"
echo "  python scripts/load_moe_debug_tensors.py --compare \\"
echo "    ${EXTRACT_DIR}/te_pad128/<tensor>/ \\"
echo "    ${EXTRACT_DIR}/mt/<tensor>/"
