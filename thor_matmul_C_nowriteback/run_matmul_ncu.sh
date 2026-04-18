#!/bin/bash

# Script to run No-WriteBack MatMulPerf with NVIDIA Nsight Compute profiling.
#
# The kernel computes C = A @ B but does NOT write C back to global memory.
# A scalar checksum per block is stored to prevent compiler DCE, so the
# profiling result reflects pure compute + read cost WITHOUT C write overhead.
#
# Sweeps M=N from 256 to 8192 (step 256) and K from 512 to 4096 (step 256).
#
# Usage:
#   ./run_matmul_ncu.sh

# Don't exit on error - continue even if individual tests fail
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT_DIR="$SCRIPT_DIR/ncu_results"

echo "No-WriteBack MatMul NCU Profiling Suite"
echo "========================================"
echo "C is NOT written back; scalar checksum prevents compiler DCE."
echo "Base output directory: $BASE_OUTPUT_DIR"
echo ""

PYTHON_CMD="python3 MatMulPerf.py"

# K dimension sweep: 512 to 4096 step 256
K_DIMS=$(seq 512 256 4096)
K_COUNT=$(echo "$K_DIMS" | wc -w)

# M and N dimensions sweep: 256 to 8192 step 256
MN_DIMS=$(seq 256 256 8192)
MN_COUNT=$(echo "$MN_DIMS" | wc -w)

# Total test cases
TOTAL_TESTS=$((K_COUNT * MN_COUNT))

echo "K sweep: 512→4096 (step 256), $K_COUNT values"
echo "M=N sweep: 256→8192 (step 256), $MN_COUNT values per K"
echo "Total test cases: $TOTAL_TESTS"
echo ""

GLOBAL_TEST_NUM=0

# Outer loop: K dimension
for k in $K_DIMS; do
    echo "========================================"
    echo "Starting K=$k sweep"
    echo "========================================"
    
    # Create K-specific output directory
    OUTPUT_DIR="$BASE_OUTPUT_DIR/K_${k}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Output directory for K=$k: $OUTPUT_DIR"
    echo ""
    
    # Inner loop: M and N dimensions (M=N)
    for dim in $MN_DIMS; do
        GLOBAL_TEST_NUM=$((GLOBAL_TEST_NUM + 1))
        m=$dim
        n=$dim
        
        # Format filename
        OUTPUT_FILE="$OUTPUT_DIR/MatMulPerf_M${m}_N${n}_K${k}.ncu-rep"
        
        echo "[${GLOBAL_TEST_NUM}/${TOTAL_TESTS}] K=$k, M=$m, N=$n"
        
        # Run with NCU profiling, continue on error
        ncu -k regex:matmul_kernel --set full -f -o "$OUTPUT_FILE" \
            $PYTHON_CMD --M $m --N $n --K $k  || echo "  Warning: Test failed but continuing..."
            # $PYTHON_CMD --M $m --N $n --K $k  --l2-persist-input || echo "  Warning: Test failed but continuing..."
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  ✓ Saved: $OUTPUT_FILE"
        else
            echo "  ✗ Error: Failed to save $OUTPUT_FILE"
        fi
    done
    
    echo ""
    echo "Completed K=$k sweep ($MN_COUNT tests)"
    echo ""
done

echo ""
echo "========================================"
echo "All profiling completed!"
echo "========================================"
echo "Total tests run: $TOTAL_TESTS"
echo "Results organized by K value in: $BASE_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Parse results:   python3 parse_ncu_results.py"
echo "  2. Analyze ratios:  python3 analyze_ddr_ratio.py"
