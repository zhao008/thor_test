#!/bin/bash

# Script to run no-writeback MatMulPerf with NVIDIA Nsight Compute profiling.
# Sweeps tm/tn together with fixed tk=64 while preserving the K/M/N sweep layout.
# Results are organized as ncu_results/tm_<tm>_tn_<tn>_tk_<tk>/K_<K>/...

# Don't exit on error - continue even if individual tests fail
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT_DIR="$SCRIPT_DIR/ncu_results"
PYTHON_CMD="python3 MatMulPerf.py"

echo "No-writeback cuTile MatMul NCU Profiling Suite"
echo "=============================================="
echo "Base output directory: $BASE_OUTPUT_DIR"
echo ""

TM_VALUES="64 128 256 512"
TK_VALUE="64"

K_DIMS=$(seq 512 512 4096)
MN_DIMS=$(seq 256 256 8192)
TM_COUNT=$(echo "$TM_VALUES" | wc -w)
K_COUNT=$(echo "$K_DIMS" | wc -w)
MN_COUNT=$(echo "$MN_DIMS" | wc -w)
TOTAL_TESTS=$((TM_COUNT * K_COUNT * MN_COUNT))

echo "tm sweep: $TM_VALUES, $TM_COUNT values"
echo "tn sweep: synced with tm"
echo "tk fixed: $TK_VALUE"
echo "K sweep: 512→4096 (step 512), $K_COUNT values"
echo "M=N sweep: 256→8192 (step 256), $MN_COUNT values per K"
echo "Total test cases: $TOTAL_TESTS"
echo ""

GLOBAL_TEST_NUM=0

for tm in $TM_VALUES; do
    tn=$tm
    tk=$TK_VALUE

    echo "========================================"
    echo "Starting tm=$tm, tn=$tn, tk=$tk sweep"
    echo "========================================"

    TILE_OUTPUT_DIR="$BASE_OUTPUT_DIR/tm_${tm}_tn_${tn}_tk_${tk}"
    mkdir -p "$TILE_OUTPUT_DIR"

    echo "Output root for tm=$tm, tn=$tn, tk=$tk: $TILE_OUTPUT_DIR"
    echo ""

    for k in $K_DIMS; do
        echo "----------------------------------------"
        echo "Starting tm=$tm, tn=$tn, tk=$tk, K=$k sweep"
        echo "----------------------------------------"

        OUTPUT_DIR="$TILE_OUTPUT_DIR/K_${k}"
        mkdir -p "$OUTPUT_DIR"

        echo "Output directory: $OUTPUT_DIR"
        echo ""

        for dim in $MN_DIMS; do
            GLOBAL_TEST_NUM=$((GLOBAL_TEST_NUM + 1))
            m=$dim
            n=$dim

            OUTPUT_FILE="$OUTPUT_DIR/MatMulPerf_tm${tm}_tn${tn}_tk${tk}_M${m}_N${n}_K${k}.ncu-rep"

            echo "[${GLOBAL_TEST_NUM}/${TOTAL_TESTS}] tm=$tm, tn=$tn, tk=$tk, K=$k, M=$m, N=$n"

            ncu -k regex:matmul_kernel_.*_no_writeback --set full -f -o "$OUTPUT_FILE" \
                $PYTHON_CMD --M $m --N $n --K $k --tile-m $tm --tile-n $tn --tile-k $tk \
                || echo "  Warning: Test failed but continuing..."
                # $PYTHON_CMD --M $m --N $n --K $k --tile-m $tm --tile-n $tn --tile-k $tk --l2-persist-input || echo "  Warning: Test failed but continuing..."

            if [ -f "$OUTPUT_FILE" ]; then
                echo "  ✓ Saved: $OUTPUT_FILE"
            else
                echo "  ✗ Error: Failed to save $OUTPUT_FILE"
            fi
        done

        echo ""
        echo "Completed tm=$tm, tn=$tn, tk=$tk, K=$k sweep ($MN_COUNT tests)"
        echo ""
    done

    echo "Finished all K sweeps for tm=$tm, tn=$tn, tk=$tk"
    echo ""
done

echo ""
echo "========================================"
echo "All profiling completed!"
echo "========================================"
echo "Total tests run: $TOTAL_TESTS"
echo "Results organized by tm/tn/tk/K in: $BASE_OUTPUT_DIR"
echo ""
echo "To parse all results:"
echo "  python3 parse_ncu_results.py"
echo ""
echo "Example output path:"
echo "  $BASE_OUTPUT_DIR/tm_256_tn_256_tk_64/K_512"
