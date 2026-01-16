#!/bin/bash
# Quick test script to verify multi-GPU setup and performance improvements

echo "=========================================="
echo "SSVAE Performance Test Script"
echo "=========================================="
echo ""

# Check for GPUs
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    # Get GPU count using nvidia-smi properly
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        # Fallback method
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    fi
    echo "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
else
    echo "No NVIDIA GPUs found (nvidia-smi not available)"
    GPU_COUNT=0
    echo ""
fi

# Check CPU cores
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
echo "CPU cores available: $CPU_COUNT"
echo ""

# Quick test parameters
DATASET="MNIST"
NUM_EPOCHS=5
N_LABELS="100,1000"
ALPHA_VALUES="0.1,1.0"
NUM_SEEDS=1

echo "Running quick performance test..."
echo "Parameters: $NUM_EPOCHS epochs, minimal configuration"
echo ""

# Test 1: Baseline (minimal optimizations)
echo "=========================================="
echo "Test 1: Baseline (single GPU, num_workers=0)"
echo "=========================================="
time python combined_robustness_experiment.py \
    --dataset $DATASET \
    --num_epochs $NUM_EPOCHS \
    --n_labels $N_LABELS \
    --alpha_values $ALPHA_VALUES \
    --num_seeds $NUM_SEEDS \
    --no_multi_gpu \
    --num_workers 0 2>&1 | tee test1_baseline.log

echo ""
echo "Test 1 complete. Log saved to test1_baseline.log"
echo ""

# Test 2: Optimized data loading
echo "=========================================="
echo "Test 2: Optimized data loading (num_workers=4)"
echo "=========================================="
time python combined_robustness_experiment.py \
    --dataset $DATASET \
    --num_epochs $NUM_EPOCHS \
    --n_labels $N_LABELS \
    --alpha_values $ALPHA_VALUES \
    --num_seeds $NUM_SEEDS \
    --no_multi_gpu \
    --num_workers 4 2>&1 | tee test2_workers.log

echo ""
echo "Test 2 complete. Log saved to test2_workers.log"
echo ""

# Test 3: Multi-GPU (if available)
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "=========================================="
    echo "Test 3: Multi-GPU ($GPU_COUNT GPUs)"
    echo "=========================================="
    time python combined_robustness_experiment.py \
        --dataset $DATASET \
        --num_epochs $NUM_EPOCHS \
        --n_labels $N_LABELS \
        --alpha_values $ALPHA_VALUES \
        --num_seeds $NUM_SEEDS \
        --num_workers 4 2>&1 | tee test3_multigpu.log
    
    echo ""
    echo "Test 3 complete. Log saved to test3_multigpu.log"
    echo ""
else
    echo "Skipping multi-GPU test (only $GPU_COUNT GPU available)"
    echo ""
fi

# Test 4: All optimizations
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "=========================================="
    echo "Test 4: All optimizations"
    echo "=========================================="
    
    # Determine optimal workers (4 per GPU or max 16)
    if [ "$GPU_COUNT" -gt 1 ]; then
        OPT_WORKERS=$((4 * GPU_COUNT))
        if [ $OPT_WORKERS -gt 16 ]; then
            OPT_WORKERS=16
        fi
    else
        OPT_WORKERS=4
    fi
    
    echo "Using num_workers=$OPT_WORKERS (4 per GPU)"
    
    time python combined_robustness_experiment.py \
        --dataset $DATASET \
        --num_epochs $NUM_EPOCHS \
        --n_labels $N_LABELS \
        --alpha_values $ALPHA_VALUES \
        --num_seeds $NUM_SEEDS \
        --num_workers $OPT_WORKERS \
        --eval_frequency 2 2>&1 | tee test4_optimized.log
    
    echo ""
    echo "Test 4 complete. Log saved to test4_optimized.log"
    echo ""
fi

# Summary
echo "=========================================="
echo "Performance Test Summary"
echo "=========================================="
echo ""
echo "Extract timing information from logs:"
echo ""
echo "Test 1 (Baseline):"
grep -i "real\|user\|sys" test1_baseline.log 2>/dev/null || echo "  (check test1_baseline.log manually)"
echo ""
echo "Test 2 (Workers=4):"
grep -i "real\|user\|sys" test2_workers.log 2>/dev/null || echo "  (check test2_workers.log manually)"
echo ""
if [ -f test3_multigpu.log ]; then
    echo "Test 3 (Multi-GPU):"
    grep -i "real\|user\|sys" test3_multigpu.log 2>/dev/null || echo "  (check test3_multigpu.log manually)"
    echo ""
fi
if [ -f test4_optimized.log ]; then
    echo "Test 4 (All optimizations):"
    grep -i "real\|user\|sys" test4_optimized.log 2>/dev/null || echo "  (check test4_optimized.log manually)"
    echo ""
fi

echo "=========================================="
echo "Recommendations based on your system:"
echo "=========================================="
echo ""
echo "System configuration:"
echo "  GPUs: $GPU_COUNT"
echo "  CPU cores: $CPU_COUNT"
echo ""

if [ "$GPU_COUNT" -ge 4 ]; then
    echo "Recommended settings for production runs:"
    echo "  --gpu_ids 0,1,2,3"
    echo "  --num_workers 8"
    echo "  --eval_frequency 5"
    echo ""
    echo "Example command:"
    echo "  python combined_robustness_experiment.py \\"
    echo "      --dataset MNIST \\"
    echo "      --num_epochs 75 \\"
    echo "      --gpu_ids 0,1,2,3 \\"
    echo "      --num_workers 8 \\"
    echo "      --eval_frequency 5"
elif [ "$GPU_COUNT" -ge 2 ]; then
    echo "Recommended settings for production runs:"
    echo "  --gpu_ids 0,1"
    echo "  --num_workers 4"
    echo "  --eval_frequency 5"
    echo ""
    echo "Example command:"
    echo "  python combined_robustness_experiment.py \\"
    echo "      --dataset MNIST \\"
    echo "      --num_epochs 75 \\"
    echo "      --gpu_ids 0,1 \\"
    echo "      --num_workers 4 \\"
    echo "      --eval_frequency 5"
elif [ "$GPU_COUNT" -eq 1 ]; then
    echo "Recommended settings for production runs:"
    echo "  --num_workers 4"
    echo "  --eval_frequency 5"
    echo ""
    echo "Example command:"
    echo "  python combined_robustness_experiment.py \\"
    echo "      --dataset MNIST \\"
    echo "      --num_epochs 75 \\"
    echo "      --num_workers 4 \\"
    echo "      --eval_frequency 5"
else
    echo "No GPUs detected. Using CPU."
    echo "Recommended settings:"
    echo "  --num_workers 2"
    echo "  --eval_frequency 10"
    echo ""
    echo "Note: Training will be significantly slower on CPU."
fi

echo ""
echo "For more details, see:"
echo "  - MULTI_GPU_GUIDE.md"
echo "  - PERFORMANCE_GUIDE.md"
echo ""
