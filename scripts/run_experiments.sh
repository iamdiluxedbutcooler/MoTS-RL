#!/usr/bin/env bash
set -e

export HYDRA_FULL_ERROR=1

echo "=========================================="
echo "MoTS-RL Experiment Runner"
echo "=========================================="

STAGES=(stage1 stage2 stage3 robustness ablations)

for ST in "${STAGES[@]}"; do
    echo ""
    echo "Running experiments for: $ST"
    echo "=========================================="
    
    for CFG in configs/$ST/*.yaml; do
        if [ -f "$CFG" ]; then
            echo "Training: $CFG"
            python scripts/train.py +experiment=$CFG
            
            echo "Evaluating: $CFG"
            python scripts/evaluate.py +experiment=$CFG
        fi
    done
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
