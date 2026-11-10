#!/usr/bin/env bash
# ==============================================================================
# S-MSDT TMLR STUDY: CONFIRMATORY EXPERIMENTAL EVALUATION (ONE-SHOT RUN)
# ==============================================================================
# This script executes exactly ONCE following the Day-14 protocol freeze.
# Evaluates all 4 core models (GenD-ours, S, ST, Full) x 5 seeds on confirmatory datasets.
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "S-MSDT TMLR: CONFIRMATORY EVALUATION RUN (TAG: prereg-freeze)"
echo "Date: $(date -u)"
echo "======================================================================"

# Confirm git tag exists
if ! git rev-parse -q --verify "refs/tags/prereg-freeze" >/dev/null; then
    echo "WARNING: git tag 'prereg-freeze' not found! Please tag before running confirmatory evaluation."
fi

MODELS=("gend_ours" "smsdt_s" "smsdt_st" "smsdt_full")
SEEDS=(1 2 3 4 5)
CONFIRMATORY_DATASETS=("dfdc" "wilddeepfake")

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="${model}_seed${seed}_confirmatory"
        out_dir="results/scores/${run_name}"
        mkdir -p "${out_dir}"
        
        ckpt_path="outputs/checkpoints_${model}/seed${seed}/best_ema.pt"
        if [ ! -f "${ckpt_path}" ]; then
            echo "Skipping missing checkpoint: ${ckpt_path}"
            continue
        fi
        
        for dataset in "${CONFIRMATORY_DATASETS[@]}"; do
            echo "Evaluating ${model} (seed ${seed}) on ${dataset}..."
            python -m smsdt.eval_cross \
                --ckpt "${ckpt_path}" \
                --dataset-name "${dataset}" \
                --split test \
                --save-scores "${out_dir}/${dataset}.csv" \
                --calib "results/calib/${model}_seed${seed}.json"
        done
    done
done

echo "======================================================================"
echo "Running Pre-Registered H1 Statistical Classification..."
python -m tools.stats --evaluate-h1 --out "results/h1_classification.csv"

echo "Regenerating Master Experiment Tables..."
python -m tools.make_tables

echo "Confirmatory Evaluation Complete."

