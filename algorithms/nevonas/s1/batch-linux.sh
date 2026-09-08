#!/usr/bin/env bash

set -u
set -o pipefail

SEEDS=(18906049 15798821 65381509 27293207 27522793)
LOG_DIR="logs_arch_search"

mkdir -p "$LOG_DIR"

for SEED in "${SEEDS[@]}"; do
    LOG_FILE="${LOG_DIR}/arch_search_seed_${SEED}.log"

    echo "=================================================="
    echo "Initializing Architecture Search seed=${SEED}"
    echo "Log: ${LOG_FILE}"
    echo "Date: $(date)"
    echo "=================================================="

    python3 arch_search.py \
        --seed "$SEED" \
        --dataset cifar10 \
        --batch_size 192 \
        --gpu 0 \
        --init_channels 8 \
        --generations 31 \
        --n_population 40 \
        --epochs_warmup 0 \
        --epochs_train_supernet 10 \
        --learning_rate 0.025 \
        --learning_rate_min 0.001 \
        --momentum 0.9 \
        --weight_decay 3e-4 \
        --mutate_rate 0.1 \
        --prob_cross 0.9 \
        --prob_mut 0.1 \
        --eta_cross 15 \
        --eta_mut 3 \
        --report_freq 50 \
        --layers 5 \
        --steps 4 \
        --multiplier 4 \
        --reduction \
        --grad_clip 5.0 \
        --timestamp_supernet 180 \
        --timestamp_individual 10 \
        --proxy_data_dir proxy-data/proxy_train/train_proxy_cifar10_resnet20_2500.npy \
        --proxy_eval_dir proxy-data/proxy_eval/eval_proxy_indices_cifar10_192_5000.npy \
        --initial_population initial/initial_population_40_continuous.npy \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -eq 0 ]; then
        echo "Seed ${SEED} completed successfully: $(date)"
    else
        echo "ERROR: seed ${SEED} finished with code ${STATUS}: $(date)"
    fi
done

echo "FINISHED: $(date)"
