#!/usr/bin/env bash

set -u
set -o pipefail

SEEDS=(18906049 15798821 65381509 27293207 27522793)
LOG_DIR="logs_evolution_search"

mkdir -p "$LOG_DIR"

for SEED in "${SEEDS[@]}"; do
    LOG_FILE="${LOG_DIR}/evolution_search_seed_${SEED}.log"

    echo "=================================================="
    echo "Initializing Evolution Search seed=${SEED}"
    echo "Log: ${LOG_FILE}"
    echo "Date: $(date)"
    echo "=================================================="

    python3 search/evolution_search.py \
        --seed "$SEED" \
        --search_space micro \
        --dataset cifar10 \
        --n_classes 10 \
        --pop_size 40 \
        --n_offspring 40 \
        --batch_size 192 \
        --n_gens 31 \
        --epochs 10 \
        --init_channels 8 \
        --loss_type tchebycheff \
        --mu 0.3 \
        --lambda_1 0.5 \
        --lambda_2 0.5 \
        --eta_mut 3 \
        --prob_mut 0.1 \
        --learning_rate 0.025 \
        --learning_rate_min 0.001 \
        --momentum 0.9 \
        --weight_decay 3e-4 \
        --grad_clip 5.0 \
        --layers 5 \
        --steps 4 \
        --multiplier 4 \
        --proxy_data_dir proxy-data/proxy_train/train_proxy_cifar10_resnet20_2500.npy \
        --proxy_eval_dir proxy-data/proxy_eval/eval_proxy_indices_cifar10_192_5000.npy \
        --initial_population initial/initial_population_40.npy \
        2>&1 | tee "$LOG_FILE"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -eq 0 ]; then
        echo "Seed ${SEED} completed successfully: $(date)"
    else
        echo "ERROR: seed ${SEED} finished with code ${STATUS}: $(date)"
    fi
done

echo "FINISHED: $(date)"
