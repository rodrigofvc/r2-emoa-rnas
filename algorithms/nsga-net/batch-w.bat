@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%S in (18906049 15798821 65381509 27293207 27522793) do (
    echo ==================================================
    echo Initializing Evolution Search seed=%%S
    echo Date: !DATE! !TIME!
    echo ==================================================

    python search/evolution_search.py ^
        --seed %%S ^
        --search_space micro ^
        --dataset cifar10 ^
        --n_classes 10 ^
        --pop_size 40 ^
        --n_offspring 40 ^
        --batch_size 192 ^
        --n_gens 31 ^
        --epochs 10 ^
        --init_channels 8 ^
        --loss_type tchebycheff ^
        --mu 0.3 ^
        --lambda_1 0.5 ^
        --lambda_2 0.5 ^
        --eta_mut 3 ^
        --prob_mut 0.1 ^
        --learning_rate 0.025 ^
        --learning_rate_min 0.001 ^
        --momentum 0.9 ^
        --weight_decay 3e-4 ^
        --grad_clip 5.0 ^
        --layers 5 ^
        --steps 4 ^
        --multiplier 4 ^
        --proxy_data_dir proxy-data/proxy_train/train_proxy_cifar10_resnet20_2500.npy ^
        --proxy_eval_dir proxy-data/proxy_eval/eval_proxy_indices_cifar10_192_5000.npy ^
        --initial_population initial/initial_population_40.npy

    set "STATUS=!ERRORLEVEL!"

    if "!STATUS!"=="0" (
        echo Seed %%S completed successfully: !DATE! !TIME!
    ) else (
        echo ERROR: seed %%S finished with code !STATUS!: !DATE! !TIME!
    )
)

echo ==================================================
echo FINISHED: %DATE% %TIME%
echo ==================================================

endlocal
pause
