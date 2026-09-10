@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%S in (18906049 15798821 65381509 27293207 27522793) do (
    echo ==================================================
    echo Initializing Main Search seed=%%S
    echo Date: !DATE! !TIME!
    echo ==================================================

    python main_search.py ^
        --seed %%S ^
        --dataset cifar10 ^
        --batch_size 192 ^
        --n_population 40 ^
        --generations 31 ^
        --epochs_warmup 0 ^
        --epochs_train_supernet 10 ^
        --prob_cross 0.9 ^
        --prob_mut 0.1 ^
        --eta_cross 15 ^
        --eta_mut 3 ^
        --learning_rate 0.025 ^
        --learning_rate_min 0.001 ^
        --momentum 0.9 ^
        --weight_decay 3e-4 ^
        --report_freq 50 ^
        --gpu 0 ^
        --init_channels 8 ^
        --reduction ^
        --layers 5 ^
        --steps 4 ^
        --multiplier 4 ^
        --attack FGSM ^
        --grad_clip 5.0 ^
        --timestamp_supernet 240 ^
        --timestamp_individual 10 ^
        --proxy_data_dir proxy-data/proxy_train/train_proxy_cifar10_resnet20_2500.npy ^
        --proxy_eval_dir proxy-data/proxy_eval/eval_proxy_indices_cifar10_192_5000.npy ^
        --initial_population initial/initial_population_40_continuous.npy

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