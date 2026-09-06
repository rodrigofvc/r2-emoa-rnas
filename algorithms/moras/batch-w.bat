@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%S in (18906049 15798821 65381509 27293207 27522793) do (
    echo ==================================================
    echo Initializing MORAS seed=%%S
    echo Date: !DATE! !TIME!
    echo ==================================================

    python moras.py ^
        --seed %%S ^
        --search_space discrete ^
        --dataset cifar10 ^
        --batch_size 192 ^
        --data ../../data ^
        --n_population 40 ^
        --epochs_train_individual 10 ^
        --generations 31 ^
        --loss_type tchebycheff ^
        --lambda_1 0.5 ^
        --lambda_2 0.5 ^
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
