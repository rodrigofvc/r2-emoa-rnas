import os
import argparse
import shutil
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "" # Disable GPU usage for this script, as it may cause issues with multiprocessing on some platforms

from utils import (create_experiment_dir, save_model,
                   save_architecture, save_archive, save_archive_accuracy,
                   save_archive_losses, plot_archive_losses,
                   plot_archive_accuracy, plot_lr_scheduler,
                   plot_hypervolume, plot_hypervolume2, plot_r2,
                   save_statistics_to_csv, save_params)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS backend.*")


# python rnas_search.py --seed 18906049 --algorithm r2-emoa-one-shot --dataset cifar10 --batch_size 96 --n_population 40 --generations 30 --epochs_warmup 100 --epochs_train_supernet 10 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 --attack FGSM --fgsm_eps 8/255 --cutout False --cutout_length 16 --drop_path_prob 0.3 --grad_clip 0.5 --train_portion 0.5
# python rnas_search.py --seed 18906049 --algorithm r2-emoa --search_space discrete --dataset cifar10 --batch_size 96 --n_population 40 --epochs_train_individual 10 --generations 30 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 --attack FGSM --cutout_length 16 --drop_path_prob 0.3 --grad_clip 0.5 --train_portion 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running R2-EMOA for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa', 'r2-emoa-one-shot'], help='algorithm to run')
    parser.add_argument('--search_space', type=str, default="continuous", choices=['continuous', 'discrete'], help='search space to use')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_population', type=int, default=40, help='population size')
    parser.add_argument('--generations', type=int, default=30, help='number of generations to search')
    parser.add_argument('--epochs_warmup', type=int, default=0, help='number of epochs to warmup supernet')
    parser.add_argument('--pretrained_supernet', type=str, default=None, help='path to pretrained supernet to load before training')
    parser.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
    parser.add_argument('--epochs_train_individual', type=int, default=1, help='number of epochs to train individual per generation')
    parser.add_argument('--objectives', type=int, default=4, help='number of objectives')
    parser.add_argument('--std_loss_index', type=int, default=0, help='index of standard loss in objectives')
    parser.add_argument('--adv_loss_index', type=int, default=1, help='index of adversarial loss in objectives')
    parser.add_argument('--flops_index', type=int, default=2, help='index of flops in objectives')
    parser.add_argument('--params_index', type=int, default=3, help='index of params in objectives')
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--prob_cross', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--prob_mut', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--eta_cross', type=int, default=15, help='crossover eta')
    parser.add_argument('--eta_mut', type=int, default=20, help='mutation eta')
    parser.add_argument('--mu', type=float, default=0.1, help='mu for thchebycheff function')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='init channels')
    parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers (cells)')
    parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell (intern nodes except input and output)')
    parser.add_argument('--multiplier', type=int, default=6, help='number of multiplier for number of channels (intern nodes to concat)')
    parser.add_argument('--attack', type=str, default='FGSM', help='adversarial attack to use')
    parser.add_argument('--fgsm_eps', type=float, default=8/255, help='attack epsilon')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--timestamp', type=int, default=6, help='timestamp in minutes for training/eval each architecture')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='Enable CUDA_LAUNCH_BLOCKING for debugging')
    parser.add_argument('--increase_epochs', action='store_true', default=False, help='Increase the number of epochs to train the supernet and individuals as generations progress')
    parser.add_argument('--reload_dir', type=str, default=None, help='Directory to reload the experiment from if --reload is set')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    print("Running with config:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.reload_dir is None:
        results_dir = create_experiment_dir(args.algorithm, args.dataset, args.seed)
    else:
        results_dir = args.reload_dir
    print(f'Results dir: {results_dir}' )
    args.save_path_final_model = results_dir
    args.save_path_final_architect = results_dir

    if args.algorithm == 'r2-emoa-one-shot':
        from r2_emoa_one_shot import r2_emoa_oneshot_nas
        # The search space is continuous because we are optimizing the architecture parameters (alphas) of the supernet
        args.search_space = 'continuous'
        save_params(args, args.save_path_final_architect)
        supernet, archive, archive_accuracy, archive_losses, statistics = r2_emoa_oneshot_nas(
            args=args
        )
        save_model(supernet, args.save_path_final_model, f"super-net.pt")
        for i, individual in enumerate(archive):
            save_architecture(i, individual, args.save_path_final_architect)
        save_archive(archive, args.save_path_final_architect)
        save_archive_accuracy(archive_accuracy, args.save_path_final_architect)
        save_archive_losses(archive_losses, args.save_path_final_architect)
        plot_archive_losses(archive_losses, args.save_path_final_architect)
        plot_archive_accuracy(archive_accuracy, args.save_path_final_architect)
        plot_lr_scheduler(statistics, args.save_path_final_architect)
        plot_hypervolume(statistics, args.save_path_final_architect)
        plot_hypervolume2(statistics, args.save_path_final_architect)
        plot_r2(statistics, args.save_path_final_architect)
        save_statistics_to_csv(statistics, args.save_path_final_architect)
        logging.info(f"Experiment completed and results saved in {results_dir}")
    elif args.algorithm == 'r2-emoa':
        from r2_emoa import r2_emoa_rnas
        if args.reload_dir is None:
            save_params(args, args.save_path_final_architect)
        archive, archive_accuracy, archive_losses, statistics = r2_emoa_rnas(
            args
        )
        for i, individual in enumerate(archive):
            save_architecture(i, individual, args.save_path_final_architect)
        save_archive(archive, args.save_path_final_architect)
        save_archive_accuracy(archive_accuracy, args.save_path_final_architect)
        save_archive_losses(archive_losses, args.save_path_final_architect)
        plot_archive_losses(archive_losses, args.save_path_final_architect)
        plot_archive_accuracy(archive_accuracy, args.save_path_final_architect)
        plot_hypervolume(statistics, args.save_path_final_architect)
        plot_hypervolume2(statistics, args.save_path_final_architect)
        plot_r2(statistics, args.save_path_final_architect)
        save_statistics_to_csv(statistics, args.save_path_final_architect)
        logging.info(f"Experiment completed and results saved in {results_dir}")