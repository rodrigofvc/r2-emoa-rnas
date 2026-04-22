import os
import argparse
import logging
import shutil
from cars_alg import cars_algorithm

import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS backend.*")


# python -X dev main_search.py --seed 18906049 --dataset cifar10 --batch_size 160 --n_population 40 --generations 30 --epochs_warmup 0 --epochs_train_supernet 10 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --report_freq 10 --gpu 0 --init_channels 8 --reduction --layers 5 --steps 4 --multiplier 4 --attack FGSM --cutout_length 16 --drop_path_prob 0.3 --grad_clip 0.5 --train_portion 0.5 --increase_epochs --timestamp 50 --pretrained_supernet results/cars/cifar10/2026-04-21_13-32-14_18906049/search/super-net.pt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running CARS for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_population', type=int, default=40, help='population size')
    parser.add_argument('--generations', type=int, default=30, help='number of generations to search')
    parser.add_argument('--epochs_warmup', type=int, default=0, help='number of epochs to warmup supernet')
    parser.add_argument('--pretrained_supernet', type=str, default=None, help='path to pretrained supernet to load before training')
    parser.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
    parser.add_argument('--objectives', type=int, default=4, help='number of objectives')
    parser.add_argument('--std_loss_index', type=int, default=0, help='index of standard loss in objectives')
    parser.add_argument('--adv_loss_index', type=int, default=1, help='index of adversarial loss in objectives')
    parser.add_argument('--flops_index', type=int, default=2, help='index of flops in objectives')
    parser.add_argument('--params_index', type=int, default=3, help='index of params in objectives')
    parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
    parser.add_argument('--prob_cross', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--prob_mut', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--eta_cross', type=int, default=15, help='crossover eta')
    parser.add_argument('--eta_mut', type=int, default=20, help='mutation eta')
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
    parser.add_argument('--timestamp', type=int, default=45, help='timestamp in minutes for training/eval each architecture')
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


    if args.reload_dir is None:
        print("Running with config:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        results_dir = utils.create_experiment_dir('cars', args.dataset, args.seed)
    else:
        results_dir = args.reload_dir
    print(f'Results dir: {results_dir}' )
    args.save_path_final_model = results_dir
    args.save_path_final_architect = results_dir
    args.search_space = 'continuous'
    if args.reload_dir is None:
        utils.save_params(args, args.save_path_final_architect)
    archive, archive_accuracy, archive_losses, statistics = cars_algorithm(
        args_=args
    )
    print("Final archive:")
    for individual in archive:
        print(individual.F, individual.std_acc, individual.adv_acc)
    for i, individual in enumerate(archive):
        utils.save_architecture(i, individual, args.save_path_final_architect)
    utils.save_archive(archive, args.save_path_final_architect)
    utils.save_archive_accuracy(archive_accuracy, args.save_path_final_architect)
    utils.save_archive_losses(archive_losses, args.save_path_final_architect)
    utils.plot_archive_losses(archive_losses, args.save_path_final_architect)
    utils.plot_archive_accuracy(archive_accuracy, args.save_path_final_architect)
    utils.plot_hypervolume(statistics, args.save_path_final_architect)
    utils.plot_hypervolume2(statistics, args.save_path_final_architect)
    utils.plot_r2(statistics, args.save_path_final_architect)
    utils.save_statistics_to_csv(statistics, args.save_path_final_architect)
    print(f"Experiment completed and results saved in {results_dir}")
