import argparse
import ssl
import random

import numpy as np
import torch

import torchvision
from torch import nn

import utils
from r2_emoa import r2_emoa_rnas_oneshot, r2_emoa_rnas
from micro_space.model_search import Network
from adversarial import get_attack_function
from micro_space.genotypes import PRIMITIVES
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

"""
 python3 rnas_search.py --seed 18906049 --algorithm r2-emoa-one-shot --dataset cifar10 --batch_size 32  \
 --n_population 10 --generations 2 --epochs_warmup 0 --epochs_train_supernet 1 \
 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 \
 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 \
 --attack FGSM --fgsm_eps 8/255 --cutout False --cutout_length 16 --drop_path_prob 0.3 \
 --grad_clip 0.5 --train_portion 0.5
"""
# Prepare all arguments and components such as model, optimizer, data loaders, weights, scheduler, attack.
def prepare_args_supernet(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = Network(
        C=args.init_channels,
        num_classes=n_classes,
        layers=args.layers,
        criterion=criterion,
        steps=args.steps,
        multiplier=args.multiplier,
        stem_multiplier=3,
        device=args.device,
    ).to(args.device)

    if args.pretrained_supernet is not None:
        print(f"Loading pretrained supernet from {args.pretrained_supernet}")
        model = utils.load_supernet(args.pretrained_supernet)
        model = model.to(args.device)

    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
        num_train = split + 32
    print(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (args.generations + 1) * args.epochs_train_supernet + args.epochs_warmup, eta_min=args.learning_rate_min)

    attack_params = {
        'name': args.attack,
        'params': {
            'eps': args.fgsm_eps
        }
    }

    attack_f = get_attack_function(attack_params)

    weights_r2 = utils.get_weights_r2(args.n_population)

    return model, criterion, optimizer, scheduler, train_queue, valid_queue, attack_f, weights_r2

"""
 python3 rnas_search.py --seed 18906049 --algorithm r2-emoa --dataset cifar10 --batch_size 32  \
 --n_population 10 --epochs_train_individual 2 --generations 2 \
 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 \
 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 \
 --attack FGSM --fgsm_eps 8/255 --cutout False --cutout_length 16 --drop_path_prob 0.3 \
 --grad_clip 0.5 --train_portion 0.5
"""
def prepare_args_standard(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
        num_train = split + 32
    print(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True)

    attack_params = {
        'name': args.attack,
        'params': {
            'eps': args.fgsm_eps
        }
    }

    attack_f = get_attack_function(attack_params)

    weights_r2 = utils.get_weights_r2(args.n_population)

    k = sum(1 for i in range(args.steps) for _ in range(2 + i))
    num_ops = len(PRIMITIVES)
    alphas_dim = (k, num_ops)

    return alphas_dim, train_queue, valid_queue, attack_f, weights_r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running R2-EMOA for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa', 'r2-emoa-one-shot'], help='algorithm to run')
    parser.add_argument('--search_space', type=str, default='continuous', choices='[continuous, discrete]', help='search space to use')
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
    parser.add_argument('--reduction', type=bool, default=True, help='use reduction cell or not')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers (cells)')
    parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell (intern nodes except input and output)')
    parser.add_argument('--multiplier', type=int, default=6, help='number of multiplier for number of channels (intern nodes to concat)')
    parser.add_argument('--attack', type=str, default='FGSM', help='adversarial attack to use')
    parser.add_argument('--fgsm_eps', type=str, default="8/255", help='attack epsilon')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--synchronize', type=bool, default=False, help='synchronize CUDA operations or not')
    args = parser.parse_args()

    print("Running with config:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True


    results_dir = utils.create_experiment_dir(args.algorithm, args.dataset, args.seed)
    print(f'Results dir: {results_dir}' )
    args.save_path_final_model = results_dir
    args.save_path_final_architect = results_dir

    if args.algorithm == 'r2-emoa-one-shot':
        # The search space is continuous because we are optimizing the architecture parameters (alphas) of the supernet
        args.search_space = 'continuous'
        model, criterion, optimizer, scheduler, train_queue, valid_queue, attack_f, weights_r2 = prepare_args_supernet(args)
        supernet, archive, archive_accuracy, archive_losses, statistics = r2_emoa_rnas_oneshot(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_queue=train_queue,
            valid_queue=valid_queue,
            attack_f=attack_f,
            weights_r2=weights_r2,
            args=args
        )
        utils.save_model(supernet, args.save_path_final_model, f"super-net.pt")
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
        utils.plot_lr_scheduler(statistics, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        utils.save_statistics_to_csv(statistics, args.save_path_final_architect)
        utils.save_params(args, args.save_path_final_architect)
        print(f"Experiment completed and results saved in {results_dir}")
    elif args.algorithm == 'r2-emoa':
        alphas_dim, train_queue, valid_queue, attack_f, weights_r2 = prepare_args_standard(args)
        archive, archive_accuracy, archive_losses, statistics = r2_emoa_rnas(
            alphas_dim=alphas_dim,
            train_queue=train_queue,
            valid_queue=valid_queue,
            attack_f=attack_f,
            weights_r2=weights_r2,
            args=args
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
        utils.save_params(args, args.save_path_final_architect)
        print(f"Experiment completed and results saved in {results_dir}")