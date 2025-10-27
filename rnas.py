import argparse
import json
import ssl
from fractions import Fraction

import numpy as np
import torch
import torchattacks
import torchvision
from torch import nn

import utils
from evolutionary import unpack_alphas
from r2_emoa import r2_emoa_rnas
from evaluation.model_search import Network


def get_attack_function(attack_params):
    attack_params['params']['eps'] = float(Fraction(attack_params['params']['eps'])) if '/' in attack_params['params']['eps'] else float(attack_params['params']['eps'])
    if attack_params['name'] == 'FGSM':
        attack_function = lambda model: torchattacks.FGSM(model, **attack_params['params'])
    elif attack_params['name'] == 'PGD':
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function

# Prepare all arguments and components such as model, optimizer, data loaders, weights, scheduler, attack.
def prepare_args(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    if args.reduction:
        reduction_layers = [args.layers//3, 2*args.layers//3]
    else:
        reduction_layers = None

    criterion = nn.CrossEntropyLoss()

    model = Network(
        C=args.init_channels,
        num_classes=args.classes,
        layers=args.layers,
        criterion=criterion,
        steps=args.steps,
        multiplier_cells=args.multiplier,
        reduction_layers=reduction_layers,
        stem_multiplier=3,
        fairdarts_eval=False,
        device=args.device,
    ).to(args.device)


    optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    print(f"Training samples: {split}, Validation samples: {num_train - split}")
    split = 32
    num_train = split + 32


    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=2, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=2, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    attack_f = get_attack_function(args.attack)

    weights_r2 = utils.get_weights_r2(args.n_population)

    return model, criterion, optimizer, scheduler, train_queue, valid_queue, attack_f, weights_r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running R2-EMOA for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa'], help='algorithm to run')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='dataset to use')
    parser.add_argument('--params_dir', type=str, required=True, help="params json dir")
    args = parser.parse_args()

    with open(args.params_dir, 'r') as f:
        config = json.load(f)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    args = argparse.Namespace(**config)
    print("Running with config:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    results_dir = utils.create_experiment_dir(args.algorithm, args.dataset, args.seed)
    args.save_path_final_model = results_dir
    args.save_path_final_architect = results_dir

    model, criterion, optimizer, scheduler, train_queue, valid_queue, attack_f, weights_r2 = prepare_args(args)
    if args.algorithm == 'r2-emoa':
        supernet, archive = r2_emoa_rnas(
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
        utils.save_model(supernet, args.save_path_final_model)
        architectures = []
        for individual in archive:
            architectures.append(unpack_alphas(individual.X, model.alphas_dim))
        utils.save_architectures(architectures, args.save_path_final_architect)
        utils.save_archive(archive, args.save_path_final_architect)

