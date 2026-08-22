import argparse
import json
import sys
import os

import logging

from micro_space.model import NetworkCIFAR
from supernet_worker import unpack_alphas

logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
)
os.environ["TORCH_LOGS"] = "-all" 
os.environ["PYTHONASYNCIODEBUG"] = "0"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_INTERFACE_LAYER"] = "LP64"
os.environ["MKL_DYNAMIC"] = "FALSE"

import time

import utils
from micro_space.micro_encoding import PRIMITIVES
from micro_space.model_search import alphas_to_genotype, discretize

import numpy as np
import torch
import torchvision

from rnas_train import infer

def get_model_from_individual(individual_X, args):

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    k = sum(2 + i for i in range(args.steps))
    alphas_dim = (k, len(PRIMITIVES))

    model = utils.load_supernet(args.supernet_path)
    model = model.to(args.device)

    individual_architect = unpack_alphas(individual_X, alphas_dim, args)
    model.update_arch_parameters(individual_architect)
    discrete = discretize(individual_architect, model.genotype(), args.device)
    model.update_arch_parameters(discrete)
    genotype_discrete = model.genotype()

    # Create a discrete model to compute FLOPs and parameters, then delete it to free memory
    discrete_model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype_discrete)
    flops, params = utils.get_model_metrics(discrete_model)
    del discrete_model

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs_train_individual, eta_min=args.learning_rate_min)

    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96

    if args.proxy_data_dir is None:
        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=args.num_workers, pin_memory=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    else:
        proxy_indices = np.load(args.proxy_data_dir)
        train_data_proxy = torch.utils.data.Subset(
            train_data,
            proxy_indices.tolist(),
        )
        train_queue = torch.utils.data.DataLoader(
            train_data_proxy, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            generator=torch.Generator().manual_seed(args.seed)
        )

    valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=args.num_workers, pin_memory=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    logging.info(f"Training {len(train_queue.dataset)} samples, validating on {len(valid_queue.dataset)} samples.")

    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, flops, params, train_queue, valid_queue, criterion, alphas_dim


"""
This script is intended to be run as a separate process for each individual in the population. 
It receives the individual's architecture encoding and other necessary information 
as command line arguments, trains the corresponding model, evaluates it, and prints the
results in a structured format that can be parsed by the main process.
"""
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gen', type=int, required=True, help='generation number')
    args.add_argument('--i', type=int, required=True, help='individual index in the population')
    args.add_argument('--seed', type=int, required=True, help='random seed')
    args.add_argument('--individual_x', type=str, required=True, help='string representation of the individual X')
    args.add_argument('--search_space', type=str, required=False, choices=['continuous'], help='search space to use')
    args.add_argument('--supernet_path', type=str, required=True, help='path to the pretrained supernet to load before training the individual')
    args.add_argument('--dataset', type=str, required=True, help='dataset to use')
    args.add_argument('--gpu', type=int, required=True, help='gpu device id')
    args.add_argument('--batch_size', type=int, required=True, help='batch size')
    args.add_argument('--epochs_train_individual', type=int, required=True, help='number of epochs to train individual')
    args.add_argument('--data', type=str, required=True, help='location of the data corpus')
    args.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    args.add_argument('--loss_type', type=str, default='tchebycheff', choices=['tchebycheff', 'ws'], help='type of loss function to use for backpropagation')
    args.add_argument('--mu', type=float, required=True, help='mu for thchebycheff function')
    args.add_argument('--lambda_1', type=float, default=0.5, help='weight for standard loss in ws scalarization')
    args.add_argument('--lambda_2', type=float, default=0.5, help='weight for adversarial loss in ws scalarization')
    args.add_argument('--optimizer', type=str, required=True, help='optimizer to use for training')
    args.add_argument('--learning_rate', type=float, required=True, help='init learning rate')
    args.add_argument('--learning_rate_min', type=float, required=True, help='min learning rate')
    args.add_argument('--momentum', type=float, required=True, help='momentum')
    args.add_argument('--weight_decay', type=float, required=True, help='weight decay')
    args.add_argument('--init_channels', type=int, required=True, help='init channels')
    args.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    args.add_argument('--layers', type=int, required=True, help='total number of layers (cells)')
    args.add_argument('--steps', type=int, required=True, help='number of steps in one cell (intern nodes except input and output)')
    args.add_argument('--multiplier', type=int, required=True, help='number of multiplier for number of channels (intern nodes to concat)')
    args.add_argument('--attack_eps', type=float, required=True, help='attack epsilon')
    args.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    args.add_argument('--cutout_length', type=int, required=True, help='cutout length')
    args.add_argument('--drop_path_prob', type=float, required=True, help='drop path probability')
    args.add_argument('--grad_clip', type=float, required=True, help='gradient clipping')
    args.add_argument('--train_portion', type=float, required=True, help='portion of training data')
    args.add_argument('--weight_individual', type=str, required=True, help='string representation of the weight individual')
    args.add_argument('--nadir_point', type=str, required=True, help='string representation of the nadir point')
    args.add_argument('--ideal_point', type=str, required=True, help='string representation of the ideal point')
    args.add_argument('--proxy_data_dir', type=str, default=None, help='Directory to load the proxy data indices (if provided)')
    args = args.parse_args()

    print(f"Running individual {args.i} with the following arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    individual_X = np.fromstring(args.individual_x.replace('[', '').replace(']', ''), sep=',')
    weight_individual = np.fromstring(args.weight_individual.replace('[', '').replace(']', ''), sep=',')
    nadir_point = np.fromstring(args.nadir_point.replace('[', '').replace(']', ''), sep=',')
    ideal_point = np.fromstring(args.ideal_point.replace('[', '').replace(']', ''), sep=',')


    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        args.device = torch.device('cuda:{}'.format(args.gpu))
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')


    model, optimizer, scheduler, individual_flops, individual_params, train_queue, valid_queue, criterion, alphas_dim = get_model_from_individual(individual_X, args)

    time_evaluation = time.time()
    std_acc, adv_acc, std_loss, adv_loss = infer(valid_queue, model, criterion, args)
    logging.info(
        f'Gen {args.gen} Evaluation {args.i + 1} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_evaluation))} (HH:MM:SS) std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, std_loss {std_loss:.4f}, adv_loss {adv_loss:.4f} ,flops {individual_flops:.2f}, params {individual_params:.2f}')
    assert np.isfinite(std_acc) and np.isfinite(adv_acc) and np.isfinite(std_loss) and np.isfinite(
        adv_loss), f"Non-finite evaluation results for individual {args.i} of generation {args.gen}: std_acc {std_acc}, adv_acc {adv_acc}, std_loss {std_loss}, adv_loss {adv_loss}"


    genotype = alphas_to_genotype(individual_X, alphas_dim, args)

    res = {"std_acc": float(std_acc),
           "adv_acc": float(adv_acc),
           "std_loss": float(std_loss),
           "adv_loss": float(adv_loss),
           "flops": float(individual_flops),
           "params": float(individual_params),
           "genotype": genotype._asdict()}
    print(f"RESULT:{json.dumps(res)}")

    output_filename = "logs" + os.sep + f"result_gen{args.gen}_ind{args.i}.json"
    with open(output_filename, 'w') as f:
        json.dump(res, f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    sys.exit(0)
