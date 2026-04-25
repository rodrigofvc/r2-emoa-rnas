import argparse
import json
import logging
import os
import random
import ssl
import torch
import time

import numpy as np
import torch.nn as nn
import torchvision
import utils_search
from adversarial import fgsm_simple
from micro_space.model_search import Network
from micro_space.model_search import discretize

logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
)

def prepare_args_supernet(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True

    ssl._create_default_https_context = ssl._create_unverified_context

    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # If the supernet path exists, we assume it contains a pretrained supernet and we load it. Otherwise, we create a new supernet model.
    if os.path.exists(args.supernet_path):
        logging.info(f"Loading model from {args.supernet_path}")
        model = utils_search.load_supernet(args.supernet_path)
        model = model.to(args.device)
    else:
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

    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils_search.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
        num_train = split + 32
    logging.info(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    epochs_scheduler = args.epochs_warmup if args.epochs_warmup > 0 else args.epochs_train_supernet
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs_scheduler, eta_min=args.learning_rate_min)


    return model, criterion, optimizer, scheduler, train_queue, valid_queue

def unpack_alphas(vec, shape_alphas, args):
    n_norm = shape_alphas[0] * shape_alphas[1]
    assert type(vec) == np.ndarray

    a_norm_np = vec[:n_norm].reshape(shape_alphas).copy()
    a_norm = torch.tensor(a_norm_np, dtype=torch.float32, device=args.device).requires_grad_(False)

    a_reduction_np = vec[n_norm:].reshape(shape_alphas).copy()
    a_reduction = torch.tensor(a_reduction_np, dtype=torch.float32, device=args.device).requires_grad_(False)
    return [a_norm, a_reduction]


def run_batch_epoch(model, inputs, target, criterion, optimizer, args):
    inputs = inputs.to(args.device)
    target = target.to(args.device)

    optimizer.zero_grad()

    adv_input = fgsm_simple(model, inputs, target)
    adv_input = adv_input.to(args.device)

    std_logits = model(inputs)
    adv_logits = model(adv_input)

    adv_loss = criterion(adv_logits, target)
    std_loss = criterion(std_logits, target)

    total_loss = std_loss * 0.5 + adv_loss * 0.5

    total_loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
    optimizer.step()

    std_predicts = std_logits.argmax(dim=1)
    adv_predicts = adv_logits.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()

def train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, gen, args, warmup=False):
    model.train()
    if warmup:
        epochs = args.epochs_warmup
    else:
        epochs = args.epochs_train_supernet
    for epoch in range(epochs):
        for n_batch, (input, target) in enumerate(train_queue):
            individual_X = pop[n_batch % args.n_population]
            individual_architect = unpack_alphas(individual_X, model.alphas_dim, args)
            model.update_arch_parameters(individual_architect)
            discrete = discretize(individual_architect, model.genotype(), args.device)
            model.update_arch_parameters(discrete)
            std_acc, adv_acc, loss = run_batch_epoch(model, input, target, criterion, optimizer, args)
            if n_batch % args.report_freq == 0:
                logging.info(
                    f'>>>> Gen {gen} | Epoch {epoch}/{epochs} | Batch {n_batch}/{len(train_queue)} | Loss {loss:.4f} | Std Acc {std_acc:.2f}% | Adv Acc {adv_acc:.2f}% ')
        scheduler.step()
        torch.save(model, args.supernet_path)
        torch.save(model, args.supernet_path.replace('.pt', '-backup.pt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gen', type=int, required=True, help='current generation')
    args.add_argument('--seed', type=int, required=True, help='random seed')
    args.add_argument('--search_space', type=str, required=True, choices=['continuous', 'discrete'], help='search space to use')
    args.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'], help='dataset to use')
    args.add_argument('--gpu', type=int, required=True, help='gpu device id')
    args.add_argument('--batch_size', type=int, required=True, help='batch size')
    args.add_argument('--data', type=str, required=True, help='location of the data corpus')
    args.add_argument('--learning_rate', type=float, required=True, help='init learning rate')
    args.add_argument('--learning_rate_min', type=float, required=True, help='min learning rate')
    args.add_argument('--momentum', type=float, required=True, help='momentum')
    args.add_argument('--weight_decay', type=float, required=True, help='weight decay')
    args.add_argument('--init_channels', type=int, required=True, help='init channels')
    args.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    args.add_argument('--layers', type=int, required=True, help='total number of layers (cells)')
    args.add_argument('--steps', type=int, required=True, help='number of steps in one cell (intern nodes except input and output)')
    args.add_argument('--multiplier', type=int, required=True, help='number of multiplier for number of channels (intern nodes to concat)')
    args.add_argument('--fgsm_eps', type=float, required=True, help='attack epsilon')
    args.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    args.add_argument('--cutout_length', type=int, required=True, help='cutout length')
    args.add_argument('--drop_path_prob', type=float, required=True, help='drop path probability')
    args.add_argument('--grad_clip', type=float, required=True, help='gradient clipping')
    args.add_argument('--train_portion', type=float, required=True, help='portion of training data')
    args.add_argument('--warmup', action='store_true', default=False, help='whether to perform warmup training of the supernet before starting the evolutionary search')
    args.add_argument('--epochs_warmup', type=int, default=0, help='number of epochs to warmup supernet')
    args.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
    args.add_argument('--supernet_path', type=str, required=False, help='path to pretrained supernet to load before training the individual')
    args.add_argument('--individuals_X_path', type=str, required=False, help='path to the file containing the individuals X values for the current generation')
    args.add_argument('--report_freq', type=float, required=False, default=45, help='report frequency during training')
    args, unknown_args = args.parse_known_args()

    with open(args.individuals_X_path, 'r') as f:
        individuals_X = []
        individuals_X_dict = json.load(f)
        for ind in individuals_X_dict:
            individuals_X.append(np.fromstring(individuals_X_dict[ind].replace('[', '').replace(']', ''), sep=',', dtype=np.float32))

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
    args.n_population = len(individuals_X)
    model, criterion, optimizer, scheduler, train_queue, valid_queue = prepare_args_supernet(args)

    train_supernet(individuals_X, train_queue, model, criterion, optimizer, scheduler, args.gen, args, warmup=args.warmup)

    os.remove(args.individuals_X_path)