import os
import argparse
import logging

import numpy as np
import torch.nn as nn
import torch
import torchvision
from resnet import resnet20, resnet56
import sys
sys.path.append('..')
import utils
from adversarial import fgsm_simple

def build_resnet18_cifar(num_classes: int) -> nn.Module:
    model = resnet20()

    # ImageNet 224x224 to CIFAR 32x32
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model_from_individual(args):
    if args.dataset == 'cifar10':
        model = resnet20(num_classes=10)
    elif args.dataset == 'cifar100':
        model = resnet56(num_classes=100)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)
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
        split = 96
        num_train = split + 96

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=args.num_workers, pin_memory=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))


    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, train_queue, criterion

def run_batch_epoch_ws(model, inputs, target, criterion, optimizer, args):
    inputs = inputs.to(args.device, non_blocking=True)
    target = target.to(args.device, non_blocking=True)

    optimizer.zero_grad()

    adv_input, std_logits = fgsm_simple(model, inputs, target, args.attack_eps)
    adv_input = adv_input.to(args.device, non_blocking=True)

    adv_logits = model(adv_input)

    adv_loss = criterion(adv_logits, target)
    std_loss = criterion(std_logits, target)

    total_loss = std_loss * args.lambda_1 + adv_loss * args.lambda_2

    total_loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
    optimizer.step()

    std_predicts = std_logits.argmax(dim=1)
    adv_predicts = adv_logits.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()


# USAGE
"""
python3 train_models.py --dataset cifar10 --seed 42 --batch_size 32\
--lambda_1 0.5 --lambda_2 0.5 --learning_rate 0.001 --learning_rate_min 1e-5\
--weight_decay 5e-4 --momentum 0.9 --grad_clip 5.0 --epochs 2 --train_portion 0.5\
--num_workers 0 --gpu 0 --output_dir models --attack_eps 0.03137254901960784
"""
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train a surrogate model on CIFAR datasets')
    args.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to train the model on')
    args.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet56'], help='Model architecture to use')
    args.add_argument('--data', type=str, default='../data', help='Directory to download/load the dataset')
    args.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    args.add_argument('--lambda_1', type=float, default=0.5, help='Lambda value for standard loss function')
    args.add_argument('--lambda_2', type=float, default=0.5, help='Lambda value for adversarial loss function')
    args.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    args.add_argument('--learning_rate_min', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    args.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    args.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    args.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping value')
    args.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
    args.add_argument('--train_portion', type=float, default=0.5, help='Portion of the dataset to use for training')
    args.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args.add_argument('--gpu', type=int, default=0, help='GPU id to use for training')
    args.add_argument('--output_dir', type=str, default='./models/', help='Directory to save the trained model')
    args.add_argument('--attack_eps', type=float, default=8/255, help='Epsilon value for adversarial training')
    args.add_argument("--cutout", action="store_true", default=False)
    args.add_argument("--cutout_length", type=int, default=16)
    args = args.parse_args()

    if torch.cuda.is_available():
        torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    model, optimizer, scheduler, train_queue, criterion = get_model_from_individual(args)

    model.to(args.device)
    model.train()
    for e in range(args.epochs):
        std_correct_total = 0
        adv_correct_total = 0
        total_size = 0
        for (step, (inputs, target)) in enumerate(train_queue):
            std_correct, adv_correct, _ = run_batch_epoch_ws(model, inputs, target, criterion, optimizer, args)
            std_correct_total += std_correct
            adv_correct_total += adv_correct
            total_size += inputs.size(0)
        logging.info(f"Epoch [{e+1}/{args.epochs}] completed. Average std_acc: {std_correct_total/total_size}, Average adv_acc: {adv_correct_total/total_size}")
        scheduler.step()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_{args.dataset}.pth"))