import argparse
import json
import logging
import os
import ssl
import time
from pathlib import Path
import re

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torchvision

import utils
import utils_train
from micro_space.model import NetworkCIFAR
from adversarial import fgsm_simple

def prepare_args(args_, genotype):
    initial_epoch = 0
    model = None
    if args_.reload_dir is not None:
        # the execution is a reload and we need to load all the variables from the previous execution as well as the most recent model
        with open(args_.reload_dir + os.sep + 'params.json', 'r') as f:
            saved_params = json.load(f)
            args = argparse.Namespace(**saved_params)
        logging.info(f">>>> Reloading training from {args_.reload_dir} with parameters:")

        pattern = r"epoch_(\d+)_model.pt"

        files = os.listdir(args_.reload_dir)

        matched_files = []
        for file in files:
            match = re.match(pattern, file)
            if match:
                epoch = int(match.group(1))
                matched_files.append((epoch, file))

        sorted_files = sorted(matched_files, key=lambda x: x[0], reverse=True)
        j = 0
        # try to load the recent model, if it fails, try the previous one
        while j < len(sorted_files):
            most_recent = sorted_files[j]
            try:
                model = torch.load(args_.reload_dir + os.sep + most_recent[1], map_location='cpu', weights_only=False)
            except Exception as e:
                logging.warning(f"Failed to load model from {most_recent[1]}: {e}, loading previous model if available.")
                j += 1
                continue
            else:
                logging.info(f"Successfully loaded model from epoch {most_recent[1]}.")
                initial_epoch = most_recent[0] + 1
                break
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs - initial_epoch, eta_min=args.learning_rate_min)
    else:
        # the execution is new and we need to initialize all the variables
        args = args_
        n_classes = 10 if args.dataset == 'cifar10' else 100
        model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype)

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            foreach=False
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.learning_rate_min)


    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    args.device = device

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
    print(f"Training samples: {split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=args.num_workers, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return args, train_queue, criterion, model, initial_epoch, optimizer, scheduler

def train(train_queue, model, criterion, scheduler, optimizer, args):
    adv_correct = 0
    total = 0
    model.to(args.device)
    model.train()
    for n_batch, (inputs, target) in enumerate(train_queue):
        inputs = inputs.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()

        adv_inputs, std_logits = fgsm_simple(model, inputs, target, args.attack_eps)

        logits_adv = model(adv_inputs)
        adv_loss = criterion(logits_adv, target)
        adv_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
        optimizer.step()

        adv_predicts = logits_adv.argmax(dim=1)
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)
        if n_batch % args.report_freq == 0:
            logging.info(
                f">>>> batch {n_batch + 1}/{len(train_queue)} : adv_acc {adv_correct / total * 100:.2f}%")
    scheduler.step()
    adv_accuracy = adv_correct / total
    return adv_accuracy * 100.0


def train_amp(train_queue, model, criterion, scheduler, optimizer, args):
    adv_correct = 0
    total = 0

    model.to(args.device)
    model.train()

    scaler = GradScaler('cuda')

    for n_batch, (inputs, target) in enumerate(train_queue):
        inputs = inputs.to(args.device, non_blocking=False)
        target = target.to(args.device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        adv_inputs, std_logits = fgsm_simple(model, inputs, target, args.attack_eps)

        with autocast(device_type="cuda"):
            logits_adv = model(adv_inputs)
            adv_loss = criterion(logits_adv, target)

        scaler.scale(adv_loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
            foreach=False
        )

        scaler.step(optimizer)
        scaler.update()

        adv_predicts = logits_adv.argmax(dim=1)
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)

        if n_batch % args.report_freq == 0:
            logging.info(f">>>> batch {n_batch + 1}/{len(train_queue)} : adv_acc {adv_correct / total * 100:.2f}%")

    scheduler.step()

    adv_accuracy = adv_correct / total
    return adv_accuracy * 100.0

def smooth_tchebycheff_sc_loss(mu, std_loss, adv_loss, flops, params, weights, z_ref_stch, nadir_point, ideal_point):
    loss_type = std_loss.dtype
    losses_grad = torch.stack([std_loss, adv_loss])
    losses_const = torch.stack([flops, params]).detach().to(dtype=loss_type)
    losses = torch.cat([losses_grad, losses_const])
    # TESTING 2 objectives
    losses = losses_grad
    ideal_point = ideal_point[:len(losses)]
    nadir_point = nadir_point[:len(losses)]
    weights = torch.tensor([0.5, 0.5], device=losses.device, dtype=loss_type)
    z_ref_stch = z_ref_stch[:len(losses)]
    # TESTING
    values = torch.abs(losses - ideal_point) / torch.clamp(torch.abs(nadir_point - ideal_point), 1e-6)
    stch_value = mu * torch.logsumexp(weights * (values - z_ref_stch) / mu, dim=-1)
    if not torch.isfinite(stch_value):
        raise ValueError(f"stch_value is not finite {stch_value.item()}")
    return stch_value

def train_individual(model, flops, params, train_queue, criterion, optimizer, args, r2_weight, nadir_point, ideal_point, scheduler):
    weight_individual = torch.tensor(r2_weight, device=args.device, dtype=torch.float32)
    model_flops = torch.tensor(float(flops), device=args.device, dtype=torch.float32)
    model_parameters = torch.tensor(float(params), device=args.device, dtype=torch.float32)
    z_ref_stch = torch.zeros(4, device=args.device, dtype=torch.float32)
    nadir_point = torch.tensor(nadir_point, device=args.device, dtype=torch.float32)
    ideal_point = torch.tensor(ideal_point, device=args.device, dtype=torch.float32)
    model.train()
    for epoch in range(args.epochs_train_individual):
        if args.loss_type == 'tchebycheff':
            for n_batch, (inputs, target) in enumerate(train_queue):
                run_batch_epoch(model, inputs, target, criterion, optimizer, args, model_flops, model_parameters, weight_individual, z_ref_stch, nadir_point, ideal_point)
        elif args.loss_type == 'ws':
            for n_batch, (inputs, target) in enumerate(train_queue):
                run_batch_epoch_ws(model, inputs, target, criterion, optimizer, args)
        scheduler.step()


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

def run_batch_epoch(model, inputs, target, criterion, optimizer, args, model_flops, model_parameters, r2_weights, z_ref_stch, nadir_point, ideal_point):

    inputs = inputs.to(args.device, non_blocking=True)
    target = target.to(args.device, non_blocking=True)

    optimizer.zero_grad()

    adv_input, std_logits = fgsm_simple(model, inputs, target, args.attack_eps)
    adv_input = adv_input.to(args.device, non_blocking=True)

    adv_logits = model(adv_input)

    adv_loss = criterion(adv_logits, target)
    std_loss = criterion(std_logits, target)
    
    total_loss = smooth_tchebycheff_sc_loss(args.mu, std_loss, adv_loss, model_flops, model_parameters, r2_weights, z_ref_stch, nadir_point, ideal_point)

    total_loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
    optimizer.step()

    std_predicts = std_logits.argmax(dim=1)
    adv_predicts = adv_logits.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()

def infer(valid_queue, model, criterion, args):
    std_correct = 0
    adv_correct = 0
    std_loss_mean = 0
    adv_loss_mean = 0
    total = 0
    model.eval()
    for step, (inputs, target) in enumerate(valid_queue):
        inputs  = inputs.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        
        adv_input, std_logits = fgsm_simple(model, inputs, target, args.attack_eps)

        with torch.no_grad():
            adv_logits = model(adv_input)

            adv_loss = criterion(adv_logits, target)
            std_loss = criterion(std_logits, target)
        
            std_predicts = std_logits.argmax(dim=1)
            adv_predicts = adv_logits.argmax(dim=1)
            std_correct += (std_predicts == target).sum().item()
            adv_correct += (adv_predicts == target).sum().item()
            total += target.size(0)

            std_loss_mean += std_loss.item()
            adv_loss_mean += adv_loss.item()
    std_accuracy = std_correct / total
    adv_accuracy = adv_correct / total
    std_loss_mean /= len(valid_queue)
    adv_loss_mean /= len(valid_queue)
    return std_accuracy * 100.0, adv_accuracy * 100.0, std_loss_mean, adv_loss_mean

# This file trains architectures found by RNAS
# it loads the architectures and supernet from specified paths
# and trains each architecture from scratch, saving results in a new directory.
if __name__ == '__main__':

    """
    python3 rnas_train.py --seed 12 --algorithm r2-emoa --search_space discrete --dataset cifar10 \
    --batch_size 32 --epochs 100 --data ./data --learning_rate 0.025 --learning_rate_min 0.001\
    --momentum 0.9 --weight_decay 3e-4 --grad_clip 5.0 --report_freq 50 --freq_save 10 --gpu 0\
    --init_channels 16 --layers 8 --steps 4 --multiplier 4 --train_portion 0.5\
    --archive_path results/r2-emoa/cifar10/2026-04-20_11-37-00_18906049/search/population_data.json\
    --amp --train_archive
    
    python3 rnas_train.py --seed 12 --algorithm r2-emoa --search_space discrete --dataset cifar10 --reload_dir auto-last
    """
    parser = argparse.ArgumentParser(description="Training architectures found by RNAS")
    parser.add_argument('--seed', type=int, default=18906049, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa', 'nevonas', 'nsganet', 'cars', 'r2-emoa-one-shot'], help='which algorithm was used to search')
    parser.add_argument('--search_space', type=str, default='discrete', help='which search space was used to search')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--freq_save', type=int, default=10, help='frequency of saving the model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--steps', type=int, default=4, help='number of steps in one cell')
    parser.add_argument('--multiplier', type=int, default=4, help='number of multiplier for channels')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='debug cuda')
    parser.add_argument('--reload_dir', type=str, default=None, help='reload from this directory')
    parser.add_argument('--amp', action='store_true', default=False, help='use automatic mixed precision')
    parser.add_argument('--train_archive', action='store_true', default=False, help='train the whole archive of architectures and store the evaluation results')
    parser.add_argument('--archive_path', type=str, default=None, help='path to the archive of architectures')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    if args.debug_cuda:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.reload_dir is None:
        base_dir = args.archive_path.split(os.sep)
        if len(base_dir) == 1:
            base_dir = args.archive_path.split('/')
        print(f'base_dir: {base_dir}')
        base_dir = base_dir[:base_dir.index('search')]
        results_dir = os.sep.join(base_dir) + os.sep + 'train' + os.sep
        args.save_path_final_model = results_dir
        utils.save_params(args, results_dir)
    elif args.reload_dir == 'auto-last':
        # reload the last training in the results directory for training
        base_dir = Path('results') / args.algorithm / args.dataset
        if not base_dir.exists():
            raise ValueError(f"No results found for algorithm {args.algorithm} and dataset {args.dataset}")
        pre_dirs = [pre_d for pre_d in base_dir.iterdir() if pre_d.is_dir()]
        dirs = [d for d in base_dir.iterdir() if d.is_dir() and 'results' + os.sep + args.algorithm + os.sep + args.dataset + os.sep + d.name + os.sep + "train" in [str(f) for f in d.iterdir()]]

        if not dirs:
            raise ValueError("No experiments found for the given algorithm and dataset")

        latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)

        results_dir = str(latest_dir) + os.sep + "train"
        args.reload_dir = results_dir
    else:
        results_dir = args.reload_dir
    print(f'Results dir: {results_dir}')

    args.save_path_final_model = results_dir

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if not args.train_archive:
        # train only the best architecture found by RNAS
        best_genotype = None
        if args.reload_dir is None:
            best_genotype = utils_train.get_best_genotype_adversarial(args.archive_path, args)

        args, train_queue, criterion, model, initial_epoch, optimizer, scheduler = prepare_args(args, best_genotype)

        time_stamp_train = time.time()
        for epoch in range(initial_epoch, args.epochs):
            logging.info(f"Epoch {epoch}/{args.epochs}")
            time_stamp = time.time()
            if args.amp:
                adv_acc = train_amp(train_queue, model, criterion, scheduler, optimizer, args)
            else:
                adv_acc = train(train_queue, model, criterion, scheduler, optimizer, args)
            logging.info(f">>>> Epoch {epoch} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))} (HH:MM:SS) adv_acc {adv_acc:.2f}% ")
            if epoch % args.freq_save == 0:
                utils.save_model(model, args.save_path_final_model, f"epoch_{epoch}_model.pt")
        utils.save_model(model, args.save_path_final_model, f"full_trained_model.pt")
        logging.info(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_train))} (HH:MM:SS)")
    else:
        # train the whole archive of architectures and store the trained models in the same directory
        archive_genotypes = utils_train.get_genotypes_from_archive(args.archive_path, args)
        for i, genotype in enumerate(archive_genotypes):
            args, train_queue, criterion, model, initial_epoch, optimizer, scheduler = prepare_args(args, genotype)
            logging.info(f">>>> Training individual {i}/{len(archive_genotypes)-1}")
            for epoch in range(initial_epoch, args.epochs):
                logging.info(f"Individual {i}/{len(archive_genotypes)-1} Epoch {epoch}/{args.epochs}")
                time_stamp = time.time()
                if args.amp:
                    adv_acc = train_amp(train_queue, model, criterion, scheduler, optimizer, args)
                else:
                    adv_acc = train(train_queue, model, criterion, scheduler, optimizer, args)
                logging.info(
                    f">>>> Individual {i}/{len(archive_genotypes)-1} Epoch {epoch} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))} (HH:MM:SS) adv_acc {adv_acc:.2f}% ")
            utils.save_model(model, args.save_path_final_model + "archive" + os.sep, f"individual_{i}_model.pt")