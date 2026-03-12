import argparse
import json
import ssl
import time

import torch
from torch import nn, amp
import numpy as np
import torchvision

import utils
import utils_train
from micro_space.model import NetworkCIFAR
from adversarial import get_attack_function

def prepare_args(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

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
        num_workers=0, pin_memory=False)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    attack_f = get_attack_function(args.attack)

    return train_queue, criterion, attack_f

def prepare_arch_genotype(architecture, algorithm):
    if algorithm == 'r2-emoa' or algorithm == 'nevonas':
        genotype = architecture.genotype
    elif algorithm == 'nsganet':
        genotype = architecture
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    model = NetworkCIFAR(args.init_channels, args.n_classes, args.layers, args.auxiliary, genotype).to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)
    return optimizer, scheduler, model

def train(train_queue, model, criterion, scheduler, optimizer, attack_f, args):
    std_correct = 0
    adv_correct = 0
    total_loss_mean = 0
    total = 0
    model.train()
    attack = attack_f(model)
    for n_batch, (input, target) in enumerate(train_queue):
        times_stamp = time.time()
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        optimizer.zero_grad()


        adv_X = attack(input, target)
        logits = model(input)

        logits_adv = model(adv_X)
        adv_loss = criterion(logits_adv, target)

        natural_loss = criterion(logits, target)

        total_loss = args.lambda_1 * natural_loss + args.lambda_2 * adv_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        std_predicts = logits.argmax(dim=1)
        adv_predicts = logits_adv.argmax(dim=1)
        std_correct += (std_predicts == target).sum().item()
        adv_correct += (adv_predicts == target).sum().item()
        total_loss_mean += total_loss.item()
        total += target.size(0)
        if n_batch % args.report_freq == 0:
            print(
                f">>>> batch {n_batch + 1}/{len(train_queue)} ({time.strftime('%H:%M:%S', time.gmtime(time.time() - times_stamp))}) (HH:MM:SS): std_acc {std_correct / total * 100:.2f}%, adv_acc {adv_correct / total * 100:.2f}%, loss {total_loss_mean:.4f}")
    std_accuracy = std_correct / total
    adv_accuracy = adv_correct / total
    total_loss_mean /= total
    return std_accuracy * 100.0, adv_accuracy * 100.0, total_loss_mean

def smooth_tchebycheff_sc_loss(mu, std_loss, adv_loss, flops, params, r2_weights, z_ref_stch, nadir_point, ideal_point):
    losses_grad = torch.stack([std_loss, adv_loss])
    losses_const = torch.stack([flops, params]).detach()
    losses = torch.cat([losses_grad, losses_const])

    ideal = ideal_point.detach()
    nadir = nadir_point.detach()
    z_ref = z_ref_stch.detach()
    weights = r2_weights.detach()

    values = torch.abs(losses - ideal) / torch.clamp(torch.abs(nadir - ideal), 1e-6)
    stch_value = mu * torch.logsumexp(weights * (values - z_ref) / mu, dim=-1)
    return stch_value

def train_individual(model, train_queue, criterion, optimizer, attack_f, args, weight_individual, nadir_point, ideal_point, scheduler):
    attack = attack_f(model)
    z_ref_stch = torch.zeros(4, device=args.device)
    model_flops, model_parameters = utils.get_model_metrics(None, model, discrete=True)
    model_flops, model_parameters = torch.tensor(float(model_flops), device=args.device), torch.tensor(
        float(model_parameters), device=args.device)
    model.train()
    time_stamp = time.time()    
    for epoch in range(args.epochs_train_individual):
        for n_batch, (input, target) in enumerate(train_queue):
            std_acc, adv_acc, loss = run_batch_epoch(model, input, target, criterion, optimizer, attack, None, args, model_flops, model_parameters, weight_individual, z_ref_stch, nadir_point, ideal_point)
        scheduler.step()

def run_batch_epoch(model, input, target, criterion, optimizer, attack, scaler, args, model_flops, model_parameters, r2_weights, z_ref_stch, nadir_point, ideal_point):

    input = input.to(args.device)
    target = target.to(args.device)

    optimizer.zero_grad(set_to_none=True)

    # for adversarial training

    adv_input = attack(input, target)
    adv_input = adv_input.to(args.device)
    concat_images = torch.cat([input, adv_input], dim=0).contiguous()

    logits = model(concat_images)
    std_logits, logits_adv = torch.split(logits, input.size(0), dim=0)

    adv_loss = criterion(logits_adv, target)
    std_loss = criterion(std_logits, target)

    total_loss = smooth_tchebycheff_sc_loss(args.mu, std_loss, adv_loss, model_flops, model_parameters, r2_weights, z_ref_stch, nadir_point, ideal_point)

    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    std_predicts = std_logits.argmax(dim=1)
    adv_predicts = logits_adv.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()

def infer(valid_queue, model, criterion, attack, args):
    std_correct = 0
    adv_correct = 0
    std_loss_mean = 0
    adv_loss_mean = 0
    total = 0
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input  = input.to(args.device)
        target = target.to(args.device)

        adv_input = attack(input, target)
        adv_input = adv_input.to(args.device)

        concat_input = torch.cat([input, adv_input], dim=0).contiguous()

        with torch.no_grad():
            logits = model(concat_input)
        std_logits, adv_logits = torch.split(logits, input.size(0), dim=0)
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
    std_loss_mean /= total
    adv_loss_mean /= total
    return std_accuracy * 100.0, adv_accuracy * 100.0, std_loss_mean, adv_loss_mean

# This file trains architectures found by RNAS
# it loads the architectures and supernet from specified paths
# and trains each architecture from scratch, saving results in a new directory.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training architectures found by RNAS")
    parser.add_argument('--seed', type=int, default=18906049, help='random seed')
    parser.add_argument('--algorithm', type=str, default='[r2-emoa, nevonas, nsganet]', help='which algorithm was used to search')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--arch_path', type=str, required=True, help="Path to the saved architecture")
    parser.add_argument('--supernet_path', type=str, help="Path to the saved supernet model")
    parser.add_argument('--trained_arch_path', type=str, required=True, help='Path to store the trained architecture')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--params_dir', type=str, required=True, help="params json dir")
    args = parser.parse_args()
    # python3 rnas_train.py --seed 18906049 --algorithm nevonas --dataset cifar10 --batch_size 32 --epochs 10 --arch_path ./results/nevonas/cifar10/search-S1-20260109-190243-cifar10/architectures --trained_arch_path /results/nevonas/cifar10/search-S1-20260109-190243-cifar10/train --params_dir params/train/params-cifar-10-r2-emoa-20.json
    with open(args.params_dir, 'r') as f:
        config = json.load(f)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)

    print('Running training with config:')
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    best_ind, best_path = utils_train.get_best_architecture_adversarial(args.arch_path, args.algorithm)

    individual = utils_train.load_architecture(best_path)
    #supernet = utils.load_model(args.supernet_path)
    logs_architectures = []


    train_queue, criterion, attack_f = prepare_args(args)

    optimizer, scheduler, model = prepare_arch_genotype(individual, args.algorithm)
    time_stamp_train = time.time()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        time_stamp = time.time()
        std_acc, adv_acc, loss_ws = train(train_queue, model, criterion, scheduler, optimizer, attack_f, args)
        print(f">>>> Epoch {epoch + 1} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))} (HH:MM:SS) std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, loss {loss_ws:.4f}")
        if (epoch + 1) % args.freq_save == 0:
            utils.save_model(model, args.trained_arch_path, f"model_epoch_{epoch}.pt")
    utils.save_params(args, args.trained_arch_path)
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_train))} (HH:MM:SS)")