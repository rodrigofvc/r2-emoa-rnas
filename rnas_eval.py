import argparse
import csv
import logging
import os
import ssl
import time

import numpy as np
import torch
import torchvision
import torchattacks

import utils

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def prepare_args(args, model):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device
    model.to(args.device)

    ssl._create_default_https_context = ssl._create_unverified_context
    set_seed(args.seed)
    _, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        test_data = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(test_data)
    indices = list(range(num_train))
    split = int(np.floor(args.test_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
    print(f"Test samples: {split}")

    test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=args.num_workers, pin_memory=False)

    logging.info(f"Test samples: {len(test_queue)* args.batch_size}")
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return test_queue, criterion

def eval_clean(test_queue, model, args):
    std_correct = 0
    total_std_loss = 0.0
    total = 0
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    for step, (inputs, target) in enumerate(test_queue):
        inputs = inputs.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            std_logits = model(inputs)
            std_loss = criterion(std_logits, target).item()

        std_predicts = std_logits.argmax(dim=1)
        std_correct += (std_predicts == target).sum().item()
        total += target.size(0)
        total_std_loss += std_loss
    std_accuracy = std_correct / total
    total_std_loss = total_std_loss / len(test_queue)
    return std_accuracy * 100.0, total_std_loss


def eval_adv(test_queue, model, attack_name, args):
    adv_correct = 0
    total_adv_loss = 0.0
    total = 0
    model.eval()
    if attack_name == 'FGSM':
        attack = torchattacks.FGSM(model, eps=8/255)
    elif attack_name == 'BIM_10':
        attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    elif attack_name == 'PGD_7':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)
    elif attack_name == 'PGD_10':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    elif attack_name == 'PGD_20':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
    elif attack_name == 'CW_0.1':
        attack = torchattacks.CW(model, c=0.1, steps=50)
    elif attack_name == 'CW_0.01':
        attack = torchattacks.CW(model, c=0.01, steps=50)
    else:
        raise ValueError(f"Unknown attack name: {attack_name}")
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    attack.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    for step, (inputs, target) in enumerate(test_queue):
        inputs = inputs.to(args.device)
        target = target.to(args.device)
        adv_input = attack(inputs, target)
        adv_input = adv_input.to(args.device)

        with torch.no_grad():
            adv_logits = model(adv_input)
            adv_loss = criterion(adv_logits, target).item()

        adv_predicts = adv_logits.argmax(dim=1)
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)
        total_adv_loss += adv_loss
    adv_accuracy = adv_correct / total
    total_adv_loss = total_adv_loss / len(test_queue)
    return adv_accuracy * 100.0, total_adv_loss


if __name__ == '__main__':

    """
    python3 -X dev rnas_eval.py --seed 12 --algorithm r2-emoa --dataset cifar10 \
    --batch_size 32 --model_path results/r2-emoa/cifar10/2026-04-20_11-37-00_18906049/train/epoch_90_model.pt 
    
    python3 -X dev rnas_eval.py --algorithm r2-emoa --dataset cifar10 --batch_size 32 --archive_path results/r2-emoa/cifar10/2026-04-20_11-37-00_18906049/train/archive
    """
    # python rnas_eval.py --seed 12 --algorithm r2-emoa --dataset cifar100 --batch_size 256 --model_path results/r2-emoa/cifar100/2026-05-08_10-04-43_18906049/train/full_trained_model.pt
    parser = argparse.ArgumentParser(description="Evaluating architectures found by RNAS")
    parser.add_argument('--seed', type=int, default=18906049, help='random seed')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset for training')
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_path', type=str, default=None, help="Path to the saved model")
    parser.add_argument('--archive_path', type=str, default=None, help="Path to the models archive (if applicable)")
    parser.add_argument('--test_portion', type=float, default=1, help='portion of test data to evaluate')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='debug cuda')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.info('Running evaluation with the following config:')
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.model_path is not None:
        model = torch.load(args.model_path, weights_only=False)

        attack_f_list = ['PGD_7', 'BIM_10', 'PGD_10', 'PGD_20', 'FGSM', 'CW_0.1', 'CW_0.01']

        test_queue, criterion = prepare_args(args, model)
        for i, attack_f in enumerate(attack_f_list):
            time_stamp = time.time()
            adv_accuracy, adv_loss = eval_adv(test_queue, model, attack_f, args)
            std_accuracy, std_loss = eval_clean(test_queue, model, args)
            logging.info(f"Attack {attack_f}: ADV accuracy {adv_accuracy:.3f}, time ({time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))})")
            with open('test-evaluations.csv', mode='a', newline='') as csvfile:
                fieldnames = ['algorithm', 'dataset', 'model', 'attack', 'std_accuracy', 'adv_accuracy']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'algorithm': args.algorithm,
                                 'dataset': args.dataset,
                                 'model': args.model_path.replace(os.sep, '/'),
                                 'attack': attack_f,
                                 'std_accuracy': std_accuracy,
                                 'adv_accuracy': adv_accuracy})
    elif args.archive_path is not None:
        models_dir = os.listdir(args.archive_path)
        models_dir = [d for d in models_dir if d.endswith('.pt')]
        attack_f_list = ['clean', 'BIM_10', 'PGD_7', 'PGD_10', 'PGD_20', 'FGSM', 'CW_0.1', 'CW_0.01']
        fieldnames = ['algorithm', 'dataset', 'model', 'flops', 'params', 'attack', 'accuracy', 'loss']
        for j, model_file in enumerate(models_dir):
            model_path = os.path.join(args.archive_path, model_file)
            model = torch.load(model_path, weights_only=False)
            test_queue, criterion = prepare_args(args, model)
            flops, params = utils.get_model_metrics(model)
            for i, attack_f in enumerate(attack_f_list):
                time_stamp = time.time()
                if attack_f == 'clean':
                    accuracy, loss = eval_clean(test_queue, model, args)
                else:
                    accuracy, loss = eval_adv(test_queue, model, attack_f, args)
                logging.info(f"Algorithm {args.algorithm} Model {j+1}/{len(models_dir)} Attack {attack_f}: accuracy {accuracy:.3f}, time ({time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))})")
                with open('test-evaluations.csv', mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'algorithm': args.algorithm,
                                     'dataset': args.dataset,
                                     'model': model_path.replace(os.sep, '/'),
                                     'flops': flops,
                                     'params': params,
                                     'attack': attack_f,
                                     'accuracy': accuracy,
                                     'loss': loss})
