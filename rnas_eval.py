import argparse
import json
import ssl
import time

import numpy as np
import torch
import torchvision

import utils
from evaluation.train_search import infer
from adversarial import get_attack_function


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
    _, valid_transform = utils.data_transforms_cifar10(args)
    test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(test_data)
    indices = list(range(num_train))
    split = int(np.floor(args.test_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 32
    print(f"Evaluation samples: {split}")

    test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=2, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    attack_f = get_attack_function(args.attack)

    return test_queue, criterion, attack_f



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluating architectures found by RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model")
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--params_dir', type=str, required=True, help="params json dir")
    args = parser.parse_args()
    with open(args.params_dir, 'r') as f:
        config = json.load(f)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = torch.load(args.model_path, weights_only=False)

    test_queue, criterion, attack_f = prepare_args(args, model)
    time_stamp = time.time()
    std_accuracy, adv_accuracy, _, _, _ = infer(test_queue, model, criterion, attack_f, args)
    utils.save_params(args, args.model_path.replace('.pt', '_eval_params.json').replace('train', 'eval'))
    print('Evaluation DONE in %.3f seconds' % (time.time() - time_stamp))
    print('Final Test Accuracy: STD %.3f ADV %.3f' % (std_accuracy, adv_accuracy))