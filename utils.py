import lzma
import time

import torchprofile
import numpy as np
import torch
from pymoo.util.ref_dirs import get_reference_directions
import torchvision.transforms as transforms
import os
import pickle

from evaluation.model import NetworkCIFAR

# Load R2 weights for the i-th population size
def get_weights_r2(n):
    file = 'r2_weights' + os.sep + 'weights_' + str(n) + '.pkl'
    with open(file, 'rb') as f:
        weights_r2 = pickle.load(f)
    return weights_r2


# Store R2 weights for the i-th population size
def store_weights(n, k):
    file = 'r2_weights' + os.sep + 'weights_' + str(n) + '.pkl'
    directions_set = {}
    for i in range(n):
        w = get_reference_directions("energy", k, 2 * n - i, seed=1)
        directions_set[i] = w
    print(directions_set[0].shape)
    print(directions_set[1].shape, type(directions_set[1]))
    with open(file, 'wb') as f:
        pickle.dump(directions_set, f)

def save_archive_accuracy(archive, archive_path):
    archive_path += 'archive_accuracy'
    np_archive = [[p.std_acc, p.adv_acc] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def save_archive(archive, archive_path):
    archive_path += 'archive'
    np_archive = [p.F for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def create_experiment_dir(algorithm, dataset, seed):
    base_dir = 'results' + os.sep + algorithm + os.sep + dataset
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    exp_dir = base_dir + os.sep + str(time.strftime('%Y-%m-%d_%H-%M-%S_')) + str(seed)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir + os.sep


def store_metrics(objective_space, algorithm):
    # TODO Obtener peores valores de hyp
    pass

def save_model(model, model_path):
    model_path += 'super-net.pt'
    torch.save(model.state_dict(), model_path)

def save_architectures(architectures, architect_path):
    architect_path += 'architectures.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(architectures, f)

def read_architectures(architect_path):
    with lzma.open(architect_path, 'rb') as f:
        architectures = pickle.load(f)
    for l_tensor in architectures:
        print(l_tensor[0].shape)
        print(l_tensor[1].shape)
    return architectures


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

# Returns the flops and number of parameters of a model given its genotype
def get_model_metrics(genotype, model):
    discretized_model = NetworkCIFAR(model.C, model.num_classes, model.layers, auxiliary=False, genotype=genotype)
    x = torch.randn(1, 3, 32, 32)
    macs = torchprofile.profile_macs(discretized_model, x) / 1e6
    flops = 2 * macs
    params = sum(v.numel() for v in filter(lambda p: p.requires_grad, discretized_model.parameters())) / 1e6
    return round(flops, 4), round(params, 4)


