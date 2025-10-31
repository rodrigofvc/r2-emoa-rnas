import csv
import lzma
import time
import matplotlib.pyplot as plt
import torchprofile
import numpy as np
import torch
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
import torchvision.transforms as transforms
import os
import pickle

from evaluation.model import NetworkCIFAR
from indicators import r2


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

# Create experiment directory structure for searching algorithms
def create_experiment_dir(algorithm, dataset, seed):
    base_dir = 'results' + os.sep + algorithm + os.sep + dataset
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    exp_dir = base_dir + os.sep + str(time.strftime('%Y-%m-%d_%H-%M-%S_')) + str(seed) + os.sep + 'search' + os.sep
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir


def store_metrics(epoch, population, objective_space, args, statistics, weights_r2):
    # get maximum and minimum for each objective
    statistics['max_f1'] = max(statistics['max_f1'], np.max(objective_space[:, 0]))
    statistics['max_f2'] = max(statistics['max_f2'], np.max(objective_space[:, 1]))
    statistics['max_f3'] = max(statistics['max_f3'], np.max(objective_space[:, 2]))
    statistics['max_f4'] = max(statistics['max_f4'], np.max(objective_space[:, 3]))
    statistics['min_f1'] = min(statistics['min_f1'], np.min(objective_space[:, 0]))
    statistics['min_f2'] = min(statistics['min_f2'], np.min(objective_space[:, 1]))
    statistics['min_f3'] = min(statistics['min_f3'], np.min(objective_space[:, 2]))
    statistics['min_f4'] = min(statistics['min_f4'], np.min(objective_space[:, 3]))
    n_obj = objective_space.shape[1]
    # compute hypervolume
    ind = HV(ref_point=np.ones(n_obj))
    population_array = np.array([ind.F_norm for ind in population])
    hyp = ind(population_array)
    statistics['hyp_log'].append(hyp)
    # compute r2
    r2_population = r2(population, weights_r2[population.size], np.zeros(n_obj))
    statistics['r2_log'].append(r2_population)
    row_hyp = [args.algorithm, args.dataset, args.attack['name'], epoch, 'hv', hyp, args.save_path_final_model]
    row_r2 = [args.algorithm, args.dataset, args.attack['name'], epoch, 'r2', r2_population, args.save_path_final_model]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    file.close()


def save_supernet(model, model_path):
    model_path += 'super-net.pt'
    torch.save(model.state_dict(), model_path)

def save_model(model, model_path, name):
    model_path += os.sep + name
    torch.save(model.state_dict(), model_path)

def save_log_train(arch_path, log):
    arch_path += 'train_log.csv'

    with open(arch_path, 'a') as f:
        log_str = ','.join([str(item) for item in log])
        f.write(log_str + '\n')

def load_model(model_path):
    model_path += 'super-net.pt'
    state_dict = torch.load(model_path)
    return state_dict

def save_architecture(i, individual, architect_path):
    architect_path += 'architectures' + os.sep
    if not os.path.exists(architect_path):
        os.makedirs(architect_path)
    architect_path += f'arch_{i}.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(individual, f)


def save_architectures(architectures, architect_path):
    architect_path += 'architectures.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(architectures, f)

def save_statistics_to_csv(statistics, csv_path):
    csv_path += 'statistics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in statistics.items():
            writer.writerow([key, value])

def load_architectures(architect_path):
    architect_path += 'architectures.xz'
    with lzma.open(architect_path, 'rb') as f:
        architectures = pickle.load(f)
    return architectures

def read_architectures(architect_path):
    with lzma.open(architect_path, 'rb') as f:
        architectures = pickle.load(f)
    for l_tensor in architectures:
        print(l_tensor[0].shape)
        print(l_tensor[1].shape)
    return architectures

def plot_archive_accuracy(archive_accuracy, archive_path):
    archive_path += 'archive_accuracy.png'
    std_acc = [p.std_acc for p in archive_accuracy]
    adv_acc = [p.adv_acc for p in archive_accuracy]
    plt.figure(figsize=(8, 6))
    plt.scatter(std_acc, adv_acc, c='blue', marker='o')
    plt.title('Archive Accuracy')
    plt.xlabel('Standard Accuracy (%)')
    plt.ylabel('Adversarial Accuracy (%)')
    plt.grid(True)
    plt.savefig(archive_path)
    plt.close()

def plot_hypervolume(statistics, path):
    path += 'hypervolume.png'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp_log'], marker='o', color='blue')
    plt.title('Hypervolume over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_r2(statistics, path):
    path += 'r2.png'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['r2_log'], marker='o', color='red')
    plt.title('R2 over Generations')
    plt.xlabel('Generation')
    plt.ylabel('R2 Indicator')
    plt.grid(True)
    plt.savefig(path)
    plt.close()



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


