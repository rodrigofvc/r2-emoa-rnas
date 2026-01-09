import csv
import json
import lzma
import time
import matplotlib.pyplot as plt
from thop import profile
import numpy as np
import torch
from pymoo.indicators.hv import HV
import torchvision.transforms as transforms
import os
import pickle

from model import NetworkCIFAR
from indicators import r2, normalize_objectives, get_dynamic_r2_reference


# Load R2 weights for the i-th population size
def get_weights_r2(n):
    file = 's1' + os.sep + 'r2_weights' + os.sep + 'weights_' + str(n) + '.pkl'
    with open(file, 'rb') as f:
        weights_r2 = pickle.load(f)
    return weights_r2


def save_archive_accuracy(archive, archive_path):
    archive_path += 'archive_accuracy'
    np_archive = [[p.std_acc, p.adv_acc] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def save_archive(archive, archive_path):
    archive_path += os.sep + 'archive'
    np_archive = [p.F for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def save_archive_2(archive, archive_path):
    archive_path += os.sep + 'archive_2'
    np_archive = [[p.F[0], p.F[1]] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def store_metrics(architectures_evaluated, population, population_2, args, weights_r2, statistics):
    max_f1 = 4 * 1.5
    max_f2 = 4 * 1.5
    max_f3 = 450 * 1.5
    max_f4 = 5 * 1.5
    # compute hypervolume
    ind = HV(ref_point=np.array([max_f1, max_f2, max_f3, max_f4]))
    population_array = np.array([ind.F for ind in population])
    hyp = ind(population_array)
    statistics['hyp_log'].append(hyp.item())
    # compute hypervolume 2 (std_loss, adv_loss)
    ind2 = HV(ref_point=np.array([max_f1, max_f2]))
    population_array2 = np.array([[ind.F[0], ind.F[1]] for ind in population_2])
    hyp2 = ind2(population_array2)
    statistics['hyp2_log'].append(hyp2.item())
    # compute r2
    normalize_objectives(population)
    z_ref = get_dynamic_r2_reference(population)
    r2_population = r2(population, weights_r2[args.pop_size], z_ref)
    statistics['r2_log'].append(r2_population.item())
    row_hyp = ['nevonas', args.dataset, 'FGSM', architectures_evaluated, 'hv', hyp, args.save_dir]
    row_r2 = ['nevonas', args.dataset, 'FGSM', architectures_evaluated, 'r2', r2_population, args.save_dir]
    row_hyp2 = ['nevonas', args.dataset, 'FGSM', architectures_evaluated, 'hv_2obj', hyp2, args.save_dir]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    writer.writerow(row_hyp2)
    file.close()
    return hyp, hyp2, r2_population

def save_supernet(model, model_path):
    print(os.getcwd())
    model_path += os.sep + 'super-net.pt'
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(cpu_state, model_path)
    if torch.cuda.is_available():
        model.to('cuda')


def load_model(model_path):
    state_dict = torch.load(model_path, weights_only=False)
    if torch.cuda.is_available():
        state_dict.to('cuda')
    return state_dict

def load_supernet(model_path, model):
    model_path += 'super-net.pt'
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.to('cuda')
    return model

def save_architecture(i, individual, architect_path):
    architect_path += os.sep + 'architectures' + os.sep
    if not os.path.exists(architect_path):
        os.makedirs(architect_path)
    architect_path += f'arch_{i}.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(individual, f)

def save_statistics_to_csv(statistics, csv_path):
    csv_path += os.sep + 'statistics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in statistics.items():
            writer.writerow([key, value])

def plot_archive_losses(archive, archive_path):
    archive_path += os.sep + 'archive.pdf'
    std_acc = [p.F[0] for p in archive]
    adv_acc = [p.F[1] for p in archive]
    plt.figure(figsize=(8, 6))
    plt.scatter(std_acc, adv_acc, c='blue', marker='o')
    plt.title('Non-dominated solutions')
    plt.xlabel('std_error')
    plt.ylabel('adv_error')
    plt.grid(True)
    plt.savefig(archive_path)
    plt.close()

def plot_hypervolume(statistics, path):
    path += os.sep + 'hypervolume.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_loss, adv_loss, flops, n_params)')
    plt.xlabel('Evaluations')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_hypervolume2(statistics, path):
    path += os.sep + 'hypervolume2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp2_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_loss, adv_loss)')
    plt.xlabel('Evaluations')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_r2(statistics, path):
    path += os.sep + 'r2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['r2_log'], marker='o', color='red')
    plt.title('R2 per evaluations (std_loss, adv_loss, flops, n_params)')
    plt.xlabel('Evaluations')
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
    discretized_model = NetworkCIFAR(model.C, model._num_classes, model._layers, auxiliary=False, genotype=genotype)
    x = torch.randn(1, 3, 32, 32)
    macs, params = profile(discretized_model, inputs=(x,), verbose=False)
    flops = (2 * macs) / 1e6
    params = params / 1e6
    return round(flops, 4), round(params, 4)

def get_best_architecture_adversarial(archs_path):
    best_adv_acc = -1.0
    best_individual = None
    best_path = ""
    for arch_path in os.listdir(archs_path):
        with lzma.open(archs_path + os.sep + arch_path, 'rb') as f:
            individual = pickle.load(f)
            if individual.adv_acc > best_adv_acc:
                best_adv_acc = individual.adv_acc
                best_individual = individual
                best_path = archs_path + os.sep + arch_path
    return best_individual, best_path

def get_best_architecture_standard(archs_path):
    best_std_acc = -1.0
    best_individual = None
    best_path = ""
    for arch_path in os.listdir(archs_path):
        with lzma.open(archs_path + os.sep + arch_path, 'rb') as f:
            individual = pickle.load(f)
            if individual.std_acc > best_std_acc:
                best_std_acc = individual.std_acc
                best_individual = individual
                best_path = archs_path + os.sep + arch_path
    return best_individual, best_path

def save_params(args, trained_arch_path):
    params_path = trained_arch_path + os.sep
    params_dict = vars(args)
    if 'device' in params_dict.keys():
        params_dict['device'] = str(params_dict['device'])
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    params_path += 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)
