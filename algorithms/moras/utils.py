import argparse
import csv
import json
import logging
import lzma
import shutil
import time

import numpy as np
from pymoo.indicators.hv import HV
import os
import pickle
from indicators import r2
import copy

from individual import create_from_json


# Load R2 weights for the i-th population size
def get_weights_r2(n):
    file = 'r2_weights' + os.sep + 'weights_' + str(n) + '.json'
    with open(file, 'r') as f:
        json_data = json.load(f)
    return {int(k): np.array(v) for k, v in json_data.items()}

def get_weights_r2_file(file):
    with open(file, 'r') as f:
        json_data = json.load(f)
    return {int(k): np.array(v) for k, v in json_data.items()}


def save_archive_accuracy(archive, archive_path):
    archive_path += os.sep + 'archive_accuracy'
    np_archive = [[p.std_acc, p.adv_acc] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def save_archive_losses(archive, archive_path):
    archive_path += os.sep + 'archive_losses'
    np_archive = [[p.F[0], p.F[1]] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def save_archive(archive, archive_path):
    archive_path += os.sep + 'archive'
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

def store_statisctics(statistics, objective_space):
    statistics['max_f1'] = max(statistics['max_f1'], np.max(objective_space[:, 0]))
    statistics['max_f2'] = max(statistics['max_f2'], np.max(objective_space[:, 1]))
    statistics['max_f3'] = max(statistics['max_f3'], np.max(objective_space[:, 2]))
    statistics['max_f4'] = max(statistics['max_f4'], np.max(objective_space[:, 3]))
    statistics['min_f1'] = min(statistics['min_f1'], np.min(objective_space[:, 0]))
    statistics['min_f2'] = min(statistics['min_f2'], np.min(objective_space[:, 1]))
    statistics['min_f3'] = min(statistics['min_f3'], np.min(objective_space[:, 2]))
    statistics['min_f4'] = min(statistics['min_f4'], np.min(objective_space[:, 3]))

def store_metrics(architectures_evaluated, population, population_2, args, weights_r2, statistics):
    max_f1 = 4 * 1.5
    max_f2 = 4 * 1.5
    max_f3 = 450 * 1.5
    max_f4 = 5 * 1.5
    # compute hypervolume
    ind = HV(ref_point=np.array([max_f1, max_f2, max_f3, max_f4]))
    population_array = np.array([ind.F for ind in population])
    hyp = ind(population_array)
    if type(hyp) == np.ndarray:
        statistics['hyp_log'].append(hyp.item())
    else:
        statistics['hyp_log'].append(hyp)
    # compute hypervolume 2 (std_loss, adv_loss)
    ind2 = HV(ref_point=np.array([max_f1, max_f2]))
    population_array2 = np.array([[ind.F[0], ind.F[1]] for ind in population_2])
    hyp2 = ind2(population_array2)
    if type(hyp2) == np.ndarray:
        statistics['hyp2_log'].append(hyp2.item())
    else:
        statistics['hyp2_log'].append(hyp2)
    # compute r2
    z_ref = np.zeros(4)
    nadir_point = np.array([max_f1, max_f2, max_f3, max_f4])
    r2_population = r2(population, weights_r2[args.n_population], nadir_point, z_ref)
    if type(r2_population) == np.ndarray:
        statistics['r2_log'].append(r2_population.item())
    else:
        statistics['r2_log'].append(r2_population)
    row_hyp = ['moras', args.dataset, args.attack, architectures_evaluated, 'hv', hyp, args.save_path_final_model.replace("\\", "/")]
    row_hyp2 = ['moras', args.dataset, args.attack, architectures_evaluated, 'hv_2obj', hyp2, args.save_path_final_model.replace("\\", "/")]
    row_r2 = ['moras', args.dataset, args.attack, architectures_evaluated, 'r2', r2_population, args.save_path_final_model.replace("\\", "/")]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    writer.writerow(row_hyp2)
    file.close()
    return hyp, hyp2, r2_population


def save_supernet(model, model_path):
    import torch
    model_path += 'super-net.pt'
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(cpu_state, model_path)
    if torch.cuda.is_available():
        model.to('cuda')

def save_model(model, model_path, name):
    import torch
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path_ = model_path + os.sep + name
    torch.save(model, model_path_)

def save_log_train(arch_path, log):
    arch_path += 'train_log.csv'

    with open(arch_path, 'a') as f:
        log_str = ','.join([str(item) for item in log])
        f.write(log_str + '\n')

# Load the supernet model from the specified path
def load_supernet(model_path):
    import torch
    try:
        model = torch.load(model_path, weights_only=False)
    except Exception as e:
        logging.error(f"Error loading supernet model from {model_path}: {e}")
        model_path_backup = model_path.replace('.pt', '-backup.pt')
        logging.info(f"Trying to load backup supernet model from {model_path_backup}...")
        model = torch.load(model_path_backup, weights_only=False)
        shutil.copy(model_path_backup, model_path)
    return model

def save_architecture(i, individual, architect_path):
    architect_path += os.sep + 'architectures' + os.sep
    if not os.path.exists(architect_path):
        os.makedirs(architect_path)
    architect_path += f'arch_{i}.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(individual, f)


def save_architectures(architectures, architect_path):
    architect_path += os.sep + 'architectures.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(architectures, f)

def save_statistics_to_csv(statistics, csv_path):
    csv_path += 'statistics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in statistics.items():
            writer.writerow([key, value])


def read_architectures(architect_path):
    with lzma.open(architect_path, 'rb') as f:
        architectures = pickle.load(f)
    for l_tensor in architectures:
        print(l_tensor[0].shape)
        print(l_tensor[1].shape)
    return architectures

def plot_archive_losses(archive_losses, archive_path):
    import matplotlib.pyplot as plt
    archive_path += os.sep + 'archive_losses.pdf'
    std_loss = [p.F[0] for p in archive_losses]
    adv_loss = [p.F[1] for p in archive_losses]
    plt.figure(figsize=(8, 6))
    plt.scatter(std_loss, adv_loss, c='blue', marker='o')
    plt.title('Archive Losses')
    plt.xlabel('Standard Loss')
    plt.ylabel('Adversarial Loss')
    plt.grid(True)
    plt.savefig(archive_path)
    plt.close()

def plot_archive_accuracy(archive_accuracy, archive_path):
    import matplotlib.pyplot as plt
    archive_path += os.sep + 'archive_accuracy.pdf'
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

def plot_lr_scheduler(statistics, path):
    import matplotlib.pyplot as plt
    path += os.sep + 'lr_scheduler.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['lr_log'], marker='o', color='blue')
    plt.title('Learning Rate Scheduler')
    plt.xlabel('Generation')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_hypervolume(statistics, path, losses=True):
    import matplotlib.pyplot as plt
    path += os.sep + 'hypervolume.pdf'
    plt.figure(figsize=(8, 6))
    if losses:
        plt.plot(statistics['hyp_log'], marker='o', color='blue')
        plt.title('Hypervolume per evaluations (std_loss, adv_loss, flops, n_params)')
    else:
        plt.plot(statistics['hyp_acc_log'], marker='o', color='blue')
        plt.title('Hypervolume per evaluations (100-std_acc, 100-adv_acc, flops, n_params)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_hypervolume2(statistics, path):
    import matplotlib.pyplot as plt
    path += os.sep + 'hypervolume2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp2_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_loss, adv_loss)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_hypervolume2_acc(statistics, path):
    import matplotlib.pyplot as plt
    path += os.sep + 'hypervolume2_acc.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp2_acc_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_acc, adv_acc)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_r2(statistics, path):
    import matplotlib.pyplot as plt
    path += os.sep + 'r2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['r2_log'], marker='o', color='red')
    plt.title('R2 per evaluations (std_loss, adv_loss, flops, n_params)')
    plt.xlabel('Generation')
    plt.ylabel('R2 Indicator')
    plt.grid(True)
    plt.savefig(path)
    plt.close()



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        import torch
        h, w = img.size(1), img.size(2)

        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        mask[y1:y2, x1:x2] = 0
        img = img * mask.expand_as(img)

        return img

def data_transforms_cifar10(args):
    import torchvision.transforms as transforms
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

def get_model_metrics(model_):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    model = copy.deepcopy(model_)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = round(float(params_num) / 1e6, 4)

    x = torch.randn(1, 3, 32, 32).to(next(model.parameters()).device)
    with FlopCounterMode(display=False) as flop_counter:
        model(x)

    flops = round(float(flop_counter.get_total_flops()) / 1e6, 4)
    del model 
    return flops, params

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
    if 'device' in params_dict:
        params_dict.pop('device')
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    params_path += 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)

def store_population_data(generation, pop_X, pop_obj, archive, archive_losses, statistics, time_search, save_path_final_architect):
    population_data_dir = save_path_final_architect + os.sep + 'population_data.json'
    population_data = {
        'generation': generation,
        'pop_obj': pop_obj.tolist(),
        'pop_X': pop_X.tolist(),
        'archive': [ind.to_dict() for ind in archive],
        'archive_losses': [ind.to_dict() for ind in archive_losses],
        'statistics': statistics,
        'time_search': time_search
    }
    with open(population_data_dir, 'w') as f:
        json.dump(population_data, f, indent=4)
    # store a backup
    population_data_dir_backup = save_path_final_architect + os.sep + 'population_data_backup.json'
    with open(population_data_dir_backup, 'w') as f:
        json.dump(population_data, f, indent=4)



# read the json file with the parameters used for training the architecture and return it as a dictionary
def load_execution(args_dir):
    with open(args_dir + os.sep + 'params.json', 'r') as f:
        args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
    try:
        with open(args_dir + os.sep + 'population_data.json', 'r') as f:
            population_data = json.load(f)
    except Exception as e:
        # if there is an error loading the main population data file, try to load the backup
        logging.info(f">>>> Error loading population data from {args_dir + os.sep + 'population_data.json'}: {e}. Trying to load backup...")
        with open(args_dir + os.sep + 'population_data_backup.json', 'r') as f:
            population_data = json.load(f)
        # after loading the backup, copy it to the main file to ensure we have a valid population data file for the next reloads
        shutil.copy(args_dir + os.sep + 'population_data_backup.json', args_dir + os.sep + 'population_data.json')
    finally:
        generation = population_data['generation']
        pop = [create_from_json(ind_json, args.search_space) for ind_json in population_data['population']]
        archive = [create_from_json(ind_json, args.search_space) for ind_json in population_data['archive']]
        archive_acc_4objs = [create_from_json(ind_json, args.search_space) for ind_json in population_data['archive_acc_4objs']]
        archive_accuracy = [create_from_json(ind_json, args.search_space) for ind_json in population_data['archive_accuracy']]
        archive_losses = [create_from_json(ind_json, args.search_space) for ind_json in population_data['archive_losses']]
        statistics = population_data['statistics']
        nadir_point = np.array(population_data['nadir_point'])
        ideal_point = np.array(population_data['ideal_point'])
        time_search = int(population_data['time_search'])
        generation = int(generation)
        generation += 1
    if 'epochs_train_individual' in args_dict:
        args.epochs_train_individual = args.epochs_train_individual + (generation // 10) * 5 if args.increase_epochs else args.epochs_train_individual
        print(f"Updated epochs for training individuals: {args.epochs_train_individual}")
    if 'epochs_train_supernet' in args_dict:
        args.epochs_train_supernet = args.epochs_train_supernet + (generation // 10) * 5 if args.increase_epochs else args.epochs_train_supernet
        print(f"Updated epochs for training supernet: {args.epochs_train_supernet}")
    return args, statistics, generation, pop, archive, archive_acc_4objs, archive_accuracy, archive_losses, nadir_point, ideal_point, time_search

if __name__ == '__main__':
    """
    best_adv = "results/r2-emoa/cifar10/2025-11-26_21-14-55_18906049/search/architectures/arch_56.xz"
    path = "results/r2-emoa/cifar10/2025-11-26_21-14-55_18906049/search/architectures/"
    best_ind, best_path = get_best_architecture_adversarial(path)
    # 22
    print(f"8. Best adversarial architecture found in {best_path} with adv acc {best_ind.adv_acc} and std acc {best_ind.std_acc}")
    best_ind, best_path = get_best_architecture_standard(path)
    # 33
    print(f"8. Best standard architecture found in {best_path} with std acc {best_ind.std_acc} and adv acc {best_ind.adv_acc}")

    # Obtaining architectures for best-5
    #path = "results/r2-emoa/cifar10/best-5/search/architectures/"
    #best_ind, best_path = get_best_architecture_adversarial(path)
    # 47
    #print(f"5. Best adversarial architecture found in {best_path} with adv acc {best_ind.adv_acc} and std acc {best_ind.std_acc}")
    # 36
    #best_ind, best_path = get_best_architecture_standard(path)
    #print(f"5. Best standard architecture found in {best_path} with std acc {best_ind.std_acc} and adv acc {best_ind.adv_acc}")
    """