import csv
import json
import lzma
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV

from indicators import normalize_objectives, get_dynamic_r2_reference, r2


def get_weights_r2(n):
    file = 'r2_weights' + os.sep + 'weights_' + str(n) + '.pkl'
    with open(file, 'rb') as f:
        weights_r2 = pickle.load(f)
    return weights_r2

def save_archive_losses(archive, archive_path):
    archive_path += 'archive_losses'
    np_archive = [[p.F[0], p.F[1]] for p in archive]
    np_archive = np.array(np_archive)
    np.savez_compressed(archive_path, np_archive)

def store_metrics(architectures_evaluated, args, population, population_2, weights_r2, statistics):
    max_f1 = 4 * 1.5
    max_f2 = 4 * 1.5
    max_f3 = 450 * 1.5
    max_f4 = 5 * 1.5
    # compute hypervolume
    ind = HV(ref_point=np.array([max_f1, max_f2, max_f3, max_f4]))
    population_array = np.array([ind.F for ind in population])
    hyp = ind(population_array)
    if type(hyp) is np.ndarray:
        statistics['hyp_log'].append(hyp.item())
    else:
        statistics['hyp_log'].append(hyp)
    # compute hypervolume 2 (std_loss, adv_loss)
    ind2 = HV(ref_point=np.array([max_f1, max_f2]))
    population_array2 = np.array([[ind.F[0], ind.F[1]] for ind in population_2])
    hyp2 = ind2(population_array2)
    if type(hyp2) is np.ndarray:
        statistics['hyp2_log'].append(hyp2.item())
    else:
        statistics['hyp2_log'].append(hyp2)
    # compute r2
    normalize_objectives(population)
    z_ref = get_dynamic_r2_reference(population)
    r2_population = r2(population, weights_r2[args['n_population']], z_ref)
    if type(r2_population) is np.ndarray:
        statistics['r2_log'].append(r2_population.item())
    else:
        statistics['r2_log'].append(r2_population)
    row_hyp = ['nsganetv2', args['dataset'], 'FGSM', architectures_evaluated, 'hv', hyp, args['save_path']]
    row_hyp2 = ['nsganetv2', args['dataset'], 'FGSM', architectures_evaluated, 'hv_2obj', hyp2, args['save_path']]
    row_r2 = ['nsganetv2', args['dataset'], 'FGSM', architectures_evaluated, 'r2', r2_population, args['save_path']]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    writer.writerow(row_hyp2)
    file.close()
    return hyp, hyp2, r2_population

def save_architecture(i, individual, objectives, save_dir):
    architect_path = save_dir + os.sep + 'architectures' + os.sep
    if not os.path.exists(architect_path):
        os.makedirs(architect_path)
    architect_path += f'arch_{i}.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump((individual, objectives), f)


def save_archive(archive, save_dir):
    save_dir += os.sep + 'archive'
    np_archive = np.array(archive)
    np.savez_compressed(save_dir, np_archive)

def save_archive_2(archive, save_dir):
    save_dir += os.sep + 'archive_2'
    np_archive = np.array(archive)
    np.savez_compressed(save_dir, np_archive)


def plot_hypervolume(statistics, path):
    path += os.sep + 'hypervolume.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_loss, adv_loss, flops, n_params)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_archive(archive, save_dir):
    save_dir += os.sep + 'archive.pdf'
    archive = np.array(archive)
    plt.figure(figsize=(8, 6))
    plt.scatter(archive[:, 0], archive[:, 1], c='blue', marker='o')
    plt.title('Non-dominated solutions')
    plt.xlabel('std_error')
    plt.ylabel('adv_error')
    plt.grid(True)
    plt.savefig(save_dir)
    plt.close()


def plot_hypervolume2(statistics, path):
    path += os.sep + 'hypervolume2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp2_log'], marker='o', color='blue')
    plt.title('Hypervolume per evaluations (std_loss, adv_loss)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_r2(statistics, path):
    path += os.sep + 'r2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['r2_log'], marker='o', color='red')
    plt.title('R2 per evaluations (std_loss, adv_loss, flops, n_params)')
    plt.xlabel('Generation')
    plt.ylabel('R2 Indicator')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def save_statistics_to_csv(statistics, save_dir):
    save_dir = save_dir + os.sep + 'statistics.csv'
    with open(save_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in statistics.items():
            writer.writerow([key, value])

def save_params(args, save_dir):
    params_path = save_dir + os.sep
    params_dict = args
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    params_path += 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)
