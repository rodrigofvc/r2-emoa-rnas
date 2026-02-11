import csv
import json
import lzma
import os
import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV

def get_weights_r2(n):
    file = 'r2_weights' + os.sep + 'weights_' + str(n) + '.pkl'
    with open(file, 'rb') as f:
        weights_r2 = pickle.load(f)
    return weights_r2

def r2(population, weights, nadir_point, z_ref):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            max_diff = max([w_j * abs((p[j] - z_ref[j]) / max(nadir_point[j] - z_ref[j], 1e-10)) for j, w_j in enumerate(w)])
            min_diff = min(min_diff, max_diff)
        assert np.isfinite(max_diff), f"Non-finite max_diff encountered in R2 calculation: {max_diff}"
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, nadir_point, z_ref):
    n = len(population)
    #full = r2(population, weights, nadir_point, z_ref)
    population_exclude = [p for p in population if np.not_equal(p, individual).any()]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(population_exclude, weights, nadir_point, z_ref)
    return excl

def store_metrics(dataset, architectures_evaluated, pop_obj, pop_obj_2, save_dir, statistics):
    max_f1 = 4 * 1.5
    max_f2 = 4 * 1.5
    max_f3 = 450 * 1.5
    max_f4 = 5 * 1.5
    # compute hypervolume 4 objectives
    ind = HV(ref_point=np.array([max_f1, max_f2, max_f3, max_f4]))
    hyp = ind(pop_obj)
    statistics['hyp_log'].append(hyp.item())
    # compute hypervolume 2 objectives (std_loss and adv_loss)
    ind_2obj = HV(ref_point=np.array([max_f1, max_f2]))
    hyp_2obj = ind_2obj(pop_obj_2)
    statistics['hyp2_log'].append(hyp_2obj.item())
    # compute r2
    weights_r2 = get_weights_r2(40)
    z_ref = np.zeros(4)
    nadir_point = np.array([max_f1, max_f2, max_f3, max_f4])
    r2_population = r2(pop_obj, weights_r2, nadir_point, z_ref)
    statistics['r2_log'].append(r2_population)
    row_hyp =  ['nsga-net', dataset, 'FGSM', architectures_evaluated, 'hv', hyp, save_dir]
    row_r2 =   ['nsga-net', dataset, 'FGSM', architectures_evaluated, 'r2', r2_population, save_dir]
    row_hyp2 = ['nsga-net', dataset, 'FGSM', architectures_evaluated, 'hv_2obj', hyp_2obj, save_dir]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    writer.writerow(row_hyp2)
    file.close()
    return hyp, hyp_2obj, r2_population

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

def plot_hypervolume(statistics, save_dir):
    save_dir += os.sep + 'hypervolume.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp_log'], marker='o', color='blue')
    plt.title('Hypervolume over generations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(save_dir)
    plt.close()
def plot_hypervolume2(statistics, save_dir):
    save_dir += os.sep + 'hypervolume_2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['hyp2_log'], marker='o', color='blue')
    plt.title('Hypervolume over generations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.savefig(save_dir)
    plt.close()

def plot_r2(statistics, save_dir):
    save_dir += os.sep + 'r2.pdf'
    plt.figure(figsize=(8, 6))
    plt.plot(statistics['r2_log'], marker='o', color='red')
    plt.title('R2 over generations')
    plt.xlabel('Generation')
    plt.ylabel('R2 Indicator')
    plt.grid(True)
    plt.savefig(save_dir)
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
    params_dict = vars(args)
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    params_path += 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)