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

def normalize_objectives(pop_obj):
    n_obj = pop_obj.shape[1]
    f_mins = np.min(pop_obj, axis=0)
    f_maxs = np.max(pop_obj, axis=0)
    for ind in pop_obj:
        for i in range(n_obj):
            assert np.isfinite(ind[i]), f"Non-finite F encountered in normalization: {ind.F[i]}"
            if f_maxs[i] - f_mins[i] > 1e-12:
                ind[i] = (ind[i] - f_mins[i]) / (f_maxs[i] - f_mins[i])
            else:
                ind[i] = 0.0
            assert np.isfinite(ind[i]), f"Non-finite F_norm encountered in normalization: {ind[i]}"
            ind[i] = np.clip(ind[i], a_min=0.0, a_max=1.0)



def r2(population, weights, z_ref):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            max_diff = max([w_j * abs(p[j] - z_ref[j]) for j, w_j in enumerate(w)])
            min_diff = min(min_diff, max_diff)
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, z_ref):
    n = population.shape[0]
    assert weights[n].shape[0] == n, f"weights shape != population size {n}"
    full = r2(population, weights[n], z_ref)
    population_exclude = [p for p in population if np.not_equal(p, individual).any()]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(population_exclude, weights[n], z_ref)
    return abs(full - excl)

def get_dynamic_r2_reference(population):
    n_obj = population.shape[1]
    z_ref = np.zeros(n_obj)
    max_f = 0
    for i in range(n_obj):
        max_f_i = np.max(population[:, i])
        min_f_i = np.min(population[:, i])
        max_f = max(max_f, max_f_i - min_f_i)
        assert np.isfinite(min_f_i), "Non-finite min_f_i encountered in dynamic R2 reference point calculation"
        assert np.isfinite(max_f_i), "Non-finite max_f_i encountered in dynamic R2 reference point calculation"
        assert np.isfinite(max_f), "Non-finite max_f encountered in dynamic R2 reference point calculation"
    for i in range(n_obj):
        min_f_i = np.min(population[:, i])
        z_ref[i] = min_f_i - max_f
        assert np.isfinite(z_ref[i]), "Non-finite z_ref encountered in dynamic R2 reference point calculation"
        assert np.isfinite(min_f_i), "Non-finite min_f_i encountered in dynamic R2 reference point calculation"
    return z_ref

def store_metrics(architectures_evaluated, pop_obj, save_dir, statistics):
    max_f1 = 2 * 1.5
    max_f2 = 2 * 1.5
    max_f3 = 110 * 1.5
    max_f4 = 2 * 1.5
    # compute hypervolume 4 objectives
    ind = HV(ref_point=np.array([max_f1, max_f2, max_f3, max_f4]))
    hyp = ind(pop_obj)
    statistics['hyp_log'].append(hyp.item())
    # compute hypervolume 2 objectives (std_loss and adv_loss)
    ind_2obj = HV(ref_point=np.array([max_f1, max_f2]))
    hyp_2obj = ind_2obj(pop_obj[:, :2])
    statistics['hyp2_log'].append(hyp_2obj.item())
    # compute r2
    weights_r2 = get_weights_r2(40)
    norm_obj = pop_obj.copy()
    normalize_objectives(norm_obj)
    z_ref = get_dynamic_r2_reference(norm_obj)
    r2_population = r2(norm_obj, weights_r2[40], z_ref)
    statistics['r2_log'].append(r2_population)
    row_hyp = ['nsga-net', 'cifar10', 'FGSM', architectures_evaluated, 'hv', hyp, save_dir]
    row_r2 = ['nsga-net', 'cifar10', 'FGSM', architectures_evaluated, 'r2', r2_population, save_dir]
    file = open('evaluations.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(row_hyp)
    writer.writerow(row_r2)
    file.close()
    row_hyp2 = ['nsga-net', 'cifar10', 'FGSM', architectures_evaluated, 'hv_2obj', hyp_2obj, save_dir]
    file2 = open('evaluations-2.csv', 'a', newline='')
    writer2 = csv.writer(file2)
    writer2.writerow(row_hyp2)
    file2.close()
    return hyp, r2_population

def save_architecture(i, individual, save_dir):
    architect_path = save_dir + os.sep + 'architectures' + os.sep
    if not os.path.exists(architect_path):
        os.makedirs(architect_path)
    architect_path += f'arch_{i}.xz'
    with lzma.open(architect_path, 'wb') as f:
        pickle.dump(individual, f)

def save_archive(archive, save_dir):
    save_dir += os.sep + 'archive'
    np_archive = np.array(archive)
    np.savez_compressed(save_dir, np_archive)
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