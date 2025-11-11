import numpy as np
from pymoo.core.population import Population


def normalize_objectives(population):
    n_obj = len(population[0].F)
    f_mins = [min([ind.F[i] for ind in population]) for i in range(n_obj)]
    f_maxs = [max([ind.F[i] for ind in population]) for i in range(n_obj)]
    for ind in population:
        for i in range(n_obj):
            if f_maxs[i] - f_mins[i] > 0:
                ind.F_norm[i] = (ind.F[i] - f_mins[i]) / (f_maxs[i] - f_mins[i])
            else:
                ind.F_norm[i] = 0.0


def r2(population, weights, z_ref):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            max_diff = max([w_j * abs(p.F_norm[j] - z_ref[j]) for j, w_j in enumerate(w)])
            min_diff = min(min_diff, max_diff)
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, z_ref):
    n = population.size
    assert weights[n].shape[0] == n, f"weights shape != population size {n}"
    full = r2(population, weights[n], z_ref)
    population_exclude = [p for p in population if p != individual]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(Population(individuals=population_exclude), weights[n], z_ref)
    return abs(full - excl)

def get_dynamic_r2_reference(population):
    n_obj = len(population[0].F)
    z_ref = np.zeros(n_obj)
    max_f = 0
    for i in range(n_obj):
        max_f_i = max([ind.F_norm[i] for ind in population])
        min_f_i = min([ind.F_norm[i] for ind in population])
        max_f = max(max_f, max_f_i - min_f_i)
    for i in range(n_obj):
        min_f_i = min([ind.F_norm[i] for ind in population])
        z_ref[i] = min_f_i - max_f
    return z_ref