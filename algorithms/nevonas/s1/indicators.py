import numpy as np

def r2(population, weights, nadir_point, z_ref):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            max_diff = max([w_j * abs((p.F[j] - z_ref[j]) / max(nadir_point[j] - z_ref[j], 1e-10)) for j, w_j in enumerate(w)])
            min_diff = min(min_diff, max_diff)
        assert np.isfinite(max_diff), f"Non-finite max_diff encountered in R2 calculation: {max_diff}"
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, nadir_point, z_ref):
    n = len(population)
    #full = r2(population, weights, nadir_point, z_ref)
    population_exclude = [p for p in population if p != individual]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(population_exclude, weights, nadir_point, z_ref)
    return excl

def get_dynamic_r2_reference(population):
    n_obj = len(population[0].F)
    z_ref = np.zeros(n_obj)
    max_f = 0
    for i in range(n_obj):
        max_f_i = max([ind.F[i] for ind in population])
        min_f_i = min([ind.F[i] for ind in population])
        max_f = max(max_f, max_f_i - min_f_i)
    print("Max_f for dynamic R2 reference point calculation:", max_f)
    for i in range(n_obj):
        min_f_i = min([ind.F[i] for ind in population])
        print("Min_f_i for objective", i, ":", min_f_i)
        z_ref[i] = min_f_i - max_f
    for i in range(n_obj):
        assert z_ref[i] <= min([ind.F[i] for ind in population]), f"Dynamic R2 reference point z_ref[{i}] is not less than the minimum objective value {z_ref[i]} >= {min([ind.F[i] for ind in population])}"
    print("Dynamic R2 reference point:", z_ref)
    return z_ref