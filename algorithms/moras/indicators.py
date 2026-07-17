import numpy as np

def normalize_objectives(population):
    n_obj = len(population[0].F)
    f_mins = [min([ind.F[i] for ind in population]) for i in range(n_obj)]
    f_maxs = [max([ind.F[i] for ind in population]) for i in range(n_obj)]
    for ind in population:
        for i in range(n_obj):
            assert np.isfinite(ind.F[i]), f"Non-finite F encountered in normalization: {ind.F[i]}"
            if f_maxs[i] - f_mins[i] > 1e-12:
                ind.F_norm[i] = (ind.F[i] - f_mins[i]) / (f_maxs[i] - f_mins[i])
            else:
                ind.F_norm[i] = 0.0
            assert np.isfinite(ind.F_norm[i]), f"Non-finite F_norm encountered in normalization: {ind.F_norm[i]}"
        ind.F_norm = np.clip(ind.F_norm, a_min=0.0, a_max=1.0)


def r2(population, weights, nadir_point, z_ref, losses=True):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            if losses:
                max_diff = max([w_j * abs((p.F[j] - z_ref[j]) / max(nadir_point[j] - z_ref[j], 1e-10)) for j, w_j in enumerate(w)])
            else:
                max_diff = max([w_j * abs((p.F_acc[j] - z_ref[j]) / max(nadir_point[j] - z_ref[j], 1e-10)) for j, w_j in enumerate(w)])
            min_diff = min(min_diff, max_diff)
        assert np.isfinite(max_diff), f"Non-finite max_diff encountered in R2 calculation: {max_diff}"
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, nadir_point, z_ref, losses=True):
    n = len(population)
    #full = r2(population, weights, nadir_point, z_ref, losses)
    population_exclude = [p for p in population if p != individual]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(population_exclude, weights, nadir_point, z_ref, losses)
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

def update_ref_points(population, nadir_point, ideal_point, losses=True):
    for ind in population:
        if losses:
            F = np.array(ind.F)
        else:
            F = np.array(ind.F_acc)
        if ind.feasible:
            nadir_point[:] = np.maximum(nadir_point, F)
            ideal_point[:] = np.minimum(ideal_point, F)