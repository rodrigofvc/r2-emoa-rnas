import numpy as np

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
    population_exclude = [p for p in population if p != individual]
    assert len(population_exclude) == n - 1, f"population_exclude size != population size - 1 {n - 1}"
    excl = r2(population_exclude, weights, nadir_point, z_ref)
    return excl
