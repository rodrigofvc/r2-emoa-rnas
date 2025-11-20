from pymoo.core.problem import Problem
import numpy as np
import torch


def flatten_alphas(individual):
    alpha_normal, alpha_reduce = individual
    return np.concatenate([alpha_normal.detach().cpu().numpy().ravel(),
                           alpha_reduce.detach().cpu().numpy().ravel()]).astype(np.float64)

def unpack_alphas(vec, shape_alphas, args):
    n_norm = shape_alphas[0] * shape_alphas[1]
    a_norm = vec[:n_norm].reshape(shape_alphas)
    a_reduction = vec[n_norm:].reshape(shape_alphas)
    return [torch.from_numpy(a_norm).float().to(args.device), torch.from_numpy(a_reduction).float().to(args.device)]
class AlphaProblem(Problem):
    def __init__(self, shape_alphas):
        n_var = shape_alphas[0] * shape_alphas[1] * 2
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu)

def tournament_r2(pop, P, **kwargs):
    n_tournaments, _ = P.shape
    winners = np.empty(P.shape[0], dtype=int)
    for i in range(n_tournaments):
        competitors = P[i]
        winner = [(j, pop[j].get("c_r2")) for j in competitors]
        winner = sorted(winner, key=lambda x: x[1], reverse=False)[0][0]
        winners[i] = winner
    return winners



