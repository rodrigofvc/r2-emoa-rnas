import numpy as np
import torch

class Individual:
    def __init__(self, X=None, k=4):
        if isinstance(X, torch.Tensor):
            self.X = X.detach().cpu().numpy().astype(np.float32).copy()
        elif isinstance(X, np.ndarray):
            self.X = X.astype(np.float32).copy()
        else:
            self.X = None
        self.F = np.zeros(k)
        self.F_norm = np.zeros(k)
        self.k = k
        self.c_r2 = 0.0
        self.std_acc = 0.0
        self.adv_acc = 0.0
        self.genotype = None


def wrap_individuals(candidates, std_loss, adv_loss, flops, params):
    individuals = []
    for i, candidate in enumerate(candidates):
        ind = Individual(k=4)
        ind.F[0] = std_loss[i]
        ind.F[1] = adv_loss[i]
        ind.F[2] = flops[i]
        ind.F[3] = params[i]
        ind.genotype = candidate
        individuals.append(ind)
    return individuals