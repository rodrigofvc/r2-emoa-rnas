import numpy as np
import torch

class Individual:
    def __init__(self, X, k):
        if isinstance(X, torch.Tensor):
            self.X = X.detach().cpu().numpy().astype(np.float32).copy()
        else:
            self.X = X.astype(np.float32).copy()
        self.F = np.zeros(k)
        self.F_norm = np.zeros(k)
        self.k = k
        self.c_r2 = 0.0
        self.std_acc = 0.0
        self.adv_acc = 0.0
        self.genotype = None


