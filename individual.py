import numpy as np


class Individual:
    def __init__(self, X, k):
        self.X = X
        self.F = np.zeros(k)
        self.F_norm = np.zeros(k)
        self.k = k
        self.c_r2 = 0.0
        self.std_acc = 0.0
        self.adv_acc = 0.0
        self.genotype = None


