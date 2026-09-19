import numpy as np

class Individual:
    def __init__(self, X, k, search_space):
        assert type(X) == np.ndarray
        if search_space == 'continuous':
            self.X = X.astype(np.float32).copy()
        else:
            self.X = X.astype(np.int32).copy()
        self.F = None
        # use accuracy instead of losses as objectives
        self.F_acc = None
        self.k = k
        self.c_r2 = 0.0
        self.std_acc = 0.0
        self.adv_acc = 0.0
        self.genotype = None
        self.feasible = False

    def to_dict(self):
        return {
            'X': self.X.tolist(),
            'F': self.F.tolist(),
            'F_acc': self.F_acc.tolist(),
            'k': self.k,
            'c_r2': self.c_r2,
            'std_acc': self.std_acc,
            'adv_acc': self.adv_acc,
            'genotype': self.genotype,
            'feasible': self.feasible
        }

def create_from_json(json_dict, search_space):
    ind = Individual(X=np.array(json_dict['X']), k=json_dict['k'], search_space=search_space)
    if json_dict['F'] is not None:
        ind.F = np.array(json_dict['F'])
    else:
        ind.F = None
    if 'F_acc' in json_dict:
        ind.F_acc = np.array(json_dict['F_acc'])
    else:
        ind.F_acc = None
    if 'k' in json_dict:
        ind.k = json_dict['k']
    if 'c_r2' in json_dict:
        ind.c_r2 = json_dict['c_r2']
    if 'feasible' in json_dict:
        ind.feasible = json_dict['feasible']
    ind.std_acc = json_dict['std_acc']
    ind.adv_acc = json_dict['adv_acc']
    ind.genotype = json_dict['genotype']
    return ind
