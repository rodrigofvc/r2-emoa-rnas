import json
import numpy as np

from pymoo.util.ref_dirs import get_reference_directions

# Store R2 weights for the i-th population size
def store_weights(n, k):
    file = 'weights_' + str(n) + '.pkl'
    directions_set = {}
    for i in range(n+1):
        size_pop = 2 * n - i
        w = get_reference_directions("energy", k, size_pop, seed=1)
        directions_set[size_pop] = w
    json_data = {str(k): v.tolist() for k, v in directions_set.items()}
    with open(file, 'w') as f:
        json.dump(json_data, f)
    print('Weights stored in file:', file)

def get_weights_ponderated(n, k):
    seed = 1
    w_p = []
    while len(w_p) < n:
        w_set = get_reference_directions("energy", k, n, seed=seed)
        for w in w_set:
            if w[0] + w[1] >= 0.75 and len(w_p) < n:
                w_p.append(w)
        seed += 1
    return np.array(w_p)

def store_weights_ponderated(n, k):
    file = 'weights_' + str(n) + '.json'
    directions_set = {}
    for i in range(2*n-9):
        size_pop = 2 * n - i
        directions_set[size_pop] = get_weights_ponderated(size_pop, k)
    json_data = {str(k): v.tolist() for k, v in directions_set.items()}
    with open(file, 'w') as f:
        json.dump(json_data, f)
    print('Weights stored in file:', file)

if __name__ == '__main__':
    n = 40
    k = 4
    #store_weights(n, k)
    store_weights_ponderated(n, k)