import pickle

from pymoo.util.ref_dirs import get_reference_directions

# Store R2 weights for the i-th population size
def store_weights(n, k):
    file = 'weights_' + str(n) + '.pkl'
    directions_set = {}
    for i in range(n+1):
        size_pop = 2 * n - i
        w = get_reference_directions("energy", k, size_pop, seed=1)
        directions_set[size_pop] = w
    for v,k in directions_set.items():
        print(v, k.shape)
    with open(file, 'wb') as f:
        pickle.dump(directions_set, f)

if __name__ == '__main__':
    n = 20
    k = 4
    store_weights(n, k)
