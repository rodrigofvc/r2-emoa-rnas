import pickle
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

def plot_weights(F, n_points):
    print('Plotting weights for population size:', n_points)
    figure = plt.figure(figsize=(8, 8))
    #ax = figure.add_subplot(1, 1, 1)
    plot = Scatter(plot_3d=False, tight_layout=True, title='Weights for population size: ' + str(n_points))
    plot.add(F, s=10)
    plot.show()
    #figure.savefig('weights_' + str(n_points) + '.pdf')
    #plt.close()

if __name__ == '__main__':
    weights_file = 'weights_' + str(40) + '.pkl'
    with open(weights_file, 'rb') as f:
        weights_dict = pickle.load(f)
    weights_40 = weights_dict[40]
    print(weights_40)
    plot_weights(weights_40, 40)

