from n_layer_neural_network import DeepNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class NeuralNetwork(DeepNeuralNetwork):

    def __init__(self, input_dim = 2, hidden_dim = 3, output_dim = 2, activation_type = 'tanh', regularization = 0, random_seed = None):
        super().__init__([input_dim, hidden_dim, output_dim], activation_type, regularization, random_seed)


def main():
    Noise_samples = [[0.025, '00'], [.050, '01'], [.10, '02']]
    Activation_fun = [['ReLU', 'relu', '00'], ['sigmoid', 'sigmoid', '01'], ['hyperbolic tangent', 'tanh', '02'],['parametric relu', 'prelu', '03'], ['exponential linear unites', 'elu', '04']]
    Data_maker = [['00', lambda x: datasets.make_moons(n_samples = 1000, shuffle = True, noise = x, random_state = None)], ['01', lambda x: datasets.make_circles(n_samples = 1000, shuffle = True, noise = x, random_state = None, factor = 0.75)]]
    N_hidden = [[10, '00'], [20, '01'], [30, '02']]

    for l in N_hidden:
        for k in Data_maker:
            for j in Activation_fun:
                for i in Noise_samples:
                    # Data
                    X, y = k[1](i[0])
                    # Pre-Processing
                    X = X.T
                    y = np.array([[i == 0, i == 1] for i in y]).T
                    # Network
                    network = NeuralNetwork(input_dim = 2, hidden_dim = l[0], output_dim = 2, activation_type = j[1], regularization =0, random_seed = None)
                    network.train(X, y, j[3], 100000, True, .2)
                    # Display
                    x_min, x_max = X.T[:, 0].min() - .5, X.T[:, 0].max() + .5
                    y_min, y_max = X.T[:, 1].min() - .5, X.T[:, 1].max() + .5
                    h = .01

                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    z = network.predict(np.c_[xx.ravel(), yy.ravel()].T)
                    z = z.reshape(xx.shape)

                    plt.figure(figsize = (5, 5))
                    plt.axis('off')
                    plt.title(j[0] + ', ' + str(l[0]) + ', ' + '{0:.3f}'.format(i[0]))
                    plt.contourf(xx, yy, z, cmap = plt.cm.Spectral)
                    plt.scatter(X.T[:, 0], X.T[:, 1], c = y[1], cmap = plt.cm.Spectral)
                    plt.savefig('C:/Users/xiaoqian chen/Documents/GitHub/images/' + i[1] + j[2] + k[0] + l[1] + '.png')
                    # plt.show(block = False)

    # plt.show()

if __name__ == '__main__':
    main()