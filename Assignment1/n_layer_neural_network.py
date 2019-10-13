import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
__author__ = 'Xiaoqian Chen'


def generate_data(quantity, seed, noise):

    np.random.seed(seed)
    x, y = datasets.make_moons(quantity, noise=noise)
    return x, y


def plot_decision_boundary(predictor, x, y):

    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    z = predictor(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class DeepNeuralNetwork(object):

    def __init__(self, dimensions=np.array([2, 3, 2]), activation_type='sigmoid', reg_lambda=0, random_seed=None):

        np.random.seed(random_seed)
        # dimensions = [nn_input_dim, nn_hidden_dim , nn_output_dim]
        self.dimensions = np.array(dimensions)
        self.activation_type = activation_type
        self.reg_lambda = reg_lambda

        self.w = dict([(i + 1, 2 * np.random.random([self.dimensions[i], self.dimensions[i - 1]]) - 1) for i in range(1, len(self.dimensions))])
        self.b = dict([(i + 1, 2 * np.random.random([self.dimensions[i], 1]) - 1) for i in range(1, len(self.dimensions))])
        self.cachew = dict([(i + 1, np.zeros([self.dimensions[i], self.dimensions[i - 1]])) for i in range(1, len(self.dimensions))])
        self.cacheb = dict([(i + 1, np.zeros([self.dimensions[i], 1])) for i in range(1, len(self.dimensions))])
        self.hidden = dict()
        self.z = dict()
        self.probs = []

    def feed_forward(self, X):

        self.hidden[1] = np.array(X)
        self.z[2] = self.w[2] @ self.hidden[1] + self.b[2]

        for i in range(2, len(self.dimensions)):
            self.hidden[i] = self.activation(self.z[i])
            self.z[i + 1] = self.w[i + 1] @ self.hidden[i] + self.b[i + 1]

        self.hidden[len(self.dimensions)] = self.z[len(self.dimensions)]
        self.probs = self.soft_max(self.z[len(self.dimensions)])



    def activation(self, x):

        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(- x))
        elif self.activation_type == 'relu':
            return (x > 0) * x
        elif self.activation_type == 'prelu':
            return np.where(x > 0, x, x * 0.01)
        elif self.activation_type == 'elu':
            return np.where(x > 0, x, 0.01 * (np.exp(x)-1))


    def activation_derivative(self, x):

        if self.activation_type == 'tanh':
            y = self.activation(x)
            return 1-y**2
        elif self.activation_type == 'sigmoid':
            y = self.activation(x)
            return y * (1 - y)
        elif self.activation_type == 'relu':
            return x > 0
        elif self.activation_type == 'prelu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation_type == 'elu':
            return np.where(x > 0, 1, 0.01 * np.exp(x))

    def soft_max(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def back_propagation(self, X, y):

        self.hidden[0] = np.array(X)
        d_b = dict()
        d_w = dict()
        delta = dict()
       # delta[len(self.dimensions)] =  self.hidden[len(self.dimensions)] - y
        delta[len(self.dimensions)] = self.probs - y
        for l in range(len(self.dimensions) - 1, 1, - 1):
            delta[l] = self.w[l+1].T @ delta[l + 1] * self.activation_derivative(self.z[l])
        for l in range(2, len(self.dimensions)+1):
            d_b[l] = np.sum(delta[l], axis=1, keepdims=True) / len(X[0])
            d_w[l] = np.einsum('ik, jk -> ij', delta[l], self.hidden[l - 1]) / len(X[0])

        return d_w, d_b


    def train(self, X, y, learning_rate, passes, print_loss, print_rate):
        for i in range(0, passes + 1):
            self.feed_forward(X)
            d_w, d_b = self.back_propagation(X, y)


            for j in range(2, len(self.dimensions)+1):
                d_w[j] += self.reg_lambda * self.w[j]
                self.cachew[j] = 0.9 * self.cachew[j] + 0.1 * d_w[j] ** 2
                self.cacheb[j] = 0.9 * self.cacheb[j] + 0.1 * d_b[j] ** 2

                self.b[j] -= learning_rate * d_b[j] / (np.sqrt(self.cacheb[j]) + 0.00001)
                self.w[j] -= learning_rate * d_w[j] / (np.sqrt(self.cachew[j]) + 0.00001)

            if print_loss and i % (int(passes * print_rate)) == 0:
                print("Epoch: {0}, training loss: {1:.5f}". format(i, self.calculate_loss(X, y)))
        print('------------------------train finished---------------------------------')

    def calculate_loss(self, X, y):
        data_loss = - np.sum(y * np.log(self.probs))
        l2norm = 0
        for key in self.w:
            l2norm += self.reg_lambda / 2 * np.linalg.norm(key)
        data_loss += l2norm
        return data_loss / len(X[0])


    def predict(self, X):
        self.feed_forward(X)
        return np.argmax(self.probs, axis=0)


def main():
    Noise_samples = [[0.025, '00'], [.050, '01'], [.100, '02']]
    Activation_fun = [['Parametric ReLU', 'prelu', '03', 0.01]]#, ['Exponential Linear Unites', 'elu', '04',0.01]], ['ReLU', 'relu', '00', 0.01], ['Sigmoid', 'sigmoid', '01', 0.01], ['Hyperbolic Tangent', 'tanh', '02',0.01],
    Data_maker = [['00', lambda x: datasets.make_moons(n_samples = 1000, shuffle = True, noise = x, random_state = None)], ['01', lambda x: datasets.make_circles(n_samples=1000, shuffle=True, noise=x, random_state=None, factor=0.75)]]
    Dimensions = [[[2, 12, 2], '00'], [[2, 6,6, 2], '01'], [[2, 4,4,4, 2], '02']]

    for l in Dimensions:
        for k in Data_maker:
            for j in Activation_fun:
                for i in Noise_samples:
                    # Data
                    X, y = k[1](i[0])

                    # Pre-Processing
                    X = X.T
                    y = np.array([[i == 0, i == 1] for i in y]).T

                    # Network
                    network = DeepNeuralNetwork(dimensions=np.array(l[0]), activation_type=j[1], reg_lambda=0.0001)
                    network.train(X, y, j[3], 10000, True, .25)

                    # Display
                    x_min, x_max = X.T[:, 0].min() - .5, X.T[:, 0].max() + .5
                    y_min, y_max = X.T[:, 1].min() - .5, X.T[:, 1].max() + .5
                    h = .01
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    z = network.predict(np.c_[xx.ravel(), yy.ravel()].T)
                    z = z.reshape(xx.shape)
                    plt.figure(figsize=(5, 5))
                    plt.axis('off')
                    plt.title(j[0] + ', ' + str(l[0][1: - 1]) + ', ' + '{0:.3f}'.format(i[0]))
                    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
                    plt.scatter(X.T[:, 0], X.T[:, 1], c=y[1], cmap=plt.cm.Spectral)
                    plt.savefig('C:/Users/xiaoqian chen/Documents/GitHub/images/' +'d'+ i[1] + j[2] + k[0] + l[1] + '.png')
                    # plt.show(block = False)
                    #plt.show()


if __name__ == '__main__':
    main()