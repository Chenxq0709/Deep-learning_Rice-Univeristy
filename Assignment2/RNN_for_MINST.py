from Functions import *
from tensorflow.python.ops import rnn, rnn_cell
from matplotlib import pyplot as plt

# 3 Build and Train an RNN on MNIST

# 3.1 Setup an RNN

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

x_train = mnist.train.images.reshape((-1, 28, 28))
y_train = mnist.train.labels

x_test = mnist.test.images.reshape((-1, 28, 28))
y_test = mnist.test.labels

rnn_graph, rnn_vars = make_rnn(tf.contrib.rnn.BasicRNNCell, 1024)

optimizer = tf.train.AdamOptimizer(1e-4)
epochs = 5
batch_size = 100

eval_train, eval_test, sess = train_network(x_train, y_train, x_test, y_test, rnn_graph, rnn_vars.x, rnn_vars.y, optimizer, epochs, batch_size, rnn_vars.cross_entropy, rnn_vars.accuracy)
sess.close()
print('Train Accuracy: %.4f' % eval_train)
print('Test Accuracy: %.4f' % eval_test)

# 3.2 How about using an LSTM or GRU

cell_cpy_pairs = [(tf.contrib.rnn.BasicRNNCell, 1), (tf.contrib.rnn.BasicLSTMCell, 2), (tf.contrib.rnn.GRUCell, 1)]
cell_sizes = [256]
monitor_iters = 100
rnn_results = []

for rnn_cell, init_cpy in cell_cpy_pairs:
    rnn_cell_results = []

    for cell_size in cell_sizes:
        print(rnn_cell.__name__, cell_size)
        rnn_graph, rnn_vars = make_rnn(rnn_cell, cell_size, init_cpy)
        eval_train, eval_test, sess, monitor_train, monitor_test = train_network(x_train, y_train, x_test, y_test, rnn_graph, rnn_vars.x, rnn_vars.y, optimizer, epochs, batch_size, rnn_vars.cross_entropy, rnn_vars.accuracy,
                                                                                 monitor_vars = (rnn_vars.accuracy, rnn_vars.cross_entropy), monitor_iters = monitor_iters)
        sess.close()
        rnn_cell_results.append((eval_train, eval_test, monitor_train, monitor_test))

    rnn_results.append(rnn_cell_results)

plt.figure(figsize = (7, 7))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for j, cell_size in enumerate(cell_sizes):
    for i, (rnn_cell, init_cpy) in enumerate(cell_cpy_pairs):
        eval_train, eval_test, monitor_train, monitor_test = rnn_results[i][j]
        iters, train_vals = zip(*monitor_train)
        train_accs, train_losses = zip(*train_vals)
        _, test_vals = zip(*monitor_test)
        test_accs, test_losses = zip(*test_vals)

        plt.subplot(3, 2, j * 2 + 1)
        plt.plot(np.array(iters) / x_train.shape[0], train_accs, color = colors[i], label = '%s Train' % rnn_cell.__name__)
        plt.plot(np.array(iters) / x_train.shape[0], test_accs, color = colors[i], linestyle = '--', label = '%s Test' % rnn_cell.__name__)
        plt.title('Epoch vs Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.2)

        plt.subplot(3, 2, j * 2 + 2)
        plt.plot(np.array(iters) / x_train.shape[0], train_losses, color = colors[i], label = '%s Train' % rnn_cell.__name__)
        plt.plot(np.array(iters) / x_train.shape[0], test_losses, color = colors[i], linestyle = '--', label = '%s Test' % rnn_cell.__name__)
        plt.title('Epoch vs Cross Entropy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy')
        plt.ylim(0, 2.5)

plt.tight_layout()
plt.show()