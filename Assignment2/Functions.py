import re
import numpy as np
import tensorflow as tf
from skimage import io
from sklearn.preprocessing import OneHotEncoder
from collections import namedtuple

initializer = tf.contrib.layers.xavier_initializer()

def weight(shape):
    return tf.Variable(lambda: initializer(shape = shape))


def train_network(x_train, y_train, x_test, y_test, graph, x, y, optimizer, epochs, batch_size, train_metric, eval_metric = None, monitor_vars = None, monitor_iters = 100, train_feed_dict = { }, eval_feed_dict = None):
    session = tf.Session(graph = graph)

    with graph.as_default():
        n_train_samples = x_train.shape[0]
        train_step = optimizer.minimize(train_metric)

        if eval_metric is None:
            eval_metric = train_metric

        if eval_feed_dict is None:
            eval_feed_dict = train_feed_dict

        monitor_train = []
        monitor_test = []
        global_variables_initializer = tf.global_variables_initializer()
        session.run(global_variables_initializer)

        for n in range(epochs):
            print('epoch ' + str(n) + ' of ' + str(epochs))
            permutation = np.arange(n_train_samples)
            np.random.shuffle(permutation)

            for i, k in enumerate(range(0, n_train_samples, batch_size)):
                x_batch, y_batch = x_train[permutation[k: k + batch_size]], y_train[permutation[k: k + batch_size]]
                session.run(train_step, feed_dict = { x: x_batch, y: y_batch, **train_feed_dict })

                if monitor_vars is not None and i % monitor_iters == 0:
                    monitor_train.append((n * n_train_samples + k, session.run(monitor_vars, feed_dict = { x: x_train, y: y_train, **eval_feed_dict })))
                    monitor_test.append((n * n_train_samples + k, session.run(monitor_vars, feed_dict = { x: x_test, y: y_test, **eval_feed_dict })))

        eval_train = session.run(eval_metric, feed_dict = { x: x_train, y: y_train, **eval_feed_dict })
        eval_test = session.run(eval_metric, feed_dict = { x: x_test, y: y_test, **eval_feed_dict })

    if monitor_vars is None:
        return eval_train, eval_test, session
    else:
        return eval_train, eval_test, session, monitor_train, monitor_test


def make_rnn(rnn_cell, rnn_cell_size, rnn_cell_init_copies = 1):
    mnist_graph = tf.Graph()

    with mnist_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 28, 28))
        y = tf.placeholder(tf.float32, (None, 10))
        rnn = rnn_cell(rnn_cell_size)
        state = tf.fill(tf.stack((tf.shape(x)[0], rnn_cell_size)), 0.0)

        if rnn_cell_init_copies > 1:
            state = (state,) * rnn_cell_init_copies

        for i in range(28):
            output, state = rnn(x[:, i, :], state)

        W = weight((rnn_cell_size, 10))
        b = weight((10,))
        y_out = tf.matmul(output, W) + b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_out))
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    MNISTVars = namedtuple('MNISTGraph', ('x', 'y', 'output', 'W', 'b', 'y_out', 'cross_entropy', 'accuracy'))
    mnist_vars = MNISTVars(x = x, y = y, output = output, W = W, b = b, y_out = y_out, cross_entropy = cross_entropy, accuracy = accuracy)

    return mnist_graph, mnist_vars