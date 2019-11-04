import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
sess = tf.InteractiveSession()


def weight_variable(name, shape):
    weights = tf.get_variable(name = name, shape = shape, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed = None, dtype = tf.float32))
    return weights


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape = shape, dtype = tf.float32)
    bias = tf.get_variable(name = name, dtype = tf.float32, initializer = initial)
    return bias


def layer_calc(layer_name, x, w_shape, stride = [1, 1, 1, 1], max_pool_shape = [1, 2, 2, 1], max_pool_stride = [1, 2, 2, 1], act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable('W' + layer_name, w_shape)
        #    variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable('b' + layer_name, [w_shape[-1]])
        #    variable_summaries(weights)
        with tf.name_scope('linearity'):
            pre_activation = tf.nn.conv2d(x, weights, strides = stride, padding = 'SAME') + biases
        #    tf.summary.histogram('pre-activations', pre_activation)
        activation = act(pre_activation, name = 'activation')
        tf.summary.histogram('conv', activation)
        max_pool = tf.nn.max_pool(value = activation, ksize = max_pool_shape, strides = max_pool_stride, padding = 'SAME')
        tf.summary.histogram('pool', max_pool)
        return max_pool


def flatLayer(layer_name, conv_layer_out, flat_dim, act = tf.nn.relu):
    shape_of_conv_layer = conv_layer_out.get_shape()
    n_features = shape_of_conv_layer[1:4].num_elements()
    conv_layer_out = tf.reshape(conv_layer_out, [-1, n_features])
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable('W' + layer_name, [n_features, flat_dim])
        #    variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable('b' + layer_name, [flat_dim])
        #    variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(conv_layer_out, weights) + biases
        #    tf.summary.histogram('pre_activations', preactivate)
        flat = act(preactivate, name = 'activation')
        tf.summary.histogram('flat', flat)
        return flat


def output_layer(layer_name, flat_layer_in, out_dim):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable('W' + layer_name, [flat_layer_in.get_shape()[1], out_dim])
        #    variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable('b' + layer_name, [out_dim])
        #    variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            output = tf.matmul(flat_layer_in, weights) + biases
            tf.summary.histogram('output', output)
            return output

def variable_summaries(var):
    with tf.name_scope('statistics'):
        tf.summary.scalar('mean', tf.reduce_mean(var))
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var)))))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('hist', var)


def main():
    result_dir = './cnnresult/'
    max_step = 5500  # Total number of training epochs

    start_time = time.time()
    n_classes = 10
    x = tf.placeholder(dtype = tf.float32, shape = [None, 784])
    y_ = tf.placeholder(dtype = tf.int64, shape = [None, n_classes])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    l1_out = layer_calc('Layer1', x_image, [5, 5, 1, 32])
    l2_out = layer_calc('Layer2', l1_out, [5, 5, 32, 64])
    flat_out = flatLayer('flat' + 'layer', l2_out, 1024)
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(flat_out, keep_prob)

    y = output_layer('output' + 'layer', dropped, n_classes)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='loss')
        tf.summary.scalar('loss', loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.SGDOptimizer(learning_rate=1e-3).minimize(loss)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name = 'correct_preds')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
        tf.summary.scalar('accuracy', accuracy)

    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    sess.run(init)

    for i in range(max_step):
        batch = mnist.train.next_batch(50)
        optimizer.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            loss_batch, acc_batch, summary_tr = sess.run([loss, accuracy, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_tr, i)
            summary_writer.flush()
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(i, loss_batch, acc_batch))

    stop_time = time.time()
    print('The training takes %f second to finish' % (stop_time - start_time))


if __name__ == "__main__":
    main()