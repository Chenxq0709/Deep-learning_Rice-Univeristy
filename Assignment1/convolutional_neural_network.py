import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_data(mode='train'):

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels
        x_train, _ = reformat(x_train, y_train)
        x_valid, _ = reformat(x_valid, y_valid)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test, _ = reformat(x_test, y_test)
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y):

    img_size, num_ch, num_class = 28, 1, 10
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


# weight and bais wrappers
def weight_variable(shape):

    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=initer)

def bias_variable(shape):

    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=initial)


def fc_layer(x, num_units, name, use_relu=True):

    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(shape=[in_dim, num_units])
        b = bias_variable(shape=[num_units])
        tf.summary.histogram('Weights', W)
        tf.summary.histogram('bias', b)

        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


def conv_layer(x, filter_size, num_filters, stride, name):

    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = weight_variable(shape=shape)
        b = bias_variable(shape=[num_filters])
        tf.summary.histogram('Weights', W)
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],padding="SAME")
        layer += b
        return tf.nn.relu(layer)


def flatten_layer(layer):

    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):

    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME", name=name)


def variable_summaries(x, name):
    with tf.name_scope(name+'stat'):
        mean = tf.reduce_mean(x)
        tf.summary.scalar('mean',tf.reduce_mean(x))
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
        tf.summary.scalar('std', stddev)
        tf.summary.scalar('max', tf.reduce_max(x))
        tf.summary.scalar('min', tf.reduce_min(x))
        tf.summary.histogram('hist', x)



def main():
    logs_path = "./logs"  # path to the folder that we want to save the logs for TensorBoard
    img_h = img_w = 28  # MNIST images are 28x28
    n_classes = 10  # Number of classes, one class per digit
    n_channels = 1

    # Load MNIST data
    x_train, y_train, x_valid, y_valid = load_data(mode='train')

    # Hyper-parameters
    lr = 0.001  # The optimization initial learning rate
    epochs = 5  # Total number of training epochs
    batch_size = 50  # Training batch size
    display_freq = 1000

    # Network Configuration
    # 1st Convolutional Layer
    filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters1 = 32  # There are 32 of these filters.
    stride1 = 1  # The stride of the sliding window

    # 2nd Convolutional Layer
    filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters2 = 64  # There are 64 of these filters.
    stride2 = 1  # The stride of the sliding window

    # Create the network graph
    # Placeholders for inputs (x), outputs(y)
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
        y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
    conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name='conv1')
    pool1 = max_pool(conv1, ksize=2, stride=2, name='pool1')
    conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name='conv2')
    pool2 = max_pool(conv2, ksize=2, stride=2, name='pool2')
    layer_flat = flatten_layer(pool2)
    dropped = tf.nn.dropout(tf.nn.relu(layer_flat), 0.5)
    output_logits = fc_layer(dropped, n_classes, 'OUT', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.variable_scope('Train'):

        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
        tf.summary.scalar('loss', loss)
        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        # Creating the op for initializing all variables and Merge all summaries
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        # Launch the graph (session)
        with tf.Session() as sess:
            sess.run(init)
            global_step = 0
            summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
            # Number of training iterations in each epoch
            num_tr_iter = int(len(y_train) / batch_size)
            for epoch in range(epochs):
                print('Training epoch: {}'.format(epoch + 1))
                x_train, y_train = randomize(x_train, y_train)
                for iteration in range(num_tr_iter):
                    global_step += 1
                    start = iteration * batch_size
                    end = (iteration + 1) * batch_size
                    x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

                    # Run optimization op (backprop)
                    feed_dict_batch = {x: x_batch, y: y_batch}
                    sess.run(optimizer, feed_dict=feed_dict_batch)

                    if iteration % display_freq == 0:
                        # Calculate and display the batch loss and accuracy
                        loss_batch, acc_batch, summary_tr = sess.run([loss, accuracy, merged], feed_dict=feed_dict_batch)
                        summary_writer.add_summary(summary_tr, global_step)

                        print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
                # Run validation after every epoch
                feed_dict_valid = {x: x_valid, y: y_valid}
                loss_valid, acc_valid, summary_val = sess.run([loss, accuracy, merged], feed_dict=feed_dict_valid)
                summary_writer.add_summary(summary_val, global_step)
                print('---------------------------------------------------------')
                print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
                      format(epoch + 1, loss_valid, acc_valid))
                print('---------------------------------------------------------')
        sess.close()
if __name__ == '__main__':
    main()
