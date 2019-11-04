import tensorflow as tf
import numpy as np
import re
from skimage import io


def load_modified_cifar10():
    class_regex = re.compile(r'.*\\(\d)\\.*')
    train_data = io.imread_collection('CIFAR10\\Train\\*\\*.png')
    test_data = io.imread_collection('CIFAR10\\Test\\*\\*.png')

    class_encoder = OneHotEncoder(10)
    train_classes = class_encoder.fit_transform(np.array([int(class_regex.match(path).group(1)) for path in train_data.files])[:, None]).toarray()
    test_classes = class_encoder.transform(np.array([int(class_regex.match(path).group(1)) for path in test_data.files])[:, None]).toarray()
    train_data_processed = np.stack(train_data).astype(float) / 255

    test_data_processed = np.stack(test_data).astype(float) / 255
    return train_data_processed, train_classes, test_data_processed, test_classes


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


# 1 Visualizing a CNN with CIFAR10
def main():
    logs_path = "./logs"  # path to the folder that we want to save the logs for TensorBoard
    img_h = img_w = 28  # CIFAR10 greyscaled images are 28x28
    n_classes = 10  # Number of classes, one class per digit
    n_channels = 1


    # 1.1 CIFAR10 Dataset
    x_train, y_train, x_test, y_test = load_modified_cifar10()

    # Hyper-parameters
    lr = 0.001  # The optimization initial learning rate
    epochs = 5  # Total number of training epochs
    batch_size = 50  # Training batch size
    display_freq = 1000

    # Network Configuration
    # Create the network graph

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
        y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
    conv1 = conv_layer(x, filter_size=5, num_filters=32, stride=1, name='conv1')
    pool1 = max_pool(conv1, ksize=2, stride=2, name='pool1')
    conv2 = conv_layer(pool1, filter_size=5, num_filters=64, stride=1,  name='conv2')
    pool2 = max_pool(conv2, ksize=2, stride=2, name='pool2')
    layer_flat = flatten_layer(pool2)
    fc1 = fc_layer(layer_flat, num_units=1024, 'FC1', use_relu=True)
    fc2 = fc_layer(fc1, num_units=10, 'FC2', use_relu=True)
    output_logits = fc_layer(fc2, n_classes, 'OUT', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    with tf.variable_scope('Train'):

        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
        tf.summary.scalar('loss', loss)
        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.SGDOptimizer(learning_rate=lr, name='SGD-op').minimize(loss)
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
                feed_dict_valid = {x: x_test, y: y_test}
                loss_test, acc_test, summary_val = sess.run([loss, accuracy, merged], feed_dict=feed_dict_valid)
                summary_writer.add_summary(summary_val, global_step)
                print('---------------------------------------------------------')
                print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
                      format(epoch + 1, loss_test, acc_test))
                print('---------------------------------------------------------')
        sess.close()
if __name__ == '__main__':
    main()
