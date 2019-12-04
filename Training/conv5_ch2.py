import tensorflow as tf


weights = {
    'w_conv1': tf.Variable(tf.truncated_normal([5, 5, 2, 64], stddev=5e-2), name='wc1'),
    'w_conv2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-2), name='wc2'),
    'w_conv3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=5e-2), name='wc3'),
    'w_conv4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=5e-2), name='wc4'),
    'w_conv5': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=5e-2), name='wc5'),
    'w_fc1': tf.Variable(tf.truncated_normal(shape=[4*4*128, 384], stddev=5e-2)),
    'w_fc2': tf.Variable(tf.truncated_normal(shape=[384, 50], stddev=5e-2))
}

biases = {
    'b_conv1': tf.Variable(tf.constant(0.1, shape=[64])),
    'b_conv2': tf.Variable(tf.constant(0.1, shape=[64])),
    'b_conv3': tf.Variable(tf.constant(0.1, shape=[128])),
    'b_conv4': tf.Variable(tf.constant(0.1, shape=[128])),
    'b_conv5': tf.Variable(tf.constant(0.1, shape=[128])),
    'b_fc1': tf.Variable(tf.constant(0.1, shape=[384])),
    'b_fc2': tf.Variable(tf.constant(0.1, shape=[50]))
}

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):

    h_conv1 = conv2d(x, weights['w_conv1'], biases['b_conv1'])
    h_pool1 = maxpool2d(h_conv1)

    h_conv2 = conv2d(h_pool1, weights['w_conv2'], biases['b_conv2'])
    h_pool2 = maxpool2d(h_conv2)

    h_conv3 = conv2d(h_pool2, weights['w_conv3'], biases['b_conv3'])
    h_conv4 = conv2d(h_conv3, weights['w_conv4'], biases['b_conv4'])
    h_conv5 = conv2d(h_conv4, weights['w_conv5'], biases['b_conv5'])

    h_conv5_flat = tf.reshape(h_conv5, [-1, 4*4*128])
    h_fc1 = tf.nn.leaky_relu(tf.add(tf.matmul(h_conv5_flat, weights['w_fc1']), biases['b_fc1']))
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    out = tf.add(tf.matmul(h_fc1_drop, weights['w_fc2']), biases['b_fc2'])

    return out