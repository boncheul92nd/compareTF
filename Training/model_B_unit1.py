import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    seed=None,
    dtype=tf.float32
)

weights = {
    'w_conv0': tf.get_variable(initializer=initializer, shape=[1, 1, 2, 32], name='wc0'),
    'w_conv1': tf.get_variable(initializer=initializer, shape=[3, 3, 32, 32], name='wc1'),
    'w_conv2': tf.get_variable(initializer=initializer, shape=[3, 3, 32, 32], name='wc2'),

    'w_fc1': tf.get_variable(initializer=initializer, shape=[4 * 4 * 32, 384], name='wf1'),
    'w_fc2': tf.get_variable(initializer=initializer, shape=[384, 50], name='wf2')
}

biases = {
    'b_conv0': tf.Variable(tf.constant(0.1, shape=[32]), name='bc0'),
    'b_conv1': tf.Variable(tf.constant(0.1, shape=[32]), name='bc1'),
    'b_conv2': tf.Variable(tf.constant(0.1, shape=[32]), name='bc2'),

    'b_fc1': tf.Variable(tf.constant(0.1, shape=[384]), name='bf1'),
    'b_fc2': tf.Variable(tf.constant(0.1, shape=[50]), name='bf2')
}


def conv2d(x, W, b):
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)


def downscale(x):
    return tf.nn.max_pool(x, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):

    conv = tf.nn.conv2d(input=x, filters=weights['w_conv0'], strides=[1, 1, 1, 1], padding='SAME')
    h_conv0 = tf.nn.bias_add(conv, biases['b_conv0'])

    h_conv1 = conv2d(h_conv0, weights['w_conv1'], biases['b_conv1'])
    h_conv2 = conv2d(h_conv1, weights['w_conv2'], biases['b_conv2'])
    h_pool2 = downscale(h_conv2)

    h_conv8_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 32])

    h_fc1 = tf.nn.leaky_relu(tf.add(tf.matmul(h_conv8_flat, weights['w_fc1']), biases['b_fc1']))
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    out = tf.add(tf.matmul(h_fc1_drop, weights['w_fc2']), biases['b_fc2'])

    return out
