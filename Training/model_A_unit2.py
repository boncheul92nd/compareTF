import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    seed=None,
    dtype=tf.float32
)

weights = {
    'w_conv1': tf.get_variable(initializer=initializer, shape=[3, 3, 2, 64], name='wc1'),
    'w_conv2': tf.get_variable(initializer=initializer, shape=[3, 3, 64, 64], name='wc2'),
    'w_fc1': tf.get_variable(initializer=initializer, shape=[16*16*64, 384], name='wf1'),
    'w_fc2': tf.get_variable(initializer=initializer, shape=[384, 50], name='wf2')
}

biases = {
    'b_conv1': tf.Variable(tf.constant(0.1, shape=[64]), name='bc1'),
    'b_conv2': tf.Variable(tf.constant(0.1, shape=[64]), name='bc2'),
    'b_fc1': tf.Variable(tf.constant(0.1, shape=[384]), name='bf1'),
    'b_fc2': tf.Variable(tf.constant(0.1, shape=[50]), name='bf2')
}

def conv2d(x, W, b):
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

def downscale(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def conv_net(x, weights, biases, dropout):

    h_conv1 = conv2d(x, weights['w_conv1'], biases['b_conv1'])
    h_pool1 = downscale(h_conv1)

    h_conv2 = conv2d(h_pool1, weights['w_conv2'], biases['b_conv2'])
    h_pool2 = downscale(h_conv2)

    h_conv5_flat = tf.reshape(h_pool2, [-1, 16*16*64])
    h_fc1 = tf.nn.leaky_relu(tf.add(tf.matmul(h_conv5_flat, weights['w_fc1']), biases['b_fc1']))
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    out = tf.add(tf.matmul(h_fc1_drop, weights['w_fc2']), biases['b_fc2'])

    return out