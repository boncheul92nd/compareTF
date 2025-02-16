import tensorflow as tf
from parameters import *

if FRE_ORIENTATION == "2D":
    k_height = K_HEIGHT
    k_input_channels = NUM_CHANNELS
    k_conv_rows = 3 # conv kernel height
    k_conv_cols = 3 # conv kernel widht
    k_pool_rows = 4
    k_downsampled_height = -(-k_height//4)
    k_downsampled_width = -(-K_NUMFRAMES//4)

elif FRE_ORIENTATION == "1D":
    k_height = K_HEIGHT
    k_input_channels = NUM_CHANNELS
    k_conv_rows = 1 # conv kernel height
    k_conv_cols = 3 # conv kernel width
    k_pool_rows = 1
    k_downsampled_height = 1
    k_downsampled_width = -(-K_NUMFRAMES//4)

# Model params common to both 1D and 2D
K_NUM_CONV_LAYERS = 1
K_CONV_STRIDE_ROWS = 1  # kernel horizontal stride
K_CONV_STRIDE_COLS = 1  # kernel vertical stride
K_POOL_STRIDE_ROWS = k_pool_rows

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k_h=k_pool_rows, k_w=4):
    # MaxPool2D warpper
    # ksize = [batch, height, width,channels]
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, dropout)

    # Fully connected layer
    # Reshape conv1 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 or freqbin input, L1_CHANNELS outputs
    'wc1': tf.Variable(tf.truncated_normal([k_conv_rows, k_conv_rows, k_input_channels, L1_CHANNELS], stddev=0.1), name='wc1'),

    # Fully connected, (37//4=10)*(50//4=13)*L1_CHANNELS inputs, 1,200 outputs
    'wd1': tf.Variable(tf.truncated_normal([k_downsampled_height * k_downsampled_width * L1_CHANNELS, FC_SIZE], stddev=0.1), name='wd1'),

    # 1,200 inputs, 50 outputs (class prediction)
    'wout': tf.Variable(tf.truncated_normal([FC_SIZE, N_LABELS], stddev=0.1), name='wout')

}

biases = {
    'bc1': tf.Variable(tf.zeros([L1_CHANNELS]), name='bc1'),
    'bd1': tf.Variable(tf.constant(1.0, shape=[FC_SIZE]), name='bd1'),
    'bout': tf.Variable(tf.constant(1.0, shape=[N_LABELS]), name='bout')
}