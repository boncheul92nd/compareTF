import os
os.environ["CUDA_DIVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import importlib
from Training.parameters import *
m = importlib.import_module(FLAGS.model) #import CNN model
import Training.utils.pickledModel as picledModel
import Training.utils.spectreader as spectreader


# some utility functions
def time_taken(elapsed):
    """ To format time taken in hh:mm:ss. Use with time.monotic() """
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" & (h, m, s)

# function to save/load numpy array to/from file
def save_set(sets, name):
    """
        Writes the data array to .npy file. Can be loaded using load_set.

        :param sets: array to be saved. can take a list
        :param name: string to name the file. follow same order as in sets
    """
    ind = 0
    for x in sets:
        np.save(save_path + '/{}.npy'.format(name[ind]), x)
        ind += 1

def load_set(sets):
    """
        Load existing data array from .npy files.
        Use if have preexisting data or when you don't have to reshuffle the dataset
    """
    return np.load('{}.npy'.format(sets))

if FRE_ORIENTATION is "2D":
    K_height = K_FREQBINS
    k_input_chhannels = 1
elif FRE_ORIENTATION is "1D":
    k_height = 1
    k_input_chhannels = K_FREQBINS
else:
    raise ValueError("please only enter '1D' or '2D'")

# Create list of parameters for serlializing so that network can be properly reconstructed, and for documentation purposes
parameters = {
    'k_height'              : k_height,
    'k_num_frames'          : K_NUMFRAMES,
    'k_input_channels'      : k_input_chhannels,
    'k_num_conv_layers'     : m.K_NUM_CONV_LAYERS,
    'L1_channels'           : L1_CHANNELS,
    'fc_size'               : FC_SIZE,
    'k_conv_rows'           : m.k_conv_rows,
    'k_conv_cols'           : m.k_conv_cols,
    'k_conv_stride_rows'    : m.K_CONV_STRIDE_ROWS,
    'k_conv_stride_cols'    : m.K_CONV_STRIDE_COLS,
    'k_pool_rows'           : m.k_pool_rows,
    'k_pool_stride_rows'    : m.K_POOL_STRIDE_ROWS,
    'k_downsampled_height'  : m.k_downsampled_height,
    'k_downsampled_width'   : m.k_downsampled_width,
    'freqorientation'       : FRE_ORIENTATION
}

def getImage(fnames, freq_orientation, n_epochs=None):
    """
        Reads data from the prepaired *list* of files in fnames of TFRecords, does some preprocessing
        :param
            fnames: List of filenames to read data from
            n_epochs: An integer(optional). Just fed to tf.string_input_producer(). Reads through all data n_epochs times before generating an OutOfRange error. None means read forever.
        :return:
    """
    label, image = spectreader.getImage(fnames, n_epochs)

    # No need to flatten - must just be explicit about shape so that shuffle_batch will work
    image = tf.reshape(image,[K_FREQBINS,K_NUMFRAMES,NUM_CHANNELS])
    if freq_orientation is "1D":
        image = tf.transpose(image, perm=[0, 3, 2, 1])  # Moves freqbins from height to channel dimension

    # Re-define label as a "one-hot" vector, it will be [0, 1] or [1, 0] here.
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, N_LABELS))
    return label, image

def get_TFR_folds(a_dir, foldnumlist):
    """
        Returns a list of files names in a_dir that start with foldX where X is from the foldnumlist
    """
    lis = []
    for num in foldnumlist:
        lis.extend([
            a_dir + '/' + name for name in os.listdir(a_dir) if name.startswith("fold" + str(num))
        ])
    return lis

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = save_path + "/filewriter/"
checkpoint_path = save_path + "/checkpoint/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# tf Graph input placeholders
if FRE_ORIENTATION is "2D":
    x = tf.placeholder(tf.float32, [BATCH_SIZE, K_FREQBINS, K_NUMFRAMES, NUM_CHANNELS])
elif FRE_ORIENTATION is "1D":
    x = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CHANNELS, K_NUMFRAMES, K_FREQBINS])

y = tf.placeholder(tf.int32, [None, N_LABELS])
keep_prob = tf.placeholder(tf.float32, (), name="keep_prob")    # dropout (keep probability)

# Construct model
pred = m.conv_net(x, m.weights, m.biases, keep_prob)

# L2 regularization
lossL2 = tf.add_n([tf.nn.l2_loss(val) for name, val in m.weights.items()]) * beta   # L2 reg on all weight layers
lossL2_onlyfull = tf.add_n([tf.nn.l2_loss(m.weights['wd1']), tf.nn.l2_loss(m.weights['wout'])]) * beta  # L2 reg on dense layers

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    if l2reg:
        if l2regfull:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + lossL2_onlyfull)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + lossL2)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Train op
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(epsilon=epsilon).minimize(loss)

# Add the loss to summary
tf.summary.scalar("cross_entropy", loss)

# Predictions
prob = tf.nn.softmax(pred)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy'. accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

NUM_THREADS = 4
foldlist = [1, 2, 3, 4, 5]
max_acc = []
max_epoch = []

start_time_log = time.monotonic()
text_file = open(save_path + "/stft-double_v2.txt", "w")    # Save training data
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

for fold in foldlist:

    test_acc_list = []

    datanumlist = [x for x in foldlist if x != fold]
    validatenumlist = [fold]

    datafnames = get_TFR_folds(INDIR, datanumlist)
    target, data = getImage(datafnames, FRE_ORIENTATION, nepochs=EPOCHS)

    validatefnames = get_TFR_folds(INDIR, validatenumlist)
    vtarget, vdata = getImage(validatefnames, FRE_ORIENTATION)

    imageBatch, labelBatch = tf.train.shuffle_batch(
        [data, target],
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        allow_smaller_final_batch=True,     # Want to finish an epoch even if datasizes doesn't divide by bachsize
        enqueue_many=False,                 # IMPORTANT to get right, default=False
        capacity=1000,
        min_after_dequeue=500
    )

    vimageBatch, vlabelBatch = tf.train.batch(
        [vdata, vtarget],
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        allow_smaller_final_batch=True,     # Want to finish an epoch even if datasize doesn't divide by batchsize
        enqueue_many=False,                 # IMPORTANT to get right, default=False
        capacity=1000
    )

    text_file.write("*** Initializing fold #%u as test set ***\n" % fold)
    print("*** Initializing fold #%u as test set ***" % fold)

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path + str(fold))

    gpu_options = tf.GPUOptions(allow_growth=True)