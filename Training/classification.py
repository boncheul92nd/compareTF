import os
os.environ["CUDA_DIVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import importlib
from parameters import *
m = importlib.import_module(FLAGS.model) #import CNN model
import utils.spectreader as spectreader


# some utility functions
def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

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
    k_height = K_FREQBINS
    k_input_chhannels = 1
elif FRE_ORIENTATION is "1D":
    k_height = 1
    k_input_chhannels = K_FREQBINS
else:
    raise ValueError("please only enter '1D' or '2D'")

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
        lis.extend(
            [
                a_dir + '/' + name for name in os.listdir(a_dir)if name.startswith("fold" + str(num))
            ]
        )
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
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

NUM_THREADS = 4
foldlist = [1, 2, 3, 4, 5]
max_acc = []
max_epochs = []

start_time_long = time.monotonic()
text_file = open(save_path + "/stft-double_v2.txt", "w")    # Save training data
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

for fold in foldlist:

    test_acc_list = []

    datanumlist = [x for x in foldlist if x != fold]
    validatenumlist = [fold]

    datafnames = get_TFR_folds(INDIR, datanumlist)
    target, data = getImage(datafnames, FRE_ORIENTATION, n_epochs=EPOCHS)

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
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:

        # Initialize all variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("{} Start training...".format(datetime.now()))
        start_time = time.monotonic()

        try:
            if coord.should_stop():
                print("coord should stop")

            e = 1
            step = 1
            print("{} Epoch number: {}".format(datetime.now(), e))

            while True: # For each mini-batch until data runs out after specified number of epochs
                if coord.should_stop():
                    print("data feed done, quitting")
                    break

                # Create training mini-batch here
                batch_data, batch_labels = sess.run([imageBatch, labelBatch])
                # Train and backprop
                sess.run(optimizer, feed_dict={x:batch_data, y:batch_labels, keep_prob:dropout})

                if (step % display_step == 0):
                    s = sess.run(merged_summary, feed_dict={x:batch_data, y:batch_labels, keep_prob: 1.})
                    writer.add_summary(s, step)

                if (step % test_N_steps == 0):
                    test_acc = 0.
                    test_count = 0

                    for j in range(test_batches_per_epoch):
                        try:
                            # Prepare test mini-batch
                            test_batch, label_batch = sess.run([vimageBatch, vlabelBatch])

                            acc = sess.run(accuracy, feed_dict={x: test_batch, y:label_batch, keep_prob: 1.})
                            test_acc += acc * BATCH_SIZE
                            test_count += 1 * BATCH_SIZE
                        except (Exception) as ex:
                            print(ex)
                    # Calculate total test accuracy
                    test_acc /= test_count
                    print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))
                    text_file.write("{} Test Accuracy = {:.4f}\n".format(datetime.now(), test_acc))
                    test_acc_list.append(test_acc)

                if (step % train_batches_per_epoch == 0):
                    e += 1
                    print("{} Epoch number: {}".format(datetime.now(), e))
                    # Save checkpoint of the model
                    if (e % checkpoint_epoch == 0):
                        checkpoint_name = os.path.join(checkpoint_path, dataset_name + 'model_fold' + str(fold) + '_epoch' + str(e) + '.ckpt')
                        saver.save(sess, checkpoint_name)
                        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                step += 1

        except (tf.errors.OutOfRangeError) as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)

        # Find the max test score and the epoch it belons to
        max_acc.append(max(test_acc_list))
        max_epoch = test_acc_list.index(max(test_acc_list)) + 1
        max_epochs.append(max_epoch)

        elapsed_time = time.monotonic() - start_time
        print(elapsed_time)
        text_file.write("--- Training time taken: {} ---\n".format(time_taken(elapsed_time)))
        print("--- Training time taken: ", time_taken(elapsed_time), "---")
        print("------------------------")

        # Return the max accuracies of each fold and their respective epochs
        print(max_acc)
        print(max_epochs)

    sess.close()
writer.close()
elapsed_time_long = time.monotonic() - start_time_long
print("*** All runs completed ***")
text_file.write("Total time taken:")
text_file.write(time_taken(elapsed_time_long))
print("Total time taken:",time_taken(elapsed_time_long))
text_file.close()