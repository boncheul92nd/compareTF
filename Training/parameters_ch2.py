import argparse
import os

# Pass some user input as flags
FLAGS = None
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datafolder', type=str, help='basename of folder where TFRecords are kept', default='res/IF-Mel')
parser.add_argument('--fold', type=int, help='fold used as test set for k-fold cross validation', default=1)
parser.add_argument('--freqorientation', type=str, help='convolution over 1D or 2D. If 1D, then freq bins treated as channels. If 2D, then freq bins is the height of input', default='2D')
parser.add_argument('--model', type=str, help='load the model to train', default='model_B_unit4')
parser.add_argument('--freqbins', type=int, help='number of frequency bins in the spectrogram input', default=64)
parser.add_argument('--num_frames', type=int, help='number of frames in the spectrogram input (must divisible by 3)', default=64)
parser.add_argument('--batchsize', type=int, help='number of data records per training batch', default=100)
parser.add_argument('--n_epochs', type=int, help='number of epochs to use for training', default=200)
parser.add_argument('--l1channels', type=int, help='number of channels in the first convolutional layer', default=180)
parser.add_argument('--l2channels', type=int, help='number of channels in the second convolutional layer (ignored if numconvlayers is 1)', default=48)
parser.add_argument('--l3channels', type=int, help='number of channels in the third convolutional layer (ignored if numconvlayers is 1)', default=96)
parser.add_argument('--fcsize', type=int, help='dimension of the final fully-connected layer', default=800)
parser.add_argument('--num_labels', type=int, help='number of classes in data', choices=[2, 50], default=50)
parser.add_argument('--files_per_fold', type=int, help='number of samples per fold', choices=[2, 400], default=400)
parser.add_argument('--save_path', type=str, help='output root directory for logging', default='../Results')

FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed:  {0}'.format(FLAGS))

# Data location
dataset_name = "ESC50"  # supports ESC50 and US8K
TRAINING_FOLDS = 4

dataset_path = "../" + FLAGS.datafolder
INDIR = dataset_path
save_path = FLAGS.save_path # path to save output
if not os.path.isdir(save_path): os.mkdir(save_path)

text_file_name = "/" + FLAGS.datafolder + FLAGS.freqorientation # name of text file to output with result
print(text_file_name)

# Image/Data Parameters
K_FREQBINS = FLAGS.freqbins     # pixel height ie. frequency bins
K_NUMFRAMES = FLAGS.num_frames   # pixel with ie. time bins
N_LABELS = FLAGS.num_labels     # number of classes

FRE_ORIENTATION = FLAGS.freqorientation # supports 2D and 1D
if FRE_ORIENTATION in ["2D", "1D"]:
    pass
else:
    raise ValueError("Please only enter '1D' or '2D'")

if FRE_ORIENTATION == "1D":
    NUM_CHANNELS = K_FREQBINS   # Number of image channels
    K_HEIGHT = 1
    print("Orientation is 1D, so setting NUM_CHANNELS to " + str(K_FREQBINS))
elif FRE_ORIENTATION == "2D":
    NUM_CHANNELS = 2            # Number of image channels
    K_HEIGHT = K_FREQBINS
    print("Orientation is 2D, so setting NUM_CHANNELS to " + str(1))

# See threading and queueing info: https://www.tensorflow.org/programmers_guide/reading_data
files_per_fold = FLAGS.files_per_fold   # Number of samples per fold
NUM_THREADS = 4                         # Threads to read in TFReocrds; don't want more threads then there are

# Model parameters
L1_CHANNELS = FLAGS.l1channels
L2_CHANNELS = FLAGS.l2channels
L3_CHANNELS = FLAGS.l3channels
FC_SIZE = FLAGS.fcsize

# Learning parameters
BATCH_SIZE = FLAGS.batchsize
EPOCHS = FLAGS.n_epochs
TOTAL_RUNS = 1  # number of rounds of k-fold cross validation done

test_batches_per_epoch = max(1, int(files_per_fold / BATCH_SIZE)) # Include check for batch_size > files_per_fold
train_batches_per_epoch = max(1, int(files_per_fold * TRAINING_FOLDS / BATCH_SIZE)) # Equivalent to steps per epoch
test_N_steps = train_batches_per_epoch  # test every n step
print("Batch_size = " + str(BATCH_SIZE) + ", and files_per_fold is " + str(files_per_fold))
print("Will test every " + str(test_N_steps) + " batches.")

# Network parameters
epsilon = 1e-08     # Epsilon value for Adam optimizer
dropout = .5        # Dropout, probability to keep units
l2reg = True        # If want L2 regularization
l2regfull = False   # If want L2 regularization only on dense layers, else L2 regularization on all weight layers
beta = 0.001        # L2-regularization

# Tensorboard checkpoint parameters
display_step = 4        # How often we want to write the tf.summary data to disk. Each step denotes 1 mini-batch
checkpoint_epoch = 250  # Checkpoint and save model every checkpoint_epoch

# Train/Test holdout split parameters. Can ignore if not using holdout
hold_prop = 0.4
rand_seed = 14