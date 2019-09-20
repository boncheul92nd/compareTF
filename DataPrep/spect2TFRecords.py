#
#    Before running this program, mv some of the data/train shapes to the validate directory
#    e.g.
#        > mv data/train/squares/data3???.* data/validate/squares/.
#        > mv data/train/triangles/data3???.* data/validate/triangles/.
#
#    Then run the conversion (see the runcmd)
#
#    Then you are ready to use the record files as input to your program
#
#    The original copyright notice below, but this code has been modified to
#    read in tiff files, convert them to jpegs to then be converted to TFRecords
# ----------------------------------------------------------------------------------------------


# Copyright 2016 Google Inc. All Right Reserved.
#
# Licensed under the Apache license, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------------------------

"""
    Converts image data to TFRecords file format with Example protos.
    The image data set is expected to reside in JPEG files located in the
    following directory structure.

        data_dir/label_0/image0.jpeg
        data_dir/label_0/image1.jpeg
        ...
        data_dir/label_1/weird-image.jpeg
        data_dir/label_1/my-image.jpeg
        ...

    Where the sub-directory is the unique label associated with these images.
    This TensorFlow script converts the training and evaluation data into
    a shared data set consisting of TFRecord files

        train_directory/train-00000-of-00128
        train_directory/train-00001-of-00128
        ...
        train_directory/train-00127-of-00128

    and

        validation_directory/validation-00000-of-00128
        validation_directory/validation-00001-of-00128
        ...
        validation_directory/validation-00127-of-00128

    Where we have selected 1024 and 128 shards for each data set. Each record
    within the TFRecord file is a serialized Example proto. The Example proto
    contains the following fields:

        image/encoded: string containing JPEG encoded image in RGB colorspace
        image/height: integer, image height in pixels
        image/width: integer, image width in pixels
        image/colorspace: string, specifying the colorspace, always 'RGB'
        image/channels: integer, specifying the number of channels, always 3
        image/format: string, specifying the format, always 'JPEG'
        image/filename: string containing the basename of the image file

            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'

        image/class/label: integer specifying the index in a classification layer

            The label ranges from [0, num_labels] where 0 is unused and left as
            the background class.

        image/class/text: string specifying the human-readable version of the label

            e.g. 'dog'

    If your data set involves bounding boxes, please look at build_imagenet_data.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import math
import numpy as np
import tensorflow as tf

k_colorspace = 'GrayScale'  # https://www.tensorflow.org/api_guides/python/image
k_channels = 1              # saving sonogram as 2D GrayScale image for now - try 256 channels later
K_image_format = 'PNG'      # also not used for sonogram TFRecord reading and writing

tf.app.flags.DEFINE_string('main_dir', '.', 'Directory that holds all folds')
tf.app.flags.DEFINE_string('fold1_dir', tf.app.flags.FLAGS.main_dir + '/1', 'Training data fold1 directory')
tf.app.flags.DEFINE_string('fold2_dir', tf.app.flags.FLAGS.main_dir + '/2', 'Training data fold2 directory')
tf.app.flags.DEFINE_string('fold3_dir', tf.app.flags.FLAGS.main_dir + '/3', 'Training data fold3 directory')
tf.app.flags.DEFINE_string('fold4_dir', tf.app.flags.FLAGS.main_dir + '/4', 'Training data fold4 directory')
tf.app.flags.DEFINE_string('fold5_dir', tf.app.flags.FLAGS.main_dir + '/5', 'Training data fold5 directory')
tf.app.flags.DEFINE_string('output_dir', tf.app.flags.FLAGS.main_dir, 'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 2, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')


# The labels file contains a list of valid labels are held in this file.
# Assume that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', 'labels.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS



