"""
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'

  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

k_colorspace = 'GrayScale'
k_channels = 1  # saving songram as 2D GrayScale image for now - try 256 channels later
k_image_format = 'PNG'  # also not used for sonogram TFRecord reading and writing

tf.app.flags.DEFINE_string('main_dir', '../res/Phsae-Mel+H',
                           'Directory that holds all folds')

tf.app.flags.DEFINE_string('fold1_dir', tf.app.flags.FLAGS.main_dir + '/1',
                           'Training data fold1 directory')
tf.app.flags.DEFINE_string('fold2_dir', tf.app.flags.FLAGS.main_dir + '/2',
                           'Training data fold2 directory')
tf.app.flags.DEFINE_string('fold3_dir', tf.app.flags.FLAGS.main_dir + '/3',
                           'Training data fold3 directory')
tf.app.flags.DEFINE_string('fold4_dir', tf.app.flags.FLAGS.main_dir + '/4',
                           'Training data fold4 directory')
tf.app.flags.DEFINE_string('fold5_dir', tf.app.flags.FLAGS.main_dir + '/5',
                           'Training data fold5 directory')
tf.app.flags.DEFINE_string('output_dir', tf.app.flags.FLAGS.main_dir,
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', 'labels.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    # if not isinstance(value, list):
    #  value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(filename, image_buffer_ch1, image_buffer_ch2, label, text, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/colorspace': _bytes_feature(tf.compat.as_bytes(k_colorspace)),
                'image/channels': _int64_feature(k_channels+1),
                'image/class/label': _int64_feature(label),
                'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
                'image/format': _bytes_feature(tf.compat.as_bytes(k_image_format)),
                'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
                'image/encoded_ch1': _bytes_feature(tf.compat.as_bytes(image_buffer_ch1)),
                'image/encoded_ch2': _bytes_feature(tf.compat.as_bytes(image_buffer_ch2))
            }
        )
    )
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes Grayscale PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=k_channels)

    def decode_png(self, image_data):
        # Decode the image data as a png image
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3  # "PNG needs to have height x width x channels"
        assert image.shape[2] == k_channels  # "Our spectrograms have 1 channel (Grayscale)"
        return image


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.png'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, png encoding of grayscale image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    print(filename)
    image_data_channel1 = tf.gfile.FastGFile(filename, 'rb').read()

    a = os.path.basename(filename).split('.')
    b = a[0] + '_IF'
    c = b + '.' + a[1]
    d = filename[:len(filename) - len(os.path.basename(filename))]
    e = d + c
    image_data_channel2 = tf.gfile.FastGFile(e, 'rb').read()

    # Decode the Grayscale PNG.
    image = coder.decode_png(image_data_channel1)

    # Check that image converted to Grayscale
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == k_channels

    return image_data_channel1, image_data_channel2, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer_ch1, image_buffer_ch2, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer_ch1, image_buffer_ch2, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0

    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """
    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [
        l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()
    ]

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of files and labels.
    for text in unique_labels:
        file_path = '%s/%s/*[^_IF].png' % (data_dir, text)
        matching_files = tf.gfile.Glob(file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d PNG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):

    # Run it!
    _process_dataset(
        name='fold1',
        directory=FLAGS.fold1_dir,
        num_shards=FLAGS.train_shards,
        labels_file=FLAGS.labels_file
    )

    _process_dataset(
        name='fold2',
        directory=FLAGS.fold2_dir,
        num_shards=FLAGS.train_shards,
        labels_file=FLAGS.labels_file
    )

    _process_dataset(
        name='fold3',
        directory=FLAGS.fold3_dir,
        num_shards=FLAGS.train_shards,
        labels_file=FLAGS.labels_file
    )

    _process_dataset(
        name='fold4',
        directory=FLAGS.fold4_dir,
        num_shards=FLAGS.train_shards,
        labels_file=FLAGS.labels_file
    )

    _process_dataset(
        name='fold5',
        directory=FLAGS.fold5_dir,
        num_shards=FLAGS.train_shards,
        labels_file=FLAGS.labels_file
    )


if __name__ == '__main__':
    tf.app.run()
