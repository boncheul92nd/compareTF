# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers as contrib_layers
from demo.GANSynth.lib import layers
import tensorflow as tf
import six


class ResolutionSchedule(object):
    """ Image resolution upscaling schedule. """

    def __init__(self,
                 scale_mode='ALL',
                 start_resolutions=(4, 4),
                 scale_base=2,
                 num_resolutions=4):
        """ Initializer.

        Args:
            scale_mode:             'ALL' (along both H and W) or 'H' (along H).
            start_resolutions:      An tuple of integers of HxW format for start image resolutions.
                                    Defaults to (4, 4)
            scale_base:             An integer of resolution base multiplier. Defaults to 2.
            num_resolutions:        An integer of how many progressive resolutions (including
                                    `start_resolutions`). Defaults to 4.
        """
        self._scale_mode = scale_mode
        self._start_resolutions = start_resolutions
        self._scale_base = scale_base
        self._num_resolutions = num_resolutions

    @property
    def scale_mode(self):
        return self._scale_mode

    @property
    def start_resolutions(self):
        return tuple(self._start_resolutions)

    @property
    def num_resolutions(self):
        return self._num_resolutions

    @property
    def scale_base(self):
        return self._scale_base

    def downscale(self, images, scale):
        if self._scale_mode == 'ALL':
            return layers.downscale(images, scale)

    def scale_factor(self, block_id):
        """ Returns the scale factor for network  """


def _discriminator_alpha(block_id, progress):
    """ Returns the block input parameter for discriminator network.

    The discriminator has N blocks with `block_id` = 1, 2, ..., N.
    Each block bloc_id accept an
        - input(block_id) transformed from the real data and
        - the output of block_id + 1, i.e. output(block_id + 1)
    The final input is a linear combination of them,
    i.e. alpha * input(block_id) + (1 - alpha) * output(block_id + 1)
    where alpha = _discriminator_alpha(block_id, progress).

    With a fixed block_id, alpha(block_id, progress) stays to be 1
    when progress <= block_id - 1, then linear decays to 0 when
    block_id - 1 < progress <= block_id, and finally stays at 0
    when progress > block_id.

    Args:
        block_id:   An integer of generator block id.
        progress:   A scalar float `Tensor` of training progress.

    Returns:
        A scalar float `Tensor` of block input parameter.
    """
    return tf.clip_by_value(block_id - progress, 0.0, 1.0)


def block_name(block_id):
    """ Returns the scope name for the network block `block_id`. """
    return 'progressive_gan_block{}'.format(block_id)


def discriminator(x,
                  progress,
                  num_filters_fn,
                  resolution_schedule,
                  num_blocks=None,
                  kernel_size=3,
                  simple_arch=False,
                  scope='progressive_gan_discriminator',
                  reuse=None):
    """ Discriminator network for the progressive GAN model.

    Args:
        x:                      A `Tensor` of NHWC format representing images of size `resolution`.
        progress:               A scalar float `Tensor` of training progress.
        num_filters_fn:         A function that maps `block_id` to # of filters for the block.
        resolution_schedule:    An object of `ResolutionSchedule`.
        num_blocks:             An integer of number of blocks. None means maximum number of blocks,
                                i.e. `resolution.schedule.num_resolutions`. Defaults to None.
        kernel_size:            An integer of convolution kernel size.
        simple_arch:            Bool, use a simple architecture.
        scope:                  A string or variable scope.
        reuse:                  Whether to reuse `scope`. Defaults to None which means to inherit
                                the reuse option of the parent scope.

    Returns
        A `Tensor` of model output and a dictionary of model end points.
    """

    """ 
        Confirm that Xavier weight initial value setting shows inefficient result
        when using ReLu as activation function
        
        In this case, the initialization method used, and this method also uses two
        methods, normal distribution and equal distribution
    """
    he_init = contrib_layers.variance_scaling_initializer()

    if num_blocks is None:
        num_blocks = resolution_schedule.num_resolutions

    def _conv2d(scope, x, kernel_size, filters, padding='SAME'):
        return layers.custom_conv2d(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=tf.nn.leaky_relu,
            he_initializer_slope=0.0,
            scope=scope)

    def _from_rgb(x, block_id):
        return _conv2d('from_rgb', x, 1, num_filters_fn(block_id))

    if resolution_schedule.scale_mode == 'H':
        strides = (resolution_schedule.scale_base, 1)
    else:
        strides = (resolution_schedule.scale_base, resolution_schedule.scale_base)

    end_points = {}

    with tf.variable_scope(scope, reuse=reuse):
        x0 = x
        end_points['rgb'] = x0

        lods = []
        for block_id in range(num_blocks, 0, -1):
            with tf.variable_scope(block_name(block_id)):
                scale = resolution_schedule.scale_factor(block_id)
                lod = resolution_schedule.downscale(x0, scale)
                end_points['downscaled_rgb_{}'.format(block_id)] = lod
                if simple_arch:
                    lod = tf.layers.conv2d(
                        lod,
                        num_filters_fn(block_id),
                        kernel_size=1,
                        padding='SAME',
                        name='from_rgb',
                        kernel_initializer=he_init)
                    lod = tf.nn.relu(lod)
                else:
                    lod = _from_rgb(lod, block_id)
                # alpha_i is used to replace lod_select.
                alpha = _discriminator_alpha(block_id, progress)
                end_points['alpha_{}'.format(block_id)] = alpha
            lods.append((lod, alpha))

        lods_iter = iter(lods)
        x, _ = six.next(lods_iter)

        for block_id in range(num_blocks, 1, -1):
            with tf.variable_scope(block_name(block_id)):
                if simple_arch:
                    x = tf.layers.conv2d(
                        x,
                        num_filters_fn(block_id - 1),
                        strides=strides,
                        kernel_size=kernel_size,
                        padding='SAME',
                        name='conv',
                        kernel_initializer=he_init)
                    x = tf.nn.relu(x)
                else:
                    x = _conv2d('conv0', x, kernel_size, num_filters_fn(block_id))
                    x = _conv2d('conv1', x, kernel_size, num_filters_fn(block_id - 1))
                    x = resolution_schedule.downscale(x, resolution_schedule.scale_base)
                lod, alpha = six.next(lods_iter)
                x = alpha * lod + (1.0 - alpha) * x

        with tf.variable_scope(block_name(1)):
            x = layers.scalar_concat(x, layers.minibatch_mean_stddev(x))
            if simple_arch:
                x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
                x = tf.layers.dense(x, num_filters_fn(0), name='last_conv', kernel_initializer=he_init)
                x = tf.reshape(x, [tf.shape(x)[0], 1, 1, num_filters_fn(0)])
                x = tf.nn.relu(x)
            else:
                x = _conv2d('conv0', x, kernel_size, num_filters_fn(1))
                x = _conv2d('conv1', x, resolution_schedule.start_resolutions, num_filters_fn(0), 'VALID')
            end_points['last_conv'] = x
            if simple_arch:
                logits = tf.layers.dense(x, 1, name='logits', kernel_initializer=he_init)
            else:
                logits = tf.layers.dense(x=x, units=1, scope='logits')
            end_points['logits'] = logits

    return logits, end_points