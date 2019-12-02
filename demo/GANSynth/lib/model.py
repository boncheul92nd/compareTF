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

""" GANSynth Model class definition.
    Exposes external API for generating samples and evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def set_flags(flags):
    """ Set default hyperparameters. """
    flags.set_if_empty('train_root_dir', '/tmp/gansynth/train')
    flags.set_if_empty('train_data_path', '/tmp/gansynth/nsynth-train.tfrecord')

    # Dataset
    flags.set_if_empty('dataset_name', 'nsynth_tfrecord')
    flags.set_if_empty('data_type', 'mel')
    flags.set_if_empty('audio_length', 64000)
    flags.set_if_empty('sample_rate', 16000)

    # specgram_simple_normalizer, specgram_freq_normalizer
    flags.set_if_empty('data_normalizer', 'specgrams_prespecified_normalizer')
    flags.set_if_empty('normalizer_margin', 0.8)
    flags.set_if_empty('normalizer_num_examples', 1000)
    flags.set_if_empty('mag_normalizer_a', 0.0661371661726)
    flags.set_if_empty('mag_normalizer_b', 0.113718730221)
    flags.set_if_empty('p_normalizer_a', 0.8)
    flags.set_if_empty('p_normalizer_b', 0.)

    # Losses
    # Gradient norm target for wasserstein loss
    flags.set_if_empty('gradient_penalty_target', 1.0)
    flags.set_if_empty()