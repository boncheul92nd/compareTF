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

import importlib

from demo.GANSynth.lib import model as lib_model
from demo.GANSynth.lib import flags as lib_flags
import absl.flags
import tensorflow as tf

absl.flags.DEFINE_string('hparams', '{}', 'Flags dict as JSON string.')
absl.flags.DEFINE_string('config', '', 'Name of config module')
FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    absl.flags.FLAGS.alsologtostderr = True
    # Set hyperparams from json args and defauls
    flags = lib_flags.Flags()

    # Config hparams
    config_module = importlib.import_module('demo.GANSynth.mel_prog_hires')
    flags.load(config_module.hparams)

    # Command line hparams
    flags.load_json(FLAGS.hparams)

    # Set default flags
    lib_model.set_flags(flags)

def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()