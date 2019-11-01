import os
import time
import librosa
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean
from specgrams_helper import SpecgramsHelper

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class PreProcessor(object):

    def __init__(self):

        # Special class variables
        self._audio_original_stack = None
        self._audio_restored_stack = None
        self._specgram_stack = None
        self._stft_stack = None

        self._SpecgramsHelper = SpecgramsHelper(
            audio_length=64000,
            spec_shape=(256, 512),
            overlap=0.75,
            sample_rate=16000,
            mel_downscale=2
        )

    def _list_dir(self, dir, file_form):
        fname_list = [
            os.path.normcase(f) for f in os.listdir(dir)
            if (not (f.startswith('.')))
        ]

        file_list = [
            os.path.join(dir, f) for f in fname_list
            if os.path.splitext(f)[1] in file_form
        ]

        return file_list

    def _two_to_three(self, two_dim):
        three_dim = np.expand_dims(two_dim, axis=-1)
        return three_dim

    def _load_audio(self, wav_dir, fname):
        audio, sr = librosa.load(
            path='./' + wav_dir + '/' + fname,
            sr=16000,
            mono=True,
            duration=4
        )
        return audio, sr

    def _save_audio(self, wav_dir, audio, sr):
        librosa.output.write_wav('./' + wav_dir + '.wav', audio, sr=sr)

    def data_stack(self, wav_dir):
        full_path = self._list_dir(wav_dir, '.wav')
        two_dim = None
        for idx in range(len(full_path)):
            fname = os.path.basename(full_path[idx])
            audio_original, sample_rate = self._load_audio(wav_dir, fname)
            if two_dim is None:
                two_dim = audio_original
            else:
                two_dim = np.vstack((two_dim, audio_original))
            print(str(idx+1) + ": " + fname)
            del fname
            del audio_original
            del sample_rate

        self._audio_original_stack = self._two_to_three(two_dim)
        np.save('./audio_original_stack.npy', self._audio_original_stack)
        print("1D audio data stacked...")

    def _get_dist(self, original_vector, restored_vector):
        dist = euclidean(original_vector, restored_vector)
        return dist

    # TODO: Only Log Mag difference
    def experiment_A(self, wav_dir):
        self._audio_original_stack = np.load('./' + wav_dir + '_audio_original_stack.npy')
        self._stft_stack = session.run(
            self._SpecgramsHelper.waves_to_stfts(
                tf.convert_to_tensor(
                    self._audio_original_stack
                )
            )
        )
        self._audio_restored_stack = session.run(
            self._SpecgramsHelper.stfts_to_waves(
                self._stft_stack
            )
        )
        np.save('./' + wav_dir + '_audio_restored_stack.npy', self._audio_restored_stack)

    # TODO: Log Mag + InstFreq difference
    def experiment_B(self, wav_dir):
        self._audio_original_stack = np.load('./' + wav_dir + '_audio_original_stack.npy')
        self._specgram_stack = session.run(
            self._SpecgramsHelper.waves_to_specgrams(
                tf.convert_to_tensor(
                    self._audio_original_stack
                )
            )
        )
        self._audio_restored_stack = session.run(
            self._SpecgramsHelper.specgrams_to_waves(
                self._specgram_stack
            )
        )
        np.save('./' + wav_dir + '_audio_restored_stack.npy',  self._audio_restored_stack)


    def get_dist_meanstd(self, dist_list):
        n_sample = self._audio_original_stack.shape[0]
        for idx in range(n_sample):
            original_vector = self._audio_original_stack[idx,:,0]
            restored_vector = self._audio_restored_stack[idx, :, 0]
            dist = self._get_dist(original_vector, restored_vector)
            dist_list.append(dist)
        mean = np.mean(dist_list)
        std = np.std(dist_list)
        print("# of dataset:\t\t\t" + str(n_sample))
        print("The MEAN of Euclidean distance between the Original and Retored vectors.:\t" + str(mean))
        print("The STD of Euclidean distance between the Original and Retored vectors.:\t" + str(std) + "\n\n\n\n\n")