import librosa
import librosa.display
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean
from specgrams_helper import SpecgramsHelper

class PreProcessor(object):

    def __init__(self, fname):
        self._fname = fname

        self._audio_original, self._sample_rate = librosa.load(
            path='./' + self._fname,
            sr=16000,
            mono=True,
            duration=None
        )

        self._specgram = None
        self._stft = None
        self._audio_restored = None

        self._SpecgramsHelper = SpecgramsHelper(
            audio_length=64000,
            spec_shape=(256, 512),
            overlap=0.75,
            sample_rate=16000,
            mel_downscale=2
        )

    def _one_to_three(self, one_dim):
        two_dim = np.expand_dims(one_dim, axis=0)
        three_dim = np.expand_dims(two_dim, axis=-1)
        return three_dim

    def _calc(self):
        self._specgram = self._SpecgramsHelper.waves_to_specgrams(self._one_to_three(self._audio_original))
        self._stft = self._SpecgramsHelper.specgrams_to_stfts(self._specgram)
        self._audio_restored = tf.Session().run(self._SpecgramsHelper.stfts_to_waves(self._stft))


    def get_dist(self):
        self._calc()
        dist = euclidean(self._audio_original, self._audio_restored)
        return dist