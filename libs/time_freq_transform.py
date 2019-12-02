import os
import librosa
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
print(tf.executing_eagerly())

import time
from libs import png_spec
from configs import configuration as config
from libs.specgrams_helper import SpecgramsHelper

class PreProcessor(object):

    def __init__(self, srate, fft_size, fft_hop, scale_height, scale_width, trans, n_mels, out_dir):
        self._sample_rate = srate
        self._fft_size = fft_size
        self._fft_hop = fft_hop
        self._height = scale_height
        self._width = scale_width
        self._transform = trans
        self._mel_filterbank = n_mels
        self._wav_dir = config.WAV_DIR
        self._out_dir = out_dir

    def _time_taken(self, elapsed):
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _get_sub_dirs(self, dir):
        return [
            name for name in os.listdir(dir)
            if (os.path.isdir(os.path.join(dir, name))) and not (name.startswith('.'))
        ]

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

    def _esc50_get_fold(self, fname):
        return fname.split('-')[0]

    def _load_audio(self, fname):
        try:
            audio_data, sample_rate = librosa.load(
                path=fname,
                sr=self._sample_rate,
                mono=True,
                duration=None
            )
        except:
            print("Can not read " + fname)
            return

        return audio_data, sample_rate

    def _wav_to_stft(self, fname):

        audio_data, _ = self._load_audio(fname)

        D = librosa.stft(
            y=audio_data,
            n_fft=self._fft_size,
            hop_length=self._fft_hop,
            win_length=self._fft_size,
            window='hann',
            center=False
        )
        magnitude = np.abs(D)
        phase = np.angle(D)

        return magnitude, phase

    def _diff_tf(self, x, axis=1):
        shape = x.get_shape()
        if axis >= len(shape):
            raise ValueError('Invalid axis index: %d for tensor with only %d axes.'
                             % (axis, len(shape)))
        begin_back = [0 for unused_s in range(len(shape))]
        begin_front = [0 for unused_s in range(len(shape))]
        begin_front[axis] = 1

        size = shape.as_list()
        size[axis] -= 1
        slice_front = tf.slice(x, begin_front, size)
        slice_back = tf.slice(x, begin_back, size)
        d = slice_front - slice_back
        return d

    def _unwrap_tf(self, phase_angle, discont=np.pi, axis=-1):
        dd = self._diff_tf(phase_angle, axis=axis)
        ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
        idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
        ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
        ph_correct = ddmod - dd
        idx = tf.less(tf.abs(dd), discont)
        ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
        ph_cumsum = tf.cumsum(ph_correct, axis=axis)

        shape = phase_angle.get_shape().as_list()
        shape[axis] = 1
        ph_cumsum = tf.concat([tf.zeros(shape, dtype=phase_angle.dtype), ph_cumsum], axis=axis)
        unwrapped = phase_angle + ph_cumsum
        return unwrapped

    def _instantaneous_frequency_tf(self, phase_angle, time_axis=-2):
        phase_unwrapped = self._unwrap_tf(phase_angle, axis=time_axis)
        dphase = self._diff_tf(phase_unwrapped, axis=time_axis)

        # Add an initial phase to dphase
        size = phase_unwrapped.get_shape().as_list()
        size[time_axis] = 1
        begin = [0 for unused_s in size]
        phase_slice = tf.slice(phase_unwrapped, begin, size)
        dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
        return dphase

    def _wav_to_mel(self, fname):

        _, phase = self._wav_to_stft(fname=fname)
        audio_data, _ = self._load_audio(fname)

        mag = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self._sample_rate,
            n_fft=self._fft_size,
            hop_length=self._fft_hop,
            n_mels=self._mel_filterbank
        )
        mag = librosa.power_to_db(S=mag, ref=np.max)

        tensor_phase = tf.convert_to_tensor(phase)
        tensor_instntfreq = self._instantaneous_frequency_tf(tensor_phase)
        instantfreq = tensor_instntfreq.numpy()

        return mag, instantfreq

    def wav_to_spectrogram(self):

        sub_dirs = self._get_sub_dirs(self._wav_dir)
        count = 0

        for sub_dir in sub_dirs:

            full_paths = self._list_dir(self._wav_dir + '/' + sub_dir, '.wav')
            for idx in range(len(full_paths)):
                fname = os.path.basename(full_paths[idx])
                fold = self._esc50_get_fold(fname=fname)

                if self._transform == 'stft':
                    D = self._wav_to_stft(fname=full_paths[idx])

                elif self._transform == 'mel':
                    mag, instantfreq = self._wav_to_mel(fname=full_paths[idx])
                elif self._transform == 'cqt':
                    # TODO: CQT method
                    break
                elif self._transform == 'cwt':
                    # TODO: CWT method
                    break
                elif self._transform == 'mfcc':
                    # TODO: MFCC method
                    break

                # TODO: transform code that it's in below's comment box as class method
                ########################################################################
                png_dir = self._out_dir + fold + '/' + sub_dir
                try:
                    os.stat(png_dir)
                except:
                    os.makedirs(png_dir)

                png_spec.logspec_to_png(
                    out_img=mag,
                    fname=png_dir + '/' + os.path.splitext(fname)[0] + '.png',
                    scale_height=self._height,
                    scale_width=self._width
                )

                # png_spec.logspec_to_png(
                #     out_img=instantfreq,
                #     fname=png_dir + '/' + os.path.splitext(fname)[0] + '_IF.png',
                #     scale_height=self._height,
                #     scale_width=self._width
                # )
                ########################################################################

                print(str(count) + ": " + sub_dir + "/" + os.path.splitext(fname)[0])
                count += 1
        print("COMPLETE")

class PreProcessorTF(PreProcessor):

    def __init__(self, srate, dur, t_ax, f_ax, trans, ifreq, out_dir):
        self._SpecHelper = SpecgramsHelper(
            audio_length=(srate*dur),
            spec_shape=(t_ax, f_ax),
            overlap=config.OVERLAP_RATIO,
            sample_rate=srate,
            mel_downscale=config.MEL_DOWNSCALE,
            ifreq=ifreq,
            discard_dc=True
        )
        self._time_axis = t_ax
        self._freq_axis = f_ax
        self._sample_rate = srate
        self._transform = trans
        self._wav_dir = config.WAV_DIR
        self._out_dir = out_dir

    def _load_audio(self, fname):
        try:
            audio_data, sample_rate = librosa.load(
                path=fname,
                sr=self._sample_rate,
                mono=True,
                duration=4
            )
        except:
            print("Can not read " + fname)
            return
        return audio_data

    def _one_to_three(self, one_dim):
        two_dim = np.expand_dims(one_dim, axis=0)
        three_dim = np.expand_dims(two_dim, axis=-1)
        return three_dim

    def wav_to_spectrogram(self):

        sub_dirs = self._get_sub_dirs(self._wav_dir)
        count = 0

        start_time = time.monotonic()
        for sub_dir in sub_dirs:
            full_paths = self._list_dir(self._wav_dir + '/' + sub_dir, '.wav')

            for idx in range(len(full_paths)):
                fname = os.path.basename(full_paths[idx])
                fold = self._esc50_get_fold(fname=fname)
                specgram = None

                if self._transform == 'stft':
                    wav = self._one_to_three(
                        self._load_audio(full_paths[idx])
                    )
                    specgram = self._SpecHelper.waves_to_specgrams(
                        tf.convert_to_tensor(wav)
                    ).numpy()
                    del wav

                elif self._transform == 'mel':
                    wav = self._one_to_three(
                        self._load_audio(full_paths[idx])
                    )
                    specgram = self._SpecHelper.waves_to_melspecgrams(
                        tf.convert_to_tensor(wav)
                    ).numpy()
                    del wav

                png_dir = self._out_dir + fold + '/' + sub_dir

                try:
                    os.stat(png_dir)
                except:
                    os.makedirs(png_dir)
                mag = specgram[0, :, :, 0]
                phase = specgram[0, :, :, 1]
                png_spec.logspec_to_png(
                    out_img=mag,
                    fname=png_dir + '/' + os.path.splitext(fname)[0] + '.png',
                    scale_height=(mag.shape[1])//4,
                    scale_width=(mag.shape[0])//4
                )
                png_spec.logspec_to_png(
                    out_img=phase,
                    fname=png_dir + '/' + os.path.splitext(fname)[0] + '_IF.png',
                    scale_height=(phase.shape[1])//4,
                    scale_width=(phase.shape[0])//4
                )

                print(str(count) + ": " + sub_dir + "/" + os.path.splitext(fname)[0])
                count += 1
                del specgram
                del mag
                del phase

        print("COMPLETE")
        elapsed_time = time.monotonic() - start_time
        print("Total time taken:", self._time_taken(elapsed_time))