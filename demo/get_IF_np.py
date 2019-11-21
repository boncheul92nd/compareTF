import numpy as np
import librosa
import tensorflow as tf

def diff_np(x, axis=1):
    shape = x.shape
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.'
                         % (axis, len(shape)))
    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1

    size = [shape[0], shape[1]]
    size[axis] -= 1
    slice_front = tf.slice(x, begin_front, size)
    slice_back = tf.slice(x, begin_back, size)
    d = slice_front - slice_back
    return d


def unwrap_np(phase_angle, discont=np.pi, axis=-1):
    dd = diff_np(phase_angle, axis=axis)
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


def instantaneous_frequency_np(phase_angle, time_axis=-2):
    phase_unwrapped = unwrap_np(phase_angle, axis=time_axis)
    dphase = diff_np(phase_unwrapped, axis=time_axis)

    # Add an initial phase to dphase
    size = phase_unwrapped.get_shape().as_list()
    size[time_axis] = 1
    begin = [0 for unused_s in size]
    phase_slice = tf.slice(phase_unwrapped, begin, size)
    dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase

if __name__ == "__main__":
    fname = 'Hen.wav'
    n_mels = 128
    fft_size = 512
    hop_size = fft_size//2

    audio_data, sample_rate = librosa.load(
        path='./' + fname,
        sr=22050,
        mono=True,
        duration=None
    )

    D = librosa.stft(
        y=audio_data,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window='hann',
        center=False
    )
    mag = np.abs(D)
    phase = np.angle(D)

    unwrapped = unwrap_np(phase, axis=-1)
    # tensor_instntfreq = instantaneous_frequency_np(tensor_phase)
    # unwrapped = (tf.Session().run(tensor_unwrapped))
    # instantfreq = (tf.Session().run(tensor_instntfreq))