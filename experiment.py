import os
import librosa
import librosa.display
import numpy as np
from libs.time_freq_transform import PreProcessor
from scipy.spatial.distance import euclidean

def median_absolute_deviation(data):
    return np.median(np.absolute(data - np.median(data, axis=0)), axis=0)

def main():
    dist_list = []
    wav_path = 'res/X-PROJECT/'
    high_definition = PreProcessor(
        srate=16000,
        fft_size=2048,
        fft_hop=2048,
        scale_height=37,
        scale_width=50,
        trans='mel',
        n_mels=128
    )

    full_path = high_definition._list_dir(wav_path, '.wav')
    for idx in range(len(full_path)):
        fname = os.path.basename(full_path[idx])
        original_audio_data, original_sample_rate = high_definition._load_audio(fname=wav_path + fname)

        D = librosa.stft(
            y=original_audio_data,
            n_fft=high_definition._fft_size,
            hop_length=high_definition._fft_hop,
            win_length=high_definition._fft_size//2,
            window='hann',
            center=True
        )
        mag = np.abs(D)
        mag_dB_sqrt = librosa.power_to_db(mag**2, ref=np.max)
        restored_audio_data = librosa.istft(
            stft_matrix=D,
            hop_length=high_definition._fft_hop,
            win_length=high_definition._fft_size//2,
            window='hann',
            dtype=original_audio_data.dtype,
            length=original_sample_rate
        )

        dist = euclidean(original_audio_data, restored_audio_data)
        dist_list.append(dist)
        print(str(idx+1) + ': ' + fname )
    median = np.median(dist_list)
    MAD = median_absolute_deviation(dist_list)
    print(median)
    print(MAD)
    print("Median & Median Absolute Deviation(MAD):\t%f±%f" % (median, MAD))

if __name__ == "__main__":
    main()