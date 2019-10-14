import os
import librosa
import numpy as np
import png_spec
import configuration as exprt_config

class PreProcessor(object):

    def __init__(self, srate, fft_size, fft_hop, scale_width, scale_height, trans):
        self._sample_rate = srate
        self._fft_size = fft_size
        self._fft_hop = fft_hop
        self._width = scale_width
        self._height = scale_height
        self._transform = trans
        self._wav_dir = exprt_config.WAV_DIR
        self._out_dir = None

        if self._transform == 'stft':
            self._out_dir = exprt_config.MAIN_DIR + 'stft'
            if (self._width != None):
                self._out_dir = self._out_dir + '.' + str(self._width)
            if (self._height != None):
                self._out_dir = self._out_dir + '.' + str(self._height)
        elif self._transform == 'mel':
            self._out_dir = exprt_config.MAIN_DIR + 'mel'
        elif self._transform == 'cqt':
            self._out_dir = exprt_config.MAIN_DIR + 'cqt'
        elif self._transform == 'cwt':
            self._out_dir = exprt_config.MAIN_DIR + 'cwt'
        elif self._transform == 'mfcc':
            self._out_dir = exprt_config.MAIN_DIR + 'mfcc'
        else:
            raise ValueError("Transform not supported! Please choose from ['stft','mel','cqt','cwt','mfcc']")

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

    def _wav_to_mel(self, fname, n_mels=128):
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

        S = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=self._fft_size,
            hop_length=self._fft_hop,
            n_mels=n_mels
        )

        D = librosa.power_to_db(S=S, ref=np.max)

        return D

    def wav_to_spectrogram(self):

        sub_dirs = self._get_sub_dirs(self._wav_dir)
        count = 0

        for sub_dir in sub_dirs:

            full_paths = self._list_dir(self._wav_dir + '/' + sub_dir, '.wav')
            for idx in range(len(full_paths)):
                fname = os.path.basename(full_paths[idx])
                fold = self._esc50_get_fold(fname=fname)

                if self._transform == 'stft':
                    # TODO: STFT method
                    break
                elif self._transform == 'mel':
                    D = self._wav_to_mel(fname=full_paths[idx])
                elif self._transform == 'cqt':
                    # TODO: CQT method
                    break
                elif self._transform == 'cwt':
                    # TODO: CWT method
                    break
                elif self._transform == 'mfcc':
                    # TODO: MFCC method
                    break

                png_dir = self._out_dir + '_png/' + fold + '/' + sub_dir
                try:
                    os.stat(png_dir)
                except:
                    os.makedirs(png_dir)

                png_spec.logspec_to_png(
                    out_img=D,
                    fname=png_dir + '/' + os.path.splitext(fname)[0] + '.png',
                    scale_width=self._width,
                    scale_height=self._height
                )
                print(str(count) + ": " + sub_dir + "/" + os.path.splitext(fname)[0])
                count += 1
        print("COMPLETE")