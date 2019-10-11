import os

import librosa
import numpy as np
import png_spec

main_dir = '/home/201850854/compareTF/DataPrep/'
WAV_DIR = main_dir + 'ESC-50-wav'
sample_rate = 22050
fft_size = 512
fft_hop = 256
scale_width = 37
scale_height = 50
transform = 'mel'

if transform == 'stft':
    SPEC_DIR = main_dir + 'stft'
    if (scale_width != None):
        SPEC_DIR = SPEC_DIR + '.' + str(scale_width)
    if (scale_height != None):
        SPEC_DIR = SPEC_DIR + '.' + str(scale_height)
elif transform == 'mel':
    SPEC_DIR = main_dir + 'mel'
elif transform == 'cqt':
    SPEC_DIR = main_dir + 'cqt'
elif transform == 'cwt':
    SPEC_DIR = main_dir + 'cwt'
elif transform == 'mfcc':
    SPEC_DIR = main_dir + 'mfcc'
else:
    raise ValueError("Transform not supported! Please choose from ['stft','mel','cqt','cwt','mfcc']")

def get_sub_dirs(dir):
    return [
        name for name in os.listdir(dir)
        if (os.path.isdir(os.path.join(dir, name))) and not (name.startswith('.'))
    ]

def list_dir(dir, file_form):
    fname_list = [
        os.path.normcase(f) for f in os.listdir(dir)
        if (not(f.startswith('.')))
    ]

    file_list = [
        os.path.join(dir, f) for f in fname_list
        if os.path.splitext(f)[1] in file_form
    ]

    return file_list

def esc50_get_fold(fname):
   return fname.split('-')[0]

def wav_to_mel(fname, srate, fft_size, fft_hop, n_mels=128):
    try:
        audio_data, sample_rate = librosa.load(
            path=fname,
            sr=srate,
            mono=True,
            duration=None
        )
    except:
        print("Can not read " + fname)
        return
    S = librosa.feature.melspectrogram(
        y=audio_data,
        sr=srate,
        n_fft=fft_size,
        hop_length=fft_hop,
        n_mels=n_mels
    )
    D = librosa.power_to_db(
        S=S,
        ref=np.max
    )
    return D

def wav_to_spec(wav_dir, out_dir, srate, fft_size, fft_hop, trans, scale_width=None, scale_height=None):

    sub_dirs = get_sub_dirs(wav_dir)
    count = 0

    for sub_dir in sub_dirs:
        full_paths = list_dir(wav_dir + '/' + sub_dir, '.wav')
        for idx in range(len(full_paths)):
            fname = os.path.basename(full_paths[idx])
            fold = esc50_get_fold(fname)

            if trans == 'stft':
                break
            elif trans == 'mel':
                D = wav_to_mel(
                    fname=full_paths[idx],
                    srate=srate,
                    fft_size=fft_size,
                    fft_hop=fft_hop
                )
            elif trans == 'cqt':
                break
            elif trans == 'cwt':
                break
            elif trans == 'mfcc':
                break

            png_dir = out_dir + '_png/' + fold + '/' + sub_dir
            try:
                os.stat(png_dir)
            except:
                os.makedirs(png_dir)

            png_spec.log_spec_to_png(
                out_img=png_dir+'/'+os.path.splitext(fname)[0] + '.png',
                scale_width=scale_width,
                scale_height=scale_height
            )


wav_to_spec(
    wav_dir=WAV_DIR,
    out_dir=SPEC_DIR,
    srate=sample_rate,
    fft_size=fft_size,
    fft_hop=fft_hop,
    trans=transform,
    scale_width=scale_width,
    scale_height=scale_height
)