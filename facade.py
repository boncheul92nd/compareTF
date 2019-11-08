import os
os.environ['CUDA_DIVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

from libs.time_freq_transform import PreProcessorTF
from configs import configuration as config

def main():

    IF_H = PreProcessorTF(
        srate=16000,
        dur=4,
        t_ax=128,
        f_ax=1024,
        trans='stft',
        ifreq=True,
        out_dir=config.IF_H
    )
    IF_H.wav_to_spectrogram()
    del IF_H

if __name__ == "__main__":
    main()