import os
os.environ['CUDA_DIVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

from libs.time_freq_transform import PreProcessorTF
from configs import configuration as config

def main():

    IF = PreProcessorTF(
        srate=16000,
        dur=4,
        t_ax=256,
        f_ax=512,
        trans='stft',
        ifreq=True,
        out_dir=config.IF
    )
    IF.wav_to_spectrogram()
    del IF


if __name__ == "__main__":
    main()