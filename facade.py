import os
os.environ['CUDA_DIVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

from libs.time_freq_transform import PreProcessorTF
from configs import configuration as config

def main():

    PHASE_MEL_H = PreProcessorTF(
        srate=16000,
        dur=4,
        t_ax=128,
        f_ax=1024,
        trans='mel',
        ifreq=False,
        out_dir=config.PHASE_MEL_H
    )
    PHASE_MEL_H.wav_to_spectrogram()
    del PHASE_MEL_H

if __name__ == "__main__":
    main()