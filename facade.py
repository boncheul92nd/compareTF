import os
os.environ['CUDA_DIVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

from libs.time_freq_transform import PreProcessor
from libs.time_freq_transform import PreProcessorTF
from configs import configuration as config

def main():

    NARROWBAND = PreProcessor(
        srate=22050,
        fft_size=512,
        fft_hop=256,
        scale_height=37,
        scale_width=50,
        trans='mel',
        n_mels=128,
        out_dir=config.NARROWBAND
    )
    NARROWBAND.wav_to_spectrogram()
    del NARROWBAND

if __name__ == "__main__":
    main()