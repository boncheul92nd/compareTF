from libs.time_freq_transform import PreProcessor
from libs.specgrams_helper import SpecgramsHelper

def main():
    narrowband = PreProcessor(
        srate=22050,
        fft_size=512,
        fft_hop=256,
        scale_height=37,
        scale_width=50,
        trans='mel',
        n_mels=128
    )

    wideband = PreProcessor(
        srate=22050,
        fft_size=2048,
        fft_hop=1024,
        scale_height=154,
        scale_width=12,
        trans='mel',
        n_mels=512
    )

    del narrowband
    del wideband

    A = SpecgramsHelper(
        audio_length=64000,
        spec_shape=(128, 1024),
        overlap=0.75,
        sample_rate=16000,
        mel_downscale=2
    )
    del A

if __name__ == "__main__":
    main()