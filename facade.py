from libs.time_freq_transform import PreProcessor

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
    narrowband.wav_to_spectrogram()

    wideband = PreProcessor(
        srate=22050,
        fft_size=2048,
        fft_hop=1024,
        scale_height=154,
        scale_width=12,
        trans='mel',
        n_mels=512
    )
    wideband.wav_to_spectrogram()

    del narrowband
    del wideband

if __name__ == "__main__":
    main()