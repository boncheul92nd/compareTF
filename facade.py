from libs.time_freq_transform import PreProcessor

def main():
    narrowband = PreProcessor(
        srate=22050,
        fft_size=512,
        fft_hop=32,
        scale_height=37,
        scale_width=50,
        trans='mel',
        n_mels=128
    )
    narrowband.wav_to_spectrogram()

if __name__ == "__main__":
    main()