from libs.time_freq_transform import PreProcessor

def main():
    tmp = PreProcessor(
        srate=22050,
        fft_size=2048,
        fft_hop=1024,
        scale_width=37,
        scale_height=50,
        trans='mel'
    )
    tmp.wav_to_spectrogram()

if __name__ == "__main__":
    main()