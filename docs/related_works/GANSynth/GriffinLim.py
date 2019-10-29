from specgrams_helper import SpecgramsHelper

A = SpecgramsHelper(
    audio_length=64000,
    spec_shape=(256, 512),
    overlap=0.75,
    sample_rate=16000,
    mel_downscale=2
)