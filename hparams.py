from collections import namedtuple

from typing import TypedDict
from enum import Enum


class AudioCompression(Enum):
    RAW = "raw"
    MU = "mu_law"
    MU_QUANT = "mu_law_quantize"


class HParams(dict):
    # Audio
    sample_rate: int
    ref_level_db: int

    # Audio compression
    audio_compression: AudioCompression

    # Mel spectrogram
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int

    emphasis = 0.97

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value
