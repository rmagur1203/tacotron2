# coding: utf-8
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile

from hparams import HParams, AudioCompression


def preemphasis(audio, k):
    return signal.lfilter([1, -k], [1], audio)


def deemphasis(audio, k):
    return signal.lfilter([1], [1, -k], audio)


def audio2mel(audio, hparams: HParams):
    # if hparams.audio_compression == AudioCompression.MU:
    #     audio = librosa.mu_compress(audio, mu=255, quantize=False)
    # if hparams.audio_compression == AudioCompression.MU_QUANT:
    #     audio = librosa.mu_compress(audio, mu=255, quantize=True)
    D = librosa.stft(
        audio,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
    )
    M = librosa.feature.melspectrogram(
        S=librosa.util.abs2(D),
        sr=hparams.sample_rate,
        n_mels=hparams.n_mels,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
    )
    # S = librosa.amplitude_to_db(M) - hparams.ref_level_db
    return M


def griffin_lim(mel, hparams: HParams):
    # A = librosa.db_to_amplitude(mel + hparams.ref_level_db)
    S = librosa.feature.inverse.mel_to_stft(
        mel,
        sr=hparams.sample_rate,
        n_fft=hparams.n_fft,
    )
    W = librosa.griffinlim(S)
    return W
