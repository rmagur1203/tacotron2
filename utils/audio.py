import librosa
import numpy as np


def griffin_lim(mag, n_fft, hop_length, win_length, num_iters):
    librosa.gri
    """Reconstruct an audio signal from a spectrogram.
    """
    # mag = np.clip(mag, 0, np.inf)
    phase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_spec = mag * phase
    for i in range(num_iters):
        wav = librosa.istft(complex_spec, hop_length=hop_length, win_length=win_length)
        if i != num_iters - 1:
            complex_spec = librosa.stft(
                wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )
            _, phase = librosa.magphase(complex_spec)
            phase = np.exp(1j * np.angle(phase))
            complex_spec = mag * phase
    return wav
