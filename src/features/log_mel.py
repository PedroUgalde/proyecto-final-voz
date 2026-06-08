"""Filterbanks log-Mel (MFCC sin la etapa DCT)."""

import numpy as np
from .mfcc import preemphasis, framing, hamming_window, mel_filterbank


def compute_log_mel(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 40,
    n_fft: int = 512,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    preemph_coef: float = 0.97,
) -> np.ndarray:
    """Retorna espectrograma log-Mel (n_frames, n_mels)."""
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    signal = preemphasis(signal.astype(np.float64), preemph_coef)
    frames = framing(signal, frame_length, frame_shift) * hamming_window(frame_length)

    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft

    fbank = mel_filterbank(n_mels, n_fft, sample_rate)
    mel_energies = np.dot(power, fbank.T)
    return np.log(mel_energies + 1e-10).astype(np.float32)
