"""
Extracción de MFCC implementada paso a paso.

Pipeline: preénfasis → ventaneo → FFT → banco Mel → log → DCT
"""

import numpy as np
from scipy.fftpack import dct


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convierte Hz a escala Mel (percepción auditiva)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convierte escala Mel a Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Construye banco de filtros triangulares en escala Mel.
    Retorna matriz (n_mels, n_fft//2 + 1).
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    n_freqs = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_freqs), dtype=np.float64)

    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left or right == center:
            continue
        for k in range(left, center):
            if 0 <= k < n_freqs:
                fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < n_freqs:
                fbank[m - 1, k] = (right - k) / (right - center)

    return fbank


def preemphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Realza altas frecuencias: y[n] = x[n] - coef * x[n-1]."""
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def framing(
    signal: np.ndarray,
    frame_length: int,
    frame_shift: int,
) -> np.ndarray:
    """Divide la señal en tramas con solapamiento."""
    n_frames = 1 + (len(signal) - frame_length) // frame_shift
    if n_frames <= 0:
        pad = frame_length - len(signal)
        signal = np.pad(signal, (0, pad))
        n_frames = 1

    frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = signal[start : start + frame_length]
    return frames


def hamming_window(length: int) -> np.ndarray:
    """Ventana de Hamming: w(n) = 0.54 - 0.46*cos(2πn/(N-1))."""
    n = np.arange(length)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))


def compute_mfcc(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    preemph_coef: float = 0.97,
) -> np.ndarray:
    """
    Calcula MFCC de una señal de audio.

    Returns:
        Array (n_frames, n_mfcc)
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    signal = preemphasis(signal.astype(np.float64), preemph_coef)
    frames = framing(signal, frame_length, frame_shift)
    window = hamming_window(frame_length)
    frames *= window

    # FFT de potencia
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft

    # Banco Mel + log
    fbank = mel_filterbank(n_mels, n_fft, sample_rate)
    mel_energies = np.dot(power, fbank.T)
    log_mel = np.log(mel_energies + 1e-10)

    # DCT tipo-II (coeficientes cepstrales)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc.astype(np.float32)
