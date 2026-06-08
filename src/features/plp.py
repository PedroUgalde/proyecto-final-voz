"""
PLP (Perceptual Linear Prediction) simplificado.

Combina escala Bark, curva de equalización auditiva y predicción lineal.
"""

import numpy as np
from scipy.fftpack import dct
from .mfcc import framing, hamming_window, preemphasis


def hz_to_bark(hz: float) -> float:
    """Convierte Hz a escala Bark (psicoacústica)."""
    return 6.0 * np.arcsinh(hz / 600.0)


def bark_filterbank(n_barks: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Banco de filtros en escala Bark."""
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    bark_freqs = np.array([hz_to_bark(f) for f in freqs])

    bark_max = hz_to_bark(sample_rate / 2)
    bark_points = np.linspace(0, bark_max, n_barks + 2)
    fbank = np.zeros((n_barks, n_freqs), dtype=np.float64)

    for m in range(1, n_barks + 1):
        left, center, right = bark_points[m - 1], bark_points[m], bark_points[m + 1]
        for k, b in enumerate(bark_freqs):
            if left <= b <= center and center > left:
                fbank[m - 1, k] = (b - left) / (center - left)
            elif center < b <= right and right > center:
                fbank[m - 1, k] = (right - b) / (right - center)

    return fbank


def compute_plp(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_plp: int = 13,
    n_barks: int = 22,
    n_fft: int = 512,
    order: int = 12,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """
    Calcula coeficientes PLP por trama.

    Returns:
        Array (n_frames, n_plp)
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    signal = preemphasis(signal.astype(np.float64), 0.97)
    frames = framing(signal, frame_length, frame_shift) * hamming_window(frame_length)

    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    fbank = bark_filterbank(n_barks, n_fft, sample_rate)
    bark_energies = np.dot(spectrum, fbank.T)

    # Equalización loudness simplificada
    loudness = bark_energies ** 0.33
    plp_raw = dct(np.log(loudness + 1e-10), type=2, axis=1, norm="ortho")[:, :order]
    return plp_raw[:, :n_plp].astype(np.float32)
