"""
LPC (Linear Predictive Coding) - modelado fuente-filtro del tracto vocal.

Estima coeficientes que predicen la muestra actual a partir de muestras pasadas.
"""

import numpy as np
from scipy.linalg import solve_toeplitz


def autocorrelation(signal: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelación para el método de Levinson-Durbin."""
    n = len(signal)
    r = np.zeros(order + 1, dtype=np.float64)
    for lag in range(order + 1):
        r[lag] = np.sum(signal[: n - lag] * signal[lag:])
    return r


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """
    Algoritmo de Levinson-Durbin para coeficientes LPC.

    Returns:
        (coeficientes a, error de predicción)
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(i))
        k = -acc / (e + 1e-12)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= 1.0 - k * k

    return a[1:], e


def compute_lpc(
    signal: np.ndarray,
    order: int = 12,
    frame_length: int = 256,
    frame_shift: int = 128,
) -> np.ndarray:
    """
    Calcula coeficientes LPC por trama.

    Returns:
        Array (n_frames, order)
    """
    n_frames = max(1, 1 + (len(signal) - frame_length) // frame_shift)
    lpc_features = np.zeros((n_frames, order), dtype=np.float32)

    for i in range(n_frames):
        start = i * frame_shift
        frame = signal[start : start + frame_length].astype(np.float64)
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame -= frame.mean()
        r = autocorrelation(frame, order)
        if r[0] < 1e-8:
            continue
        r /= r[0]
        try:
            a = solve_toeplitz(r[:-1], -r[1:])
            lpc_features[i] = a
        except np.linalg.LinAlgError:
            a, _ = levinson_durbin(r * r[0], order)
            lpc_features[i] = a

    return lpc_features
