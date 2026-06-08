"""Carga y preprocesamiento básico de audio."""

import numpy as np
import soundfile as sf


def load_wav(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Carga un archivo WAV y lo convierte a mono flotante."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_linear(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Remuestreo lineal simple (sin librerías de ASR)."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    n_samples = int(duration * target_sr)
    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, n_samples, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normaliza amplitud a [-1, 1]."""
    peak = np.max(np.abs(audio)) + 1e-8
    return audio / peak
