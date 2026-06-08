"""Utilidades para cargar el dataset Speech Commands."""

import os
import random
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, SAMPLES_PER_CLASS, TARGET_WORDS
from .audio_io import load_wav, normalize_audio
from .features.mfcc import compute_mfcc


def find_dataset_root(base_path: str | Path) -> Path:
    """
    Localiza la carpeta con subdirectorios por palabra.
    kagglehub puede descargar en rutas anidadas.
    """
    base = Path(base_path)
    for root, dirs, _ in os.walk(base):
        root_path = Path(root)
        if any(w in dirs for w in TARGET_WORDS):
            return root_path
    raise FileNotFoundError(
        f"No se encontraron carpetas de palabras en {base_path}. "
        "Descarga el dataset con kagglehub primero."
    )


def collect_samples(
    dataset_root: Path,
    words: list[str] | None = None,
    max_per_class: int | None = SAMPLES_PER_CLASS,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Recopila rutas (archivo, etiqueta) para las palabras objetivo."""
    words = words or TARGET_WORDS
    rng = random.Random(seed)
    samples: list[tuple[str, str]] = []

    for word in words:
        word_dir = dataset_root / word
        if not word_dir.exists():
            # Buscar en subcarpetas (estructura jerárquica del dataset)
            for sub in dataset_root.rglob(word):
                if sub.is_dir() and sub.name == word:
                    word_dir = sub
                    break

        if not word_dir.exists():
            continue

        files = [str(f) for f in word_dir.glob("*.wav")]
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        samples.extend((f, word) for f in files)

    return samples


def word_to_char_indices(word: str) -> list[int]:
    """Convierte palabra a índices de caracteres para CTC."""
    from .config import CHAR_TO_IDX
    return [CHAR_TO_IDX[c] for c in word.lower()]


def extract_mfcc_file(path: str) -> np.ndarray:
    """Carga audio y extrae MFCC."""
    audio, _ = load_wav(path, SAMPLE_RATE)
    audio = normalize_audio(audio)
    return compute_mfcc(audio, sample_rate=SAMPLE_RATE)


def train_val_split(
    samples: list[tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Divide por palabra manteniendo proporción."""
    rng = random.Random(seed)
    by_word: dict[str, list[str]] = {}
    for path, word in samples:
        by_word.setdefault(word, []).append(path)

    train, val = [], []
    for word, paths in by_word.items():
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        for p in paths[n_val:]:
            train.append((p, word))
        for p in paths[:n_val]:
            val.append((p, word))
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
