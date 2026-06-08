"""
Preprocesamiento para entrenamiento CTC.

Extrae MFCC de todos los archivos y prepara metadatos.
"""

from pathlib import Path

import numpy as np

from .config import CACHE_DIR, SAMPLE_RATE
from .dataset import collect_samples, extract_mfcc_file, find_dataset_root, train_val_split
from .features.mfcc import compute_mfcc


def preprocess_dataset(
    dataset_path: str | Path,
    cache_name: str = "mfcc_cache.npz",
    max_per_class: int = 300,
) -> dict:
    """
    Procesa dataset completo y guarda cache de MFCC.

    Returns:
        dict con train/val paths, labels y shapes
    """
    root = find_dataset_root(dataset_path)
    samples = collect_samples(root, max_per_class=max_per_class)
    train_samples, val_samples = train_val_split(samples)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name

    data = {
        "train_paths": [s[0] for s in train_samples],
        "train_labels": [s[1] for s in train_samples],
        "val_paths": [s[0] for s in val_samples],
        "val_labels": [s[1] for s in val_samples],
        "dataset_root": str(root),
    }
    np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in data.items()})
    return data


def load_mfcc_batch(paths: list[str]) -> list[np.ndarray]:
    """Carga MFCC para una lista de archivos."""
    return [extract_mfcc_file(p) for p in paths]
