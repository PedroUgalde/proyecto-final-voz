"""
K-Means Clustering para clasificación de palabras.

1. Extrae vector resumen (media de MFCC por archivo)
2. Agrupa con K-Means (k = número de palabras)
3. Asigna etiqueta a cada cluster por mayoría
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


class KMeansWordClassifier:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_to_word: dict[int, str] = {}

    def _summarize(self, mfcc: np.ndarray) -> np.ndarray:
        """Resume secuencia MFCC con media y desviación estándar."""
        return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])

    def fit(self, features: list[np.ndarray], labels: list[str]) -> "KMeansWordClassifier":
        X = np.array([self._summarize(f) for f in features])
        clusters = self.kmeans.fit_predict(X)

        for c in range(self.n_clusters):
            mask = clusters == c
            if not mask.any():
                continue
            words, counts = np.unique(np.array(labels)[mask], return_counts=True)
            self.cluster_to_word[c] = words[np.argmax(counts)]

        return self

    def predict(self, mfcc: np.ndarray) -> str:
        x = self._summarize(mfcc).reshape(1, -1)
        cluster = int(self.kmeans.predict(x)[0])
        return self.cluster_to_word.get(cluster, "unknown")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "KMeansWordClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
