"""
GMM (Gaussian Mixture Model) - enfoque estadístico clásico para ASR.

Entrena un GMM por palabra sobre vectores resumidos de MFCC.
Clasificación por máxima verosimilitud.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMWordClassifier:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.models: dict[str, GaussianMixture] = {}

    def _summarize(self, mfcc: np.ndarray) -> np.ndarray:
        return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])

    def fit(self, features: list[np.ndarray], labels: list[str]) -> "GMMWordClassifier":
        words = sorted(set(labels))
        for word in words:
            X = np.array([
                self._summarize(f)
                for f, lbl in zip(features, labels)
                if lbl == word
            ])
            gmm = GaussianMixture(
                n_components=min(self.n_components, len(X)),
                covariance_type="diag",
                random_state=42,
            )
            gmm.fit(X)
            self.models[word] = gmm
        return self

    def predict(self, mfcc: np.ndarray) -> str:
        x = self._summarize(mfcc).reshape(1, -1)
        best_word, best_score = None, -np.inf
        for word, gmm in self.models.items():
            score = gmm.score(x)
            if score > best_score:
                best_score = score
                best_word = word
        return best_word or "unknown"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GMMWordClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
