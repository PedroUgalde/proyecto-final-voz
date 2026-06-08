"""
HMM simplificado (Hidden Markov Model) para reconocimiento de palabras.

Estados = frames temporales discretizados
Observaciones = índices de clusters de MFCC (vector cuantizado)
"""

import pickle
from pathlib import Path

import numpy as np


def quantize_features(mfcc: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Asigna cada frame al centroide más cercano del codebook."""
    indices = []
    for frame in mfcc:
        dists = np.sum((codebook - frame) ** 2, axis=1)
        indices.append(int(np.argmin(dists)))
    return np.array(indices, dtype=int)


class SimpleHMMClassifier:
    """
    HMM por palabra con transiciones left-to-right.
    Emisiones modeladas como histogramas de símbolos cuantizados.
    """

    def __init__(self, n_symbols: int = 64, n_states: int = 5):
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.codebook: np.ndarray | None = None
        self.models: dict[str, dict] = {}

    def _build_codebook(self, all_mfcc: list[np.ndarray]) -> np.ndarray:
        """Codebook K-Means manual para cuantización."""
        frames = np.vstack(all_mfcc)
        n = min(self.n_symbols, len(frames))
        idx = np.random.choice(len(frames), n, replace=False)
        centroids = frames[idx].copy()

        for _ in range(20):
            labels = np.argmin(
                np.sum((frames[:, None, :] - centroids[None, :, :]) ** 2, axis=2),
                axis=1,
            )
            for k in range(n):
                mask = labels == k
                if mask.any():
                    centroids[k] = frames[mask].mean(axis=0)
        return centroids

    def _train_word_hmm(self, sequences: list[np.ndarray]) -> dict:
        """Entrena emisiones por estado (uniforme left-to-right)."""
        obs_seqs = [quantize_features(s, self.codebook) for s in sequences]
        emission = np.ones((self.n_states, self.n_symbols)) * 1e-6

        for obs in obs_seqs:
            state_per_frame = np.linspace(0, self.n_states - 1, len(obs)).astype(int)
            for t, o in enumerate(obs):
                emission[state_per_frame[t], o] += 1

        emission /= emission.sum(axis=1, keepdims=True) + 1e-10
        return {"emission": emission}

    def fit(self, features: list[np.ndarray], labels: list[str]) -> "SimpleHMMClassifier":
        self.codebook = self._build_codebook(features)
        words = sorted(set(labels))
        for word in words:
            seqs = [f for f, lbl in zip(features, labels) if lbl == word]
            self.models[word] = self._train_word_hmm(seqs)
        return self

    def _log_likelihood(self, obs: np.ndarray, model: dict) -> float:
        """Forward algorithm en escala log simplificada."""
        emission = model["emission"]
        n_states = emission.shape[0]
        log_pi = np.log(np.ones(n_states) / n_states)

        log_alpha = log_pi + np.log(emission[:, obs[0]] + 1e-10)
        for t in range(1, len(obs)):
            trans = np.tril(np.ones((n_states, n_states)))  # left-to-right
            trans /= trans.sum(axis=1, keepdims=True)
            log_trans = np.log(trans + 1e-10)
            log_emit = np.log(emission[:, obs[t]] + 1e-10)

            new_alpha = np.zeros(n_states)
            for j in range(n_states):
                new_alpha[j] = log_emit[j] + np.logaddexp.reduce(
                    log_alpha + log_trans[:, j]
                )
            log_alpha = new_alpha

        return float(np.logaddexp.reduce(log_alpha))

    def predict(self, mfcc: np.ndarray) -> str:
        obs = quantize_features(mfcc, self.codebook)
        best_word, best_ll = None, -np.inf
        for word, model in self.models.items():
            ll = self._log_likelihood(obs, model)
            if ll > best_ll:
                best_ll = ll
                best_word = word
        return best_word or "unknown"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SimpleHMMClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
