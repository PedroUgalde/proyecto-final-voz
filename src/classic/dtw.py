"""
DTW (Dynamic Time Warping) - comparación de patrones temporales.

Encuentra la alineación óptima entre dos secuencias de características
minimizando la distancia acumulada (programación dinámica).
"""

import numpy as np


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana entre dos vectores de características."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Calcula distancia DTW entre dos secuencias (n_frames, n_features).

    Matriz de costo acumulado:
        D[i,j] = d(a_i, b_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    """
    n, m = len(seq_a), len(seq_b)
    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_dist(seq_a[i - 1], seq_b[j - 1])
            dtw_mat[i, j] = cost + min(
                dtw_mat[i - 1, j],
                dtw_mat[i, j - 1],
                dtw_mat[i - 1, j - 1],
            )

    return dtw_mat[n, m]


def dtw_predict(
    features: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> str:
    """
    Clasifica comparando contra plantillas DTW por palabra.

    Args:
        features: MFCC del audio de prueba (T, D)
        templates: {palabra: [lista de secuencias MFCC de entrenamiento]}
    """
    best_word, best_dist = None, np.inf

    for word, seq_list in templates.items():
        for template in seq_list:
            dist = dtw_distance(features, template)
            if dist < best_dist:
                best_dist = dist
                best_word = word

    return best_word or "unknown"
