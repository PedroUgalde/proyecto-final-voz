"""Métricas de evaluación para clasificadores de voz."""

from collections import defaultdict

import numpy as np


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat


def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """Precisión, recall y F1 por clase."""
    report = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(t == label for t in y_true),
        }
    return report
