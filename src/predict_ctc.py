"""
Decodificación Greedy para CTC.

En cada frame selecciona el carácter más probable, luego:
1. Elimina repeticiones consecutivas
2. Elimina el token blank
"""

import torch

from .config import CHAR_TO_IDX, IDX_TO_CHAR, TARGET_WORDS
from .dataset import extract_mfcc_file
from .model_ctc import BiLSTMCTC


def greedy_decode(logits: torch.Tensor) -> str:
    """
    Decodifica logits (T, num_classes) con búsqueda voraz.

    CTC collapse: aa<blank>bb → ab
    """
    blank = CHAR_TO_IDX["<blank>"]
    indices = logits.argmax(dim=-1).tolist()

    collapsed = []
    prev = None
    for idx in indices:
        if idx != blank and idx != prev:
            collapsed.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(collapsed)


def predict_greedy(model: BiLSTMCTC, audio_path: str, device=None) -> str:
    """Predice palabra desde archivo de audio."""
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    mfcc = extract_mfcc_file(audio_path)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    x_len = torch.tensor([mfcc.shape[0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, x_len)[0]

    return greedy_decode(logits)


def word_accuracy(predictions: list[str], references: list[str]) -> float:
    """Exactitud a nivel palabra."""
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / max(len(references), 1)


def match_to_vocabulary(text: str, vocabulary: list[str] | None = None) -> str:
    """Mapea texto decodificado a la palabra válida más cercana."""
    vocabulary = vocabulary or TARGET_WORDS
    if text in vocabulary:
        return text
    # Distancia de edición simple
    def edit_dist(a, b):
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, 1):
                temp = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
                prev = temp
        return dp[-1]

    return min(vocabulary, key=lambda w: edit_dist(text, w))
