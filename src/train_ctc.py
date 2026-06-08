"""Entrenamiento del modelo Bi-LSTM + CTC."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    CHAR_TO_IDX,
    CTC_MODEL_PATH,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    N_MFCC,
    NUM_LSTM_LAYERS,
)
from .dataset import extract_mfcc_file, word_to_char_indices
from .model_ctc import BiLSTMCTC


class CTCAudioDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[str]):
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        mfcc = extract_mfcc_file(self.paths[idx])
        chars = word_to_char_indices(self.labels[idx])
        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(chars, dtype=torch.long),
            self.labels[idx],
        )


def collate_ctc(batch):
    """Padding variable para secuencias MFCC y etiquetas."""
    features, labels, words = zip(*batch)
    feat_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    max_t = max(feat_lengths)
    max_l = max(label_lengths)
    n_feat = features[0].size(1)

    padded_x = torch.zeros(len(batch), max_t, n_feat)
    padded_y = torch.zeros(len(batch), max_l, dtype=torch.long)

    for i, (f, l) in enumerate(zip(features, labels)):
        padded_x[i, : f.size(0)] = f
        padded_y[i, : len(l)] = l

    return padded_x, padded_y, feat_lengths, label_lengths, words


def train_ctc_model(
    train_paths: list[str],
    train_labels: list[str],
    val_paths: list[str] | None = None,
    val_labels: list[str] | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_path: Path = CTC_MODEL_PATH,
) -> BiLSTMCTC:
    """Entrena modelo CTC y guarda pesos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CHAR_TO_IDX)

    model = BiLSTMCTC(
        input_size=N_MFCC,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CTCLoss(blank=CHAR_TO_IDX["<blank>"], zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = CTCAudioDataset(train_paths, train_labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ctc
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, x_len, y_len, _ in loader:
            x, y = x.to(device), y.to(device)
            x_len, y_len = x_len.to(device), y_len.to(device)

            optimizer.zero_grad()
            logits = model(x, x_len)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)

            loss = criterion(log_probs, y, x_len, y_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if val_paths and val_labels:
            val_loss = evaluate_ctc_loss(model, val_paths, val_labels, criterion, device)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_size": N_MFCC,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LSTM_LAYERS,
                "num_classes": num_classes,
            },
            "history": history,
        },
        save_path,
    )
    return model


def evaluate_ctc_loss(model, paths, labels, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for path, word in zip(paths, labels):
            mfcc = extract_mfcc_file(path)
            x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
            x_len = torch.tensor([mfcc.shape[0]], dtype=torch.long)
            y = torch.tensor([word_to_char_indices(word)], dtype=torch.long).to(device)
            y_len = torch.tensor([len(word)], dtype=torch.long)

            logits = model(x, x_len)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, y.squeeze(0), x_len, y_len)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def load_ctc_model(path: Path = CTC_MODEL_PATH) -> tuple[BiLSTMCTC, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = BiLSTMCTC(**cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
