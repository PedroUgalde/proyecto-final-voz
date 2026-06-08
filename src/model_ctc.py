"""
Modelo acústico Bi-LSTM con CTC (Connectionist Temporal Classification).

Arquitectura:
    MFCC (T, 13) → Bi-LSTM → Linear → Logits (T, n_chars+1)
"""

import torch
import torch.nn as nn


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 28,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_time, input_size)
            lengths: longitudes reales de cada secuencia
        Returns:
            logits: (batch, max_time, num_classes)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)
