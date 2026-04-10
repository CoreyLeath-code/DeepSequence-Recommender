"""Deep sequence recommender model.

Architecture
------------
- Item embedding layer
- Bidirectional LSTM encoder
- Dot-product attention over encoder outputs
- Linear projection to item vocabulary (logits)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Scaled dot-product self-attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim*2)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_dim*2)
        return context


class DeepSequenceModel(nn.Module):
    """Bidirectional LSTM + attention sequence recommender."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 2, num_items)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Return logits over the item catalogue.

        Parameters
        ----------
        item_seq:
            LongTensor of shape ``(batch_size, seq_len)`` containing item IDs.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch_size, num_items)``.
        """
        emb = self.dropout(self.embedding(item_seq))  # (B, L, E)
        lstm_out, _ = self.lstm(emb)  # (B, L, H*2)
        context = self.attention(lstm_out)  # (B, H*2)
        logits = self.output_proj(self.dropout(context))  # (B, num_items)
        return logits

    @torch.no_grad()
    def recommend(
        self,
        item_seq: torch.Tensor,
        top_k: int = 10,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Return top-k recommended item IDs for a single sequence."""
        self.eval()
        logits = self.forward(item_seq)
        if exclude_ids:
            for idx in exclude_ids:
                logits[:, idx] = float("-inf")
        scores = torch.topk(logits, k=top_k, dim=-1)
        return scores.indices[0].tolist()
