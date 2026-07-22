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

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim*2)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights.unsqueeze(-1)  # (batch, seq_len, 1)
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
        self.num_items = num_items
        self.padding_idx = padding_idx
        self.output_proj = nn.Linear(hidden_dim * 2, num_items + 1)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Return logits over the item catalogue.

        Parameters
        ----------
        item_seq:
            LongTensor of shape ``(batch_size, seq_len)`` containing item IDs.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch_size, num_items + 1)``. Index zero
            is reserved for padding and is never eligible for recommendation.
        """
        emb = self.dropout(self.embedding(item_seq))  # (B, L, E)
        lstm_out, _ = self.lstm(emb)  # (B, L, H*2)
        mask = item_seq.ne(self.padding_idx)
        context = self.attention(lstm_out, mask)  # (B, H*2)
        logits = self.output_proj(self.dropout(context))  # (B, num_items + 1)
        logits[:, self.padding_idx] = float("-inf")
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
        if item_seq.ndim != 2 or not item_seq.ne(self.padding_idx).any():
            raise ValueError("A recommendation requires at least one known item")
        if not 1 <= top_k <= self.num_items:
            raise ValueError(f"top_k must be between 1 and {self.num_items}")
        excluded = {idx for idx in (exclude_ids or []) if 1 <= idx <= self.num_items}
        if top_k > self.num_items - len(excluded):
            raise ValueError("top_k exceeds the remaining eligible catalogue")
        logits = self.forward(item_seq)
        for idx in excluded:
            logits[:, idx] = float("-inf")
        scores = torch.topk(logits, k=top_k, dim=-1)
        return scores.indices[0].tolist()
