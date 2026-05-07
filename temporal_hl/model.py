from __future__ import annotations

import torch
from torch import nn


class TemporalHLBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.static_head = nn.Linear(d_model, 40 * 26)
        self.motion_head = nn.Linear(d_model, 40 * 27)
        self.keyframe_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> dict:
        hidden = self.input_proj(features)
        hidden = self.encoder(hidden)
        static_logits = self.static_head(hidden).view(hidden.size(0), hidden.size(1), 40, 26)
        motion_logits = self.motion_head(hidden).view(hidden.size(0), hidden.size(1), 40, 27)
        keyframe_logits = self.keyframe_head(hidden).squeeze(-1)
        return {
            "static_logits": static_logits,
            "motion_logits": motion_logits,
            "keyframe_logits": keyframe_logits,
        }


class StaticHLBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.static_head = nn.Linear(d_model, 40 * 26)

    def forward(self, features: torch.Tensor) -> dict:
        hidden = self.input_proj(features)
        hidden = self.encoder(hidden)
        static_logits = self.static_head(hidden).view(hidden.size(0), hidden.size(1), 40, 26)
        return {
            "static_logits": static_logits,
        }

