from __future__ import annotations

import torch
from torch import nn


class TokenReconstructionModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_motion: bool = True,
        use_keyframe: bool = True,
    ) -> None:
        super().__init__()
        self.use_motion = use_motion
        self.use_keyframe = use_keyframe
        self.static_embed = nn.Embedding(27, d_model)
        if use_motion:
            self.motion_embed = nn.Embedding(28, d_model)
        if use_keyframe:
            self.keyframe_embed = nn.Embedding(2, d_model)
        self.region_pos = nn.Embedding(40, d_model)
        self.frame_proj = nn.Sequential(
            nn.Linear(d_model * 40, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
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
        self.coord_head = nn.Linear(d_model, 42 * 3)

    def forward(
        self,
        static_tokens: torch.Tensor,
        motion_tokens: torch.Tensor | None = None,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        static_tokens = static_tokens.clamp(min=-1) + 1
        static_feat = self.static_embed(static_tokens)
        feat = static_feat
        if self.use_motion and motion_tokens is not None:
            motion_tokens = motion_tokens.clamp(min=-1) + 1
            feat = feat + self.motion_embed(motion_tokens)
        if self.use_keyframe and keyframe_mask is not None:
            feat = feat + self.keyframe_embed(keyframe_mask.long()).unsqueeze(2)
        feat = feat + self.region_pos.weight.unsqueeze(0).unsqueeze(0)
        feat = feat.reshape(feat.size(0), feat.size(1), -1)
        hidden = self.frame_proj(feat)
        hidden = self.encoder(hidden)
        return self.coord_head(hidden)
