from __future__ import annotations

import torch
from torch import nn


class TokenEmbeddingMixin(nn.Module):
    def __init__(self, d_model: int, use_motion: bool, use_keyframe: bool) -> None:
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

    def embed_tokens(
        self,
        static_tokens: torch.Tensor,
        motion_tokens: torch.Tensor | None = None,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        static_tokens = static_tokens.clamp(min=-1) + 1
        feat = self.static_embed(static_tokens)
        if self.use_motion and motion_tokens is not None:
            motion_tokens = motion_tokens.clamp(min=-1) + 1
            feat = feat + self.motion_embed(motion_tokens)
        if self.use_keyframe and keyframe_mask is not None:
            feat = feat + self.keyframe_embed(keyframe_mask.long()).unsqueeze(2)
        feat = feat + self.region_pos.weight.unsqueeze(0).unsqueeze(0)
        feat = feat.reshape(feat.size(0), feat.size(1), -1)
        return self.frame_proj(feat)


class GRUTokenReconstructionModel(TokenEmbeddingMixin):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_motion: bool = True,
        use_keyframe: bool = True,
    ) -> None:
        super().__init__(d_model=d_model, use_motion=use_motion, use_keyframe=use_keyframe)
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.coord_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 42 * 3),
        )

    def forward(
        self,
        static_tokens: torch.Tensor,
        motion_tokens: torch.Tensor | None = None,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(static_tokens, motion_tokens, keyframe_mask)
        hidden, _ = self.encoder(hidden)
        return self.coord_head(hidden)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.block(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(residual + y)


class TCNTokenReconstructionModel(TokenEmbeddingMixin):
    def __init__(
        self,
        d_model: int = 256,
        depth: int = 6,
        dropout: float = 0.1,
        use_motion: bool = True,
        use_keyframe: bool = True,
    ) -> None:
        super().__init__(d_model=d_model, use_motion=use_motion, use_keyframe=use_keyframe)
        dilations = [1, 2, 4, 8, 1, 2][:depth]
        self.encoder = nn.ModuleList([ConvBlock(d_model, dilation=d, dropout=dropout) for d in dilations])
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 42 * 3),
        )

    def forward(
        self,
        static_tokens: torch.Tensor,
        motion_tokens: torch.Tensor | None = None,
        keyframe_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(static_tokens, motion_tokens, keyframe_mask)
        for block in self.encoder:
            hidden = block(hidden)
        return self.coord_head(hidden)
