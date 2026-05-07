from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalHLTokenDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, hold_out: float = 0.1, train: bool = True) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = [item for item in manifest if item["split"] == split]
        items.sort(key=lambda x: x["clip_id"])
        split_idx = max(1, int(len(items) * (1.0 - hold_out)))
        if split == "train":
            items = items[:split_idx] if train else items[split_idx:]
        self.items: List[dict] = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        data = np.load(item["path"])
        coords = data["coords"].astype(np.float32)
        joint_valid = data["joint_valid"].astype(np.float32)
        coords = self._normalize(coords, joint_valid)
        return {
            "clip_id": item["clip_id"],
            "static_tokens": torch.from_numpy(data["static_tokens"].astype(np.int64)),
            "motion_tokens": torch.from_numpy(data["motion_tokens"].astype(np.int64)),
            "keyframe_mask": torch.from_numpy(data["keyframe_mask"].astype(np.int64)),
            "coords": torch.from_numpy(coords.reshape(coords.shape[0], -1)),
            "joint_valid": torch.from_numpy(joint_valid.astype(np.float32)),
        }

    @staticmethod
    def _normalize(coords: np.ndarray, joint_valid: np.ndarray) -> np.ndarray:
        coords = coords.copy()
        wrist_r = coords[:, 0]
        wrist_l = coords[:, 21]
        valid_r = (joint_valid[:, 0] > 0.5).astype(np.float32)
        valid_l = (joint_valid[:, 21] > 0.5).astype(np.float32)
        ref = wrist_r * valid_r[:, None] + wrist_l * valid_l[:, None]
        denom = np.clip(valid_r + valid_l, 1.0, None)
        ref = ref / denom[:, None]
        coords = coords - ref[:, None, :]

        valid_points = joint_valid[..., None] > 0.5
        norms = np.linalg.norm(np.where(valid_points, coords, 0.0), axis=-1)
        scale = np.maximum(norms.max(axis=1, keepdims=True), 1.0)
        coords = coords / scale.reshape(-1, 1, 1)
        coords = np.where(valid_points, coords, 0.0)
        return coords
