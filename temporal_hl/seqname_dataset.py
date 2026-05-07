from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def load_seqname_map(data_root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split in ("train", "test"):
        with (data_root / split / "annotations.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                mapping[obj["clip_id"]] = obj["source_group"]["seq_name"]
    return mapping


class SeqNameTokenDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, split: str, mode: str) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seqname_map = load_seqname_map(data_root)

        items = [item for item in manifest if item["split"] == split]
        labels = sorted({seqname_map[item["clip_id"]] for item in items})
        overlap_labels = set(labels)
        other_split = "test" if split == "train" else "train"
        other_items = [item for item in manifest if item["split"] == other_split]
        other_labels = {seqname_map[item["clip_id"]] for item in other_items}
        overlap_labels &= other_labels

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(overlap_labels))}
        self.items: List[dict] = []
        self.mode = mode
        for item in items:
            seq_name = seqname_map[item["clip_id"]]
            if seq_name in self.label_to_idx:
                self.items.append(
                    {
                        "path": item["path"],
                        "clip_id": item["clip_id"],
                        "label": self.label_to_idx[seq_name],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        data = np.load(item["path"])
        static_tokens = data["static_tokens"].astype(np.int64)
        motion_tokens = data["motion_tokens"].astype(np.int64)
        keyframe_mask = data["keyframe_mask"].astype(np.int64)

        return {
            "clip_id": item["clip_id"],
            "static_tokens": torch.from_numpy(static_tokens),
            "motion_tokens": torch.from_numpy(motion_tokens),
            "keyframe_mask": torch.from_numpy(keyframe_mask),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }

