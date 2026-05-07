from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from temporal_hl.token_dataset import TemporalHLTokenDataset
from temporal_hl.token_reconstruction_model import TokenReconstructionModel


def load_manifest_item(manifest_path: Path, clip_id: str) -> tuple[int, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for idx, item in enumerate([x for x in manifest if x["split"] == "test"]):
        if item["clip_id"] == clip_id:
            return idx, item
    raise KeyError(clip_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstruction trajectories.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--clip-id", type=str, default="test_000000")
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = TemporalHLTokenDataset(args.manifest, split="test", train=False)
    idx, _ = load_manifest_item(args.manifest, args.clip_id)
    sample = dataset[idx]

    static_model = TokenReconstructionModel(use_motion=False, use_keyframe=False)
    static_ckpt = torch.load("temporal_hl_cache/runs/token_recon_static/best_static.pt", map_location="cpu")
    static_model.load_state_dict(static_ckpt["model_state"])

    temporal_model = TokenReconstructionModel(use_motion=True, use_keyframe=True)
    temporal_ckpt = torch.load("temporal_hl_cache/runs/token_recon_temporal_kf/best_temporal.pt", map_location="cpu")
    temporal_model.load_state_dict(temporal_ckpt["model_state"])

    with torch.no_grad():
        static_pred = static_model(sample["static_tokens"].unsqueeze(0)).squeeze(0).numpy().reshape(-1, 42, 3)
        temporal_pred = temporal_model(
            sample["static_tokens"].unsqueeze(0),
            sample["motion_tokens"].unsqueeze(0),
            sample["keyframe_mask"].unsqueeze(0),
        ).squeeze(0).numpy().reshape(-1, 42, 3)

    gt = sample["coords"].numpy().reshape(-1, 42, 3)
    wrist_idx = 8

    plt.figure(figsize=(10, 4))
    plt.plot(gt[:, wrist_idx, 0], label="GT", linewidth=2)
    plt.plot(static_pred[:, wrist_idx, 0], label="Static-HL recon", linestyle="--")
    plt.plot(temporal_pred[:, wrist_idx, 0], label="Temporal-HL recon", linestyle="-.")
    plt.xlabel("Frame")
    plt.ylabel("Normalized X")
    plt.title(f"Trajectory Comparison on Joint {wrist_idx} ({args.clip_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / f"{args.clip_id}_reconstruction_compare.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
