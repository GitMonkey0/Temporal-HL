from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_item(manifest_path: Path, clip_id: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(x for x in manifest if x["clip_id"] == clip_id)
    data = np.load(item["path"])
    return item, data


def plot_token_strips(data, save_path: Path) -> None:
    static_tokens = data["static_tokens"]
    motion_tokens = data["motion_tokens"]
    keyframe_mask = data["keyframe_mask"][None, :]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(np.where(static_tokens.T < 0, np.nan, static_tokens.T), aspect="auto", interpolation="nearest")
    axes[0].set_title("Static Tokens (40 regions x T)")
    axes[0].set_ylabel("Region")

    axes[1].imshow(np.where(motion_tokens.T < 0, np.nan, motion_tokens.T), aspect="auto", interpolation="nearest")
    axes[1].set_title("Motion Tokens (40 regions x T)")
    axes[1].set_ylabel("Region")

    axes[2].imshow(keyframe_mask, aspect="auto", interpolation="nearest", cmap="gray_r")
    axes[2].set_title("Keyframe Mask")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Frame")

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_motion_energy(data, save_path: Path) -> None:
    coords = data["coords"]
    valid = data["joint_valid"] > 0.5
    delta = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    frame_energy = np.zeros(coords.shape[0], dtype=np.float32)
    frame_energy[1:] = (delta * pair_valid).sum(axis=1) / np.maximum(pair_valid.sum(axis=1), 1)
    keyframes = data["keyframe_mask"].astype(bool)

    plt.figure(figsize=(12, 3))
    plt.plot(frame_energy, label="motion energy", linewidth=2)
    plt.scatter(np.where(keyframes)[0], frame_energy[keyframes], c="red", s=30, label="keyframes")
    plt.xlabel("Frame")
    plt.ylabel("Energy")
    plt.title("Motion Energy and Detected Keyframes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from Temporal-HL data.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--clip-id", type=str, default="test_000000")
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, data = load_item(args.manifest, args.clip_id)
    plot_token_strips(data, args.out_dir / f"{args.clip_id}_token_strips.png")
    plot_motion_energy(data, args.out_dir / f"{args.clip_id}_motion_energy.png")


if __name__ == "__main__":
    main()
