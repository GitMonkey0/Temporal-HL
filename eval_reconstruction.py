from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_hl.reconstruction import (
    reconstruct_from_static_and_motion,
    reconstruct_from_static_tokens,
)


def evaluate_split(manifest: list[dict], split: str) -> dict:
    items = [item for item in manifest if item["split"] == split]
    static_err = []
    temporal_err = []
    for item in items:
        data = np.load(item["path"])
        coords = data["coords"]
        joint_valid = data["joint_valid"]
        static_tokens = data["static_tokens"]
        motion_tokens = data["motion_tokens"]
        static_res = reconstruct_from_static_tokens(static_tokens, coords, joint_valid)
        temporal_res = reconstruct_from_static_and_motion(static_tokens, motion_tokens, coords, joint_valid)
        static_err.append(static_res.mpjpe)
        temporal_err.append(temporal_res.mpjpe)
    return {
        "split": split,
        "static_mpjpe": float(np.mean(static_err)),
        "temporal_mpjpe": float(np.mean(temporal_err)),
        "improvement": float(np.mean(static_err) - np.mean(temporal_err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate notation-to-motion reconstruction.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [evaluate_split(manifest, split) for split in ("train", "test")]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
