from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_split(manifest_path: Path, split: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [x for x in manifest if x["split"] == split]
    items.sort(key=lambda x: x["clip_id"])
    arrays = []
    clip_ids = []
    for item in items:
        data = np.load(item["path"])
        arrays.append(
            {
                "static": data["static_tokens"],
                "motion": data["motion_tokens"],
                "keyframe": data["keyframe_mask"],
            }
        )
        clip_ids.append(item["clip_id"])
    return clip_ids, arrays


def token_match_score(a: np.ndarray, b: np.ndarray) -> float:
    valid = (a >= 0) & (b >= 0)
    if valid.sum() == 0:
        return 0.0
    return float(((a == b) & valid).sum() / valid.sum())


def keyframe_score(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def sequence_similarity(x: dict, y: dict, mode: str) -> float:
    static_score = token_match_score(x["static"], y["static"])
    if mode == "static":
        return static_score
    if mode == "keyframe":
        return 0.7 * static_score + 0.3 * keyframe_score(x["keyframe"], y["keyframe"])
    if mode == "temporal":
        motion_score = token_match_score(x["motion"], y["motion"])
        kf_score = keyframe_score(x["keyframe"], y["keyframe"])
        return 0.45 * static_score + 0.35 * motion_score + 0.20 * kf_score
    raise ValueError(mode)


def retrieval_metrics(arrays, mode: str) -> dict:
    n = len(arrays)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            s = sequence_similarity(arrays[i], arrays[j], mode)
            sim[i, j] = s
            sim[j, i] = s
    np.fill_diagonal(sim, -1.0)
    neighbor = np.max(sim, axis=1)
    return {
        "nn_sim_mean": float(neighbor.mean()),
        "nn_sim_std": float(neighbor.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sequence-aware token retrieval.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    args = parser.parse_args()

    _, arrays = load_split(args.manifest, "test")
    results = {}
    for mode in ("static", "keyframe", "temporal"):
        results[mode] = retrieval_metrics(arrays, mode)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
