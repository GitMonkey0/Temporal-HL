from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from temporal_hl.notation import extract_temporal_hl


def load_annotation_stream(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def convert_clip(record: dict) -> dict:
    frames = record["frames"]
    coords = np.asarray([frame["joint_world_coord"] for frame in frames], dtype=np.float32)
    valid = np.asarray([frame["joint_valid_3d"] for frame in frames], dtype=np.float32).squeeze(-1) > 0.5
    hand_type = [frame["hand_type"] for frame in frames]

    labels = extract_temporal_hl(coords=coords, joint_valid=valid, hand_type=hand_type)
    return {
        "clip_id": record["clip_id"],
        "split": record["split"],
        "video": record["video"],
        "fps": record["fps"],
        "num_frames": record["num_frames"],
        "coords": labels.coords,
        "joint_valid": labels.joint_valid,
        "static_tokens": labels.static_tokens,
        "motion_tokens": labels.motion_tokens,
        "keyframe_mask": labels.keyframe_mask,
    }


def write_processed_record(output_dir: Path, record: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{record['clip_id']}.npz",
        coords=record["coords"],
        joint_valid=record["joint_valid"],
        static_tokens=record["static_tokens"],
        motion_tokens=record["motion_tokens"],
        keyframe_mask=record["keyframe_mask"],
        fps=np.asarray(record["fps"], dtype=np.int64),
        num_frames=np.asarray(record["num_frames"], dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Temporal-HL labels from raw annotations.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke"))
    args = parser.parse_args()

    manifest = []
    for split in ("train", "test"):
        annotation_path = args.data_root / split / "annotations.jsonl"
        split_output = args.output_root / split
        for record in load_annotation_stream(annotation_path):
            processed = convert_clip(record)
            write_processed_record(split_output, processed)
            manifest.append(
                {
                    "clip_id": processed["clip_id"],
                    "split": split,
                    "video": processed["video"],
                    "path": str((split_output / f"{processed['clip_id']}.npz").as_posix()),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
