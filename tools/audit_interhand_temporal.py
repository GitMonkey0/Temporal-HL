#!/usr/bin/env python3
"""Audit InterHand annotations for temporal HL experiments.

This script inspects the compact COCO-style metadata and the frame-wise 3D/MANO
annotations shipped with InterHand. It is intended to answer a narrow question:
does the local dataset support sequence-native, multi-view, temporal symbolic
experiments without additional data collection?
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def summarize_sequence_lengths(values: list[int]) -> str:
    if not values:
        return "n/a"
    values_sorted = sorted(values)
    return (
        f"min={values_sorted[0]}, "
        f"median={statistics.median(values_sorted):.0f}, "
        f"max={values_sorted[-1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot-root",
        type=Path,
        default=Path("/opt/tiger/InterHand/annotations/machine_annot"),
        help="Root directory containing InterHand annotation JSON files.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="InterHand split to inspect.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="Number of sequence samples to print.",
    )
    args = parser.parse_args()

    data_path = args.annot_root / f"InterHand2.6M_{args.split}_data.json"
    joint_path = args.annot_root / f"InterHand2.6M_{args.split}_joint_3d.json"
    mano_path = args.annot_root / f"InterHand2.6M_{args.split}_MANO_NeuralAnnot.json"

    data = load_json(data_path)
    joint_3d = load_json(joint_path)
    mano = load_json(mano_path)

    seq_frames: dict[tuple[int, str], set[int]] = defaultdict(set)
    seq_cameras: dict[tuple[int, str], set[str]] = defaultdict(set)
    seq_subjects: dict[tuple[int, str], set[int]] = defaultdict(set)
    hand_types = Counter()
    image_ids_per_seq: dict[tuple[int, str], int] = Counter()

    images = data["images"]
    annotations = data["annotations"]
    for img, ann in zip(images, annotations):
        seq_key = (img["capture"], img["seq_name"])
        seq_frames[seq_key].add(img["frame_idx"])
        seq_cameras[seq_key].add(img["camera"])
        seq_subjects[seq_key].add(img["subject"])
        image_ids_per_seq[seq_key] += 1
        hand_types[ann["hand_type"]] += 1

    frame_counts = [len(v) for v in seq_frames.values()]
    camera_counts = [len(v) for v in seq_cameras.values()]
    frame_deltas = Counter()
    for frame_set in seq_frames.values():
        frames = sorted(frame_set)
        for a, b in zip(frames, frames[1:]):
            frame_deltas[b - a] += 1

    capture_ids = sorted(joint_3d.keys(), key=int)
    joint_hand_types = Counter()
    joint_validity = Counter()
    capture_frame_counts = {}
    for capture_id in capture_ids:
        capture_frames = joint_3d[capture_id]
        capture_frame_counts[capture_id] = len(capture_frames)
        for item in capture_frames.values():
            joint_hand_types[item["hand_type"]] += 1
            joint_validity[item["hand_type_valid"]] += 1

    mano_right = 0
    mano_left = 0
    mano_both = 0
    for capture_frames in mano.values():
        for item in capture_frames.values():
            has_right = item["right"] is not None
            has_left = item["left"] is not None
            mano_right += int(has_right)
            mano_left += int(has_left)
            mano_both += int(has_right and has_left)

    print(f"split: {args.split}")
    print(f"data_path: {data_path}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")
    print(f"sequences: {len(seq_frames)}")
    print(f"frames_per_sequence: {summarize_sequence_lengths(frame_counts)}")
    print(f"cameras_per_sequence: {summarize_sequence_lengths(camera_counts)}")
    print(f"annotation_hand_types: {dict(hand_types)}")
    print(f"frame_delta_histogram_top5: {frame_deltas.most_common(5)}")
    print(f"joint_3d_captures: {capture_frame_counts}")
    print(f"joint_3d_hand_types: {dict(joint_hand_types)}")
    print(f"joint_3d_hand_type_valid: {dict(joint_validity)}")
    print(
        "mano_availability: "
        f"right={mano_right}, left={mano_left}, both={mano_both}"
    )

    print("\nsequence_samples:")
    for seq_key in sorted(seq_frames.keys())[: args.show_samples]:
        frames = sorted(seq_frames[seq_key])
        subjects = sorted(seq_subjects[seq_key])
        print(
            f"  seq={seq_key}, subjects={subjects}, "
            f"n_frames={len(frames)}, frame_range=({frames[0]}, {frames[-1]}), "
            f"n_cameras={len(seq_cameras[seq_key])}, "
            f"n_images={image_ids_per_seq[seq_key]}"
        )


if __name__ == "__main__":
    main()
