#!/usr/bin/env python3
"""Evaluate old HL vs temporal HL on a sequence-level retrieval task.

Protocol:
- build one symbolic prototype per sequence from a gallery split
- query each overlapping sequence from a second split
- compare state-only features (old HL) against temporal features
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def l2_normalize(features: dict[str, float]) -> dict[str, float]:
    denom = math.sqrt(sum(value * value for value in features.values()))
    if denom <= 0:
        return features
    return {key: value / denom for key, value in features.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


def iter_hand_records(frame: dict[str, object]) -> Iterable[dict[str, object]]:
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is not None:
            yield hand_record


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def scalar_feature(value: float) -> dict[str, float]:
    return {"value": value}


def build_features(sequence: dict[str, object], mode: str) -> dict[str, float]:
    return build_weighted_features(
        sequence,
        state_weight=1.0,
        transition_weight=1.0,
        hand_motion_weight=1.0,
        interaction_weight=1.0,
        tempo_weight=1.0,
        include_temporal=(mode == "temporal"),
    )


def extract_channel_features(
    sequence: dict[str, object], include_temporal: bool
) -> dict[str, dict[str, float]]:
    state_hist: Counter[str] = Counter()
    transition_hist: Counter[str] = Counter()
    hand_motion_hist: Counter[str] = Counter()
    interaction_hist: Counter[str] = Counter()
    frame_indices = []
    non_stay_transitions = 0
    major_transitions = 0
    total_transition_labels = 0
    active_hand_motion = 0
    total_hand_motion = 0
    interaction_changes = 0
    total_interaction = 0

    for frame in sequence["frames"]:
        frame_indices.append(frame["frame_idx"])
        interaction = frame.get("interaction_motion")
        if interaction not in (None, "unknown"):
            interaction_hist[f"interaction::{interaction}"] += 1
            total_interaction += 1
            if interaction != "steady":
                interaction_changes += 1
        for hand_record in iter_hand_records(frame):
            for token in hand_record["token_labels"]:
                state_hist[f"state::{token}"] += 1
            if include_temporal:
                for token in hand_record["transition_labels"]:
                    if token == "start":
                        continue
                    transition_hist[f"transition::{token}"] += 1
                    total_transition_labels += 1
                    if token != "stay":
                        non_stay_transitions += 1
                    if token == "major_shift":
                        major_transitions += 1
                motion = hand_record.get("hand_motion")
                if motion not in (None, "start", "unknown"):
                    hand_motion_hist[f"hand_motion::{motion}"] += 1
                    total_hand_motion += 1
                    if motion != "steady":
                        active_hand_motion += 1

    channels = {
        "state": normalize_counter(state_hist),
        "transition": normalize_counter(transition_hist),
        "hand_motion": normalize_counter(hand_motion_hist),
        "interaction": normalize_counter(interaction_hist),
    }
    if not include_temporal:
        channels["transition"] = {}
        channels["hand_motion"] = {}
        channels["interaction"] = {}
        channels["tempo"] = {}
        return channels

    frame_deltas = [
        b - a for a, b in zip(frame_indices, frame_indices[1:]) if (b - a) > 0
    ]
    effective_stride = statistics.median(frame_deltas) if frame_deltas else 0.0
    num_frames = len(frame_indices)
    steps = max(num_frames - 1, 1)
    duration_units = max(steps * effective_stride, 1.0)
    tempo_features = {
        "tempo::effective_stride": effective_stride / 18.0,
        "tempo::frames": num_frames / 1000.0,
        "tempo::transition_density": safe_ratio(non_stay_transitions, duration_units),
        "tempo::major_ratio": safe_ratio(major_transitions, total_transition_labels),
        "tempo::hand_change_ratio": safe_ratio(active_hand_motion, total_hand_motion),
        "tempo::interaction_change_ratio": safe_ratio(
            interaction_changes, total_interaction
        ),
    }
    channels["tempo"] = tempo_features
    return channels


def build_weighted_features(
    sequence: dict[str, object],
    state_weight: float,
    transition_weight: float,
    hand_motion_weight: float,
    interaction_weight: float,
    tempo_weight: float,
    include_temporal: bool,
) -> dict[str, float]:
    channels = extract_channel_features(sequence, include_temporal)
    feature_counter = Counter()
    for key, value in channels["state"].items():
        feature_counter[key] = state_weight * value
    if include_temporal:
        for key, value in channels["transition"].items():
            feature_counter[key] = transition_weight * value
        for key, value in channels["hand_motion"].items():
            feature_counter[key] = hand_motion_weight * value
        for key, value in channels["interaction"].items():
            feature_counter[key] = interaction_weight * value
        for key, value in channels["tempo"].items():
            feature_counter[key] = tempo_weight * value

    return l2_normalize(dict(feature_counter))


def build_prototypes(
    dataset: dict[str, object],
    mode: str,
    state_weight: float,
    transition_weight: float,
    hand_motion_weight: float,
    interaction_weight: float,
    tempo_weight: float,
    fusion_mode: str,
) -> dict[str, dict[str, float]]:
    prototypes = {}
    for sequence in dataset["sequences"]:
        if fusion_mode == "feature":
            prototypes[sequence["seq_name"]] = build_weighted_features(
                sequence,
                state_weight=state_weight,
                transition_weight=transition_weight,
                hand_motion_weight=hand_motion_weight,
                interaction_weight=interaction_weight,
                tempo_weight=tempo_weight,
                include_temporal=(mode == "temporal"),
            )
        else:
            prototypes[sequence["seq_name"]] = extract_channel_features(
                sequence, include_temporal=(mode == "temporal")
            )
    return prototypes


def late_fusion_similarity(
    query_channels: dict[str, dict[str, float]],
    gallery_channels: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> float:
    score = 0.0
    for channel_name, channel_weight in weights.items():
        if channel_weight <= 0:
            continue
        left = query_channels.get(channel_name, {})
        right = gallery_channels.get(channel_name, {})
        if not left or not right:
            continue
        left = l2_normalize(left)
        right = l2_normalize(right)
        score += channel_weight * cosine_similarity(left, right)
    return score


def evaluate_retrieval(
    gallery: dict[str, object],
    query: dict[str, object],
    mode: str,
    state_weight: float,
    transition_weight: float,
    hand_motion_weight: float,
    interaction_weight: float,
    tempo_weight: float,
    fusion_mode: str,
) -> dict[str, object]:
    gallery_prototypes = build_prototypes(
        gallery,
        mode,
        state_weight,
        transition_weight,
        hand_motion_weight,
        interaction_weight,
        tempo_weight,
        fusion_mode,
    )
    gallery_labels = set(gallery_prototypes.keys())
    query_sequences = [
        sequence for sequence in query["sequences"] if sequence["seq_name"] in gallery_labels
    ]

    results = []
    correct = 0
    late_weights = {
        "state": state_weight,
        "transition": transition_weight,
        "hand_motion": hand_motion_weight,
        "interaction": interaction_weight,
        "tempo": tempo_weight,
    }
    for sequence in query_sequences:
        if fusion_mode == "feature":
            features = build_weighted_features(
                sequence,
                state_weight=state_weight,
                transition_weight=transition_weight,
                hand_motion_weight=hand_motion_weight,
                interaction_weight=interaction_weight,
                tempo_weight=tempo_weight,
                include_temporal=(mode == "temporal"),
            )
            ranked = sorted(
                (
                    (seq_name, cosine_similarity(features, prototype))
                    for seq_name, prototype in gallery_prototypes.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        else:
            query_channels = extract_channel_features(
                sequence, include_temporal=(mode == "temporal")
            )
            ranked = sorted(
                (
                    (
                        seq_name,
                        late_fusion_similarity(query_channels, prototype, late_weights),
                    )
                    for seq_name, prototype in gallery_prototypes.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        prediction = ranked[0][0]
        correct += int(prediction == sequence["seq_name"])
        results.append(
            {
                "seq_name": sequence["seq_name"],
                "prediction": prediction,
                "top3": ranked[:3],
                "correct": prediction == sequence["seq_name"],
            }
        )

    accuracy = correct / len(query_sequences) if query_sequences else 0.0
    per_class = Counter()
    per_class_correct = Counter()
    for result in results:
        per_class[result["seq_name"]] += 1
        per_class_correct[result["seq_name"]] += int(result["correct"])
    per_class_accuracy = {
        key: per_class_correct[key] / per_class[key] for key in sorted(per_class)
    }

    return {
        "mode": mode,
        "fusion_mode": fusion_mode,
        "weights": {
            "state": state_weight,
            "transition": transition_weight,
            "hand_motion": hand_motion_weight,
            "interaction": interaction_weight,
            "tempo": tempo_weight,
        },
        "num_gallery_classes": len(gallery_labels),
        "num_queries": len(query_sequences),
        "top1_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_retrieval_summary.json"),
    )
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=0.5)
    parser.add_argument("--hand-motion-weight", type=float, default=0.5)
    parser.add_argument("--interaction-weight", type=float, default=0.2)
    parser.add_argument("--tempo-weight", type=float, default=0.2)
    parser.add_argument(
        "--fusion-mode",
        choices=["feature", "late"],
        default="late",
        help="Feature concatenation or channel-wise late fusion for temporal HL.",
    )
    args = parser.parse_args()

    gallery = load_json(args.gallery)
    query = load_json(args.query)
    summary = {
        "state_only": evaluate_retrieval(
            gallery,
            query,
            mode="state",
            state_weight=1.0,
            transition_weight=0.0,
            hand_motion_weight=0.0,
            interaction_weight=0.0,
            tempo_weight=0.0,
            fusion_mode="feature",
        ),
        "temporal_hl": evaluate_retrieval(
            gallery,
            query,
            mode="temporal",
            state_weight=args.state_weight,
            transition_weight=args.transition_weight,
            hand_motion_weight=args.hand_motion_weight,
            interaction_weight=args.interaction_weight,
            tempo_weight=args.tempo_weight,
            fusion_mode=args.fusion_mode,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"output: {args.output}")
    for key in ("state_only", "temporal_hl"):
        result = summary[key]
        print(
            f"{key}: top1={result['top1_accuracy']:.4f}, "
            f"num_queries={result['num_queries']}, "
            f"num_gallery_classes={result['num_gallery_classes']}"
        )


if __name__ == "__main__":
    main()
