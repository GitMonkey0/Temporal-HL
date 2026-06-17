#!/usr/bin/env python3
"""Train lightweight symbolic classifiers on old HL vs temporal HL features.

Protocol:
- use overlapping ROM classes between a train split and a test split
- slice each sequence into fixed-length windows
- train a simple classifier on window features
- aggregate window probabilities back to the sequence level
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


def iter_hand_records(frame: dict[str, object]):
    for hand_name in ("right", "left"):
        record = frame.get(hand_name)
        if record is not None:
            yield record


def canonical_label(seq_name: str) -> str:
    fixes = {
        "ROM07_RT_Finger_Occlusions": "ROM07_Rt_Finger_Occlusions",
        "ROM08_LT_Finger_Occlusions": "ROM08_Lt_Finger_Occlusions",
    }
    return fixes.get(seq_name, seq_name)


def extract_channel_features(
    frames: list[dict[str, object]],
    include_temporal: bool,
    include_wrist_features: bool = False,
):
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
    proximal_nonstay = 0
    proximal_major = 0
    proximal_angles = []
    distal_nonstay = 0
    distal_major = 0
    distal_angles = []
    proximal_coherent_frames = 0
    temporal_hand_frames = 0
    proximal_indices = [3, 7, 11, 15, 19]
    state_persistence_hist: Counter[str] = Counter()
    activity_persistence_hist: Counter[str] = Counter()
    interaction_persistence_hist: Counter[str] = Counter()
    interaction_activity_persistence_hist: Counter[str] = Counter()

    for frame in frames:
        frame_indices.append(frame["frame_idx"])
        interaction = frame.get("interaction_motion")
        if interaction not in (None, "unknown"):
            interaction_hist[f"interaction::{interaction}"] += 1
            total_interaction += 1
            if interaction != "steady":
                interaction_changes += 1
        interaction_persist = frame.get("interaction_persistence_label")
        if interaction_persist not in (None, "unknown"):
            interaction_persistence_hist[
                f"tempo::interaction_persist::{interaction_persist}"
            ] += 1
        interaction_active_persist = frame.get("interaction_activity_persistence_label")
        if interaction_active_persist not in (None, "unknown"):
            interaction_activity_persistence_hist[
                f"tempo::interaction_activity_persist::{interaction_active_persist}"
            ] += 1
        for hand_record in iter_hand_records(frame):
            for token in hand_record["token_labels"]:
                state_hist[f"state::{token}"] += 1
            state_persist = hand_record.get("state_persistence_label")
            if state_persist not in (None, "unknown"):
                state_persistence_hist[f"tempo::state_persist::{state_persist}"] += 1
            activity_persist = hand_record.get("activity_persistence_label")
            if activity_persist not in (None, "unknown"):
                activity_persistence_hist[
                    f"tempo::activity_persist::{activity_persist}"
                ] += 1
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
                labels = hand_record.get("transition_labels", [])
                angles = hand_record.get("transition_angles_deg", [])
                if include_wrist_features and labels and angles:
                    temporal_hand_frames += 1
                    prox_labels = [labels[idx] for idx in proximal_indices]
                    prox_angles = [angles[idx] for idx in proximal_indices]
                    dist_labels = [
                        labels[idx] for idx in range(len(labels)) if idx not in proximal_indices
                    ]
                    dist_angles = [
                        angles[idx] for idx in range(len(angles)) if idx not in proximal_indices
                    ]
                    prox_nonstay_local = sum(label != "stay" for label in prox_labels)
                    dist_nonstay_local = sum(label != "stay" for label in dist_labels)
                    proximal_nonstay += prox_nonstay_local
                    distal_nonstay += dist_nonstay_local
                    proximal_major += sum(label == "major_shift" for label in prox_labels)
                    distal_major += sum(label == "major_shift" for label in dist_labels)
                    proximal_angles.extend(prox_angles)
                    distal_angles.extend(dist_angles)
                    active_prox_labels = [label for label in prox_labels if label != "stay"]
                    if len(active_prox_labels) >= 3 and len(set(active_prox_labels)) == 1:
                        proximal_coherent_frames += 1

    channels = {"state": normalize_counter(state_hist)}
    if not include_temporal:
        channels.update(
            {"transition": {}, "hand_motion": {}, "interaction": {}, "tempo": {}}
        )
        return channels

    frame_deltas = [b - a for a, b in zip(frame_indices, frame_indices[1:]) if b > a]
    effective_stride = statistics.median(frame_deltas) if frame_deltas else 0.0
    num_frames = len(frame_indices)
    steps = max(num_frames - 1, 1)
    duration_units = max(steps * effective_stride, 1.0)
    channels.update(
        {
            "transition": normalize_counter(transition_hist),
            "hand_motion": normalize_counter(hand_motion_hist),
            "interaction": normalize_counter(interaction_hist),
            "tempo": {
                "tempo::effective_stride": effective_stride / 18.0,
                "tempo::frames": num_frames / 128.0,
                "tempo::transition_density": non_stay_transitions / duration_units,
                "tempo::major_ratio": (
                    major_transitions / total_transition_labels
                    if total_transition_labels
                    else 0.0
                ),
                "tempo::hand_change_ratio": (
                    active_hand_motion / total_hand_motion if total_hand_motion else 0.0
                ),
                "tempo::interaction_change_ratio": (
                    interaction_changes / total_interaction if total_interaction else 0.0
                ),
            },
        }
    )
    if include_wrist_features:
        channels["tempo"].update(
            {
                "tempo::proximal_nonstay_ratio": (
                    proximal_nonstay / max(proximal_nonstay + distal_nonstay, 1)
                ),
                "tempo::proximal_major_ratio": (
                    proximal_major / max(proximal_nonstay, 1)
                ),
                "tempo::proximal_angle_mean": (
                    statistics.mean(proximal_angles) / 90.0 if proximal_angles else 0.0
                ),
                "tempo::distal_angle_mean": (
                    statistics.mean(distal_angles) / 90.0 if distal_angles else 0.0
                ),
                "tempo::proximal_angle_ratio": (
                    (statistics.mean(proximal_angles) / max(statistics.mean(distal_angles), 1e-6))
                    if proximal_angles and distal_angles
                    else 0.0
                ),
                "tempo::proximal_coherence_ratio": (
                    proximal_coherent_frames / temporal_hand_frames if temporal_hand_frames else 0.0
                ),
            }
        )
    if state_persistence_hist:
        channels["tempo"].update(normalize_counter(state_persistence_hist))
    if activity_persistence_hist:
        channels["tempo"].update(normalize_counter(activity_persistence_hist))
    if interaction_persistence_hist:
        channels["tempo"].update(normalize_counter(interaction_persistence_hist))
    if interaction_activity_persistence_hist:
        channels["tempo"].update(
            normalize_counter(interaction_activity_persistence_hist)
        )
    return channels


def merge_channels(
    channels: dict[str, dict[str, float]],
    state_weight: float,
    transition_weight: float,
    hand_motion_weight: float,
    interaction_weight: float,
    tempo_weight: float,
    event_weight: float = 0.0,
) -> dict[str, float]:
    merged = {}
    for weight, channel_name in [
        (state_weight, "state"),
        (transition_weight, "transition"),
        (hand_motion_weight, "hand_motion"),
        (interaction_weight, "interaction"),
        (event_weight, "event"),
        (tempo_weight, "tempo"),
    ]:
        if weight <= 0:
            continue
        for key, value in channels.get(channel_name, {}).items():
            merged[key] = weight * value
    return l2_normalize(merged)


def build_vocabulary(samples: list[dict[str, object]]) -> list[str]:
    vocab = set()
    for sample in samples:
        vocab.update(sample["features"].keys())
    return sorted(vocab)


def vectorize(samples: list[dict[str, object]], vocab: list[str]) -> np.ndarray:
    feat_index = {key: idx for idx, key in enumerate(vocab)}
    matrix = np.zeros((len(samples), len(vocab)), dtype=np.float32)
    for row_idx, sample in enumerate(samples):
        for key, value in sample["features"].items():
            matrix[row_idx, feat_index[key]] = value
    return matrix


def overlap_labels(train_data: dict[str, object], test_data: dict[str, object]) -> set[str]:
    train_labels = {canonical_label(seq["seq_name"]) for seq in train_data["sequences"]}
    test_labels = {canonical_label(seq["seq_name"]) for seq in test_data["sequences"]}
    return train_labels & test_labels


def make_window_samples(
    dataset: dict[str, object],
    allowed_labels: set[str],
    mode: str,
    window_size: int,
    stride: int,
    weights: dict[str, float],
    include_wrist_features: bool = False,
) -> list[dict[str, object]]:
    samples = []
    for seq_idx, sequence in enumerate(dataset["sequences"]):
        label = canonical_label(sequence["seq_name"])
        if label not in allowed_labels:
            continue
        frames = sequence["frames"]
        if len(frames) < window_size:
            starts = [0]
        else:
            starts = list(range(0, len(frames) - window_size + 1, stride))
            if starts[-1] != len(frames) - window_size:
                starts.append(len(frames) - window_size)
        for window_idx, start in enumerate(starts):
            window = frames[start : start + window_size]
            channels = extract_channel_features(
                window,
                include_temporal=(mode == "temporal"),
                include_wrist_features=include_wrist_features,
            )
            features = merge_channels(
                channels,
                state_weight=weights["state"],
                transition_weight=weights["transition"],
                hand_motion_weight=weights["hand_motion"],
                interaction_weight=weights["interaction"],
                tempo_weight=weights["tempo"],
            )
            samples.append(
                {
                    "sample_id": f"{seq_idx}:{window_idx}",
                    "seq_name": sequence["seq_name"],
                    "label": label,
                    "features": features,
                }
            )
    return samples


def subsample_by_fraction(
    samples: list[dict[str, object]], fraction: float, seed: int
) -> list[dict[str, object]]:
    if fraction >= 0.999:
        return list(samples)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    subset = []
    for label, label_samples in grouped.items():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        keep = max(1, int(round(len(label_samples) * fraction)))
        subset.extend(label_samples[:keep])
    return subset


def aggregate_sequence_predictions(
    probs: np.ndarray,
    pred_labels: np.ndarray,
    labels: list[str],
    samples: list[dict[str, object]],
    method: str,
) -> tuple[list[str], list[str]]:
    by_sequence_probs = defaultdict(list)
    by_sequence_votes = defaultdict(list)
    gt = {}
    for prob, pred, sample in zip(probs, pred_labels, samples):
        seq_name = sample["seq_name"]
        by_sequence_probs[seq_name].append(prob)
        by_sequence_votes[seq_name].append(int(pred))
        gt[seq_name] = sample["label"]
    y_true, y_pred = [], []
    for seq_name in sorted(by_sequence_probs):
        probs_arr = np.asarray(by_sequence_probs[seq_name], dtype=np.float64)
        if method == "mean_log_prob":
            aggregated = np.mean(np.log(np.clip(probs_arr, 1e-8, 1.0)), axis=0)
            pred_idx = int(np.argmax(aggregated))
        elif method == "vote":
            vote_counter = Counter(by_sequence_votes[seq_name])
            pred_idx = vote_counter.most_common(1)[0][0]
        else:
            aggregated = np.mean(probs_arr, axis=0)
            pred_idx = int(np.argmax(aggregated))
        pred = labels[pred_idx]
        y_true.append(gt[seq_name])
        y_pred.append(pred)
    return y_true, y_pred


def run_experiment(
    train_samples: list[dict[str, object]],
    test_samples: list[dict[str, object]],
    fraction: float,
    seed: int,
    c_value: float,
    aggregation: str,
) -> dict[str, object]:
    train_subset = subsample_by_fraction(train_samples, fraction, seed)
    vocab = build_vocabulary(train_subset + test_samples)
    x_train = vectorize(train_subset, vocab)
    x_test = vectorize(test_samples, vocab)

    label_names = sorted({sample["label"] for sample in train_subset})
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    y_train = np.array([label_to_idx[sample["label"]] for sample in train_subset], dtype=int)
    y_test = np.array([label_to_idx[sample["label"]] for sample in test_samples], dtype=int)

    clf = LogisticRegression(
        max_iter=2000,
        C=c_value,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    window_pred = clf.predict(x_test)
    window_acc = accuracy_score(y_test, window_pred)
    seq_true, seq_pred = aggregate_sequence_predictions(
        clf.predict_proba(x_test), window_pred, label_names, test_samples, aggregation
    )
    seq_acc = accuracy_score(seq_true, seq_pred)
    return {
        "fraction": fraction,
        "seed": seed,
        "num_train_windows": len(train_subset),
        "num_test_windows": len(test_samples),
        "aggregation": aggregation,
        "window_accuracy": float(window_acc),
        "sequence_accuracy": float(seq_acc),
    }


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    by_fraction = defaultdict(list)
    for result in results:
        by_fraction[result["fraction"]].append(result)
    summary = {}
    for fraction, items in sorted(by_fraction.items()):
        summary[str(fraction)] = {
            "num_runs": len(items),
            "avg_train_windows": statistics.mean(item["num_train_windows"] for item in items),
            "window_accuracy_mean": statistics.mean(item["window_accuracy"] for item in items),
            "window_accuracy_std": (
                statistics.pstdev(item["window_accuracy"] for item in items)
                if len(items) > 1
                else 0.0
            ),
            "sequence_accuracy_mean": statistics.mean(
                item["sequence_accuracy"] for item in items
            ),
            "sequence_accuracy_std": (
                statistics.pstdev(item["sequence_accuracy"] for item in items)
                if len(items) > 1
                else 0.0
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_classifier_summary.json"),
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--wrist-features", action="store_true")
    parser.add_argument(
        "--aggregation",
        choices=["mean_prob", "mean_log_prob", "vote"],
        default="mean_log_prob",
    )
    args = parser.parse_args()

    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)
    labels = overlap_labels(train_data, test_data)
    base_weights = {"state": 1.0, "transition": 0.0, "hand_motion": 0.0, "interaction": 0.0, "tempo": 0.0}
    temporal_weights = {"state": 1.0, "transition": 0.5, "hand_motion": 0.5, "interaction": 0.2, "tempo": 0.2}

    all_results = {}
    for mode, weights in [("state", base_weights), ("temporal", temporal_weights)]:
        train_samples = make_window_samples(
            train_data, labels, mode, args.window_size, args.stride, weights, args.wrist_features
        )
        test_samples = make_window_samples(
            test_data, labels, mode, args.window_size, args.stride, weights, args.wrist_features
        )
        results = []
        for fraction in args.fractions:
            for seed in args.seeds:
                results.append(
                    run_experiment(
                        train_samples=train_samples,
                        test_samples=test_samples,
                        fraction=fraction,
                        seed=seed,
                        c_value=args.c_value,
                        aggregation=args.aggregation,
                    )
                )
        all_results[mode] = {
            "weights": weights,
            "num_train_windows_full": len(train_samples),
            "num_test_windows": len(test_samples),
            "results": results,
            "summary": summarize_results(results),
        }

    payload = {
        "labels": sorted(labels),
        "window_size": args.window_size,
        "stride": args.stride,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "c_value": args.c_value,
        "aggregation": args.aggregation,
        "state_only": all_results["state"],
        "temporal_hl": all_results["temporal"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for key in ("state_only", "temporal_hl"):
        print(f"\n{key}")
        for fraction, item in payload[key]["summary"].items():
            print(
                f"fraction={fraction} "
                f"seq_acc={item['sequence_accuracy_mean']:.4f}±{item['sequence_accuracy_std']:.4f} "
                f"win_acc={item['window_accuracy_mean']:.4f}±{item['window_accuracy_std']:.4f} "
                f"avg_train_windows={item['avg_train_windows']:.1f}"
            )


if __name__ == "__main__":
    main()
