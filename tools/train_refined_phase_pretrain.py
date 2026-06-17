#!/usr/bin/env python3
"""Pretrain -> finetune classifier on compressed refined-phase HL features."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from tools.eval_sequence_symbolic_retrieval import (
    classify_hand_phase_refined,
    classify_interaction_phase_refined,
    refined_phase_group_key,
    segment_duration_bucket,
)
from tools.train_symbolic_classifier import (
    build_vocabulary,
    canonical_label,
    overlap_labels,
    subsample_by_fraction,
    vectorize,
)
from tools.train_symbolic_pretrain import (
    MLPClassifier,
    aggregate_sequence_predictions,
    predict_probs,
    resolve_split_window_args,
    split_train_holdout,
    train_model,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def extract_refined_phase_channels(frames: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    hand_phase_hist: Counter[str] = Counter()
    interaction_phase_hist: Counter[str] = Counter()
    segdur_hist: Counter[str] = Counter()
    phase_event_hist: Counter[str] = Counter()
    frame_indices = [int(frame["frame_idx"]) for frame in frames]

    prev_phase_key = None
    phase_event_runs: list[tuple[str, int]] = []
    run_count = 0

    for frame in frames:
        for hand_name in ("right", "left"):
            phase = classify_hand_phase_refined(frame.get(hand_name))
            hand_phase_hist[f"{hand_name}:phase::{phase}"] += 1
            hand_record = frame.get(hand_name)
            if hand_record is not None:
                segdur_hist[
                    f"{hand_name}:state_segdur::{hand_record.get('state_segment_duration_label', 'missing')}"
                ] += 1
                segdur_hist[
                    f"{hand_name}:activity_segdur::{hand_record.get('activity_segment_duration_label', 'missing')}"
                ] += 1
        interaction_phase = classify_interaction_phase_refined(frame)
        interaction_phase_hist[f"interaction:phase::{interaction_phase}"] += 1
        segdur_hist[
            f"interaction:segdur::{frame.get('interaction_segment_duration_label', 'missing')}"
        ] += 1
        segdur_hist[
            "interaction:activity_segdur::"
            f"{frame.get('interaction_activity_segment_duration_label', 'missing')}"
        ] += 1

        phase_key = refined_phase_group_key(frame, mode="temporal")
        if prev_phase_key is None or phase_key == prev_phase_key:
            run_count += 1
        else:
            phase_event_runs.append((prev_phase_key, run_count))
            run_count = 1
        prev_phase_key = phase_key
    if prev_phase_key is not None:
        phase_event_runs.append((prev_phase_key, run_count))

    for phase_key, run_count in phase_event_runs:
        phase_event_hist[
            f"phase_event::{phase_key}::dur::{segment_duration_bucket(run_count)}"
        ] += 1

    total_frames = max(len(frames), 1)
    total_events = max(len(phase_event_runs), 1)
    frame_deltas = [b - a for a, b in zip(frame_indices, frame_indices[1:]) if b > a]
    effective_stride = statistics.median(frame_deltas) if frame_deltas else 0.0
    phase_lengths = [run for _, run in phase_event_runs]
    channels = {
        "hand_phase": normalize_counter(hand_phase_hist),
        "interaction_phase": normalize_counter(interaction_phase_hist),
        "segdur": normalize_counter(segdur_hist),
        "phase_event": normalize_counter(phase_event_hist),
        "tempo": {
            "tempo::phase_event_count": total_events / total_frames,
            "tempo::phase_run_mean": statistics.mean(phase_lengths) / total_frames
            if phase_lengths
            else 0.0,
            "tempo::phase_run_max": max(phase_lengths) / total_frames if phase_lengths else 0.0,
            "tempo::effective_stride": effective_stride / 18.0,
        },
    }
    return channels


def merge_phase_channels(channels: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    merged = {}
    for channel_name in ["hand_phase", "interaction_phase", "segdur", "phase_event", "tempo"]:
        weight = weights.get(channel_name, 0.0)
        if weight <= 0:
            continue
        for key, value in channels.get(channel_name, {}).items():
            merged[key] = weight * value
    return merged


def make_window_phase_samples(
    dataset: dict[str, object],
    allowed_labels: set[str],
    window_size: int,
    stride: int,
    window_span_units: int | None = None,
    window_step_units: int | None = None,
):
    samples = []
    for seq_idx, sequence in enumerate(dataset["sequences"]):
        label = canonical_label(sequence["seq_name"])
        if label not in allowed_labels:
            continue
        frames = sequence["frames"]
        window_ranges = []
        if window_span_units is None:
            if len(frames) < window_size:
                starts = [0]
            else:
                starts = list(range(0, len(frames) - window_size + 1, stride))
                if starts[-1] != len(frames) - window_size:
                    starts.append(len(frames) - window_size)
            for start in starts:
                window_ranges.append((start, min(start + window_size, len(frames))))
        else:
            step_units = window_step_units if window_step_units is not None else window_span_units
            frame_positions = [int(frame["frame_idx"]) for frame in frames]
            target = frame_positions[0]
            last_frame = frame_positions[-1]
            starts = []
            cursor = 0
            while target <= last_frame:
                while cursor + 1 < len(frame_positions) and frame_positions[cursor + 1] <= target:
                    cursor += 1
                starts.append(cursor)
                target += step_units
            tail_start = max(0, len(frames) - 1)
            if not starts or starts[-1] != tail_start:
                starts.append(tail_start)
            starts = sorted(set(starts))
            for start in starts:
                start_pos = frame_positions[start]
                end = start + 1
                while end < len(frames) and frame_positions[end] - start_pos < window_span_units:
                    end += 1
                window_ranges.append((start, end))

        for window_idx, (start, end) in enumerate(window_ranges):
            window = frames[start:end]
            samples.append(
                {
                    "sample_id": f"{seq_idx}:{window_idx}",
                    "seq_name": sequence["seq_name"],
                    "label": label,
                    "channels": extract_refined_phase_channels(window),
                }
            )
    return samples


def run_once(
    pre_samples,
    ft_samples,
    test_samples,
    fraction,
    seed,
    hidden_dim,
    pretrain_epochs,
    finetune_epochs,
    lr,
    weight_decay,
    device,
    aggregation,
):
    ft_fraction = subsample_by_fraction(ft_samples, fraction, seed)
    ft_train, ft_holdout = split_train_holdout(ft_fraction, 0.2, seed)

    all_for_vocab = pre_samples + ft_train + ft_holdout + test_samples
    vocab = build_vocabulary(all_for_vocab)

    pre_labels = sorted({s["label"] for s in pre_samples})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    pre_train, pre_holdout = split_train_holdout(pre_samples, 0.1, seed)
    x_pre_train = vectorize(pre_train, vocab)
    y_pre_train = np.asarray([pre_map[s["label"]] for s in pre_train], dtype=np.int64)
    x_pre_hold = vectorize(pre_holdout, vocab)
    y_pre_hold = np.asarray([pre_map[s["label"]] for s in pre_holdout], dtype=np.int64)

    ft_labels = sorted({s["label"] for s in ft_train + ft_holdout + test_samples})
    ft_map = {k: i for i, k in enumerate(ft_labels)}
    x_ft_train = vectorize(ft_train, vocab)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train], dtype=np.int64)
    x_ft_hold = vectorize(ft_holdout, vocab)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_holdout], dtype=np.int64)
    x_test = vectorize(test_samples, vocab)
    y_test = np.asarray([ft_map[s["label"]] for s in test_samples], dtype=np.int64)

    scratch = MLPClassifier(len(vocab), hidden_dim, len(ft_labels)).to(device)
    train_model(
        scratch,
        x_ft_train,
        y_ft_train,
        x_ft_hold,
        y_ft_hold,
        device,
        finetune_epochs,
        lr,
        weight_decay,
    )
    scratch_probs = predict_probs(scratch, x_test, device)
    scratch_pred = scratch_probs.argmax(axis=1)
    scratch_win = accuracy_score(y_test, scratch_pred)
    st, sp = aggregate_sequence_predictions(
        scratch_probs, scratch_pred, ft_labels, test_samples, aggregation
    )
    scratch_seq = accuracy_score(st, sp)

    pretrained = MLPClassifier(len(vocab), hidden_dim, len(pre_labels)).to(device)
    train_model(
        pretrained,
        x_pre_train,
        y_pre_train,
        x_pre_hold,
        y_pre_hold,
        device,
        pretrain_epochs,
        lr,
        weight_decay,
    )
    encoder_state = {
        k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()
    }
    finetune = MLPClassifier(len(vocab), hidden_dim, len(ft_labels)).to(device)
    finetune.encoder.load_state_dict(encoder_state)
    train_model(
        finetune,
        x_ft_train,
        y_ft_train,
        x_ft_hold,
        y_ft_hold,
        device,
        finetune_epochs,
        lr,
        weight_decay,
    )
    ft_probs = predict_probs(finetune, x_test, device)
    ft_pred = ft_probs.argmax(axis=1)
    ft_win = accuracy_score(y_test, ft_pred)
    tt, tp = aggregate_sequence_predictions(ft_probs, ft_pred, ft_labels, test_samples, aggregation)
    ft_seq = accuracy_score(tt, tp)

    return {
        "fraction": fraction,
        "seed": seed,
        "num_finetune_train_windows": len(ft_train),
        "feature_dim": len(vocab),
        "scratch_window_accuracy": float(scratch_win),
        "scratch_sequence_accuracy": float(scratch_seq),
        "pretrained_window_accuracy": float(ft_win),
        "pretrained_sequence_accuracy": float(ft_seq),
    }


def summarize(results):
    by_fraction = defaultdict(list)
    for r in results:
        by_fraction[r["fraction"]].append(r)
    out = {}
    for fraction, items in sorted(by_fraction.items()):
        out[str(fraction)] = {
            "num_runs": len(items),
            "avg_feature_dim": statistics.mean(x["feature_dim"] for x in items),
            "avg_finetune_train_windows": statistics.mean(
                x["num_finetune_train_windows"] for x in items
            ),
            "scratch_window_accuracy_mean": statistics.mean(
                x["scratch_window_accuracy"] for x in items
            ),
            "scratch_sequence_accuracy_mean": statistics.mean(
                x["scratch_sequence_accuracy"] for x in items
            ),
            "pretrained_window_accuracy_mean": statistics.mean(
                x["pretrained_window_accuracy"] for x in items
            ),
            "pretrained_sequence_accuracy_mean": statistics.mean(
                x["pretrained_sequence_accuracy"] for x in items
            ),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pretrain-json", type=Path, required=True)
    parser.add_argument("--train-finetune-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--window-span-units", type=int, default=None)
    parser.add_argument("--window-step-units", type=int, default=None)
    parser.add_argument("--pretrain-window-span-units", type=int, default=None)
    parser.add_argument("--pretrain-window-step-units", type=int, default=None)
    parser.add_argument("--finetune-window-span-units", type=int, default=None)
    parser.add_argument("--finetune-window-step-units", type=int, default=None)
    parser.add_argument("--test-window-span-units", type=int, default=None)
    parser.add_argument("--test-window-step-units", type=int, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/refined_phase_pretrain.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_data = load_json(args.train_pretrain_json)
    ft_data = load_json(args.train_finetune_json)
    test_data = load_json(args.test_json)
    allowed = overlap_labels(ft_data, test_data)

    pre_span, pre_step = resolve_split_window_args(
        args.window_span_units,
        args.window_step_units,
        args.pretrain_window_span_units,
        args.pretrain_window_step_units,
    )
    ft_span, ft_step = resolve_split_window_args(
        args.window_span_units,
        args.window_step_units,
        args.finetune_window_span_units,
        args.finetune_window_step_units,
    )
    test_span, test_step = resolve_split_window_args(
        args.window_span_units,
        args.window_step_units,
        args.test_window_span_units,
        args.test_window_step_units,
    )

    base_weights = {
        "hand_phase": 1.0,
        "interaction_phase": 0.7,
        "segdur": 0.6,
        "phase_event": 1.0,
        "tempo": 0.4,
    }
    pre_samples = make_window_phase_samples(
        pre_data,
        {canonical_label(s["seq_name"]) for s in pre_data["sequences"]},
        args.window_size,
        args.stride,
        pre_span,
        pre_step,
    )
    ft_samples = make_window_phase_samples(
        ft_data,
        allowed,
        args.window_size,
        args.stride,
        ft_span,
        ft_step,
    )
    test_samples = make_window_phase_samples(
        test_data,
        allowed,
        args.window_size,
        args.stride,
        test_span,
        test_step,
    )
    pre_samples = [{**s, "features": merge_phase_channels(s["channels"], base_weights)} for s in pre_samples]
    ft_samples = [{**s, "features": merge_phase_channels(s["channels"], base_weights)} for s in ft_samples]
    test_samples = [{**s, "features": merge_phase_channels(s["channels"], base_weights)} for s in test_samples]

    results = []
    for fraction in args.fractions:
        for seed in args.seeds:
            set_seed(seed)
            results.append(
                run_once(
                    pre_samples,
                    ft_samples,
                    test_samples,
                    fraction,
                    seed,
                    args.hidden_dim,
                    args.pretrain_epochs,
                    args.finetune_epochs,
                    args.lr,
                    args.weight_decay,
                    device,
                    args.aggregation,
                )
            )

    payload = {
        "weights": base_weights,
        "window_size": args.window_size,
        "stride": args.stride,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "results": results,
        "summary": summarize(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    for fraction, summary in payload["summary"].items():
        print(
            f"fraction={fraction} "
            f"scratch_seq={summary['scratch_sequence_accuracy_mean']:.4f} "
            f"pretrained_seq={summary['pretrained_sequence_accuracy_mean']:.4f} "
            f"scratch_win={summary['scratch_window_accuracy_mean']:.4f} "
            f"pretrained_win={summary['pretrained_window_accuracy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
