#!/usr/bin/env python3
"""Aggregate multi-seed sequence errors for grouped symbolic pretraining experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.analyze_symbolic_pretrain_grouped import train_eval_details
from tools.train_symbolic_classifier import canonical_label, overlap_labels, subsample_by_fraction
from tools.train_symbolic_pretrain import (
    build_finetune_class_weights,
    load_json,
    make_window_channel_samples,
    parse_boost_labels,
    resolve_split_window_args,
    set_seed,
    split_train_holdout,
)
from tools.train_symbolic_pretrain_grouped import prepare_grouped_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pretrain-json", type=Path, required=True)
    parser.add_argument("--train-finetune-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["state", "temporal"], default="temporal")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument("--wrist-features", action="store_true")
    parser.add_argument("--duration-features", action="store_true")
    parser.add_argument("--event-features", action="store_true")
    parser.add_argument("--coordination-features", action="store_true")
    parser.add_argument("--event-weight", type=float, default=0.3)
    parser.add_argument("--state-weight", type=float, default=None)
    parser.add_argument("--transition-weight", type=float, default=None)
    parser.add_argument("--hand-motion-weight", type=float, default=None)
    parser.add_argument("--interaction-weight", type=float, default=None)
    parser.add_argument("--tempo-weight", type=float, default=None)
    parser.add_argument("--occlusion-features", action="store_true")
    parser.add_argument("--finger-detail-features", action="store_true")
    parser.add_argument("--finger-shape-features", action="store_true")
    parser.add_argument("--fusion-type", choices=["gated", "concat", "residual_aux"], default="concat")
    parser.add_argument("--boost-label", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_pretrain_grouped_multiseed_analysis.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_data = load_json(args.train_pretrain_json)
    ft_data = load_json(args.train_finetune_json)
    test_data = load_json(args.test_json)
    allowed = overlap_labels(ft_data, test_data)
    if args.mode == "temporal":
        weights = {
            "state": 1.0 if args.state_weight is None else args.state_weight,
            "transition": 0.5 if args.transition_weight is None else args.transition_weight,
            "hand_motion": 0.5 if args.hand_motion_weight is None else args.hand_motion_weight,
            "interaction": 0.2 if args.interaction_weight is None else args.interaction_weight,
            "tempo": 0.2 if args.tempo_weight is None else args.tempo_weight,
            "event": args.event_weight if args.event_features else 0.0,
        }
    else:
        weights = {"state": 1.0, "transition": 0.0, "hand_motion": 0.0, "interaction": 0.0, "tempo": 0.0, "event": 0.0}
    boost_map = parse_boost_labels(args.boost_label)
    pre_span, pre_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.pretrain_window_span_units, args.pretrain_window_step_units)
    ft_span, ft_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.finetune_window_span_units, args.finetune_window_step_units)
    test_span, test_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.test_window_span_units, args.test_window_step_units)

    pre_samples = make_window_channel_samples(pre_data, {canonical_label(s["seq_name"]) for s in pre_data["sequences"]}, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, pre_span, pre_step)
    ft_samples = make_window_channel_samples(ft_data, allowed, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, ft_span, ft_step)
    test_samples = make_window_channel_samples(test_data, allowed, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, test_span, test_step)
    pre_samples = prepare_grouped_samples(pre_samples, weights)
    ft_samples = prepare_grouped_samples(ft_samples, weights)
    test_samples = prepare_grouped_samples(test_samples, weights)

    scratch_error_counter = Counter()
    pretrained_error_counter = Counter()
    scratch_per_seq = defaultdict(list)
    pretrained_per_seq = defaultdict(list)
    runs = []

    for seed in args.seeds:
        set_seed(seed)
        ft_fraction_preview = subsample_by_fraction(ft_samples, args.fraction, seed)
        ft_train_preview, ft_holdout_preview = split_train_holdout(ft_fraction_preview, 0.2, seed)
        ft_labels_preview = sorted({s["label"] for s in ft_train_preview + ft_holdout_preview + test_samples})
        finetune_class_weights = build_finetune_class_weights(ft_labels_preview, boost_map)
        details = train_eval_details(
            pre_samples,
            ft_samples,
            test_samples,
            args.fraction,
            seed,
            args.hidden_dim,
            args.pretrain_epochs,
            args.finetune_epochs,
            args.lr,
            args.weight_decay,
            device,
            args.aggregation,
            finetune_class_weights,
            args.fusion_type,
            export_family_scores=False,
        )
        run = {
            "seed": seed,
            "scratch_sequence_accuracy": float(np.mean([r["correct"] for r in details["scratch"]["sequence_rows"]])),
            "pretrained_sequence_accuracy": float(np.mean([r["correct"] for r in details["pretrained"]["sequence_rows"]])),
            "scratch_errors": [],
            "pretrained_errors": [],
        }
        for mode_name, rows, counter, per_seq in [
            ("scratch", details["scratch"]["sequence_rows"], scratch_error_counter, scratch_per_seq),
            ("pretrained", details["pretrained"]["sequence_rows"], pretrained_error_counter, pretrained_per_seq),
        ]:
            for row in rows:
                seq_name = row["seq_name"]
                per_seq[seq_name].append({"seed": seed, "correct": row["correct"], "prediction": row["prediction"]})
                if not row["correct"]:
                    key = (row["target"], row["prediction"])
                    counter[key] += 1
                    run[f"{mode_name}_errors"].append(row)
        runs.append(run)

    seq_summary = {}
    all_seq = sorted(set(scratch_per_seq) | set(pretrained_per_seq))
    for seq_name in all_seq:
        seq_summary[seq_name] = {
            "scratch_correct_runs": sum(item["correct"] for item in scratch_per_seq[seq_name]),
            "pretrained_correct_runs": sum(item["correct"] for item in pretrained_per_seq[seq_name]),
            "scratch_predictions": scratch_per_seq[seq_name],
            "pretrained_predictions": pretrained_per_seq[seq_name],
        }

    payload = {
        "mode": args.mode,
        "fraction": args.fraction,
        "seeds": args.seeds,
        "fusion_type": args.fusion_type,
        "weights": weights,
        "wrist_features": args.wrist_features,
        "duration_features": args.duration_features,
        "occlusion_features": args.occlusion_features,
        "finger_detail_features": args.finger_detail_features,
        "finger_shape_features": args.finger_shape_features,
        "window_span_units": args.window_span_units,
        "window_step_units": args.window_step_units,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "boost_map": boost_map,
        "runs": runs,
        "scratch_error_counts": [{"target": t, "prediction": p, "count": c} for (t, p), c in scratch_error_counter.most_common()],
        "pretrained_error_counts": [{"target": t, "prediction": p, "count": c} for (t, p), c in pretrained_error_counter.most_common()],
        "sequence_consistency": seq_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    print("pretrained_error_counts", payload["pretrained_error_counts"][:10])


if __name__ == "__main__":
    main()
