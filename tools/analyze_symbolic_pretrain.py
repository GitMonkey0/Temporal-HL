#!/usr/bin/env python3
"""Detailed sequence-level analysis for symbolic pretrain experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from tools.train_symbolic_pretrain import (
    MLPClassifier,
    aggregate_sequence_predictions,
    build_finetune_class_weights,
    load_json,
    make_window_channel_samples,
    merge_sample_features,
    overlap_labels,
    parse_boost_labels,
    predict_probs,
    resolve_split_window_args,
    set_seed,
    split_train_holdout,
    train_model,
)
from tools.train_symbolic_classifier import (
    build_vocabulary,
    canonical_label,
    merge_channels,
    subsample_by_fraction,
    vectorize,
)


FAMILY_LABELS = [
    "ROM03_LT_No_Occlusion",
    "ROM03_RT_No_Occlusion",
    "ROM04_LT_Occlusion",
    "ROM04_RT_Occlusion",
    "ROM05_LT_Wrist_ROM",
    "ROM05_RT_Wrist_ROM",
    "ROM07_Rt_Finger_Occlusions",
    "ROM08_Lt_Finger_Occlusions",
]


def sequence_prob_table(probs, pred_labels, label_names, samples, method, export_family_scores=False):
    by_sequence_probs = defaultdict(list)
    by_sequence_votes = defaultdict(list)
    gt = {}
    for prob, pred, sample in zip(probs, pred_labels, samples):
        seq_name = sample["seq_name"]
        by_sequence_probs[seq_name].append(prob)
        by_sequence_votes[seq_name].append(int(pred))
        gt[seq_name] = sample["label"]
    rows = []
    for seq_name in sorted(by_sequence_probs):
        probs_arr = np.asarray(by_sequence_probs[seq_name], dtype=np.float64)
        if method == "mean_log_prob":
            agg = np.mean(np.log(np.clip(probs_arr, 1e-8, 1.0)), axis=0)
        elif method == "vote":
            counts = Counter(by_sequence_votes[seq_name])
            agg = np.full(probs_arr.shape[1], -1e9)
            for idx, count in counts.items():
                agg[idx] = count
        else:
            agg = np.mean(probs_arr, axis=0)
        ranking = np.argsort(agg)[::-1]
        row = {
            "seq_name": seq_name,
            "target": gt[seq_name],
            "prediction": label_names[int(ranking[0])],
            "top3": [
                [label_names[int(idx)], float(agg[int(idx)])] for idx in ranking[:3]
            ],
            "correct": label_names[int(ranking[0])] == gt[seq_name],
        }
        if export_family_scores:
            score_map = {label_names[int(idx)]: float(agg[int(idx)]) for idx in range(len(label_names))}
            row["family_scores"] = {
                label: score_map[label]
                for label in FAMILY_LABELS
                if label in score_map
            }
        rows.append(row)
    return rows


def train_eval_details(
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
    finetune_class_weights,
    export_family_scores=False,
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
        scratch, x_ft_train, y_ft_train, x_ft_hold, y_ft_hold, device,
        finetune_epochs, lr, weight_decay, class_weights=finetune_class_weights
    )
    scratch_probs = predict_probs(scratch, x_test, device)
    scratch_pred = scratch_probs.argmax(axis=1)

    pretrained = MLPClassifier(len(vocab), hidden_dim, len(pre_labels)).to(device)
    train_model(pretrained, x_pre_train, y_pre_train, x_pre_hold, y_pre_hold, device, pretrain_epochs, lr, weight_decay)
    encoder_state = {k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()}
    finetune = MLPClassifier(len(vocab), hidden_dim, len(ft_labels)).to(device)
    finetune.encoder.load_state_dict(encoder_state)
    train_model(
        finetune, x_ft_train, y_ft_train, x_ft_hold, y_ft_hold, device,
        finetune_epochs, lr, weight_decay, class_weights=finetune_class_weights
    )
    ft_probs = predict_probs(finetune, x_test, device)
    ft_pred = ft_probs.argmax(axis=1)

    scratch_seq_rows = sequence_prob_table(
        scratch_probs, scratch_pred, ft_labels, test_samples, aggregation, export_family_scores
    )
    pretrained_seq_rows = sequence_prob_table(
        ft_probs, ft_pred, ft_labels, test_samples, aggregation, export_family_scores
    )

    scratch_true = [row["target"] for row in scratch_seq_rows]
    scratch_est = [row["prediction"] for row in scratch_seq_rows]
    pretrained_true = [row["target"] for row in pretrained_seq_rows]
    pretrained_est = [row["prediction"] for row in pretrained_seq_rows]

    return {
        "labels": ft_labels,
        "scratch": {
            "window_accuracy": float((scratch_pred == y_test).mean()),
            "sequence_rows": scratch_seq_rows,
            "confusion_matrix": confusion_matrix(scratch_true, scratch_est, labels=ft_labels).tolist(),
        },
        "pretrained": {
            "window_accuracy": float((ft_pred == y_test).mean()),
            "sequence_rows": pretrained_seq_rows,
            "confusion_matrix": confusion_matrix(pretrained_true, pretrained_est, labels=ft_labels).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pretrain-json", type=Path, required=True)
    parser.add_argument("--train-finetune-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["state", "temporal"], default="temporal")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--occlusion-features", action="store_true")
    parser.add_argument("--finger-detail-features", action="store_true")
    parser.add_argument("--finger-shape-features", action="store_true")
    parser.add_argument(
        "--boost-label",
        action="append",
        default=[],
        help="Optional finetune class boost as LABEL:WEIGHT. Can be repeated.",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_pretrain_analysis.json"))
    parser.add_argument("--export-family-scores", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    pre_data = load_json(args.train_pretrain_json)
    ft_data = load_json(args.train_finetune_json)
    test_data = load_json(args.test_json)
    allowed = overlap_labels(ft_data, test_data)
    weights = {"state": 1.0, "transition": 0.5, "hand_motion": 0.5, "interaction": 0.2, "tempo": 0.2, "event": args.event_weight if args.event_features else 0.0} if args.mode == "temporal" else {"state": 1.0, "transition": 0.0, "hand_motion": 0.0, "interaction": 0.0, "tempo": 0.0, "event": 0.0}
    boost_map = parse_boost_labels(args.boost_label)
    pre_span, pre_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.pretrain_window_span_units, args.pretrain_window_step_units)
    ft_span, ft_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.finetune_window_span_units, args.finetune_window_step_units)
    test_span, test_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.test_window_span_units, args.test_window_step_units)

    pre_samples = make_window_channel_samples(pre_data, {canonical_label(s["seq_name"]) for s in pre_data["sequences"]}, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, pre_span, pre_step)
    ft_samples = make_window_channel_samples(ft_data, allowed, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, ft_span, ft_step)
    test_samples = make_window_channel_samples(test_data, allowed, args.mode, args.window_size, args.stride, args.wrist_features, args.duration_features, args.event_features, args.coordination_features, args.occlusion_features, args.finger_detail_features, args.finger_shape_features, test_span, test_step)
    pre_samples = merge_sample_features(pre_samples, weights)
    ft_samples = merge_sample_features(ft_samples, weights)
    test_samples = merge_sample_features(test_samples, weights)
    ft_fraction_preview = subsample_by_fraction(ft_samples, args.fraction, args.seed)
    ft_train_preview, ft_holdout_preview = split_train_holdout(
        ft_fraction_preview, 0.2, args.seed
    )
    ft_labels_preview = sorted(
        {s["label"] for s in ft_train_preview + ft_holdout_preview + test_samples}
    )
    finetune_class_weights = build_finetune_class_weights(
        ft_labels_preview, boost_map
    )

    details = train_eval_details(
        pre_samples,
        ft_samples,
        test_samples,
        args.fraction,
        args.seed,
        args.hidden_dim,
        args.pretrain_epochs,
        args.finetune_epochs,
        args.lr,
        args.weight_decay,
        device,
        args.aggregation,
        finetune_class_weights,
        args.export_family_scores,
    )
    payload = {
        "mode": args.mode,
        "fraction": args.fraction,
        "seed": args.seed,
        "window_size": args.window_size,
        "stride": args.stride,
        "aggregation": args.aggregation,
        "wrist_features": args.wrist_features,
        "duration_features": args.duration_features,
        "event_features": args.event_features,
        "coordination_features": args.coordination_features,
        "event_weight": args.event_weight,
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
        "export_family_scores": args.export_family_scores,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    print(f"scratch_window={details['scratch']['window_accuracy']:.4f}")
    print(f"pretrained_window={details['pretrained']['window_accuracy']:.4f}")
    scratch_seq = np.mean([row['correct'] for row in details['scratch']['sequence_rows']])
    pretrained_seq = np.mean([row['correct'] for row in details['pretrained']['sequence_rows']])
    print(f"scratch_sequence={scratch_seq:.4f}")
    print(f"pretrained_sequence={pretrained_seq:.4f}")


if __name__ == "__main__":
    main()
