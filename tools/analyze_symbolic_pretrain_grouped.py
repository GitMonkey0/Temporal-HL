#!/usr/bin/env python3
"""Detailed sequence-level analysis for grouped symbolic pretrain experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from tools.analyze_symbolic_pretrain import FAMILY_LABELS
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
from tools.train_symbolic_pretrain_grouped import (
    GroupedEncodedSet,
    build_grouped_model,
    build_group_vocab,
    encode_grouped_samples,
    evaluate_model,
    prepare_grouped_samples,
    train_model,
)


def sequence_prob_table(probs, pred_labels, label_names, seq_names, gt_indices, method, export_family_scores=False):
    by_sequence_probs = defaultdict(list)
    by_sequence_votes = defaultdict(list)
    gt = {}
    for prob, pred, seq_name, gt_idx in zip(probs, pred_labels, seq_names, gt_indices):
        by_sequence_probs[seq_name].append(prob)
        by_sequence_votes[seq_name].append(int(pred))
        gt[seq_name] = label_names[int(gt_idx)]
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
            "top3": [[label_names[int(idx)], float(agg[int(idx)])] for idx in ranking[:3]],
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


def predict_probs_grouped(model, encoded: GroupedEncodedSet, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = {
            "state": torch.from_numpy(encoded.state.astype(np.float32)).to(device),
            "dynamics": torch.from_numpy(encoded.dynamics.astype(np.float32)).to(device),
            "context": torch.from_numpy(encoded.context.astype(np.float32)).to(device),
        }
        logits, _ = model(batch)
        return torch.softmax(logits, dim=-1).cpu().numpy()


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
    fusion_type,
    export_family_scores=False,
):
    ft_fraction = subsample_by_fraction(ft_samples, fraction, seed)
    ft_train, ft_holdout = split_train_holdout(ft_fraction, 0.2, seed)
    pre_train, pre_holdout = split_train_holdout(pre_samples, 0.1, seed)

    all_for_vocab = pre_train + pre_holdout + ft_train + ft_holdout + test_samples
    vocabs = {name: build_group_vocab(all_for_vocab, name) for name in ("state", "dynamics", "context")}

    pre_labels = sorted({s["label"] for s in pre_samples})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    ft_labels = sorted({s["label"] for s in ft_train + ft_holdout + test_samples})
    ft_map = {k: i for i, k in enumerate(ft_labels)}

    pre_train_encoded = encode_grouped_samples(pre_train, vocabs, pre_map)
    pre_holdout_encoded = encode_grouped_samples(pre_holdout, vocabs, pre_map)
    ft_train_encoded = encode_grouped_samples(ft_train, vocabs, ft_map)
    ft_holdout_encoded = encode_grouped_samples(ft_holdout, vocabs, ft_map)
    test_encoded = encode_grouped_samples(test_samples, vocabs, ft_map)

    dims = {name: getattr(ft_train_encoded, name).shape[1] for name in ("state", "dynamics", "context")}

    scratch = build_grouped_model(dims, hidden_dim, len(ft_labels), fusion_type).to(device)
    train_model(
        scratch,
        ft_train_encoded,
        ft_holdout_encoded,
        ft_labels,
        device,
        finetune_epochs,
        lr,
        weight_decay,
        aggregation,
        class_weights=finetune_class_weights,
    )
    scratch_probs = predict_probs_grouped(scratch, test_encoded, device)
    scratch_pred = scratch_probs.argmax(axis=1)
    _, _, scratch_gates = evaluate_model(scratch, test_encoded, ft_labels, device, aggregation)

    pretrained = build_grouped_model(dims, hidden_dim, len(pre_labels), fusion_type).to(device)
    train_model(
        pretrained,
        pre_train_encoded,
        pre_holdout_encoded,
        pre_labels,
        device,
        pretrain_epochs,
        lr,
        weight_decay,
        aggregation,
    )
    finetune = build_grouped_model(dims, hidden_dim, len(ft_labels), fusion_type).to(device)
    finetune.encoders.load_state_dict(pretrained.encoders.state_dict())
    if fusion_type == "gated":
        finetune.gate.load_state_dict(pretrained.gate.state_dict())
    elif fusion_type == "residual_aux":
        finetune.aux_gate.load_state_dict(pretrained.aux_gate.state_dict())
        finetune.aux_proj.load_state_dict(pretrained.aux_proj.state_dict())
    train_model(
        finetune,
        ft_train_encoded,
        ft_holdout_encoded,
        ft_labels,
        device,
        finetune_epochs,
        lr,
        weight_decay,
        aggregation,
        class_weights=finetune_class_weights,
    )
    ft_probs = predict_probs_grouped(finetune, test_encoded, device)
    ft_pred = ft_probs.argmax(axis=1)
    _, _, pretrained_gates = evaluate_model(finetune, test_encoded, ft_labels, device, aggregation)

    scratch_rows = sequence_prob_table(
        scratch_probs, scratch_pred, ft_labels, test_encoded.seq_names, test_encoded.labels, aggregation, export_family_scores
    )
    pretrained_rows = sequence_prob_table(
        ft_probs, ft_pred, ft_labels, test_encoded.seq_names, test_encoded.labels, aggregation, export_family_scores
    )

    scratch_true = [row["target"] for row in scratch_rows]
    scratch_est = [row["prediction"] for row in scratch_rows]
    pretrained_true = [row["target"] for row in pretrained_rows]
    pretrained_est = [row["prediction"] for row in pretrained_rows]

    return {
        "labels": ft_labels,
        "scratch": {
            "window_accuracy": float((scratch_pred == test_encoded.labels).mean()),
            "sequence_rows": scratch_rows,
            "confusion_matrix": confusion_matrix(scratch_true, scratch_est, labels=ft_labels).tolist(),
            "gate_means_test": scratch_gates,
        },
        "pretrained": {
            "window_accuracy": float((ft_pred == test_encoded.labels).mean()),
            "sequence_rows": pretrained_rows,
            "confusion_matrix": confusion_matrix(pretrained_true, pretrained_est, labels=ft_labels).tolist(),
            "gate_means_test": pretrained_gates,
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
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_pretrain_grouped_analysis.json"))
    parser.add_argument("--export-family-scores", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
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

    ft_fraction_preview = subsample_by_fraction(ft_samples, args.fraction, args.seed)
    ft_train_preview, ft_holdout_preview = split_train_holdout(ft_fraction_preview, 0.2, args.seed)
    ft_labels_preview = sorted({s["label"] for s in ft_train_preview + ft_holdout_preview + test_samples})
    finetune_class_weights = build_finetune_class_weights(ft_labels_preview, boost_map)

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
        args.fusion_type,
        args.export_family_scores,
    )

    payload = {
        "mode": args.mode,
        "fraction": args.fraction,
        "seed": args.seed,
        "device": str(device),
        "fusion_type": args.fusion_type,
        "weights": weights,
        "boost_map": boost_map,
        "window_size": args.window_size,
        "stride": args.stride,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "export_family_scores": args.export_family_scores,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    print(f"scratch_window={details['scratch']['window_accuracy']:.4f}")
    print(f"pretrained_window={details['pretrained']['window_accuracy']:.4f}")
    print(f"scratch_seq={np.mean([row['correct'] for row in details['scratch']['sequence_rows']]):.4f}")
    print(f"pretrained_seq={np.mean([row['correct'] for row in details['pretrained']['sequence_rows']]):.4f}")
    if details["pretrained"]["gate_means_test"]:
        print(f"pretrained_gates={details['pretrained']['gate_means_test']}")


if __name__ == "__main__":
    main()
