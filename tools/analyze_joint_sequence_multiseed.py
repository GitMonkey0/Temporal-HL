#!/usr/bin/env python3
"""Aggregate multi-seed sequence errors for joint sequence experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.train_joint_sequence_student import (
    JointSequenceEncoder,
    predict_probs,
    train_model,
)
from tools.train_joint_student import load_json, make_window_joint_samples, set_seed
from tools.train_symbolic_classifier import canonical_label, overlap_labels, subsample_by_fraction
from tools.train_symbolic_pretrain import (
    aggregate_sequence_predictions,
    resolve_split_window_args,
    split_train_holdout,
)


def sequence_prob_table(probs, pred_labels, label_names, samples, method):
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
        rows.append(
            {
                "seq_name": seq_name,
                "target": gt[seq_name],
                "prediction": label_names[int(ranking[0])],
                "top3": [
                    [label_names[int(idx)], float(agg[int(idx)])] for idx in ranking[:3]
                ],
                "correct": label_names[int(ranking[0])] == gt[seq_name],
            }
        )
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
):
    ft_fraction = subsample_by_fraction(ft_samples, fraction, seed)
    ft_train, ft_holdout = split_train_holdout(ft_fraction, 0.2, seed)

    pre_labels = sorted({s["label"] for s in pre_samples})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    pre_train, pre_holdout = split_train_holdout(pre_samples, 0.1, seed)
    x_pre_train = np.stack([s["frame_features"] for s in pre_train], axis=0)
    y_pre_train = np.asarray([pre_map[s["label"]] for s in pre_train], dtype=np.int64)
    x_pre_hold = np.stack([s["frame_features"] for s in pre_holdout], axis=0)
    y_pre_hold = np.asarray([pre_map[s["label"]] for s in pre_holdout], dtype=np.int64)

    ft_labels = sorted({s["label"] for s in ft_train + ft_holdout + test_samples})
    ft_map = {k: i for i, k in enumerate(ft_labels)}
    x_ft_train = np.stack([s["frame_features"] for s in ft_train], axis=0)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train], dtype=np.int64)
    x_ft_hold = np.stack([s["frame_features"] for s in ft_holdout], axis=0)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_holdout], dtype=np.int64)
    x_test = np.stack([s["frame_features"] for s in test_samples], axis=0)
    y_test = np.asarray([ft_map[s["label"]] for s in test_samples], dtype=np.int64)

    frame_dim = x_ft_train.shape[-1]

    scratch = JointSequenceEncoder(frame_dim, hidden_dim, len(ft_labels)).to(device)
    train_model(
        scratch, x_ft_train, y_ft_train, x_ft_hold, y_ft_hold,
        device, finetune_epochs, lr, weight_decay
    )
    scratch_probs = predict_probs(scratch, x_test, device)
    scratch_pred = scratch_probs.argmax(axis=1)

    pretrained = JointSequenceEncoder(frame_dim, hidden_dim, len(pre_labels)).to(device)
    train_model(
        pretrained, x_pre_train, y_pre_train, x_pre_hold, y_pre_hold,
        device, pretrain_epochs, lr, weight_decay
    )
    proj_state = {k: v.detach().cpu().clone() for k, v in pretrained.input_proj.state_dict().items()}
    enc_state = {k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()}
    finetune = JointSequenceEncoder(frame_dim, hidden_dim, len(ft_labels)).to(device)
    finetune.input_proj.load_state_dict(proj_state)
    finetune.encoder.load_state_dict(enc_state)
    train_model(
        finetune, x_ft_train, y_ft_train, x_ft_hold, y_ft_hold,
        device, finetune_epochs, lr, weight_decay
    )
    ft_probs = predict_probs(finetune, x_test, device)
    ft_pred = ft_probs.argmax(axis=1)

    return {
        "scratch_window_accuracy": float((scratch_pred == y_test).mean()),
        "pretrained_window_accuracy": float((ft_pred == y_test).mean()),
        "scratch_sequence_rows": sequence_prob_table(scratch_probs, scratch_pred, ft_labels, test_samples, aggregation),
        "pretrained_sequence_rows": sequence_prob_table(ft_probs, ft_pred, ft_labels, test_samples, aggregation),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pretrain-json", type=Path, required=True)
    parser.add_argument("--train-finetune-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--train-pretrain-joint-json", type=Path, required=True)
    parser.add_argument("--train-finetune-joint-json", type=Path, required=True)
    parser.add_argument("--test-joint-json", type=Path, required=True)
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
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/joint_sequence_multiseed_analysis.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_sym = load_json(args.train_pretrain_json)
    ft_sym = load_json(args.train_finetune_json)
    test_sym = load_json(args.test_json)
    pre_joint_json = load_json(args.train_pretrain_joint_json)
    ft_joint_json = load_json(args.train_finetune_joint_json)
    test_joint_json = load_json(args.test_joint_json)
    allowed = overlap_labels(ft_sym, test_sym)

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

    pre_samples = make_window_joint_samples(
        pre_sym,
        pre_joint_json,
        {canonical_label(s["seq_name"]) for s in pre_sym["sequences"]},
        args.window_size,
        args.stride,
        pre_span,
        pre_step,
    )
    ft_samples = make_window_joint_samples(
        ft_sym, ft_joint_json, allowed, args.window_size, args.stride, ft_span, ft_step
    )
    test_samples = make_window_joint_samples(
        test_sym, test_joint_json, allowed, args.window_size, args.stride, test_span, test_step
    )

    scratch_error_counter = Counter()
    pretrained_error_counter = Counter()
    scratch_per_seq = defaultdict(list)
    pretrained_per_seq = defaultdict(list)
    runs = []

    for seed in args.seeds:
        set_seed(seed)
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
        )
        run = {
            "seed": seed,
            "scratch_sequence_accuracy": float(np.mean([r["correct"] for r in details["scratch_sequence_rows"]])),
            "pretrained_sequence_accuracy": float(np.mean([r["correct"] for r in details["pretrained_sequence_rows"]])),
            "scratch_errors": [],
            "pretrained_errors": [],
        }
        for mode_name, rows, counter, per_seq in [
            ("scratch", details["scratch_sequence_rows"], scratch_error_counter, scratch_per_seq),
            ("pretrained", details["pretrained_sequence_rows"], pretrained_error_counter, pretrained_per_seq),
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
            "scratch_predictions": scratch_per_seq[seq_name],
            "pretrained_predictions": pretrained_per_seq[seq_name],
        }

    payload = {
        "fraction": args.fraction,
        "seeds": args.seeds,
        "window_span_units": args.window_span_units,
        "window_step_units": args.window_step_units,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "runs": runs,
        "scratch_error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in scratch_error_counter.most_common()
        ],
        "pretrained_error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in pretrained_error_counter.most_common()
        ],
        "sequence_consistency": seq_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    print("scratch_error_counts", payload["scratch_error_counts"][:10])
    print("pretrained_error_counts", payload["pretrained_error_counts"][:10])


if __name__ == "__main__":
    main()
