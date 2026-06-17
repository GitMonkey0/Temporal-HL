#!/usr/bin/env python3
"""Lightweight learned-token control baseline on joint windows.

This script fits a MiniBatchKMeans codebook on normalized 3D-joint frame
features, converts each joint window into a discrete token sequence, and uses
simple token statistics as input to the same pretrain -> finetune classification
protocol used in the symbolic experiments.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

from tools.train_joint_student import (
    load_json,
    make_window_joint_samples,
    set_seed,
)
from tools.train_symbolic_classifier import canonical_label, overlap_labels, subsample_by_fraction
from tools.train_symbolic_pretrain import (
    MLPClassifier,
    aggregate_sequence_predictions,
    predict_probs,
    resolve_split_window_args,
    split_train_holdout,
    train_model,
)


def fit_codebook(
    train_samples: list[dict[str, object]],
    num_tokens: int,
    max_frames: int,
    seed: int,
) -> MiniBatchKMeans:
    frame_bank = np.concatenate([sample["frame_features"] for sample in train_samples], axis=0)
    if len(frame_bank) > max_frames:
        rng = np.random.default_rng(seed)
        choice = rng.choice(len(frame_bank), size=max_frames, replace=False)
        frame_bank = frame_bank[choice]
    kmeans = MiniBatchKMeans(
        n_clusters=num_tokens,
        random_state=seed,
        batch_size=min(4096, max(512, num_tokens * 32)),
        n_init="auto",
        max_iter=200,
    )
    kmeans.fit(frame_bank)
    return kmeans


def run_lengths(items: list[int]) -> list[int]:
    runs = []
    prev = None
    count = 0
    for item in items:
        if prev is None or item == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
        prev = item
    if count > 0:
        runs.append(count)
    return runs


def encode_window_features(
    frame_features: np.ndarray,
    codebook: MiniBatchKMeans,
    num_tokens: int,
) -> dict[str, float]:
    token_ids = codebook.predict(frame_features).tolist()
    hist = Counter(token_ids)
    features: dict[str, float] = {}
    length = max(len(token_ids), 1)
    for token_id, count in hist.items():
        features[f"tok::{token_id}"] = count / length

    if len(token_ids) > 1:
        bigrams = Counter(zip(token_ids[:-1], token_ids[1:]))
        denom = max(len(token_ids) - 1, 1)
        for (left, right), count in bigrams.items():
            features[f"bigram::{left}->{right}"] = count / denom
        transition_count = sum(int(a != b) for a, b in zip(token_ids[:-1], token_ids[1:]))
    else:
        transition_count = 0

    runs = run_lengths(token_ids)
    features["tempo::token_transition_rate"] = transition_count / max(len(token_ids) - 1, 1)
    features["tempo::token_run_mean"] = statistics.mean(runs) / length if runs else 0.0
    features["tempo::token_run_max"] = max(runs) / length if runs else 0.0
    features["tempo::token_uniqueness"] = len(hist) / max(min(length, num_tokens), 1)
    return features


def build_vocab(samples: list[dict[str, object]]) -> list[str]:
    vocab = set()
    for sample in samples:
        vocab.update(sample["features"].keys())
    return sorted(vocab)


def vectorize(samples: list[dict[str, object]], vocab: list[str]) -> np.ndarray:
    feat_index = {key: idx for idx, key in enumerate(vocab)}
    mat = np.zeros((len(samples), len(vocab)), dtype=np.float32)
    for row_idx, sample in enumerate(samples):
        for key, value in sample["features"].items():
            if key in feat_index:
                mat[row_idx, feat_index[key]] = value
    return mat


def encode_samples(
    samples: list[dict[str, object]],
    codebook: MiniBatchKMeans,
    num_tokens: int,
) -> list[dict[str, object]]:
    encoded = []
    for sample in samples:
        encoded.append(
            {
                "sample_id": sample["sample_id"],
                "seq_name": sample["seq_name"],
                "label": sample["label"],
                "features": encode_window_features(sample["frame_features"], codebook, num_tokens),
            }
        )
    return encoded


def run_once(
    pre_samples,
    ft_samples,
    test_samples,
    fraction,
    seed,
    num_tokens,
    max_codebook_frames,
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
    pre_train, pre_holdout = split_train_holdout(pre_samples, 0.1, seed)

    codebook = fit_codebook(pre_train, num_tokens, max_codebook_frames, seed)

    pre_train_enc = encode_samples(pre_train, codebook, num_tokens)
    pre_hold_enc = encode_samples(pre_holdout, codebook, num_tokens)
    ft_train_enc = encode_samples(ft_train, codebook, num_tokens)
    ft_hold_enc = encode_samples(ft_holdout, codebook, num_tokens)
    test_enc = encode_samples(test_samples, codebook, num_tokens)

    vocab = build_vocab(pre_train_enc + pre_hold_enc + ft_train_enc + ft_hold_enc + test_enc)

    pre_labels = sorted({s["label"] for s in pre_train_enc + pre_hold_enc})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    ft_labels = sorted({s["label"] for s in ft_train_enc + ft_hold_enc + test_enc})
    ft_map = {k: i for i, k in enumerate(ft_labels)}

    x_pre_train = vectorize(pre_train_enc, vocab)
    y_pre_train = np.asarray([pre_map[s["label"]] for s in pre_train_enc], dtype=np.int64)
    x_pre_hold = vectorize(pre_hold_enc, vocab)
    y_pre_hold = np.asarray([pre_map[s["label"]] for s in pre_hold_enc], dtype=np.int64)

    x_ft_train = vectorize(ft_train_enc, vocab)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train_enc], dtype=np.int64)
    x_ft_hold = vectorize(ft_hold_enc, vocab)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_hold_enc], dtype=np.int64)
    x_test = vectorize(test_enc, vocab)
    y_test = np.asarray([ft_map[s["label"]] for s in test_enc], dtype=np.int64)

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
        scratch_probs, scratch_pred, ft_labels, test_enc, aggregation
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
        key: value.detach().cpu().clone()
        for key, value in pretrained.encoder.state_dict().items()
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
    tt, tp = aggregate_sequence_predictions(ft_probs, ft_pred, ft_labels, test_enc, aggregation)
    ft_seq = accuracy_score(tt, tp)

    return {
        "fraction": fraction,
        "seed": seed,
        "num_tokens": num_tokens,
        "feature_dim": len(vocab),
        "num_finetune_train_windows": len(ft_train_enc),
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
    parser.add_argument("--train-pretrain-joint-json", type=Path, required=True)
    parser.add_argument("--train-finetune-joint-json", type=Path, required=True)
    parser.add_argument("--test-joint-json", type=Path, required=True)
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
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--max-codebook-frames", type=int, default=100000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--aggregation",
        choices=["mean_prob", "mean_log_prob", "vote"],
        default="mean_log_prob",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/joint_token_baseline.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_sym = load_json(args.train_pretrain_json)
    ft_sym = load_json(args.train_finetune_json)
    test_sym = load_json(args.test_json)
    pre_joint = load_json(args.train_pretrain_joint_json)
    ft_joint = load_json(args.train_finetune_joint_json)
    test_joint = load_json(args.test_joint_json)
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
        pre_joint,
        {canonical_label(s["seq_name"]) for s in pre_sym["sequences"]},
        args.window_size,
        args.stride,
        pre_span,
        pre_step,
    )
    ft_samples = make_window_joint_samples(
        ft_sym,
        ft_joint,
        allowed,
        args.window_size,
        args.stride,
        ft_span,
        ft_step,
    )
    test_samples = make_window_joint_samples(
        test_sym,
        test_joint,
        allowed,
        args.window_size,
        args.stride,
        test_span,
        test_step,
    )

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
                    args.num_tokens,
                    args.max_codebook_frames,
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
        "num_tokens": args.num_tokens,
        "max_codebook_frames": args.max_codebook_frames,
        "window_size": args.window_size,
        "stride": args.stride,
        "window_span_units": args.window_span_units,
        "window_step_units": args.window_step_units,
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
