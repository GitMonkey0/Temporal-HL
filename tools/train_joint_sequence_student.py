#!/usr/bin/env python3
"""Train sequence encoders on 3D-joint windows with optional pretraining."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.train_joint_student import (
    load_json,
    make_window_joint_samples,
    set_seed,
)
from tools.train_symbolic_classifier import canonical_label, overlap_labels, subsample_by_fraction
from tools.train_symbolic_pretrain import (
    aggregate_sequence_predictions,
    resolve_split_window_args,
    split_train_holdout,
)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class JointSequenceEncoder(nn.Module):
    def __init__(self, frame_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out, _ = self.encoder(x)
        return out.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_sequence(x))


def train_model(
    model: JointSequenceEncoder,
    train_x: np.ndarray,
    train_y: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    loader = DataLoader(SequenceDataset(train_x, train_y), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    patience = 20
    left = patience
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        holdout_acc = eval_window_acc(model, holdout_x, holdout_y, device)
        if holdout_acc > best_acc:
            best_acc = holdout_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def eval_window_acc(model, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32)).to(device))
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
    return float((pred == y).mean())


def predict_probs(model, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32)).to(device))
        return torch.softmax(logits, dim=-1).cpu().numpy()


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
    scratch_win = accuracy_score(y_test, scratch_pred)
    st, sp = aggregate_sequence_predictions(scratch_probs, scratch_pred, ft_labels, test_samples, aggregation)
    scratch_seq = accuracy_score(st, sp)

    pretrained = JointSequenceEncoder(frame_dim, hidden_dim, len(pre_labels)).to(device)
    train_model(
        pretrained, x_pre_train, y_pre_train, x_pre_hold, y_pre_hold,
        device, pretrain_epochs, lr, weight_decay
    )
    encoder_state = {k: v.detach().cpu().clone() for k, v in pretrained.input_proj.state_dict().items()}
    rnn_state = {k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()}
    finetune = JointSequenceEncoder(frame_dim, hidden_dim, len(ft_labels)).to(device)
    finetune.input_proj.load_state_dict(encoder_state)
    finetune.encoder.load_state_dict(rnn_state)
    train_model(
        finetune, x_ft_train, y_ft_train, x_ft_hold, y_ft_hold,
        device, finetune_epochs, lr, weight_decay
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
            "avg_finetune_train_windows": statistics.mean(x["num_finetune_train_windows"] for x in items),
            "scratch_window_accuracy_mean": statistics.mean(x["scratch_window_accuracy"] for x in items),
            "scratch_sequence_accuracy_mean": statistics.mean(x["scratch_sequence_accuracy"] for x in items),
            "pretrained_window_accuracy_mean": statistics.mean(x["pretrained_window_accuracy"] for x in items),
            "pretrained_sequence_accuracy_mean": statistics.mean(x["pretrained_sequence_accuracy"] for x in items),
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
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/joint_sequence_student.json"))
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
        "device": str(device),
        "fractions": args.fractions,
        "seeds": args.seeds,
        "window_span_units": args.window_span_units,
        "window_step_units": args.window_step_units,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    for fraction, item in payload["summary"].items():
        print(
            f"fraction={fraction} "
            f"scratch_seq={item['scratch_sequence_accuracy_mean']:.4f} "
            f"pretrained_seq={item['pretrained_sequence_accuracy_mean']:.4f} "
            f"scratch_win={item['scratch_window_accuracy_mean']:.4f} "
            f"pretrained_win={item['pretrained_window_accuracy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
