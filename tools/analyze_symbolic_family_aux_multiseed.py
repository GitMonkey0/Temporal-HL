#!/usr/bin/env python3
"""Train-time family-auxiliary symbolic baseline with multi-seed analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.analyze_symbolic_pretrain import FAMILY_LABELS, sequence_prob_table
from tools.train_symbolic_pretrain import (
    MLPClassifier,
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
)
from tools.train_symbolic_classifier import (
    build_vocabulary,
    canonical_label,
    subsample_by_fraction,
    vectorize,
)


OTHER_FAMILY = "__other__"
FAMILY_AUX_LABELS = FAMILY_LABELS + [OTHER_FAMILY]


class DenseMultiTaskDataset(Dataset):
    def __init__(self, x: np.ndarray, y_main: np.ndarray, y_family: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y_main = torch.from_numpy(y_main.astype(np.int64))
        self.y_family = torch.from_numpy(y_family.astype(np.int64))

    def __len__(self):
        return len(self.y_main)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_main[idx], self.y_family[idx]


class FamilyAuxMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_main_classes: int, num_family_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(hidden_dim, num_main_classes)
        self.family_head = nn.Linear(hidden_dim, num_family_classes)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        return self.head(feat), self.family_head(feat)


def family_target_index(label: str, family_map: dict[str, int]) -> int:
    return family_map[label] if label in family_map else family_map[OTHER_FAMILY]


def eval_window_acc(model, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x.astype(np.float32)).to(device))
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
    return float((pred == y).mean())


def predict_main_probs(model, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x.astype(np.float32)).to(device))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def train_family_aux_model(
    model: FamilyAuxMLPClassifier,
    train_x: np.ndarray,
    train_y_main: np.ndarray,
    train_y_family: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y_main: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    family_loss_weight: float,
    class_weights: np.ndarray | None = None,
):
    loader = DataLoader(
        DenseMultiTaskDataset(train_x, train_y_main, train_y_family),
        batch_size=128,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(device)
    main_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    family_criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    patience = 20
    left = patience
    for _ in range(epochs):
        model.train()
        for xb, yb_main, yb_family in loader:
            xb = xb.to(device)
            yb_main = yb_main.to(device)
            yb_family = yb_family.to(device)
            opt.zero_grad()
            logits_main, logits_family = model(xb)
            loss = main_criterion(logits_main, yb_main) + family_loss_weight * family_criterion(logits_family, yb_family)
            loss.backward()
            opt.step()
        holdout_acc = eval_window_acc(model, holdout_x, holdout_y_main, device)
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
    family_loss_weight,
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
    family_map = {label: idx for idx, label in enumerate(FAMILY_AUX_LABELS)}
    x_ft_train = vectorize(ft_train, vocab)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train], dtype=np.int64)
    y_family_train = np.asarray([family_target_index(s["label"], family_map) for s in ft_train], dtype=np.int64)
    x_ft_hold = vectorize(ft_holdout, vocab)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_holdout], dtype=np.int64)
    y_family_hold = np.asarray([family_target_index(s["label"], family_map) for s in ft_holdout], dtype=np.int64)
    x_test = vectorize(test_samples, vocab)
    y_test = np.asarray([ft_map[s["label"]] for s in test_samples], dtype=np.int64)

    pretrained = MLPClassifier(len(vocab), hidden_dim, len(pre_labels)).to(device)
    from tools.train_symbolic_pretrain import train_model  # local import to avoid circular at module load

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
    encoder_state = {k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()}
    finetune = FamilyAuxMLPClassifier(len(vocab), hidden_dim, len(ft_labels), len(FAMILY_AUX_LABELS)).to(device)
    finetune.encoder.load_state_dict(encoder_state)
    train_family_aux_model(
        finetune,
        x_ft_train,
        y_ft_train,
        y_family_train,
        x_ft_hold,
        y_ft_hold,
        device,
        finetune_epochs,
        lr,
        weight_decay,
        family_loss_weight,
        class_weights=finetune_class_weights,
    )
    ft_probs = predict_main_probs(finetune, x_test, device)
    ft_pred = ft_probs.argmax(axis=1)
    pretrained_seq_rows = sequence_prob_table(
        ft_probs,
        ft_pred,
        ft_labels,
        test_samples,
        aggregation,
        export_family_scores=False,
    )
    return {
        "labels": ft_labels,
        "pretrained": {
            "window_accuracy": float((ft_pred == y_test).mean()),
            "sequence_rows": pretrained_seq_rows,
        },
    }


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
    parser.add_argument(
        "--boost-label",
        action="append",
        default=[],
        help="Optional finetune class boost as LABEL:WEIGHT. Can be repeated.",
    )
    parser.add_argument("--family-loss-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_family_aux_analysis.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_data = load_json(args.train_pretrain_json)
    ft_data = load_json(args.train_finetune_json)
    test_data = load_json(args.test_json)
    allowed = overlap_labels(ft_data, test_data)
    weights = {"state": 1.0, "transition": 0.5, "hand_motion": 0.5, "interaction": 0.2, "tempo": 0.2, "event": 0.0} if args.mode == "temporal" else {"state": 1.0, "transition": 0.0, "hand_motion": 0.0, "interaction": 0.0, "tempo": 0.0, "event": 0.0}
    boost_map = parse_boost_labels(args.boost_label)
    pre_span, pre_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.pretrain_window_span_units, args.pretrain_window_step_units)
    ft_span, ft_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.finetune_window_span_units, args.finetune_window_step_units)
    test_span, test_step = resolve_split_window_args(args.window_span_units, args.window_step_units, args.test_window_span_units, args.test_window_step_units)

    pre_samples = make_window_channel_samples(
        pre_data,
        {canonical_label(s["seq_name"]) for s in pre_data["sequences"]},
        args.mode,
        args.window_size,
        args.stride,
        args.wrist_features,
        args.duration_features,
        False,
        False,
        False,
        False,
        False,
        pre_span,
        pre_step,
    )
    ft_samples = make_window_channel_samples(
        ft_data,
        allowed,
        args.mode,
        args.window_size,
        args.stride,
        args.wrist_features,
        args.duration_features,
        False,
        False,
        False,
        False,
        False,
        ft_span,
        ft_step,
    )
    test_samples = make_window_channel_samples(
        test_data,
        allowed,
        args.mode,
        args.window_size,
        args.stride,
        args.wrist_features,
        args.duration_features,
        False,
        False,
        False,
        False,
        False,
        test_span,
        test_step,
    )
    pre_samples = merge_sample_features(pre_samples, weights)
    ft_samples = merge_sample_features(ft_samples, weights)
    test_samples = merge_sample_features(test_samples, weights)

    error_counter = Counter()
    per_seq = defaultdict(list)
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
            args.family_loss_weight,
        )
        rows = details["pretrained"]["sequence_rows"]
        run = {
            "seed": seed,
            "pretrained_sequence_accuracy": float(np.mean([r["correct"] for r in rows])),
            "pretrained_errors": [],
        }
        for row in rows:
            per_seq[row["seq_name"]].append({"seed": seed, "correct": row["correct"], "prediction": row["prediction"]})
            if not row["correct"]:
                key = (row["target"], row["prediction"])
                error_counter[key] += 1
                run["pretrained_errors"].append(row)
        runs.append(run)

    seq_summary = {}
    for seq_name in sorted(per_seq):
        seq_summary[seq_name] = {
            "pretrained_correct_runs": sum(item["correct"] for item in per_seq[seq_name]),
            "pretrained_predictions": per_seq[seq_name],
        }

    payload = {
        "mode": args.mode,
        "fraction": args.fraction,
        "seeds": args.seeds,
        "family_loss_weight": args.family_loss_weight,
        "wrist_features": args.wrist_features,
        "duration_features": args.duration_features,
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
        "pretrained_error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in error_counter.most_common()
        ],
        "sequence_consistency": seq_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    print("pretrained_error_counts", payload["pretrained_error_counts"][:10])


if __name__ == "__main__":
    main()
