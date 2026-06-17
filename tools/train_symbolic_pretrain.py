#!/usr/bin/env python3
"""Supervised pretrain + finetune for symbolic HL features."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.train_symbolic_classifier import (
    build_vocabulary,
    canonical_label,
    merge_channels,
    overlap_labels,
    subsample_by_fraction,
    vectorize,
)
from tools.train_symbolic_torch import extract_channel_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def aggregate_sequence_predictions(
    probs: np.ndarray,
    pred_labels: np.ndarray,
    labels: list[str],
    samples: list[dict[str, object]],
    method: str,
):
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
            agg = np.mean(np.log(np.clip(probs_arr, 1e-8, 1.0)), axis=0)
            pred_idx = int(np.argmax(agg))
        elif method == "vote":
            pred_idx = max(set(by_sequence_votes[seq_name]), key=by_sequence_votes[seq_name].count)
        else:
            agg = np.mean(probs_arr, axis=0)
            pred_idx = int(np.argmax(agg))
        y_true.append(gt[seq_name])
        y_pred.append(labels[pred_idx])
    return y_true, y_pred


class DenseDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
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
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def train_model(
    model: MLPClassifier,
    train_x: np.ndarray,
    train_y: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: np.ndarray | None = None,
):
    loader = DataLoader(DenseDataset(train_x, train_y), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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


def split_train_holdout(samples, holdout_fraction, seed):
    grouped = defaultdict(list)
    for s in samples:
        grouped[s["label"]].append(s)
    rng = random.Random(seed)
    train, holdout = [], []
    for _, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        n = max(1, int(round(len(items) * holdout_fraction)))
        holdout.extend(items[:n])
        train.extend(items[n:])
    return train, holdout


def merge_sample_features(samples, weights):
    merged = []
    for sample in samples:
        sample = dict(sample)
        sample["features"] = merge_channels(
            sample["channels"],
            state_weight=weights["state"],
            transition_weight=weights["transition"],
            hand_motion_weight=weights["hand_motion"],
            interaction_weight=weights["interaction"],
            tempo_weight=weights["tempo"],
            event_weight=weights.get("event", 0.0),
        )
        merged.append(sample)
    return merged


def make_window_channel_samples(
    dataset: dict[str, object],
    allowed_labels: set[str],
    mode: str,
    window_size: int,
    stride: int,
    include_wrist_features: bool = False,
    include_duration_features: bool = False,
    include_event_features: bool = False,
    include_coordination_features: bool = False,
    include_occlusion_features: bool = False,
    include_finger_detail_features: bool = False,
    include_finger_shape_features: bool = False,
    window_span_units: int | None = None,
    window_step_units: int | None = None,
):
    include_temporal = mode == "temporal"
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
                    "channels": extract_channel_features(
                        window,
                    include_temporal,
                    include_wrist_features,
                    include_duration_features,
                    include_event_features,
                    include_coordination_features,
                    include_occlusion_features,
                    include_finger_detail_features,
                    include_finger_shape_features,
                    ),
                }
            )
    return samples


def run_once(
    train_pre_samples,
    train_ft_samples,
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
):
    ft_fraction = subsample_by_fraction(train_ft_samples, fraction, seed)
    ft_train, ft_holdout = split_train_holdout(ft_fraction, 0.2, seed)

    all_for_vocab = train_pre_samples + ft_train + ft_holdout + test_samples
    vocab = build_vocabulary(all_for_vocab)

    # Pretrain split labels
    pre_labels = sorted({s["label"] for s in train_pre_samples})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    x_pre = vectorize(train_pre_samples, vocab)
    y_pre = np.asarray([pre_map[s["label"]] for s in train_pre_samples], dtype=np.int64)
    pre_train, pre_holdout = split_train_holdout(train_pre_samples, 0.1, seed)
    x_pre_train = vectorize(pre_train, vocab)
    y_pre_train = np.asarray([pre_map[s["label"]] for s in pre_train], dtype=np.int64)
    x_pre_hold = vectorize(pre_holdout, vocab)
    y_pre_hold = np.asarray([pre_map[s["label"]] for s in pre_holdout], dtype=np.int64)

    # Finetune labels
    ft_labels = sorted({s["label"] for s in ft_train + ft_holdout + test_samples})
    ft_map = {k: i for i, k in enumerate(ft_labels)}
    x_ft_train = vectorize(ft_train, vocab)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train], dtype=np.int64)
    x_ft_hold = vectorize(ft_holdout, vocab)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_holdout], dtype=np.int64)
    x_test = vectorize(test_samples, vocab)
    y_test = np.asarray([ft_map[s["label"]] for s in test_samples], dtype=np.int64)

    # from scratch
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
        class_weights=finetune_class_weights,
    )
    scratch_probs = predict_probs(scratch, x_test, device)
    scratch_pred = scratch_probs.argmax(axis=1)
    scratch_win = accuracy_score(y_test, scratch_pred)
    st, sp = aggregate_sequence_predictions(scratch_probs, scratch_pred, ft_labels, test_samples, aggregation)
    scratch_seq = accuracy_score(st, sp)

    # pretrain then finetune
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
    encoder_state = {k: v.detach().cpu().clone() for k, v in pretrained.encoder.state_dict().items()}
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
        class_weights=finetune_class_weights,
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


def parse_boost_labels(boost_args: list[str]) -> dict[str, float]:
    boosts = {}
    for item in boost_args:
        if ":" not in item:
            raise ValueError(f"Invalid --boost-label value: {item}")
        label, value = item.rsplit(":", 1)
        boosts[label] = float(value)
    return boosts


def build_finetune_class_weights(label_names: list[str], boosts: dict[str, float]) -> np.ndarray:
    weights = np.ones(len(label_names), dtype=np.float32)
    for idx, label in enumerate(label_names):
        if label in boosts:
            weights[idx] = boosts[label]
    return weights


def resolve_split_window_args(
    default_span: int | None,
    default_step: int | None,
    split_span: int | None,
    split_step: int | None,
) -> tuple[int | None, int | None]:
    return (
        default_span if split_span is None else split_span,
        default_step if split_step is None else split_step,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pretrain-json", type=Path, required=True)
    parser.add_argument("--train-finetune-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["state", "temporal"], default="temporal")
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
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument(
        "--boost-label",
        action="append",
        default=[],
        help="Optional finetune class boost as LABEL:WEIGHT. Can be repeated.",
    )
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_pretrain_summary.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_data = load_json(args.train_pretrain_json)
    ft_data = load_json(args.train_finetune_json)
    test_data = load_json(args.test_json)
    allowed = overlap_labels(ft_data, test_data)
    include_temporal = args.mode == "temporal"
    if include_temporal:
        weights = {
            "state": 1.0 if args.state_weight is None else args.state_weight,
            "transition": 0.5 if args.transition_weight is None else args.transition_weight,
            "hand_motion": 0.5 if args.hand_motion_weight is None else args.hand_motion_weight,
            "interaction": 0.2 if args.interaction_weight is None else args.interaction_weight,
            "tempo": 0.2 if args.tempo_weight is None else args.tempo_weight,
            "event": args.event_weight if args.event_features else 0.0,
        }
    else:
        weights = {
            "state": 1.0 if args.state_weight is None else args.state_weight,
            "transition": 0.0,
            "hand_motion": 0.0,
            "interaction": 0.0,
            "tempo": 0.0,
            "event": 0.0,
        }
    boost_map = parse_boost_labels(args.boost_label)
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

    pre_samples = make_window_channel_samples(
        pre_data,
        {canonical_label(s["seq_name"]) for s in pre_data["sequences"]},
        args.mode,
        args.window_size,
        args.stride,
        args.wrist_features,
        args.duration_features,
        args.event_features,
        args.coordination_features,
        args.occlusion_features,
        args.finger_detail_features,
        args.finger_shape_features,
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
        args.event_features,
        args.coordination_features,
        args.occlusion_features,
        args.finger_detail_features,
        args.finger_shape_features,
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
        args.event_features,
        args.coordination_features,
        args.occlusion_features,
        args.finger_detail_features,
        args.finger_shape_features,
        test_span,
        test_step,
    )
    pre_samples = merge_sample_features(pre_samples, weights)
    ft_samples = merge_sample_features(ft_samples, weights)
    test_samples = merge_sample_features(test_samples, weights)

    results = []
    for fraction in args.fractions:
        for seed in args.seeds:
            set_seed(seed)
            ft_fraction_preview = subsample_by_fraction(ft_samples, fraction, seed)
            ft_train_preview, ft_holdout_preview = split_train_holdout(
                ft_fraction_preview, 0.2, seed
            )
            ft_labels_preview = sorted(
                {s["label"] for s in ft_train_preview + ft_holdout_preview + test_samples}
            )
            finetune_class_weights = build_finetune_class_weights(
                ft_labels_preview, boost_map
            )
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
                    finetune_class_weights,
                )
            )
    payload = {
        "mode": args.mode,
        "device": str(device),
        "window_size": args.window_size,
        "stride": args.stride,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "summary": summarize(results),
        "results": results,
        "wrist_features": args.wrist_features,
        "duration_features": args.duration_features,
        "event_features": args.event_features,
        "coordination_features": args.coordination_features,
        "event_weight": args.event_weight,
        "weights": weights,
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
