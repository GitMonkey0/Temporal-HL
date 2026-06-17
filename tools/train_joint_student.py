#!/usr/bin/env python3
"""Train 3D-joint students with optional temporal-HL teacher supervision."""

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
from tools.train_symbolic_pretrain import (
    aggregate_sequence_predictions,
    make_window_channel_samples,
    merge_sample_features,
    resolve_split_window_args,
    split_train_holdout,
)


RIGHT_WRIST = 20
LEFT_WRIST = 41
RIGHT_MIDDLE_BASE = 11
LEFT_MIDDLE_BASE = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def normalize_hand(coords: np.ndarray, wrist_idx: int, middle_idx: int, joint_slice: slice):
    wrist = coords[wrist_idx]
    middle = coords[middle_idx]
    scale = np.linalg.norm(middle - wrist)
    if scale < 1e-6:
        scale = 1.0
    hand = (coords[joint_slice] - wrist) / scale
    return hand


def frame_to_joint_features(world_coord: list[list[float]], joint_valid: list[list[bool]]) -> np.ndarray:
    coords = np.asarray(world_coord, dtype=np.float32)
    valid = np.asarray([float(flag[0]) for flag in joint_valid], dtype=np.float32)

    right = normalize_hand(coords, RIGHT_WRIST, RIGHT_MIDDLE_BASE, slice(0, 21))
    left = normalize_hand(coords, LEFT_WRIST, LEFT_MIDDLE_BASE, slice(21, 42))
    pos = np.concatenate([right.reshape(-1), left.reshape(-1)], axis=0)
    return np.concatenate([pos, valid], axis=0)


def make_joint_lookup(joint_json: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    lookup = {}
    for capture_str, frames in joint_json.items():
        capture = int(capture_str)
        for frame_idx_str, item in frames.items():
            lookup[(capture, int(frame_idx_str))] = item
    return lookup


def make_window_joint_samples(
    sequence_json: dict[str, object],
    joint_json: dict[str, object],
    allowed_labels: set[str],
    window_size: int,
    stride: int,
    window_span_units: int | None = None,
    window_step_units: int | None = None,
) -> list[dict[str, object]]:
    lookup = make_joint_lookup(joint_json)
    samples = []
    for seq_idx, sequence in enumerate(sequence_json["sequences"]):
        label = canonical_label(sequence["seq_name"])
        if label not in allowed_labels:
            continue
        frames = sequence["frames"]
        capture = int(sequence["capture"])
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
            feat_frames = []
            ok = True
            for frame in window:
                item = lookup.get((capture, int(frame["frame_idx"])))
                if item is None:
                    ok = False
                    break
                feat_frames.append(frame_to_joint_features(item["world_coord"], item["joint_valid"]))
            if not ok:
                continue
            if not feat_frames:
                continue
            if len(feat_frames) >= window_size:
                if len(feat_frames) == window_size:
                    sampled_frames = feat_frames
                else:
                    indices = np.linspace(0, len(feat_frames) - 1, num=window_size)
                    sampled_frames = [feat_frames[int(round(idx))] for idx in indices]
            else:
                sampled_frames = list(feat_frames)
                pad_frame = feat_frames[-1]
                sampled_frames.extend([pad_frame.copy() for _ in range(window_size - len(feat_frames))])
            samples.append(
                {
                    "sample_id": f"{seq_idx}:{window_idx}",
                    "seq_name": sequence["seq_name"],
                    "label": label,
                    "frame_features": np.stack(sampled_frames, axis=0),
                    "joint_features": np.concatenate(sampled_frames, axis=0),
                }
            )
    return samples


def attach_teacher_features(
    joint_samples: list[dict[str, object]],
    teacher_samples: list[dict[str, object]],
) -> list[dict[str, object]]:
    teacher_lookup = {sample["sample_id"]: sample["features"] for sample in teacher_samples}
    merged = []
    missing = []
    for sample in joint_samples:
        if sample["sample_id"] not in teacher_lookup:
            missing.append(sample["sample_id"])
            continue
        sample = dict(sample)
        sample["teacher_features"] = teacher_lookup[sample["sample_id"]]
        merged.append(sample)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing teacher features for {len(missing)} joint samples. First ids: {preview}")
    return merged


class JointTeacherDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, teacher: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.teacher = torch.from_numpy(teacher.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.teacher[idx]


class JointStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, teacher_dim: int):
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
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.teacher_head = nn.Linear(hidden_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        return self.classifier(feat), self.teacher_head(feat)


def train_student(
    model: JointStudent,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_teacher: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    holdout_teacher: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    teacher_weight: float,
):
    ds = JointTeacherDataset(train_x, train_y, train_teacher)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    best_acc = -1.0
    best_state = None
    patience = 20
    left = patience
    for _ in range(epochs):
        model.train()
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad()
            logits, teacher_pred = model(xb)
            loss = ce(logits, yb)
            if teacher_weight > 0:
                loss = loss + teacher_weight * mse(teacher_pred, tb)
            loss.backward()
            opt.step()
        holdout_acc, _ = eval_student(
            model, holdout_x, holdout_y, holdout_teacher, device
        )
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


def eval_student(
    model: JointStudent,
    x: np.ndarray,
    y: np.ndarray,
    teacher: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x.astype(np.float32)).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
    return float((pred == y).mean()), probs


def summarize(results):
    by_fraction = defaultdict(list)
    for result in results:
        by_fraction[result["fraction"]].append(result)
    out = {}
    for fraction, items in sorted(by_fraction.items()):
        out[str(fraction)] = {
            "num_runs": len(items),
            "avg_train_windows": statistics.mean(item["num_train_windows"] for item in items),
            "baseline_window_accuracy_mean": statistics.mean(item["baseline_window_accuracy"] for item in items),
            "baseline_sequence_accuracy_mean": statistics.mean(item["baseline_sequence_accuracy"] for item in items),
            "teacher_window_accuracy_mean": statistics.mean(item["teacher_window_accuracy"] for item in items),
            "teacher_sequence_accuracy_mean": statistics.mean(item["teacher_sequence_accuracy"] for item in items),
        }
    return out


def run_once(
    pre_joint,
    ft_joint,
    test_joint,
    fraction,
    seed,
    hidden_dim,
    pretrain_epochs,
    finetune_epochs,
    lr,
    weight_decay,
    pretrain_teacher_weight,
    finetune_teacher_weight,
    device,
    aggregation,
):
    ft_fraction = subsample_by_fraction(ft_joint, fraction, seed)
    ft_train, ft_hold = split_train_holdout(ft_fraction, 0.2, seed)
    all_teacher = pre_joint + ft_train + ft_hold + test_joint
    teacher_vocab = build_vocabulary(
        [{"features": s["teacher_features"]} for s in all_teacher]
    )

    pre_labels = sorted({s["label"] for s in pre_joint})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    pre_train, pre_hold = split_train_holdout(pre_joint, 0.1, seed)
    x_pre_train = np.stack([s["joint_features"] for s in pre_train], axis=0)
    y_pre_train = np.asarray([pre_map[s["label"]] for s in pre_train], dtype=np.int64)
    t_pre_train = vectorize(
        [{"features": s["teacher_features"]} for s in pre_train], teacher_vocab
    )
    x_pre_hold = np.stack([s["joint_features"] for s in pre_hold], axis=0)
    y_pre_hold = np.asarray([pre_map[s["label"]] for s in pre_hold], dtype=np.int64)
    t_pre_hold = vectorize(
        [{"features": s["teacher_features"]} for s in pre_hold], teacher_vocab
    )

    ft_labels = sorted({s["label"] for s in ft_train + ft_hold + test_joint})
    ft_map = {k: i for i, k in enumerate(ft_labels)}
    x_ft_train = np.stack([s["joint_features"] for s in ft_train], axis=0)
    y_ft_train = np.asarray([ft_map[s["label"]] for s in ft_train], dtype=np.int64)
    t_ft_train = vectorize(
        [{"features": s["teacher_features"]} for s in ft_train], teacher_vocab
    )
    x_ft_hold = np.stack([s["joint_features"] for s in ft_hold], axis=0)
    y_ft_hold = np.asarray([ft_map[s["label"]] for s in ft_hold], dtype=np.int64)
    t_ft_hold = vectorize(
        [{"features": s["teacher_features"]} for s in ft_hold], teacher_vocab
    )
    x_test = np.stack([s["joint_features"] for s in test_joint], axis=0)
    y_test = np.asarray([ft_map[s["label"]] for s in test_joint], dtype=np.int64)
    t_test = vectorize(
        [{"features": s["teacher_features"]} for s in test_joint], teacher_vocab
    )

    input_dim = x_ft_train.shape[1]
    teacher_dim = t_ft_train.shape[1]

    # baseline
    baseline = JointStudent(input_dim, hidden_dim, len(ft_labels), teacher_dim).to(device)
    train_student(baseline, x_ft_train, y_ft_train, np.zeros_like(t_ft_train), x_ft_hold, y_ft_hold, np.zeros_like(t_ft_hold), device, finetune_epochs, lr, weight_decay, teacher_weight=0.0)
    baseline_win, baseline_probs = eval_student(baseline, x_test, y_test, np.zeros_like(t_test), device)
    baseline_pred = baseline_probs.argmax(axis=1)
    st, sp = aggregate_sequence_predictions(baseline_probs, baseline_pred, ft_labels, test_joint, aggregation)
    baseline_seq = accuracy_score(st, sp)

    # teacher-guided pretrain + finetune
    pre_model = JointStudent(input_dim, hidden_dim, len(pre_labels), teacher_dim).to(device)
    train_student(pre_model, x_pre_train, y_pre_train, t_pre_train, x_pre_hold, y_pre_hold, t_pre_hold, device, pretrain_epochs, lr, weight_decay, teacher_weight=pretrain_teacher_weight)
    encoder_state = {k: v.detach().cpu().clone() for k, v in pre_model.encoder.state_dict().items()}
    student = JointStudent(input_dim, hidden_dim, len(ft_labels), teacher_dim).to(device)
    student.encoder.load_state_dict(encoder_state)
    train_student(student, x_ft_train, y_ft_train, t_ft_train, x_ft_hold, y_ft_hold, t_ft_hold, device, finetune_epochs, lr, weight_decay, teacher_weight=finetune_teacher_weight)
    teacher_win, teacher_probs = eval_student(student, x_test, y_test, t_test, device)
    teacher_pred = teacher_probs.argmax(axis=1)
    tt, tp = aggregate_sequence_predictions(teacher_probs, teacher_pred, ft_labels, test_joint, aggregation)
    teacher_seq = accuracy_score(tt, tp)

    return {
        "fraction": fraction,
        "seed": seed,
        "num_train_windows": len(ft_train),
        "baseline_window_accuracy": float(baseline_win),
        "baseline_sequence_accuracy": float(baseline_seq),
        "teacher_window_accuracy": float(teacher_win),
        "teacher_sequence_accuracy": float(teacher_seq),
    }


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
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--pretrain-teacher-weight", type=float, default=None)
    parser.add_argument("--finetune-teacher-weight", type=float, default=None)
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/joint_student_summary.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_teacher_weight = args.teacher_weight if args.pretrain_teacher_weight is None else args.pretrain_teacher_weight
    finetune_teacher_weight = args.teacher_weight if args.finetune_teacher_weight is None else args.finetune_teacher_weight
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

    pre_joint = make_window_joint_samples(
        pre_sym, pre_joint_json, {canonical_label(s["seq_name"]) for s in pre_sym["sequences"]},
        args.window_size, args.stride, pre_span, pre_step
    )
    ft_joint = make_window_joint_samples(
        ft_sym, ft_joint_json, allowed, args.window_size, args.stride, ft_span, ft_step
    )
    test_joint = make_window_joint_samples(
        test_sym, test_joint_json, allowed, args.window_size, args.stride, test_span, test_step
    )
    pre_teacher = make_window_channel_samples(
        pre_sym,
        {canonical_label(s["seq_name"]) for s in pre_sym["sequences"]},
        "temporal",
        args.window_size,
        args.stride,
        include_wrist_features=False,
        include_duration_features=False,
        include_occlusion_features=False,
        window_span_units=pre_span,
        window_step_units=pre_step,
    )
    ft_teacher = make_window_channel_samples(
        ft_sym,
        allowed,
        "temporal",
        args.window_size,
        args.stride,
        include_wrist_features=False,
        include_duration_features=False,
        include_occlusion_features=False,
        window_span_units=ft_span,
        window_step_units=ft_step,
    )
    test_teacher = make_window_channel_samples(
        test_sym,
        allowed,
        "temporal",
        args.window_size,
        args.stride,
        include_wrist_features=False,
        include_duration_features=False,
        include_occlusion_features=False,
        window_span_units=test_span,
        window_step_units=test_step,
    )
    teacher_weights = {
        "state": 1.0,
        "transition": 0.5,
        "hand_motion": 0.5,
        "interaction": 0.2,
        "tempo": 0.2,
    }
    pre_teacher = merge_sample_features(pre_teacher, teacher_weights)
    ft_teacher = merge_sample_features(ft_teacher, teacher_weights)
    test_teacher = merge_sample_features(test_teacher, teacher_weights)
    pre_joint = attach_teacher_features(pre_joint, pre_teacher)
    ft_joint = attach_teacher_features(ft_joint, ft_teacher)
    test_joint = attach_teacher_features(test_joint, test_teacher)

    payload = {
        "device": str(device),
        "teacher_weight": args.teacher_weight,
        "pretrain_teacher_weight": pretrain_teacher_weight,
        "finetune_teacher_weight": finetune_teacher_weight,
        "window_span_units": args.window_span_units,
        "window_step_units": args.window_step_units,
        "pretrain_window_span_units": pre_span,
        "pretrain_window_step_units": pre_step,
        "finetune_window_span_units": ft_span,
        "finetune_window_step_units": ft_step,
        "test_window_span_units": test_span,
        "test_window_step_units": test_step,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "summary": {},
        "results": [],
    }
    results = []
    for fraction in args.fractions:
        for seed in args.seeds:
            set_seed(seed)
            results.append(
                run_once(
                    pre_joint,
                    ft_joint,
                    test_joint,
                    fraction, seed, args.hidden_dim, args.pretrain_epochs, args.finetune_epochs,
                    args.lr, args.weight_decay, pretrain_teacher_weight, finetune_teacher_weight,
                    device, args.aggregation
                )
            )
    payload["results"] = results
    payload["summary"] = summarize(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    for fraction, item in payload["summary"].items():
        print(
            f"fraction={fraction} "
            f"baseline_seq={item['baseline_sequence_accuracy_mean']:.4f} "
            f"teacher_seq={item['teacher_sequence_accuracy_mean']:.4f} "
            f"baseline_win={item['baseline_window_accuracy_mean']:.4f} "
            f"teacher_win={item['teacher_window_accuracy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
