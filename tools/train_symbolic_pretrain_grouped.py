#!/usr/bin/env python3
"""Supervised symbolic pretrain + finetune with grouped branch fusion."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.train_symbolic_classifier import (
    build_vocabulary,
    canonical_label,
    overlap_labels,
    subsample_by_fraction,
    vectorize,
)
from tools.train_symbolic_pretrain import (
    build_finetune_class_weights,
    load_json,
    make_window_channel_samples,
    parse_boost_labels,
    resolve_split_window_args,
    set_seed,
    split_train_holdout,
    summarize,
)


GROUP_SPECS = {
    "state": ("state",),
    "dynamics": ("transition", "hand_motion"),
    "context": ("interaction", "tempo", "event"),
}


def l2_normalize(features: dict[str, float]) -> dict[str, float]:
    denom = sum(value * value for value in features.values()) ** 0.5
    if denom <= 0:
        return features
    return {key: value / denom for key, value in features.items()}


def merge_group_channels(
    channels: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    grouped = {}
    for group_name, channel_names in GROUP_SPECS.items():
        merged = {}
        for channel_name in channel_names:
            weight = weights.get(channel_name, 0.0)
            if weight <= 0:
                continue
            for key, value in channels.get(channel_name, {}).items():
                merged[key] = weight * value
        grouped[group_name] = l2_normalize(merged)
    return grouped


@dataclass
class GroupedEncodedSet:
    state: np.ndarray
    dynamics: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    seq_names: list[str]


def build_group_vocab(samples: list[dict[str, object]], group_name: str) -> list[str]:
    wrapped = [{"features": sample["grouped_channels"][group_name]} for sample in samples]
    return build_vocabulary(wrapped)


def vectorize_group(samples: list[dict[str, object]], group_name: str, vocab: list[str]) -> np.ndarray:
    if not vocab:
        return np.zeros((len(samples), 1), dtype=np.float32)
    wrapped = [{"features": sample["grouped_channels"][group_name]} for sample in samples]
    return vectorize(wrapped, vocab)


def encode_grouped_samples(
    samples: list[dict[str, object]],
    vocabs: dict[str, list[str]],
    label_to_idx: dict[str, int],
) -> GroupedEncodedSet:
    return GroupedEncodedSet(
        state=vectorize_group(samples, "state", vocabs["state"]),
        dynamics=vectorize_group(samples, "dynamics", vocabs["dynamics"]),
        context=vectorize_group(samples, "context", vocabs["context"]),
        labels=np.asarray([label_to_idx[s["label"]] for s in samples], dtype=np.int64),
        seq_names=[s["seq_name"] for s in samples],
    )


class GroupedDataset(Dataset):
    def __init__(self, encoded: GroupedEncodedSet):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded.labels)

    def __getitem__(self, idx: int):
        return {
            "state": torch.from_numpy(self.encoded.state[idx]),
            "dynamics": torch.from_numpy(self.encoded.dynamics[idx]),
            "context": torch.from_numpy(self.encoded.context[idx]),
            "label": torch.tensor(self.encoded.labels[idx], dtype=torch.long),
        }


class BranchEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupedFusionNet(nn.Module):
    def __init__(self, dims: dict[str, int], hidden_dim: int, gate_hidden_dim: int, num_classes: int):
        super().__init__()
        self.branch_names = ["state", "dynamics", "context"]
        self.encoders = nn.ModuleDict(
            {name: BranchEncoder(dims[name], hidden_dim) for name in self.branch_names}
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, len(self.branch_names)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_feats = {name: self.encoders[name](batch[name]) for name in self.branch_names}
        gate_input = torch.cat([branch_feats[name] for name in self.branch_names], dim=-1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
        stacked = torch.stack([branch_feats[name] for name in self.branch_names], dim=1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        return logits, {
            "gate_weights": gate_weights.detach(),
            **branch_feats,
        }


class GroupedConcatNet(nn.Module):
    def __init__(self, dims: dict[str, int], hidden_dim: int, num_classes: int):
        super().__init__()
        self.branch_names = ["state", "dynamics", "context"]
        self.encoders = nn.ModuleDict(
            {name: BranchEncoder(dims[name], hidden_dim) for name in self.branch_names}
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_feats = {name: self.encoders[name](batch[name]) for name in self.branch_names}
        fused = torch.cat([branch_feats[name] for name in self.branch_names], dim=-1)
        logits = self.classifier(fused)
        return logits, branch_feats


class GroupedResidualAuxNet(nn.Module):
    def __init__(self, dims: dict[str, int], hidden_dim: int, gate_hidden_dim: int, num_classes: int):
        super().__init__()
        self.branch_names = ["state", "dynamics", "context"]
        self.aux_branch_names = ["dynamics", "context"]
        self.encoders = nn.ModuleDict(
            {name: BranchEncoder(dims[name], hidden_dim) for name in self.branch_names}
        )
        self.aux_gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, len(self.aux_branch_names)),
        )
        self.aux_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_feats = {name: self.encoders[name](batch[name]) for name in self.branch_names}
        gate_input = torch.cat([branch_feats[name] for name in self.branch_names], dim=-1)
        aux_gate_weights = torch.softmax(self.aux_gate(gate_input), dim=-1)
        aux_stack = torch.stack([branch_feats[name] for name in self.aux_branch_names], dim=1)
        aux_mix = (aux_stack * aux_gate_weights.unsqueeze(-1)).sum(dim=1)
        state_feat = branch_feats["state"]
        residual_mix = state_feat + self.aux_proj(aux_mix)
        fused = torch.cat([state_feat, residual_mix], dim=-1)
        logits = self.classifier(fused)
        return logits, {
            "gate_weights": aux_gate_weights.detach(),
            **branch_feats,
        }


def build_grouped_model(
    dims: dict[str, int],
    hidden_dim: int,
    num_classes: int,
    fusion_type: str,
) -> nn.Module:
    if fusion_type == "concat":
        return GroupedConcatNet(dims, hidden_dim, num_classes)
    if fusion_type == "gated":
        return GroupedFusionNet(dims, hidden_dim, hidden_dim, num_classes)
    if fusion_type == "residual_aux":
        return GroupedResidualAuxNet(dims, hidden_dim, hidden_dim, num_classes)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")


def aggregate_grouped_sequence_predictions(
    probs: np.ndarray,
    pred_labels: np.ndarray,
    labels: list[str],
    seq_names: list[str],
    gt_indices: np.ndarray,
    method: str,
):
    by_sequence_probs = defaultdict(list)
    by_sequence_votes = defaultdict(list)
    gt = {}
    for prob, pred, seq_name, gt_idx in zip(probs, pred_labels, seq_names, gt_indices):
        by_sequence_probs[seq_name].append(prob)
        by_sequence_votes[seq_name].append(int(pred))
        gt[seq_name] = labels[int(gt_idx)]
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


def evaluate_model(
    model: nn.Module,
    encoded: GroupedEncodedSet,
    label_names: list[str],
    device: torch.device,
    aggregation: str,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    loader = DataLoader(GroupedDataset(encoded), batch_size=256, shuffle=False)
    probs_all, preds_all, labels_all = [], [], []
    gate_sum = None
    gate_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits, aux = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            if "gate_weights" in aux:
                gates = aux["gate_weights"].cpu().numpy()
                gate_sum = gates.sum(axis=0) if gate_sum is None else gate_sum + gates.sum(axis=0)
                gate_count += gates.shape[0]
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    labels_idx = np.concatenate(labels_all, axis=0)
    window_acc = accuracy_score(labels_idx, preds)
    y_true, y_pred = aggregate_grouped_sequence_predictions(
        probs, preds, label_names, encoded.seq_names, labels_idx, aggregation
    )
    seq_acc = accuracy_score(y_true, y_pred)
    gate_means = {}
    if gate_sum is not None and gate_count > 0:
        gate_names = ["dynamics", "context"] if len(gate_sum / gate_count) == 2 else ["state", "dynamics", "context"]
        for name, value in zip(gate_names, gate_sum / gate_count):
            gate_means[name] = float(value)
    return float(window_acc), float(seq_acc), gate_means


def train_model(
    model: nn.Module,
    train_encoded: GroupedEncodedSet,
    holdout_encoded: GroupedEncodedSet,
    label_names: list[str],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    aggregation: str,
    class_weights: np.ndarray | None = None,
):
    loader = DataLoader(GroupedDataset(train_encoded), batch_size=128, shuffle=True)
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
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            opt.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
        _, holdout_seq, _ = evaluate_model(model, holdout_encoded, label_names, device, aggregation)
        if holdout_seq > best_acc:
            best_acc = holdout_seq
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def prepare_grouped_samples(samples: list[dict[str, object]], weights: dict[str, float]):
    prepared = []
    for sample in samples:
        row = dict(sample)
        row["grouped_channels"] = merge_group_channels(sample["channels"], weights)
        prepared.append(row)
    return prepared


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
    fusion_type,
):
    ft_fraction = subsample_by_fraction(train_ft_samples, fraction, seed)
    ft_train, ft_holdout = split_train_holdout(ft_fraction, 0.2, seed)
    pre_train, pre_holdout = split_train_holdout(train_pre_samples, 0.1, seed)

    all_for_vocab = pre_train + pre_holdout + ft_train + ft_holdout + test_samples
    vocabs = {name: build_group_vocab(all_for_vocab, name) for name in GROUP_SPECS}

    pre_labels = sorted({s["label"] for s in train_pre_samples})
    pre_map = {k: i for i, k in enumerate(pre_labels)}
    ft_labels = sorted({s["label"] for s in ft_train + ft_holdout + test_samples})
    ft_map = {k: i for i, k in enumerate(ft_labels)}

    pre_train_encoded = encode_grouped_samples(pre_train, vocabs, pre_map)
    pre_holdout_encoded = encode_grouped_samples(pre_holdout, vocabs, pre_map)
    ft_train_encoded = encode_grouped_samples(ft_train, vocabs, ft_map)
    ft_holdout_encoded = encode_grouped_samples(ft_holdout, vocabs, ft_map)
    test_encoded = encode_grouped_samples(test_samples, vocabs, ft_map)

    dims = {name: getattr(ft_train_encoded, name).shape[1] for name in GROUP_SPECS}

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
    scratch_win, scratch_seq, scratch_gates = evaluate_model(
        scratch, test_encoded, ft_labels, device, aggregation
    )

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
    transferred = build_grouped_model(dims, hidden_dim, len(ft_labels), fusion_type).to(device)
    transferred.encoders.load_state_dict(pretrained.encoders.state_dict())
    if fusion_type == "gated":
        transferred.gate.load_state_dict(pretrained.gate.state_dict())
    elif fusion_type == "residual_aux":
        transferred.aux_gate.load_state_dict(pretrained.aux_gate.state_dict())
        transferred.aux_proj.load_state_dict(pretrained.aux_proj.state_dict())
    train_model(
        transferred,
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
    ft_win, ft_seq, ft_gates = evaluate_model(
        transferred, test_encoded, ft_labels, device, aggregation
    )

    return {
        "fraction": fraction,
        "seed": seed,
        "num_finetune_train_windows": len(ft_train),
        "scratch_window_accuracy": scratch_win,
        "scratch_sequence_accuracy": scratch_seq,
        "pretrained_window_accuracy": ft_win,
        "pretrained_sequence_accuracy": ft_seq,
        "scratch_gate_means_test": scratch_gates,
        "pretrained_gate_means_test": ft_gates,
    }


def summarize_with_gates(results):
    payload = summarize(results)
    by_fraction = defaultdict(list)
    for row in results:
        by_fraction[row["fraction"]].append(row)
    for fraction, rows in by_fraction.items():
        scratch_gate_keys = set()
        pretrained_gate_keys = set()
        for row in rows:
            scratch_gate_keys.update(row.get("scratch_gate_means_test", {}).keys())
            pretrained_gate_keys.update(row.get("pretrained_gate_means_test", {}).keys())
        if scratch_gate_keys:
            payload[str(fraction)]["scratch_gate_means_test"] = {
                key: statistics.mean(row["scratch_gate_means_test"].get(key, 0.0) for row in rows)
                for key in sorted(scratch_gate_keys)
            }
        if pretrained_gate_keys:
            payload[str(fraction)]["pretrained_gate_means_test"] = {
                key: statistics.mean(row["pretrained_gate_means_test"].get(key, 0.0) for row in rows)
                for key in sorted(pretrained_gate_keys)
            }
    return payload


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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_pretrain_grouped_summary.json"),
    )
    parser.add_argument("--fusion-type", choices=["gated", "concat", "residual_aux"], default="concat")
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
    pre_samples = prepare_grouped_samples(pre_samples, weights)
    ft_samples = prepare_grouped_samples(ft_samples, weights)
    test_samples = prepare_grouped_samples(test_samples, weights)

    results = []
    for fraction in args.fractions:
        for seed in args.seeds:
            set_seed(seed)
            ft_fraction_preview = subsample_by_fraction(ft_samples, fraction, seed)
            ft_train_preview, ft_holdout_preview = split_train_holdout(ft_fraction_preview, 0.2, seed)
            ft_labels_preview = sorted({s["label"] for s in ft_train_preview + ft_holdout_preview + test_samples})
            finetune_class_weights = build_finetune_class_weights(ft_labels_preview, boost_map)
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
                    args.fusion_type,
                )
            )

    payload = {
        "mode": args.mode,
        "device": str(device),
        "window_size": args.window_size,
        "stride": args.stride,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "summary": summarize_with_gates(results),
        "results": results,
        "wrist_features": args.wrist_features,
        "duration_features": args.duration_features,
        "event_features": args.event_features,
        "coordination_features": args.coordination_features,
        "event_weight": args.event_weight,
        "weights": weights,
        "group_specs": GROUP_SPECS,
        "fusion_type": args.fusion_type,
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
            f"pretrained_seq={item['pretrained_sequence_accuracy_mean']:.4f}"
        )
        if "pretrained_gate_means_test" in item:
            print(f"pretrained_gate_means={item['pretrained_gate_means_test']}")


if __name__ == "__main__":
    main()
