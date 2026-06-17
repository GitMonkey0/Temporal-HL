#!/usr/bin/env python3
"""Sweep interaction-aware realized editing mechanisms on hard right-hand slices.

This is an experiment driver, not paper text.

It compares three axes under the same strict interaction-aware joint criterion:

- preserve-hand composition:
  - delta composition from a preserve donor
  - absolute-state transplant from a preserve donor
- preserve-donor ordering:
  - raw family ordering
  - HGB pair-guided ordering
  - torch MLP pair-guided ordering
- pair search objective within the selected pool:
  - binary joint score
  - surrogate agreement score

The goal is to test whether the current interaction bottleneck is mainly due to:

- preserve-hand composition
- preserve-donor ordering capacity
- overly discrete pair-selection objectives
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tools.build_interaction_realized_pairguided_editor import (
    DEPTHS,
    TASKS,
    fmt,
)
from tools.build_pairguided_reranker_multislice import (
    candidate_pool_for_task,
    build_examples,
    collect_slice_frames,
    opposite_hand_name,
    pair_feature_vector,
    relaxed_left_family_candidates_with_meta,
    target_right_features,
)
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    compose_target_hand_transition,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pick_best_symbolic_pair,
    summarize_composed_motif,
)
from tools.build_weak_slice_joint_editor_prototype import current_grouped_signature


ROOT = Path("/opt/tiger/hand")

LEFT_MODES = ("delta", "absolute")
SEARCH_MODES = ("binary", "surrogate")
SELECTORS = ("base", "hgb", "mlp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compose_left(prev_geom, donor_row, mode: str):
    if mode not in LEFT_MODES:
        raise ValueError(f"Unsupported left composition mode: {mode}")
    donor_curr = donor_row["curr_geom"]["left"]
    if donor_curr is None:
        return None
    if mode == "absolute":
        return {
            "local_vectors": np.asarray(donor_curr["local_vectors"], dtype=np.float32).copy(),
            "flexion": np.asarray(donor_curr["flexion"], dtype=np.float32).copy(),
        }

    prev_left = prev_geom["left"]
    donor_prev = donor_row["prev_geom"]["left"]
    if prev_left is None or donor_prev is None:
        return None
    delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
    delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
    return {
        "local_vectors": np.asarray(prev_left["local_vectors"], dtype=np.float32) + delta_local,
        "flexion": np.asarray(prev_left["flexion"], dtype=np.float32) + delta_flex,
    }


def compose_variant(prev_geom, right_row, left_row, left_mode: str):
    return {
        "right": compose_target_hand_transition(prev_geom, right_row, "right_hand_motion"),
        "left": compose_left(prev_geom, left_row, left_mode),
    }


def evaluate_variant(prev_frame, curr_frame, prev_geom, right_row=None, left_row=None, left_mode: str = "delta"):
    if right_row is None:
        return {
            "available": 0,
            "right_grouped_match": 0,
            "left_preserve": 0,
            "joint_score": 0,
            "right_state_agreement": 0.0,
            "right_transition_agreement": 0.0,
            "left_state_agreement": 0.0,
            "left_transition_agreement": 0.0,
            "surrogate_score": 0.0,
        }

    if left_row is None:
        edited_right = compose_target_hand_transition(prev_geom, right_row, "right_hand_motion")
        edited_left = prev_geom["left"]
    else:
        split = compose_variant(prev_geom, right_row, left_row, left_mode)
        edited_right = split["right"]
        edited_left = split["left"]

    right_motif = None if edited_right is None else summarize_composed_motif(prev_frame, edited_right, right_row, "right_hand_motion")
    donor_grouped = "unknown"
    edited_right_group = "unknown"
    if right_motif is not None:
        donor_grouped = grouped_motif_signature(str(right_motif["donor_hand_motion"]), list(right_motif["donor_transition_labels"]))
        edited_right_group = grouped_motif_signature(str(right_motif["edited_hand_motion"]), list(right_motif["edited_transition_labels"]))

    if left_row is None:
        left_ref = curr_frame.get("left")
    else:
        left_ref = left_row["curr_frame"].get("left")
    left_motif = None if edited_left is None else summarize_composed_motif(prev_frame, edited_left, {"curr_frame": {"left": left_ref}}, "left_hand_motion")
    current_left_group = current_grouped_signature(curr_frame.get("left"))
    edited_left_group = "unknown"
    if left_motif is not None:
        edited_left_group = grouped_motif_signature(str(left_motif["edited_hand_motion"]), list(left_motif["edited_transition_labels"]))

    right_match = int(right_motif is not None and edited_right_group == donor_grouped)
    left_preserve = int(left_motif is not None and edited_left_group == current_left_group)
    right_state = 0.0 if right_motif is None else float(right_motif["state_agreement"])
    right_trans = 0.0 if right_motif is None else float(right_motif["transition_agreement"])
    left_state = 0.0 if left_motif is None else float(left_motif["state_agreement"])
    left_trans = 0.0 if left_motif is None else float(left_motif["transition_agreement"])
    surrogate = (
        4.0 * right_match
        + 4.0 * left_preserve
        + 1.5 * right_trans
        + 1.0 * right_state
        + 1.0 * left_trans
        + 0.5 * left_state
    )
    return {
        "available": 1,
        "right_grouped_match": right_match,
        "left_preserve": left_preserve,
        "joint_score": right_match * left_preserve,
        "right_state_agreement": right_state,
        "right_transition_agreement": right_trans,
        "left_state_agreement": left_state,
        "left_transition_agreement": left_trans,
        "surrogate_score": surrogate,
    }


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    keys = [
        "right_grouped_match",
        "left_preserve",
        "joint_score",
        "right_state_agreement",
        "right_transition_agreement",
        "left_state_agreement",
        "left_transition_agreement",
        "surrogate_score",
    ]
    out = {
        "num_frames": n,
        "available_rate": avail / n,
    }
    for key in keys:
        on_avail = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
        overall = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = on_avail
        out[f"{key}_overall"] = overall
    return out


def summarize_by_subtype(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        rec = {
            "other_hand_motion": key[0],
            "interaction_motion_value": key[1],
            "num_frames": len(items),
        }
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
            rec[f"{prefix}_surrogate_score_overall"] = summary["surrogate_score_overall"]
            rec[f"{prefix}_left_preserve_overall"] = summary["left_preserve_overall"]
        out.append(rec)
    return out


def train_hgb_model(train_frames, pair_bank):
    x_train, y_train, meta = build_examples(train_frames, pair_bank, "right_hand_motion")
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=30,
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model, {
        "num_examples": int(len(y_train)),
        "num_positive": pos,
        "num_negative": neg,
        "positive_weight": float(pos_weight),
        **{k: int(v) for k, v in meta.items()},
    }


def train_mlp_model(train_frames, pair_bank, seed: int):
    x_train, y_train, meta = build_examples(train_frames, pair_bank, "right_hand_motion")
    if len(y_train) == 0:
        raise RuntimeError("No training examples for MLP pair-guided model")
    x_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.float32))
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, generator=generator)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PairMLP(x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE))

    model.train()
    for _ in range(12):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, {
        "num_examples": int(len(y_train)),
        "num_positive": int(pos),
        "num_negative": int(neg),
        "positive_weight": float(pos_weight),
        "device": str(DEVICE),
        "seed": int(seed),
        **{k: int(v) for k, v in meta.items()},
    }


def score_left_pool(selector, model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool):
    opp_pool = relaxed_left_family_candidates_with_meta(
        pair_bank,
        current_left_group,
        curr_attrs,
        prev_geom,
        curr_geom,
        opposite_hand_name("right_hand_motion"),
        1,
    )
    if selector == "base" or not target_pool or not opp_pool:
        return opp_pool

    opp_pool_size = len(opp_pool)
    target_pool_size = len(target_pool)
    target_meta = {id(row): target_right_features(row, "right_hand_motion", prev_geom, curr_geom) for row in target_pool}
    scored = []
    for opp_item in opp_pool:
        feats = np.asarray(
            [pair_feature_vector(opp_item, target_meta[id(target_row)], opp_pool_size, target_pool_size) for target_row in target_pool],
            dtype=np.float32,
        )
        if selector == "hgb":
            score = float(model.predict_proba(feats)[:, 1].max())
        elif selector == "mlp":
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.from_numpy(feats).to(DEVICE))).detach().cpu().numpy()
            score = float(probs.max())
        else:
            raise ValueError(f"Unsupported selector: {selector}")
        scored.append((score, opp_item))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]


def choose_best_variant(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth: int, left_mode: str, search_mode: str, cache: dict[tuple[int, int, str], dict[str, float]]):
    best = None
    for right_row in target_pool:
        for left_item in left_pool[:depth]:
            cache_key = (id(right_row), id(left_item["row"]), left_mode)
            res = cache.get(cache_key)
            if res is None:
                res = evaluate_variant(prev_frame, curr_frame, prev_geom, right_row, left_item["row"], left_mode=left_mode)
                cache[cache_key] = res
            if search_mode == "binary":
                key = (
                    res["joint_score"],
                    res["left_preserve"],
                    res["right_grouped_match"],
                    res["right_transition_agreement"],
                    res["right_state_agreement"],
                )
            elif search_mode == "surrogate":
                key = (
                    res["surrogate_score"],
                    res["joint_score"],
                    res["left_preserve"],
                    res["right_grouped_match"],
                )
            else:
                raise ValueError(f"Unsupported search mode: {search_mode}")
            if best is None or key > best[0]:
                best = (key, right_row, left_item["row"], res)
    return best


def run_task(train_frames, test_frames, pair_bank, selector_models, task_target: str):
    rows = []
    method_specs = []
    for selector in SELECTORS:
        for left_mode in LEFT_MODES:
            for search_mode in SEARCH_MODES:
                for depth in DEPTHS:
                    method_specs.append((selector, left_mode, search_mode, depth))

    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        single = pick_best_symbolic_pair(pair_bank, curr_attrs, prev_geom, curr_geom, "right_hand_motion", task_target)
        target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        single_eval = evaluate_variant(prev_frame, curr_frame, prev_geom, single, None, left_mode="delta")
        rec.update({f"single_{k}": v for k, v in single_eval.items()})
        eval_cache: dict[tuple[int, int, str], dict[str, float]] = {}

        selector_pools = {}
        for selector in SELECTORS:
            model = selector_models.get(selector)
            selector_pools[selector] = score_left_pool(selector, model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)

        for selector, left_mode, search_mode, depth in method_specs:
            prefix = f"{selector}_{left_mode}_{search_mode}_top{depth}"
            left_pool = selector_pools[selector]
            choice = (
                choose_best_variant(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth, left_mode, search_mode, eval_cache)
                if target_pool and left_pool
                else None
            )
            eval_rec = evaluate_variant(
                prev_frame,
                curr_frame,
                prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                left_mode=left_mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})

        rows.append(rec)

    prefixes = ["single"] + [f"{selector}_{left_mode}_{search_mode}_top{depth}" for selector, left_mode, search_mode, depth in method_specs]
    return {
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "subtype_summary": summarize_by_subtype(rows, prefixes),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    selector_training = {}
    for task_field, task_target in TASKS:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        selector_models: dict[str, object] = {"base": None}
        hgb_model, hgb_stats = train_hgb_model(train_frames, pair_bank)
        mlp_model, mlp_stats = train_mlp_model(train_frames, pair_bank, args.mlp_seed)
        selector_models["hgb"] = hgb_model
        selector_models["mlp"] = mlp_model
        task_name = f"{task_field}->{task_target}"
        selector_training[task_name] = {
            "hgb": hgb_stats,
            "mlp": mlp_stats,
        }
        task_results[task_name] = run_task(train_frames, test_frames, pair_bank, selector_models, task_target)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "sweep composition mechanism, preserve-donor ordering capacity, and search objective on strict joint interaction editing",
        },
        "selector_training": selector_training,
        "task_results": task_results,
        "run_config": {
            "mlp_seed": args.mlp_seed,
            "tag": args.tag,
            "device": str(DEVICE),
        },
    }

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = GEN / f"interaction_realized_mechanism_sweep{suffix}.json"
    out_md = SUM / f"interaction_realized_mechanism_sweep{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Mechanism Sweep",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Run tag: `{args.tag}`",
        "",
        f"MLP device: `{DEVICE}`",
        "",
        "Compared axes:",
        "",
        "- preserve-hand composition: `delta` vs `absolute`",
        "- preserve-donor ordering: `base` vs `hgb` vs `mlp`",
        "- pair search objective: `binary` vs `surrogate`",
        "",
    ]

    key_methods = [
        "single",
        "base_delta_binary_top10",
        "base_absolute_binary_top10",
        "hgb_delta_binary_top10",
        "hgb_absolute_binary_top10",
        "hgb_delta_surrogate_top10",
        "hgb_absolute_surrogate_top10",
        "mlp_delta_surrogate_top10",
        "mlp_absolute_surrogate_top10",
    ]

    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "### Selector Training",
                "",
                "| selector | metric | value |",
                "| --- | --- | ---: |",
            ]
        )
        for selector, stats in selector_training[task_name].items():
            for key, value in stats.items():
                lines.append(f"| {selector} | {key} | {value} |")
        lines.extend(
            [
                "### Key Methods",
                "",
                "| method | avail | right grouped | left preserve | joint overall | surrogate overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in key_methods:
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | "
                f"{fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} | {fmt(stats['surrogate_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                "### By Subtype",
                "",
                "| other hand motion | interaction motion | frames | single joint | hgb absolute surrogate top-10 | mlp absolute surrogate top-10 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["subtype_summary"]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
                f"{fmt(row['single_joint_score_overall'])} | {fmt(row['hgb_absolute_surrogate_top10_joint_score_overall'])} | "
                f"{fmt(row['mlp_absolute_surrogate_top10_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
