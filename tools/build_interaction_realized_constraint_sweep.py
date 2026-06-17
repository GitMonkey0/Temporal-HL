#!/usr/bin/env python3
"""Constrained preserve-hand repair sweep on hard right-hand interaction slices.

This is an experiment memo, not paper text.

The sweep keeps the existing split-donor selection pipeline and asks a narrower
question:

- after selecting a right-hand target donor and a left-hand preserve donor,
  is the remaining bottleneck mainly the realized left-hand geometry?

To answer this, we apply several target-aware repair operators to the preserve
hand only and measure the strict joint interaction criterion.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from tools.build_interaction_realized_pairguided_editor import (
    DEPTHS,
    TASKS,
    choose_best_split,
    fmt,
    select_pairguided_left_pool,
    train_pairguided_model,
)
from tools.build_interaction_realized_mechanism_sweep import DEVICE, train_mlp_model
from tools.build_pairguided_reranker_multislice import (
    candidate_pool_for_task,
    collect_slice_frames,
    pair_feature_vector,
    relaxed_left_family_candidates_with_meta,
    target_right_features,
)
from tools.build_temporal_hl import EDGE_ORDER
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
from tools.build_weak_slice_joint_editor_prototype import hand_motif_from_geom


REPAIR_MODES = (
    "none",
    "blend_curr_025",
    "blend_curr_050",
    "blend_curr_075",
    "edge_token_snap",
    "edge_transition_snap",
    "finger_profile_snap",
    "full_curr_oracle",
)

FINGER_TO_EDGE_IDXS = defaultdict(list)
for idx, (finger_name, _) in enumerate(EDGE_ORDER):
    FINGER_TO_EDGE_IDXS[finger_name].append(idx)


def hand_geom_array(frame: dict[str, object], hand_name: str):
    hand = frame.get(hand_name)
    if hand is None:
        return None
    flex = hand.get("flexion_scores", {})
    return {
        "local_vectors": np.asarray(hand.get("local_vectors", []), dtype=np.float32),
        "flexion": np.asarray(
            [
                float(flex.get("thumb", 0.0)),
                float(flex.get("index", 0.0)),
                float(flex.get("middle", 0.0)),
                float(flex.get("ring", 0.0)),
                float(flex.get("pinky", 0.0)),
            ],
            dtype=np.float32,
        ),
    }


def finger_profile(transition_labels: list[str]) -> list[str]:
    out = []
    for finger_name in ("thumb", "index", "middle", "ring", "pinky"):
        labels = [transition_labels[idx] for idx in FINGER_TO_EDGE_IDXS[finger_name]]
        major = sum(label == "major_shift" for label in labels)
        minor = sum(label == "minor_shift" for label in labels)
        if major >= 1:
            out.append("active")
        elif minor >= 1:
            out.append("local")
        else:
            out.append("still")
    return out


def compose_left_delta(prev_geom, donor_row):
    prev_left = prev_geom["left"]
    donor_prev = donor_row["prev_geom"]["left"]
    donor_curr = donor_row["curr_geom"]["left"]
    if prev_left is None or donor_prev is None or donor_curr is None:
        return None
    delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
    delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
    return {
        "local_vectors": np.asarray(prev_left["local_vectors"], dtype=np.float32) + delta_local,
        "flexion": np.asarray(prev_left["flexion"], dtype=np.float32) + delta_flex,
    }


def repair_left(prev_frame, curr_frame, edited_left, mode: str):
    if edited_left is None:
        return None
    if mode == "none":
        return edited_left

    target_left = hand_geom_array(curr_frame, "left")
    if target_left is None:
        return edited_left

    repaired = {
        "local_vectors": np.asarray(edited_left["local_vectors"], dtype=np.float32).copy(),
        "flexion": np.asarray(edited_left["flexion"], dtype=np.float32).copy(),
    }
    target_raw = curr_frame.get("left")
    prev_raw = prev_frame.get("left")
    if target_raw is None or prev_raw is None:
        return repaired

    if mode.startswith("blend_curr_"):
        alpha = float(mode.split("_")[-1]) / 100.0
        repaired["local_vectors"] = (1.0 - alpha) * repaired["local_vectors"] + alpha * target_left["local_vectors"]
        repaired["flexion"] = (1.0 - alpha) * repaired["flexion"] + alpha * target_left["flexion"]
        return repaired

    if mode == "full_curr_oracle":
        return target_left

    motif = hand_motif_from_geom(prev_raw, target_raw, repaired)
    if motif is None:
        return repaired
    target_tokens = list(target_raw.get("token_labels", []))
    target_transitions = list(target_raw.get("transition_labels", []))

    if mode == "edge_token_snap":
        for idx, (edited_tok, target_tok) in enumerate(zip(motif["edited_token_labels"], target_tokens)):
            if edited_tok != target_tok:
                repaired["local_vectors"][idx] = target_left["local_vectors"][idx]
        repaired["flexion"] = target_left["flexion"].copy()
        return repaired

    if mode == "edge_transition_snap":
        for idx, (edited_tr, target_tr) in enumerate(zip(motif["edited_transition_labels"], target_transitions)):
            if edited_tr != target_tr:
                repaired["local_vectors"][idx] = target_left["local_vectors"][idx]
        repaired["flexion"] = target_left["flexion"].copy()
        return repaired

    if mode == "finger_profile_snap":
        edited_profile = finger_profile(list(motif["edited_transition_labels"]))
        target_profile = finger_profile(target_transitions)
        for finger_idx, finger_name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            if edited_profile[finger_idx] != target_profile[finger_idx]:
                for edge_idx in FINGER_TO_EDGE_IDXS[finger_name]:
                    repaired["local_vectors"][edge_idx] = target_left["local_vectors"][edge_idx]
                repaired["flexion"][finger_idx] = target_left["flexion"][finger_idx]
        return repaired

    raise ValueError(f"Unsupported repair mode: {mode}")


def evaluate_edit(prev_frame, curr_frame, prev_geom, right_row=None, left_row=None, repair_mode: str = "none"):
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
        }

    edited_right = compose_target_hand_transition(prev_geom, right_row, "right_hand_motion")
    edited_left = prev_geom["left"] if left_row is None else compose_left_delta(prev_geom, left_row)
    edited_left = repair_left(prev_frame, curr_frame, edited_left, repair_mode)

    right_motif = None if edited_right is None else summarize_composed_motif(prev_frame, edited_right, right_row, "right_hand_motion")
    donor_grouped = "unknown"
    edited_right_group = "unknown"
    if right_motif is not None:
        donor_grouped = grouped_motif_signature(str(right_motif["donor_hand_motion"]), list(right_motif["donor_transition_labels"]))
        edited_right_group = grouped_motif_signature(str(right_motif["edited_hand_motion"]), list(right_motif["edited_transition_labels"]))

    left_ref = curr_frame.get("left") if left_row is None else left_row["curr_frame"].get("left")
    left_motif = None if edited_left is None else summarize_composed_motif(prev_frame, edited_left, {"curr_frame": {"left": left_ref}}, "left_hand_motion")
    current_left_group = current_grouped_signature(curr_frame.get("left"))
    edited_left_group = "unknown"
    if left_motif is not None:
        edited_left_group = grouped_motif_signature(str(left_motif["edited_hand_motion"]), list(left_motif["edited_transition_labels"]))

    right_match = int(right_motif is not None and edited_right_group == donor_grouped)
    left_preserve = int(left_motif is not None and edited_left_group == current_left_group)
    return {
        "available": 1,
        "right_grouped_match": right_match,
        "left_preserve": left_preserve,
        "joint_score": right_match * left_preserve,
        "right_state_agreement": 0.0 if right_motif is None else float(right_motif["state_agreement"]),
        "right_transition_agreement": 0.0 if right_motif is None else float(right_motif["transition_agreement"]),
        "left_state_agreement": 0.0 if left_motif is None else float(left_motif["state_agreement"]),
        "left_transition_agreement": 0.0 if left_motif is None else float(left_motif["transition_agreement"]),
    }


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    out = {
        "num_frames": n,
        "available_rate": avail / n,
    }
    for key in (
        "right_grouped_match",
        "left_preserve",
        "joint_score",
        "right_state_agreement",
        "right_transition_agreement",
        "left_state_agreement",
        "left_transition_agreement",
    ):
        out[f"{key}_overall"] = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
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
            rec[f"{prefix}_left_preserve_overall"] = summary["left_preserve_overall"]
        out.append(rec)
    return out


def mlp_ranked_left_pool(mlp_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool):
    opp_pool = relaxed_left_family_candidates_with_meta(
        pair_bank,
        current_left_group,
        curr_attrs,
        prev_geom,
        curr_geom,
        "left",
        1,
    )
    if not target_pool or not opp_pool:
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
        with torch.no_grad():
            probs = torch.sigmoid(mlp_model(torch.from_numpy(feats).to(DEVICE))).detach().cpu().numpy()
        scored.append((float(probs.max()), opp_item))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    model, train_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    mlp_model, mlp_stats = train_mlp_model(train_frames, pair_bank, seed=0)
    rows = []
    method_specs = [("base", "none")]
    method_specs.extend([("base", mode) for mode in REPAIR_MODES if mode != "none"])
    method_specs.extend([("hgb", mode) for mode in REPAIR_MODES])
    method_specs.extend([("mlp", mode) for mode in REPAIR_MODES])

    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        single = pick_best_symbolic_pair(pair_bank, curr_attrs, prev_geom, curr_geom, "right_hand_motion", task_target)
        target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)
        base_left_pool = relaxed_left_family_candidates_with_meta(
            pair_bank,
            current_left_group,
            curr_attrs,
            prev_geom,
            curr_geom,
            "left",
            1,
        )
        hgb_left_pool = select_pairguided_left_pool(model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)
        mlp_left_pool = mlp_ranked_left_pool(mlp_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        single_eval = evaluate_edit(prev_frame, curr_frame, prev_geom, single, None, repair_mode="none")
        rec.update({f"single_{k}": v for k, v in single_eval.items()})

        for selector_name, repair_mode in method_specs:
            prefix = f"{selector_name}_{repair_mode}_top10"
            if selector_name == "base":
                left_pool = base_left_pool
            elif selector_name == "hgb":
                left_pool = hgb_left_pool
            else:
                left_pool = mlp_left_pool
            choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, left_pool, 10) if target_pool and left_pool else None
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                repair_mode=repair_mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})

        rows.append(rec)

    prefixes = ["single"] + [f"{selector}_{mode}_top10" for selector, mode in method_specs]
    return {
        "training_stats": {
            "hgb": train_stats,
            "mlp": mlp_stats,
        },
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "subtype_summary": summarize_by_subtype(rows, prefixes),
        "rows": rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for task_field, task_target in TASKS:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test preserve-hand constrained repair after split-donor pair selection on hard right-hand interaction slices",
            "repair_modes": list(REPAIR_MODES),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_constraint_sweep.json"
    out_md = SUM / "interaction_realized_constraint_sweep.md"
    out_json.write_text(json.dumps(payload, indent=2))

    key_methods = [
        "single",
        "base_none_top10",
        "base_edge_token_snap_top10",
        "base_edge_transition_snap_top10",
        "base_finger_profile_snap_top10",
        "hgb_none_top10",
        "hgb_edge_token_snap_top10",
        "hgb_edge_transition_snap_top10",
        "hgb_finger_profile_snap_top10",
        "mlp_none_top10",
        "mlp_edge_transition_snap_top10",
        "mlp_finger_profile_snap_top10",
        "hgb_full_curr_oracle_top10",
    ]

    lines = [
        "# Interaction Realized Constraint Sweep",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "We keep the split-donor selector fixed and test target-aware preserve-hand repairs.",
        "",
    ]

    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "### Training Stats",
                "",
                "| selector | metric | value |",
                "| --- | --- | ---: |",
            ]
        )
        for selector_name, stats in result["training_stats"].items():
            for key, value in stats.items():
                lines.append(f"| {selector_name} | {key} | {value} |")
        lines.extend(
            [
                "",
                "### Key Methods",
                "",
                "| method | avail | right grouped | left preserve | joint overall | left state | left transition |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in key_methods:
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | "
                f"{fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} | "
                f"{fmt(stats['left_state_agreement_overall'])} | {fmt(stats['left_transition_agreement_overall'])} |"
            )
        lines.extend(
            [
                "",
                "### By Subtype",
                "",
                "| other hand motion | interaction motion | frames | single joint | hgb none | hgb finger snap | mlp finger snap | hgb full current oracle |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["subtype_summary"]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
                f"{fmt(row['single_joint_score_overall'])} | {fmt(row['hgb_none_top10_joint_score_overall'])} | "
                f"{fmt(row['hgb_finger_profile_snap_top10_joint_score_overall'])} | {fmt(row['mlp_finger_profile_snap_top10_joint_score_overall'])} | "
                f"{fmt(row['hgb_full_curr_oracle_top10_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
