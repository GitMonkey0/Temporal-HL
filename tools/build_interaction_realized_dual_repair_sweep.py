#!/usr/bin/env python3
"""Dual-side constrained repair sweep on hard right-hand interaction slices.

This is an experiment memo, not paper text.

The current strongest line still under-serves specific hard sequence families.
This sweep keeps the strongest split-donor support/search setup and adds
target-aware repair on both sides:

- right target hand repaired toward the target donor
- left preserve hand repaired toward the current frame
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from tools.build_interaction_realized_constraint_sweep import (
    FINGER_TO_EDGE_IDXS,
    finger_profile,
    fmt,
    hand_geom_array,
    repair_left,
)
from tools.build_interaction_realized_mechanism_sweep import train_mlp_model
from tools.build_interaction_realized_pairguided_editor import (
    TASKS,
    choose_best_split,
    select_pairguided_left_pool,
    train_pairguided_model,
)
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
from tools.build_weak_slice_joint_editor_prototype import (
    current_grouped_signature,
    hand_motif_from_geom,
)


LEFT_MODE = "finger_profile_snap"
RIGHT_MODES = ("none", "edge_token_snap", "edge_transition_snap", "finger_profile_snap", "full_donor_oracle")
SELECTORS = ("hgb", "mlp", "base")
FAMILY_BUDGET = 2
DEPTH = 20


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
    import torch
    from tools.build_interaction_realized_mechanism_sweep import DEVICE
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


def reorder_to_budget(ranked_pool, raw_pool):
    raw_ids = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in raw_pool}
    out = [
        item for item in ranked_pool
        if (item["row"]["seq_name"], item["row"]["curr_frame_idx"]) in raw_ids
    ]
    seen = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in out}
    for item in raw_pool:
        key = (item["row"]["seq_name"], item["row"]["curr_frame_idx"])
        if key not in seen:
            out.append(item)
    return out


def repair_right(prev_frame, right_row, edited_right, mode: str):
    if edited_right is None or right_row is None or mode == "none":
        return edited_right
    donor_raw = right_row["curr_frame"].get("right")
    donor_right = hand_geom_array(right_row["curr_frame"], "right")
    prev_raw = prev_frame.get("right")
    if donor_raw is None or donor_right is None or prev_raw is None:
        return edited_right

    repaired = {
        "local_vectors": np.asarray(edited_right["local_vectors"], dtype=np.float32).copy(),
        "flexion": np.asarray(edited_right["flexion"], dtype=np.float32).copy(),
    }
    if mode == "full_donor_oracle":
        return donor_right

    motif = hand_motif_from_geom(prev_raw, donor_raw, repaired)
    if motif is None:
        return repaired
    donor_tokens = list(donor_raw.get("token_labels", []))
    donor_transitions = list(donor_raw.get("transition_labels", []))

    if mode == "edge_token_snap":
        for idx, (edited_tok, donor_tok) in enumerate(zip(motif["edited_token_labels"], donor_tokens)):
            if edited_tok != donor_tok:
                repaired["local_vectors"][idx] = donor_right["local_vectors"][idx]
        repaired["flexion"] = donor_right["flexion"].copy()
        return repaired

    if mode == "edge_transition_snap":
        for idx, (edited_tr, donor_tr) in enumerate(zip(motif["edited_transition_labels"], donor_transitions)):
            if edited_tr != donor_tr:
                repaired["local_vectors"][idx] = donor_right["local_vectors"][idx]
        repaired["flexion"] = donor_right["flexion"].copy()
        return repaired

    if mode == "finger_profile_snap":
        edited_profile = finger_profile(list(motif["edited_transition_labels"]))
        donor_profile = finger_profile(donor_transitions)
        for finger_idx, finger_name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            if edited_profile[finger_idx] != donor_profile[finger_idx]:
                for edge_idx in FINGER_TO_EDGE_IDXS[finger_name]:
                    repaired["local_vectors"][edge_idx] = donor_right["local_vectors"][edge_idx]
                repaired["flexion"][finger_idx] = donor_right["flexion"][finger_idx]
        return repaired

    raise ValueError(f"Unsupported right repair mode: {mode}")


def evaluate_dual_edit(prev_frame, curr_frame, prev_geom, right_row=None, left_row=None, right_mode: str = "none"):
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
    edited_right = repair_right(prev_frame, right_row, edited_right, right_mode)

    if left_row is None:
        edited_left = prev_geom["left"]
    else:
        prev_left = prev_geom["left"]
        donor_prev_left = left_row["prev_geom"]["left"]
        donor_curr_left = left_row["curr_geom"]["left"]
        if prev_left is None or donor_prev_left is None or donor_curr_left is None:
            edited_left = None
        else:
            edited_left = {
                "local_vectors": np.asarray(prev_left["local_vectors"], dtype=np.float32)
                + (
                    np.asarray(donor_curr_left["local_vectors"], dtype=np.float32)
                    - np.asarray(donor_prev_left["local_vectors"], dtype=np.float32)
                ),
                "flexion": np.asarray(prev_left["flexion"], dtype=np.float32)
                + (
                    np.asarray(donor_curr_left["flexion"], dtype=np.float32)
                    - np.asarray(donor_prev_left["flexion"], dtype=np.float32)
                ),
            }
    edited_left = repair_left(prev_frame, curr_frame, edited_left, LEFT_MODE)

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
    out = {"num_frames": n, "available_rate": avail / n}
    for key in ("right_grouped_match", "left_preserve", "joint_score", "right_state_agreement", "right_transition_agreement", "left_state_agreement", "left_transition_agreement"):
        out[f"{key}_overall"] = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
    return out


def summarize_by_sequence(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        rec = {"seq_name": seq_name, "num_frames": len(items)}
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
            rec[f"{prefix}_right_grouped_match_overall"] = summary["right_grouped_match_overall"]
            rec[f"{prefix}_left_preserve_overall"] = summary["left_preserve_overall"]
        out.append(rec)
    return out


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    mlp_model, mlp_stats = train_mlp_model(train_frames, pair_bank, seed=0)
    rows = []
    method_specs = [(selector, right_mode) for selector in SELECTORS for right_mode in RIGHT_MODES]

    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)
        raw_budget_pool = relaxed_left_family_candidates_with_meta(
            pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, "left", FAMILY_BUDGET
        )
        hgb_ranked = select_pairguided_left_pool(hgb_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)
        mlp_ranked = mlp_ranked_left_pool(mlp_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)
        selector_pools = {
            "base": raw_budget_pool,
            "hgb": reorder_to_budget(hgb_ranked, raw_budget_pool),
            "mlp": reorder_to_budget(mlp_ranked, raw_budget_pool),
        }

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        single = pick_best_symbolic_pair(pair_bank, curr_attrs, prev_geom, curr_geom, "right_hand_motion", task_target)
        single_eval = evaluate_dual_edit(prev_frame, curr_frame, prev_geom, single, None, right_mode="none")
        rec.update({f"single_{k}": v for k, v in single_eval.items()})

        for selector_name, right_mode in method_specs:
            prefix = f"{selector_name}_{right_mode}_dual"
            left_pool = selector_pools[selector_name]
            choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, left_pool, DEPTH) if target_pool and left_pool else None
            eval_rec = evaluate_dual_edit(
                prev_frame, curr_frame, prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                right_mode=right_mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)

    prefixes = ["single"] + [f"{selector}_{right_mode}_dual" for selector, right_mode in method_specs]
    return {
        "training_stats": {"hgb": hgb_stats, "mlp": mlp_stats},
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "sequence_summary": summarize_by_sequence(rows, prefixes),
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
        train_frames = [row for row in collect_slice_frames(train_data, task_field, task_target) if canonical(row["seq_name"]) in labels or row["seq_name"] in labels]
        test_frames = [row for row in collect_slice_frames(test_data, task_field, task_target) if canonical(row["seq_name"]) in labels or row["seq_name"] in labels]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether repairing both the target right hand and preserve left hand helps weak hard-sequence families",
            "family_budget": FAMILY_BUDGET,
            "depth": DEPTH,
            "left_mode": LEFT_MODE,
            "right_modes": list(RIGHT_MODES),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_dual_repair_sweep.json"
    out_md = SUM / "interaction_realized_dual_repair_sweep.md"
    out_json.write_text(json.dumps(payload, indent=2))

    key_methods = [
        "single",
        "hgb_none_dual",
        "hgb_edge_token_snap_dual",
        "hgb_edge_transition_snap_dual",
        "hgb_finger_profile_snap_dual",
        "hgb_full_donor_oracle_dual",
        "mlp_finger_profile_snap_dual",
        "base_finger_profile_snap_dual",
    ]
    lines = [
        "# Interaction Realized Dual Repair Sweep",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Budget fixed at 2 and depth fixed at 20; compare right-target repair modes on top of left finger-profile repair.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([f"## {task_name}", "", "| method | avail | right grouped | left preserve | joint overall |", "| --- | ---: | ---: | ---: | ---: |"])
        for method in key_methods:
            stats = result["summary"][method]
            lines.append(f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |")
        lines.extend(["", "### Sequence Summary", "", "| sequence | hgb none | hgb edge transition | hgb finger profile | hgb full donor oracle |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['hgb_none_dual_joint_score_overall'])} | {fmt(row['hgb_edge_transition_snap_dual_joint_score_overall'])} | "
                f"{fmt(row['hgb_finger_profile_snap_dual_joint_score_overall'])} | {fmt(row['hgb_full_donor_oracle_dual_joint_score_overall'])} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
