#!/usr/bin/env python3
"""Small-scale external support scaling for hard feasible closing.

This is an experiment memo, not paper text.

Rather than waiting for full train support to finish, probe whether the first
few high-density train closing sequences already help the selector.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
    build_frames,
    choose_cfg,
    predict_mode,
    summarize,
    paired_stats,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import pick_best_from_cache
from tools.build_interaction_realized_constraint_sweep import evaluate_edit
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_interaction_realized_pairguided_editor import select_pairguided_left_pool
from tools.build_interaction_realized_right_support_sweep import reorder_to_budget, right_candidates_mode
from tools.build_pairguided_reranker_multislice import relaxed_left_family_candidates_with_meta
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)


TASK_TARGET = "closing"
TOP_SEQUENCE_COUNTS = (0, 2, 4)
RIGHT_CAP = 5
LEFT_CAP = 5
FAMILY_BUDGET = 2
ROOT = Path("/opt/tiger/hand")
CACHE_DIR = ROOT / "experiments/generated/cache/closing_selector_support_scaling"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def all_sequence_labels(data):
    return {seq["seq_name"] for seq in data["sequences"]}


def unique_sequences_by_name(data):
    seen = set()
    out = []
    for seq in data["sequences"]:
        name = seq["seq_name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(seq)
    return {"sequences": out}


def subset_to_closing_sequences(data):
    unique = unique_sequences_by_name(data)
    keep = {row["seq_name"] for row in build_frames(unique, all_sequence_labels(unique), TASK_TARGET)}
    return {"sequences": [seq for seq in unique["sequences"] if seq["seq_name"] in keep]}


def top_n_sequences_by_closing_rows(data, top_n: int):
    unique = unique_sequences_by_name(data)
    rows = build_frames(unique, all_sequence_labels(unique), TASK_TARGET)
    counts = Counter(row["seq_name"] for row in rows)
    keep = {name for name, _ in counts.most_common(top_n)}
    return {"sequences": [seq for seq in unique["sequences"] if seq["seq_name"] in keep]}


def merge_data(*datasets):
    out = {"sequences": []}
    for dataset in datasets:
        out["sequences"].extend(dataset["sequences"])
    return out


def evaluate_selector_support(selector_train_rows, test_rows):
    best = choose_cfg(selector_train_rows)
    output_rows = []
    pred_modes = []
    for row in test_rows:
        pred_mode, _ = predict_mode(selector_train_rows, row, best["cfg"])
        pred_modes.append(pred_mode)
        rec = dict(row)
        oracle_mode = max(("none", "edge_transition_snap", "finger_profile_snap"), key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_edge", "edge_transition_snap"),
            ("temporal_window_knn", pred_mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        output_rows.append(rec)
    return {
        "selector_stats": best,
        "summary": {name: summarize(output_rows, name) for name in ("fixed_edge", "temporal_window_knn", "oracle")},
        "paired": {
            "temporal_window_knn_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "temporal_window_knn"),
            "oracle_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle"),
        },
        "pred_mode_counts": dict(Counter(pred_modes)),
    }


def collect_rows_fast(frames, pair_bank, hgb_model):
    rows = []
    for entry in frames:
        if entry["curr_frame"].get("left") is None:
            continue
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        raw_left_pool = relaxed_left_family_candidates_with_meta(
            pair_bank,
            current_left_group,
            curr_attrs,
            prev_geom,
            curr_geom,
            "left",
            FAMILY_BUDGET,
        )
        target_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, TASK_TARGET, "relax_both")[:RIGHT_CAP]
        ranked_left_pool = select_pairguided_left_pool(
            hgb_model,
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            current_left_group,
            target_pool,
        )
        left_pool = reorder_to_budget(ranked_left_pool, raw_left_pool)[:LEFT_CAP]
        choice = pick_best_from_cache(prev_frame, curr_frame, prev_geom, target_pool, left_pool, LEFT_CAP, {}) if target_pool and left_pool else None
        right_row = None if choice is None else choice[1]
        left_row = None if choice is None else choice[2]

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }
        for mode in ("none", "edge_transition_snap", "finger_profile_snap"):
            prefix = f"left_{mode}"
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                right_row=right_row,
                left_row=left_row,
                repair_mode=mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, choices=TOP_SEQUENCE_COUNTS, default=None, help="Run only one external support budget.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cached external support rows and rebuild them.")
    return parser.parse_args()


def cache_path(top_n: int) -> Path:
    return CACHE_DIR / f"external_rows_top{top_n}.json"


def load_or_build_external_rows(external_subset, ext_labels, pair_model, top_n: int, force_recompute: bool):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(top_n)
    if path.exists() and not force_recompute:
        return json.loads(path.read_text())

    ext_semantic_vocab = build_semantic_frame_vocab(external_subset, ext_labels)
    ext_pair_bank = build_pair_bank(external_subset, ext_labels, ext_semantic_vocab)
    ext_rows = attach_context(collect_rows_fast(build_frames(external_subset, ext_labels, TASK_TARGET), ext_pair_bank, pair_model))
    payload = {
        "num_external_sequences": len(external_subset["sequences"]),
        "num_external_rows": len(ext_rows),
        "external_pair_bank_size": len(ext_pair_bank),
        "rows": ext_rows,
    }
    path.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    args = parse_args()
    train_data = load_json(GEN / "temporal_hl_train.json")
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    train_subset = subset_to_closing_sequences(train_data)
    val_subset = subset_to_closing_sequences(val_data)
    test_subset = subset_to_closing_sequences(test_data)

    val_labels = all_sequence_labels(val_subset)
    val_semantic_vocab = build_semantic_frame_vocab(val_subset, val_labels)
    val_pair_bank = build_pair_bank(val_subset, val_labels, val_semantic_vocab)
    val_train_frames = build_frames(val_subset, val_labels, TASK_TARGET)
    pair_model, pair_stats = train_pairguided_model(val_train_frames, val_pair_bank, TASK_TARGET)

    val_selector_rows = attach_context(collect_rows_fast(val_train_frames, val_pair_bank, pair_model))
    test_rows = attach_context(collect_rows_fast(build_frames(test_subset, all_sequence_labels(test_subset), TASK_TARGET), val_pair_bank, pair_model))

    budgets = TOP_SEQUENCE_COUNTS if args.top_n is None else (args.top_n,)
    task_results = {}
    for top_n in budgets:
        if top_n == 0:
            selector_rows = list(val_selector_rows)
            meta = {"num_external_sequences": 0, "num_external_rows": 0}
        else:
            external_subset = top_n_sequences_by_closing_rows(train_subset, top_n)
            ext_labels = all_sequence_labels(external_subset)
            ext_payload = load_or_build_external_rows(external_subset, ext_labels, pair_model, top_n, args.force_recompute)
            ext_rows = ext_payload["rows"]
            selector_rows = ext_rows + val_selector_rows
            meta = {
                "num_external_sequences": int(ext_payload["num_external_sequences"]),
                "num_external_rows": int(ext_payload["num_external_rows"]),
                "external_pair_bank_size": int(ext_payload["external_pair_bank_size"]),
                "cache_path": str(cache_path(top_n)),
            }
        result = evaluate_selector_support(selector_rows, test_rows)
        result["support_meta"] = meta
        task_results[f"top_{top_n}"] = result

    payload = {
        "focus": {
            "goal": "small-scale selector support scaling for hard feasible closing",
            "task": "right_hand_motion->closing",
            "budgets": list(budgets),
        },
        "training_stats": {
            "pair_model": pair_stats,
            "num_val_selector_rows": len(val_selector_rows),
            "val_pair_bank_size": len(val_pair_bank),
        },
        "task_results": task_results,
    }

    suffix = "" if args.top_n is None else f"_top{args.top_n}"
    out_json = GEN / f"interaction_realized_closing_selector_support_scaling{suffix}.json"
    out_md = SUM / f"interaction_realized_closing_selector_support_scaling{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Selector Support Scaling",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Val selector rows: `{len(val_selector_rows)}`",
        f"Fast path caps: right `{RIGHT_CAP}`, left `{LEFT_CAP}`",
        f"Cache dir: `{CACHE_DIR}`",
        "",
    ]
    for name, result in task_results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"Support meta: `{result['support_meta']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_edge", "temporal_window_knn", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        paired = result["paired"]
        best = result["selector_stats"]
        lines.extend(
            [
                "",
                f"Selected config: `{best['cfg']}`, leave-one-out joint `{fmt(best['leave_one_out_joint_score'])}`",
                "",
                f"Pred mode counts: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| temporal_window_knn vs fixed_edge | {fmt(paired['temporal_window_knn_vs_fixed_edge']['delta'])} | {paired['temporal_window_knn_vs_fixed_edge']['wins']} | {paired['temporal_window_knn_vs_fixed_edge']['losses']} | {paired['temporal_window_knn_vs_fixed_edge']['ties']} |",
                f"| oracle vs fixed_edge | {fmt(paired['oracle_vs_fixed_edge']['delta'])} | {paired['oracle_vs_fixed_edge']['wins']} | {paired['oracle_vs_fixed_edge']['losses']} | {paired['oracle_vs_fixed_edge']['ties']} |",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
