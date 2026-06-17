#!/usr/bin/env python3
"""Learned chunk-level scorers for fast-path closing support.

This is an experiment memo, not paper text.

Motivation:
- learned binary run scorers improved closing from 0.6786 to 0.7143
- the remaining gap appears to come from within-run heterogeneity

Approach:
- split each temporal run into shorter chunks
- learn binary chunk-level scorers (`edge` vs `finger`)
- compare chunk lengths on the same fast-path closing regime
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tools.build_interaction_realized_closing_run_learned_scorer import (
    CACHE_DIR,
    EXTERNAL_RELAXED_RUN_GAP,
    TASK_TARGET,
    VAL_RUN_GAP,
    all_sequence_labels,
    run_feature_vector,
    summarize_rows,
    paired_stats,
    subset_to_closing_sequences,
)
from tools.build_interaction_realized_closing_run_target import build_runs, summarize_run, fmt
from tools.build_interaction_realized_closing_selector_support_scaling import collect_rows_fast
from tools.build_interaction_realized_feasible_left_temporal_window_knn import build_frames
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)


CHUNK_LENGTHS = (2, 3)
CFG = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 2,
    "max_iter": 150,
}


def split_run_into_chunks(run_rows, max_chunk_len):
    chunks = []
    cur = []
    cur_key = None
    for row in run_rows:
        key = (row["other_hand_motion"], row["interaction_motion_value"])
        if not cur:
            cur = [row]
            cur_key = key
            continue
        if len(cur) >= max_chunk_len or key != cur_key:
            chunks.append(cur)
            cur = [row]
            cur_key = key
        else:
            cur.append(row)
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(rows, run_gap, max_chunk_len):
    chunks = []
    for run in build_runs(rows, run_gap):
        chunks.extend(split_run_into_chunks(run, max_chunk_len))
    return chunks


def summarize_chunks(rows, run_gap, max_chunk_len):
    return [summarize_run(chunk) for chunk in build_chunks(rows, run_gap, max_chunk_len)]


def evaluate_modes(chunks, pred_modes):
    total = 0.0
    n = 0
    for chunk, mode in zip(chunks, pred_modes):
        total += chunk["mode_means"][mode] * chunk["num_frames"]
        n += chunk["num_frames"]
    return total / n


def fit_classifier(train_chunks):
    labels = [chunk["best_mode"] for chunk in train_chunks]
    unique = sorted(set(labels))
    if len(unique) == 1:
        return {"kind": "constant", "label": unique[0]}
    x = np.stack([run_feature_vector(chunk) for chunk in train_chunks], axis=0)
    y = np.asarray(labels)
    model = HistGradientBoostingClassifier(
        learning_rate=CFG["learning_rate"],
        max_depth=CFG["max_depth"],
        min_samples_leaf=CFG["min_samples_leaf"],
        max_iter=CFG["max_iter"],
        random_state=0,
    )
    model.fit(x, y)
    return {"kind": "hgb", "model": model}


def predict_classifier(model_obj, chunk):
    if model_obj["kind"] == "constant":
        return model_obj["label"]
    return str(model_obj["model"].predict(run_feature_vector(chunk).reshape(1, -1))[0])


def fit_regressors(train_chunks):
    x = np.stack([run_feature_vector(chunk) for chunk in train_chunks], axis=0)
    out = {}
    for mode in ("edge_transition_snap", "finger_profile_snap"):
        y = np.asarray([chunk["mode_means"][mode] for chunk in train_chunks], dtype=np.float32)
        if float(np.max(y) - np.min(y)) == 0.0:
            out[mode] = {"kind": "constant", "value": float(y[0])}
            continue
        model = HistGradientBoostingRegressor(
            learning_rate=CFG["learning_rate"],
            max_depth=CFG["max_depth"],
            min_samples_leaf=CFG["min_samples_leaf"],
            max_iter=CFG["max_iter"],
            random_state=0,
        )
        model.fit(x, y)
        out[mode] = {"kind": "hgb", "model": model}
    return out


def predict_regressor(model_map, chunk):
    x = run_feature_vector(chunk).reshape(1, -1)
    scores = {}
    for mode in ("edge_transition_snap", "finger_profile_snap"):
        model_obj = model_map[mode]
        if model_obj["kind"] == "constant":
            scores[mode] = model_obj["value"]
        else:
            scores[mode] = float(model_obj["model"].predict(x)[0])
    return max(scores, key=scores.get), scores


def train_stats_classifier(train_chunks):
    model = fit_classifier(train_chunks)
    pred = [predict_classifier(model, chunk) for chunk in train_chunks]
    return model, {
        "cfg": CFG,
        "train_joint_score": evaluate_modes(train_chunks, pred),
        "pred_mode_counts": dict(Counter(pred)),
    }


def train_stats_regressor(train_chunks):
    model = fit_regressors(train_chunks)
    pred = [predict_regressor(model, chunk)[0] for chunk in train_chunks]
    return model, {
        "cfg": CFG,
        "train_joint_score": evaluate_modes(train_chunks, pred),
        "pred_mode_counts": dict(Counter(pred)),
    }


def apply_chunk_predictions(test_chunks, pred_modes):
    output_rows = []
    for chunk, pred_mode in zip(test_chunks, pred_modes):
        for row in chunk["rows"]:
            rec = dict(row)
            oracle_binary = (
                "finger_profile_snap"
                if float(row["left_finger_profile_snap_joint_score"]) > float(row["left_edge_transition_snap_joint_score"])
                else "edge_transition_snap"
            )
            for out_name, out_mode in [
                ("fixed_edge", "edge_transition_snap"),
                ("chunk_target", pred_mode),
                ("oracle_binary", oracle_binary),
            ]:
                prefix = f"left_{out_mode}"
                for field in ("right_grouped_match", "left_preserve", "joint_score"):
                    rec[f"{out_name}_{field}"] = rec[f"{prefix}_{field}"]
            output_rows.append(rec)
    return output_rows


def evaluate_support(rows_train, rows_test, rows_external, chunk_len):
    train_chunks = summarize_chunks(rows_train, VAL_RUN_GAP, chunk_len)
    test_chunks = summarize_chunks(rows_test, VAL_RUN_GAP, chunk_len)
    ext_singletons = [summarize_run([row]) for row in rows_external]
    ext_relaxed = summarize_chunks(rows_external, EXTERNAL_RELAXED_RUN_GAP, chunk_len)

    support_sets = {
        "val_chunks_only": train_chunks,
        "val_chunks_plus_top4_singletons": train_chunks + ext_singletons,
        "val_chunks_plus_top4_relaxed_chunks": train_chunks + ext_relaxed,
    }
    results = {}
    for name, support_train in support_sets.items():
        cls_model, cls_stats = train_stats_classifier(support_train)
        reg_model, reg_stats = train_stats_regressor(support_train)
        cls_pred = [predict_classifier(cls_model, chunk) for chunk in test_chunks]
        reg_pred = [predict_regressor(reg_model, chunk)[0] for chunk in test_chunks]

        cls_rows = apply_chunk_predictions(test_chunks, cls_pred)
        reg_rows = apply_chunk_predictions(test_chunks, reg_pred)
        # reuse row lists by attaching alt predictions onto cls rows
        merged_rows = []
        for row_cls, row_reg in zip(cls_rows, reg_rows):
            rec = dict(row_cls)
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"chunk_reg_{field}"] = row_reg[f"chunk_target_{field}"]
            merged_rows.append(rec)

        results[name] = {
            "num_support_chunks": len(support_train),
            "stats": {
                "chunk_cls": cls_stats,
                "chunk_reg": reg_stats,
            },
            "summary": {
                "fixed_edge": summarize_rows(merged_rows, "fixed_edge"),
                "chunk_cls": summarize_rows(merged_rows, "chunk_target"),
                "chunk_reg": summarize_rows(merged_rows, "chunk_reg"),
                "oracle_binary": summarize_rows(merged_rows, "oracle_binary"),
            },
            "paired": {
                "chunk_cls_vs_fixed_edge": paired_stats(merged_rows, "fixed_edge", "chunk_target"),
                "chunk_reg_vs_fixed_edge": paired_stats(merged_rows, "fixed_edge", "chunk_reg"),
                "chunk_cls_vs_chunk_reg": paired_stats(merged_rows, "chunk_reg", "chunk_target"),
            },
            "pred_mode_counts": {
                "chunk_cls": dict(Counter(cls_pred)),
                "chunk_reg": dict(Counter(reg_pred)),
            },
        }
    return results


def main():
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    val_subset = subset_to_closing_sequences(val_data)
    test_subset = subset_to_closing_sequences(test_data)

    val_labels = all_sequence_labels(val_subset)
    val_sem = build_semantic_frame_vocab(val_subset, val_labels)
    val_bank = build_pair_bank(val_subset, val_labels, val_sem)
    val_frames = build_frames(val_subset, val_labels, TASK_TARGET)
    pair_model, pair_stats = train_pairguided_model(val_frames, val_bank, TASK_TARGET)

    val_rows = collect_rows_fast(val_frames, val_bank, pair_model)
    test_rows = collect_rows_fast(build_frames(test_subset, all_sequence_labels(test_subset), TASK_TARGET), val_bank, pair_model)
    cached_top4_rows = load_json(CACHE_DIR / "external_rows_top4.json")["rows"]

    chunk_results = {}
    for chunk_len in CHUNK_LENGTHS:
        chunk_results[f"chunk_len_{chunk_len}"] = evaluate_support(val_rows, test_rows, cached_top4_rows, chunk_len)

    payload = {
        "focus": {
            "goal": "learned chunk-level scorers for fast-path closing",
            "task": "right_hand_motion->closing",
        },
        "training_stats": {
            "pair_model": pair_stats,
            "num_val_rows": len(val_rows),
            "num_test_rows": len(test_rows),
            "num_cached_top4_rows": len(cached_top4_rows),
            "val_run_gap": VAL_RUN_GAP,
            "external_relaxed_run_gap": EXTERNAL_RELAXED_RUN_GAP,
            "chunk_lengths": list(CHUNK_LENGTHS),
        },
        "chunk_results": chunk_results,
    }

    out_json = GEN / "interaction_realized_closing_chunk_learned_scorer.json"
    out_md = SUM / "interaction_realized_closing_chunk_learned_scorer.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Chunk Learned Scorer",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Val rows: `{len(val_rows)}`; test rows: `{len(test_rows)}`; cached top4 rows: `{len(cached_top4_rows)}`",
        "",
    ]
    for chunk_name, chunk_payload in chunk_results.items():
        lines.extend([f"## {chunk_name}", ""])
        for support_name, result in chunk_payload.items():
            lines.extend(
                [
                    f"### {support_name}",
                    "",
                    f"Support chunks: `{result['num_support_chunks']}`",
                    f"Stats: `{result['stats']}`",
                    "",
                    "| method | right grouped | left preserve | joint overall |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for method in ("fixed_edge", "chunk_cls", "chunk_reg", "oracle_binary"):
                stats = result["summary"][method]
                lines.append(
                    f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
                )
            lines.extend(
                [
                    "",
                    f"Pred chunk modes: `{result['pred_mode_counts']}`",
                    "",
                    "| comparison | delta | wins | losses | ties |",
                    "| --- | ---: | ---: | ---: | ---: |",
                    f"| chunk_cls vs fixed_edge | {fmt(result['paired']['chunk_cls_vs_fixed_edge']['delta'])} | {result['paired']['chunk_cls_vs_fixed_edge']['wins']} | {result['paired']['chunk_cls_vs_fixed_edge']['losses']} | {result['paired']['chunk_cls_vs_fixed_edge']['ties']} |",
                    f"| chunk_reg vs fixed_edge | {fmt(result['paired']['chunk_reg_vs_fixed_edge']['delta'])} | {result['paired']['chunk_reg_vs_fixed_edge']['wins']} | {result['paired']['chunk_reg_vs_fixed_edge']['losses']} | {result['paired']['chunk_reg_vs_fixed_edge']['ties']} |",
                    f"| chunk_cls vs chunk_reg | {fmt(result['paired']['chunk_cls_vs_chunk_reg']['delta'])} | {result['paired']['chunk_cls_vs_chunk_reg']['wins']} | {result['paired']['chunk_cls_vs_chunk_reg']['losses']} | {result['paired']['chunk_cls_vs_chunk_reg']['ties']} |",
                    "",
                ]
            )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
