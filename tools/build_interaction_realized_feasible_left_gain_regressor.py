#!/usr/bin/env python3
"""Dense gain regressor for feasible left repair.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.build_interaction_realized_feasible_left_dense_knn import attach_signatures, build_frames
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    INTERACTION_VALUES,
    OTHER_VALUES,
    attach_context,
)
from tools.build_interaction_realized_pairguided_editor import TASKS, train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
    overlap_labels,
)


TASK_BEST = {
    "closing": "edge_transition_snap",
    "opening": "finger_profile_snap",
}
LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
DELTA_MODES = LEFT_REPAIR_MODES[1:]
REG_CFG = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 10, "max_iter": 100}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def parse_signature(signature: str) -> list[str]:
    return signature.split("|") if signature else []


def build_signature_vocab(rows):
    left_vocab = sorted({tok for row in rows for tok in parse_signature(row["left_state_signature"])})
    right_vocab = sorted({tok for row in rows for tok in parse_signature(row["right_state_signature"])})
    return {"left": left_vocab, "right": right_vocab}


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def signature_one_hot(tokens: list[str], vocab: list[str], length: int = 20) -> list[float]:
    out = []
    pad = "__missing__"
    full_vocab = vocab + ([pad] if pad not in vocab else [])
    padded = list(tokens[:length]) + [pad] * max(0, length - len(tokens))
    for tok in padded[:length]:
        out.extend(one_hot(tok, full_vocab))
    return out


def feature_vector(row, sig_vocab):
    feats = []
    for prefix in ("prev", "curr", "next"):
        feats.extend(one_hot(row[f"{prefix}_other_hand_motion"], OTHER_VALUES))
        feats.extend(one_hot(row[f"{prefix}_interaction_motion_value"], INTERACTION_VALUES))
        feats.extend(
            [
                float(row[f"{prefix}_none_left_state"]),
                float(row[f"{prefix}_none_left_transition"]),
                float(row[f"{prefix}_none_right_state"]),
                float(row[f"{prefix}_none_right_transition"]),
            ]
        )
    feats.extend(signature_one_hot(parse_signature(row["left_state_signature"]), sig_vocab["left"]))
    feats.extend(signature_one_hot(parse_signature(row["right_state_signature"]), sig_vocab["right"]))
    return np.asarray(feats, dtype=np.float32)


def convert_rows(rows, sig_vocab):
    out = []
    for row in rows:
        rec = dict(row)
        rec["x"] = feature_vector(row, sig_vocab)
        rec["joint_none"] = float(row["left_none_joint_score"])
        for mode in DELTA_MODES:
            rec[f"delta_{mode}"] = float(row[f"left_{mode}_joint_score"]) - rec["joint_none"]
        out.append(rec)
    return out


def fit_regressors(train_rows, cfg):
    x = np.stack([row["x"] for row in train_rows], axis=0)
    models = {}
    for mode in DELTA_MODES:
        y = np.asarray([row[f"delta_{mode}"] for row in train_rows], dtype=np.float32)
        model = HistGradientBoostingRegressor(
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_iter=cfg["max_iter"],
            random_state=0,
        )
        model.fit(x, y)
        models[mode] = model
    return models


def predict_mode(models, row):
    scores = {"none": row["joint_none"]}
    feat = row["x"].reshape(1, -1)
    for mode in DELTA_MODES:
        pred_delta = float(models[mode].predict(feat)[0])
        scores[mode] = row["joint_none"] + pred_delta
    pred_mode = max(scores, key=scores.get)
    return pred_mode, scores


def evaluate_prediction(rows, pred_modes):
    return sum(float(row[f"left_{mode}_joint_score"]) for row, mode in zip(rows, pred_modes)) / len(rows)


def summarize(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(float(row[f"{prefix}_right_grouped_match"]) for row in rows) / n,
        "left_preserve_overall": sum(float(row[f"{prefix}_left_preserve"]) for row in rows) / n,
        "joint_score_overall": sum(float(row[f"{prefix}_joint_score"]) for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    a = [float(row[f"{a_prefix}_joint_score"]) for row in rows]
    b = [float(row[f"{b_prefix}_joint_score"]) for row in rows]
    diff = [y - x for x, y in zip(a, b)]
    return {
        "delta": sum(diff) / len(diff),
        "wins": int(sum(x > 0 for x in diff)),
        "losses": int(sum(x < 0 for x in diff)),
        "ties": int(sum(x == 0 for x in diff)),
    }


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)

    train_rows_raw = attach_context(attach_signatures(collect_rows(train_frames, pair_bank, task_target, pair_model), train_frames))
    test_rows_raw = attach_context(attach_signatures(collect_rows(test_frames, pair_bank, task_target, pair_model), test_frames))
    sig_vocab = build_signature_vocab(train_rows_raw)
    train_rows = convert_rows(train_rows_raw, sig_vocab)
    test_rows = convert_rows(test_rows_raw, sig_vocab)

    models = fit_regressors(train_rows, REG_CFG)

    fixed_mode = TASK_BEST[task_target]
    pred_modes = []
    pred_scores = []
    output_rows = []
    for raw_row, row in zip(test_rows_raw, test_rows):
        pred_mode, scores = predict_mode(models, row)
        pred_modes.append(pred_mode)
        pred_scores.append(scores)
        rec = dict(raw_row)
        oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_none", "none"),
            ("fixed_task_best", fixed_mode),
            ("gain_regressor", pred_mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["pred_mode"] = pred_mode
        rec["pred_scores"] = scores
        rec["oracle_mode"] = oracle_mode
        output_rows.append(rec)

    summary = {name: summarize(output_rows, name) for name in ("fixed_none", "fixed_task_best", "gain_regressor", "oracle")}
    paired = {
        "gain_regressor_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "gain_regressor"),
        "gain_regressor_vs_fixed_none": paired_stats(output_rows, "fixed_none", "gain_regressor"),
        "oracle_vs_gain_regressor": paired_stats(output_rows, "gain_regressor", "oracle"),
    }
    return {
        "training_stats": {
            "pair_model": pair_stats,
            "sig_vocab_sizes": {k: len(v) for k, v in sig_vocab.items()},
            "gain_regressor": {
                "cfg": REG_CFG,
                "train_fit_joint_score": evaluate_prediction(
                    train_rows,
                    [predict_mode(models, row)[0] for row in train_rows],
                ),
            },
        },
        "summary": summary,
        "paired": paired,
        "pred_mode_counts": dict(Counter(pred_modes)),
        "rows": output_rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for _, task_target in TASKS:
        task_results[f"right_hand_motion->{task_target}"] = run_task(train_data, test_data, labels, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "dense gain regressor for feasible left repair",
            "tasks": [f"right_hand_motion->{target}" for _, target in TASKS],
            "repair_modes": list(LEFT_REPAIR_MODES),
            "config": REG_CFG,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_gain_regressor.json"
    out_md = SUM / "interaction_realized_feasible_left_gain_regressor.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Gain Regressor",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Dense gain regression with current/prev/next feasible context and state-signature features.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_none", "fixed_task_best", "gain_regressor", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                f"Config: `{result['training_stats']['gain_regressor']['cfg']}`, train-fit joint `{fmt(result['training_stats']['gain_regressor']['train_fit_joint_score'])}`",
                f"Pred mode counts: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| gain_regressor vs fixed_task_best | {fmt(result['paired']['gain_regressor_vs_fixed_task_best']['delta'])} | {result['paired']['gain_regressor_vs_fixed_task_best']['wins']} | {result['paired']['gain_regressor_vs_fixed_task_best']['losses']} | {result['paired']['gain_regressor_vs_fixed_task_best']['ties']} |",
                f"| gain_regressor vs fixed_none | {fmt(result['paired']['gain_regressor_vs_fixed_none']['delta'])} | {result['paired']['gain_regressor_vs_fixed_none']['wins']} | {result['paired']['gain_regressor_vs_fixed_none']['losses']} | {result['paired']['gain_regressor_vs_fixed_none']['ties']} |",
                f"| oracle vs gain_regressor | {fmt(result['paired']['oracle_vs_gain_regressor']['delta'])} | {result['paired']['oracle_vs_gain_regressor']['wins']} | {result['paired']['oracle_vs_gain_regressor']['losses']} | {result['paired']['oracle_vs_gain_regressor']['ties']} |",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
