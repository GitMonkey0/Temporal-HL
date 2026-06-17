#!/usr/bin/env python3
"""Run-level learned scorers for feasible left repair.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    INTERACTION_VALUES,
    OTHER_VALUES,
    build_frames,
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


LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
TASK_BEST = {
    "closing": "edge_transition_snap",
    "opening": "finger_profile_snap",
}
RUN_GAP = 12
CLS_CFG = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 2, "max_iter": 150}
REG_CFG = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 2, "max_iter": 150}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_runs(rows):
    rows = sorted(rows, key=lambda r: (r["seq_name"], int(r["frame_idx"])))
    runs = []
    cur = []
    prev = None
    for row in rows:
        contiguous = False
        if prev is not None:
            contiguous = row["seq_name"] == prev["seq_name"] and int(row["frame_idx"]) - int(prev["frame_idx"]) <= RUN_GAP
        if prev is None or contiguous:
            cur.append(row)
        else:
            runs.append(cur)
            cur = [row]
        prev = row
    if cur:
        runs.append(cur)
    return runs


def summarize_run(run):
    subtype_counts = Counter((row["other_hand_motion"], row["interaction_motion_value"]) for row in run)
    mode_means = {
        mode: sum(float(row[f"left_{mode}_joint_score"]) for row in run) / len(run)
        for mode in LEFT_REPAIR_MODES
    }
    return {
        "rows": run,
        "num_frames": len(run),
        "subtype_counts": subtype_counts,
        "left_state_mean": sum(float(row["left_none_left_state_agreement"]) for row in run) / len(run),
        "left_transition_mean": sum(float(row["left_none_left_transition_agreement"]) for row in run) / len(run),
        "right_state_mean": sum(float(row["left_none_right_state_agreement"]) for row in run) / len(run),
        "right_transition_mean": sum(float(row["left_none_right_transition_agreement"]) for row in run) / len(run),
        "none_mean": mode_means["none"],
        "edge_mean": mode_means["edge_transition_snap"],
        "finger_mean": mode_means["finger_profile_snap"],
        "mode_means": mode_means,
        "best_mode": max(LEFT_REPAIR_MODES, key=lambda mode: mode_means[mode]),
    }


def run_feature_vector(run):
    subtype_total = sum(run["subtype_counts"].values()) or 1
    feats = [
        float(run["num_frames"]),
        float(run["left_state_mean"]),
        float(run["left_transition_mean"]),
        float(run["right_state_mean"]),
        float(run["right_transition_mean"]),
        float(run["none_mean"]),
        float(run["edge_mean"]),
        float(run["finger_mean"]),
        float(run["edge_mean"] - run["none_mean"]),
        float(run["finger_mean"] - run["none_mean"]),
        float(run["finger_mean"] - run["edge_mean"]),
    ]
    for value in OTHER_VALUES:
        feats.append(sum(freq for key, freq in run["subtype_counts"].items() if key[0] == value) / subtype_total)
    for value in INTERACTION_VALUES:
        feats.append(sum(freq for key, freq in run["subtype_counts"].items() if key[1] == value) / subtype_total)
    return np.asarray(feats, dtype=np.float32)


def evaluate_run_modes(run_summaries, pred_modes):
    total = 0.0
    n = 0
    for summary, mode in zip(run_summaries, pred_modes):
        total += summary["mode_means"][mode] * summary["num_frames"]
        n += summary["num_frames"]
    return total / n


def fit_classifier(train_runs):
    labels = [run["best_mode"] for run in train_runs]
    unique = sorted(set(labels))
    if len(unique) == 1:
        return {"kind": "constant", "label": unique[0]}
    x = np.stack([run_feature_vector(run) for run in train_runs], axis=0)
    y = np.asarray(labels)
    model = HistGradientBoostingClassifier(
        learning_rate=CLS_CFG["learning_rate"],
        max_depth=CLS_CFG["max_depth"],
        min_samples_leaf=CLS_CFG["min_samples_leaf"],
        max_iter=CLS_CFG["max_iter"],
        random_state=0,
    )
    model.fit(x, y)
    return {"kind": "hgb", "model": model}


def predict_classifier(model_obj, run):
    if model_obj["kind"] == "constant":
        return model_obj["label"]
    return str(model_obj["model"].predict(run_feature_vector(run).reshape(1, -1))[0])


def fit_regressors(train_runs):
    x = np.stack([run_feature_vector(run) for run in train_runs], axis=0)
    model_map = {}
    for mode in LEFT_REPAIR_MODES:
        y = np.asarray([run["mode_means"][mode] for run in train_runs], dtype=np.float32)
        if float(np.max(y) - np.min(y)) == 0.0:
            model_map[mode] = {"kind": "constant", "value": float(y[0])}
            continue
        model = HistGradientBoostingRegressor(
            learning_rate=REG_CFG["learning_rate"],
            max_depth=REG_CFG["max_depth"],
            min_samples_leaf=REG_CFG["min_samples_leaf"],
            max_iter=REG_CFG["max_iter"],
            random_state=0,
        )
        model.fit(x, y)
        model_map[mode] = {"kind": "hgb", "model": model}
    return model_map


def predict_regressor(model_map, run):
    x = run_feature_vector(run).reshape(1, -1)
    scores = {}
    for mode in LEFT_REPAIR_MODES:
        obj = model_map[mode]
        scores[mode] = obj["value"] if obj["kind"] == "constant" else float(obj["model"].predict(x)[0])
    return max(scores, key=scores.get), scores


def summarize_rows(rows, prefix):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(float(r[f"{prefix}_right_grouped_match"]) for r in rows) / n,
        "left_preserve_overall": sum(float(r[f"{prefix}_left_preserve"]) for r in rows) / n,
        "joint_score_overall": sum(float(r[f"{prefix}_joint_score"]) for r in rows) / n,
    }


def paired_stats(rows, a_prefix, b_prefix):
    a = [float(r[f"{a_prefix}_joint_score"]) for r in rows]
    b = [float(r[f"{b_prefix}_joint_score"]) for r in rows]
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
    train_rows = collect_rows(train_frames, pair_bank, task_target, pair_model)
    test_rows = collect_rows(test_frames, pair_bank, task_target, pair_model)

    train_runs = [summarize_run(run) for run in build_runs(train_rows)]
    test_runs = [summarize_run(run) for run in build_runs(test_rows)]

    cls_model = fit_classifier(train_runs)
    reg_model = fit_regressors(train_runs)
    cls_pred = [predict_classifier(cls_model, run) for run in test_runs]
    reg_pred = [predict_regressor(reg_model, run)[0] for run in test_runs]

    fixed_mode = TASK_BEST[task_target]
    output_rows = []
    for run, cls_mode, reg_mode in zip(test_runs, cls_pred, reg_pred):
        run_oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: run["mode_means"][mode])
        for row in run["rows"]:
            rec = dict(row)
            frame_oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"]))
            for out_name, out_mode in [
                ("fixed_task_best", fixed_mode),
                ("run_cls", cls_mode),
                ("run_reg", reg_mode),
                ("run_oracle", run_oracle_mode),
                ("frame_oracle", frame_oracle_mode),
            ]:
                prefix = f"left_{out_mode}"
                for field in ("right_grouped_match", "left_preserve", "joint_score"):
                    rec[f"{out_name}_{field}"] = rec[f"{prefix}_{field}"]
            rec["run_cls_mode"] = cls_mode
            rec["run_reg_mode"] = reg_mode
            rec["run_oracle_mode"] = run_oracle_mode
            rec["frame_oracle_mode"] = frame_oracle_mode
            output_rows.append(rec)

    payload = {
        "training_stats": {
            "pair_model": pair_stats,
            "num_train_runs": len(train_runs),
            "num_test_runs": len(test_runs),
            "run_cls_train_fit": evaluate_run_modes(train_runs, [predict_classifier(cls_model, run) for run in train_runs]),
            "run_reg_train_fit": evaluate_run_modes(train_runs, [predict_regressor(reg_model, run)[0] for run in train_runs]),
        },
        "summary": {
            name: summarize_rows(output_rows, name)
            for name in ("fixed_task_best", "run_cls", "run_reg", "run_oracle", "frame_oracle")
        },
        "paired": {
            "run_cls_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "run_cls"),
            "run_reg_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "run_reg"),
            "run_oracle_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "run_oracle"),
            "frame_oracle_vs_run_reg": paired_stats(output_rows, "run_reg", "frame_oracle"),
        },
        "pred_mode_counts": {
            "run_cls": dict(Counter(cls_pred)),
            "run_reg": dict(Counter(reg_pred)),
        },
        "rows": output_rows,
    }
    return payload


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
            "goal": "run-level learned scorers for feasible left repair",
            "tasks": [f"right_hand_motion->{target}" for _, target in TASKS],
            "run_gap": RUN_GAP,
            "classifier_cfg": CLS_CFG,
            "regressor_cfg": REG_CFG,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_run_learned_scorer.json"
    out_md = SUM / "interaction_realized_feasible_left_run_learned_scorer.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Run Learned Scorer",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Learn run-level preserve-side mode policies directly on the feasible subset and compare them against fixed, run-oracle, and frame-oracle bounds.",
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
        for method in ("fixed_task_best", "run_cls", "run_reg", "run_oracle", "frame_oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                f"Train runs `{result['training_stats']['num_train_runs']}`, test runs `{result['training_stats']['num_test_runs']}`",
                f"Train-fit run_cls `{fmt(result['training_stats']['run_cls_train_fit'])}`, run_reg `{fmt(result['training_stats']['run_reg_train_fit'])}`",
                f"Pred mode counts run_cls: `{result['pred_mode_counts']['run_cls']}`",
                f"Pred mode counts run_reg: `{result['pred_mode_counts']['run_reg']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| run_cls vs fixed_task_best | {fmt(result['paired']['run_cls_vs_fixed_task_best']['delta'])} | {result['paired']['run_cls_vs_fixed_task_best']['wins']} | {result['paired']['run_cls_vs_fixed_task_best']['losses']} | {result['paired']['run_cls_vs_fixed_task_best']['ties']} |",
                f"| run_reg vs fixed_task_best | {fmt(result['paired']['run_reg_vs_fixed_task_best']['delta'])} | {result['paired']['run_reg_vs_fixed_task_best']['wins']} | {result['paired']['run_reg_vs_fixed_task_best']['losses']} | {result['paired']['run_reg_vs_fixed_task_best']['ties']} |",
                f"| run_oracle vs fixed_task_best | {fmt(result['paired']['run_oracle_vs_fixed_task_best']['delta'])} | {result['paired']['run_oracle_vs_fixed_task_best']['wins']} | {result['paired']['run_oracle_vs_fixed_task_best']['losses']} | {result['paired']['run_oracle_vs_fixed_task_best']['ties']} |",
                f"| frame_oracle vs run_reg | {fmt(result['paired']['frame_oracle_vs_run_reg']['delta'])} | {result['paired']['frame_oracle_vs_run_reg']['wins']} | {result['paired']['frame_oracle_vs_run_reg']['losses']} | {result['paired']['frame_oracle_vs_run_reg']['ties']} |",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
