#!/usr/bin/env python3
"""Learned run-level scorers for fast-path closing support.

This is an experiment memo, not paper text.

Replace the hand-crafted run-distance + voting mechanism with learned run-level
predictors on the same fast-path closing regime:

- reference KNN binary / trimode selectors
- learned run-level classifier
- learned run-level regressor over mode values
- compare support variants:
  - val runs only
  - val runs + cached top4 singleton rows
  - val runs + cached top4 relaxed runs
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tools.build_interaction_realized_closing_run_target import (
    CACHE_DIR,
    EXTERNAL_RELAXED_RUN_GAP,
    OTHER_VALUES,
    INTERACTION_VALUES,
    REPAIR_MODES,
    TASK_TARGET,
    VAL_RUN_GAP,
    all_sequence_labels,
    build_runs,
    choose_cfg,
    fmt,
    predict_mode,
    summarize_rows,
    summarize_run,
    paired_stats,
    subset_to_closing_sequences,
)
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


CLS_GRID = [
    {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "max_iter": 150,
    }
]

REG_GRID = [
    {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "max_iter": 150,
    }
]


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
        float(run["finger_gain_vs_edge"]),
        float(run["none_gain_vs_edge"]),
        float(run["none_gain_vs_finger"]),
    ]
    for value in OTHER_VALUES:
        feats.append(
            sum(freq for key, freq in run["subtype_counts"].items() if key[0] == value) / subtype_total
        )
    for value in INTERACTION_VALUES:
        feats.append(
            sum(freq for key, freq in run["subtype_counts"].items() if key[1] == value) / subtype_total
        )
    return np.asarray(feats, dtype=np.float32)


def evaluate_run_modes(run_summaries, pred_modes):
    total = 0.0
    n = 0
    for summary, mode in zip(run_summaries, pred_modes):
        total += summary["mode_means"][mode] * summary["num_frames"]
        n += summary["num_frames"]
    return total / n


def fit_classifier(train_runs, target_key, cfg):
    labels = [run[target_key] for run in train_runs]
    unique = sorted(set(labels))
    if len(unique) == 1:
        return {"kind": "constant", "label": unique[0]}
    x = np.stack([run_feature_vector(run) for run in train_runs], axis=0)
    y = np.asarray(labels)
    model = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_iter=cfg["max_iter"],
        random_state=0,
    )
    model.fit(x, y)
    return {"kind": "hgb", "model": model}


def predict_classifier(model_obj, run):
    if model_obj["kind"] == "constant":
        return model_obj["label"]
    x = run_feature_vector(run).reshape(1, -1)
    return str(model_obj["model"].predict(x)[0])


def choose_classifier_cfg(train_runs, target_key):
    best = None
    for cfg in CLS_GRID:
        pred = []
        for idx, run in enumerate(train_runs):
            model_obj = fit_classifier(train_runs[:idx] + train_runs[idx + 1 :], target_key, cfg)
            pred.append(predict_classifier(model_obj, run))
        score = evaluate_run_modes(train_runs, pred)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(pred)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def fit_regressors(train_runs, modes, cfg):
    x = np.stack([run_feature_vector(run) for run in train_runs], axis=0)
    out = {}
    for mode in modes:
        y = np.asarray([run["mode_means"][mode] for run in train_runs], dtype=np.float32)
        if float(np.max(y) - np.min(y)) == 0.0:
            out[mode] = {"kind": "constant", "value": float(y[0])}
            continue
        model = HistGradientBoostingRegressor(
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_iter=cfg["max_iter"],
            random_state=0,
        )
        model.fit(x, y)
        out[mode] = {"kind": "hgb", "model": model}
    return out


def predict_regressors(model_map, run, modes):
    x = run_feature_vector(run).reshape(1, -1)
    scores = {}
    for mode in modes:
        model_obj = model_map[mode]
        if model_obj["kind"] == "constant":
            scores[mode] = model_obj["value"]
        else:
            scores[mode] = float(model_obj["model"].predict(x)[0])
    return max(scores, key=scores.get), scores


def choose_regressor_cfg(train_runs, modes):
    best = None
    for cfg in REG_GRID:
        pred = []
        for idx, run in enumerate(train_runs):
            models = fit_regressors(train_runs[:idx] + train_runs[idx + 1 :], modes, cfg)
            pred_mode, _ = predict_regressors(models, run, modes)
            pred.append(pred_mode)
        score = evaluate_run_modes(train_runs, pred)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(pred)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def build_reference_predictions(train_runs, test_runs, target_key):
    best = choose_cfg(train_runs, target_key)
    pred = [predict_mode(train_runs, run, best["cfg"], target_key) for run in test_runs]
    return best, pred


def build_learned_classifier_predictions(train_runs, test_runs, target_key):
    best = choose_classifier_cfg(train_runs, target_key)
    model_obj = fit_classifier(train_runs, target_key, best["cfg"])
    pred = [predict_classifier(model_obj, run) for run in test_runs]
    return best, pred


def build_learned_regression_predictions(train_runs, test_runs, modes):
    best = choose_regressor_cfg(train_runs, modes)
    model_map = fit_regressors(train_runs, modes, best["cfg"])
    pred = [predict_regressors(model_map, run, modes)[0] for run in test_runs]
    return best, pred


def evaluate_support(name, train_runs, test_runs):
    knn_binary_stats, knn_binary_pred = build_reference_predictions(train_runs, test_runs, "best_mode")
    knn_trimode_stats, knn_trimode_pred = build_reference_predictions(train_runs, test_runs, "best_trimode")
    learned_binary_cls_stats, learned_binary_cls_pred = build_learned_classifier_predictions(train_runs, test_runs, "best_mode")
    learned_trimode_cls_stats, learned_trimode_cls_pred = build_learned_classifier_predictions(train_runs, test_runs, "best_trimode")
    learned_binary_reg_stats, learned_binary_reg_pred = build_learned_regression_predictions(
        train_runs, test_runs, ("edge_transition_snap", "finger_profile_snap")
    )
    learned_trimode_reg_stats, learned_trimode_reg_pred = build_learned_regression_predictions(
        train_runs, test_runs, REPAIR_MODES
    )

    output_rows = []
    for run, knn_binary_mode, knn_trimode_mode, cls_binary_mode, cls_trimode_mode, reg_binary_mode, reg_trimode_mode in zip(
        test_runs,
        knn_binary_pred,
        knn_trimode_pred,
        learned_binary_cls_pred,
        learned_trimode_cls_pred,
        learned_binary_reg_pred,
        learned_trimode_reg_pred,
    ):
        for row in run["rows"]:
            rec = dict(row)
            oracle_binary = (
                "finger_profile_snap"
                if float(row["left_finger_profile_snap_joint_score"]) > float(row["left_edge_transition_snap_joint_score"])
                else "edge_transition_snap"
            )
            oracle_trimode = max(REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"]))
            for out_name, out_mode in [
                ("fixed_edge", "edge_transition_snap"),
                ("knn_binary", knn_binary_mode),
                ("knn_trimode", knn_trimode_mode),
                ("learned_binary_cls", cls_binary_mode),
                ("learned_trimode_cls", cls_trimode_mode),
                ("learned_binary_reg", reg_binary_mode),
                ("learned_trimode_reg", reg_trimode_mode),
                ("oracle_binary", oracle_binary),
                ("oracle_trimode", oracle_trimode),
            ]:
                prefix = f"left_{out_mode}"
                for field in ("right_grouped_match", "left_preserve", "joint_score"):
                    rec[f"{out_name}_{field}"] = rec[f"{prefix}_{field}"]
            output_rows.append(rec)

    methods = (
        "fixed_edge",
        "knn_binary",
        "knn_trimode",
        "learned_binary_cls",
        "learned_trimode_cls",
        "learned_binary_reg",
        "learned_trimode_reg",
        "oracle_binary",
        "oracle_trimode",
    )
    return {
        "support_name": name,
        "num_support_runs": len(train_runs),
        "stats": {
            "knn_binary": knn_binary_stats,
            "knn_trimode": knn_trimode_stats,
            "learned_binary_cls": learned_binary_cls_stats,
            "learned_trimode_cls": learned_trimode_cls_stats,
            "learned_binary_reg": learned_binary_reg_stats,
            "learned_trimode_reg": learned_trimode_reg_stats,
        },
        "summary": {method: summarize_rows(output_rows, method) for method in methods},
        "paired": {
            "knn_binary_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "knn_binary"),
            "learned_binary_cls_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "learned_binary_cls"),
            "learned_trimode_cls_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "learned_trimode_cls"),
            "learned_binary_reg_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "learned_binary_reg"),
            "learned_trimode_reg_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "learned_trimode_reg"),
            "learned_binary_cls_vs_knn_binary": paired_stats(output_rows, "knn_binary", "learned_binary_cls"),
            "learned_binary_reg_vs_knn_binary": paired_stats(output_rows, "knn_binary", "learned_binary_reg"),
            "oracle_trimode_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle_trimode"),
        },
        "pred_mode_counts": {
            "knn_binary": dict(Counter(knn_binary_pred)),
            "knn_trimode": dict(Counter(knn_trimode_pred)),
            "learned_binary_cls": dict(Counter(learned_binary_cls_pred)),
            "learned_trimode_cls": dict(Counter(learned_trimode_cls_pred)),
            "learned_binary_reg": dict(Counter(learned_binary_reg_pred)),
            "learned_trimode_reg": dict(Counter(learned_trimode_reg_pred)),
        },
    }


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

    val_runs = [summarize_run(run) for run in build_runs(val_rows, VAL_RUN_GAP)]
    test_runs = [summarize_run(run) for run in build_runs(test_rows, VAL_RUN_GAP)]
    cached_singletons = [summarize_run([row]) for row in cached_top4_rows]
    cached_relaxed_runs = [summarize_run(run) for run in build_runs(cached_top4_rows, EXTERNAL_RELAXED_RUN_GAP)]

    task_results = {
        "val_runs_only": evaluate_support("val_runs_only", val_runs, test_runs),
        "val_runs_plus_top4_singletons": evaluate_support("val_runs_plus_top4_singletons", val_runs + cached_singletons, test_runs),
        "val_runs_plus_top4_relaxed_runs": evaluate_support("val_runs_plus_top4_relaxed_runs", val_runs + cached_relaxed_runs, test_runs),
    }

    payload = {
        "focus": {
            "goal": "learned run-level scorers for fast-path closing",
            "task": "right_hand_motion->closing",
        },
        "training_stats": {
            "pair_model": pair_stats,
            "num_val_rows": len(val_rows),
            "num_val_runs": len(val_runs),
            "num_test_rows": len(test_rows),
            "num_test_runs": len(test_runs),
            "num_cached_top4_rows": len(cached_top4_rows),
            "num_cached_top4_relaxed_runs": len(cached_relaxed_runs),
            "val_run_gap": VAL_RUN_GAP,
            "external_relaxed_run_gap": EXTERNAL_RELAXED_RUN_GAP,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_closing_run_learned_scorer.json"
    out_md = SUM / "interaction_realized_closing_run_learned_scorer.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Run Learned Scorer",
        "",
        "This is an experiment memo, not paper text.",
        "",
        (
            f"Val rows/runs: `{len(val_rows)}/{len(val_runs)}` @ gap `{VAL_RUN_GAP}`; "
            f"test rows/runs: `{len(test_rows)}/{len(test_runs)}` @ gap `{VAL_RUN_GAP}`; "
            f"cached top4 rows: `{len(cached_top4_rows)}`; "
            f"cached top4 relaxed runs: `{len(cached_relaxed_runs)}` @ gap `{EXTERNAL_RELAXED_RUN_GAP}`"
        ),
        "",
    ]
    for name, result in task_results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"Support runs: `{result['num_support_runs']}`",
                f"Stats: `{result['stats']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in (
            "fixed_edge",
            "knn_binary",
            "knn_trimode",
            "learned_binary_cls",
            "learned_trimode_cls",
            "learned_binary_reg",
            "learned_trimode_reg",
            "oracle_binary",
            "oracle_trimode",
        ):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                f"Pred run modes: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| knn_binary vs fixed_edge | {fmt(result['paired']['knn_binary_vs_fixed_edge']['delta'])} | {result['paired']['knn_binary_vs_fixed_edge']['wins']} | {result['paired']['knn_binary_vs_fixed_edge']['losses']} | {result['paired']['knn_binary_vs_fixed_edge']['ties']} |",
                f"| learned_binary_cls vs fixed_edge | {fmt(result['paired']['learned_binary_cls_vs_fixed_edge']['delta'])} | {result['paired']['learned_binary_cls_vs_fixed_edge']['wins']} | {result['paired']['learned_binary_cls_vs_fixed_edge']['losses']} | {result['paired']['learned_binary_cls_vs_fixed_edge']['ties']} |",
                f"| learned_trimode_cls vs fixed_edge | {fmt(result['paired']['learned_trimode_cls_vs_fixed_edge']['delta'])} | {result['paired']['learned_trimode_cls_vs_fixed_edge']['wins']} | {result['paired']['learned_trimode_cls_vs_fixed_edge']['losses']} | {result['paired']['learned_trimode_cls_vs_fixed_edge']['ties']} |",
                f"| learned_binary_reg vs fixed_edge | {fmt(result['paired']['learned_binary_reg_vs_fixed_edge']['delta'])} | {result['paired']['learned_binary_reg_vs_fixed_edge']['wins']} | {result['paired']['learned_binary_reg_vs_fixed_edge']['losses']} | {result['paired']['learned_binary_reg_vs_fixed_edge']['ties']} |",
                f"| learned_trimode_reg vs fixed_edge | {fmt(result['paired']['learned_trimode_reg_vs_fixed_edge']['delta'])} | {result['paired']['learned_trimode_reg_vs_fixed_edge']['wins']} | {result['paired']['learned_trimode_reg_vs_fixed_edge']['losses']} | {result['paired']['learned_trimode_reg_vs_fixed_edge']['ties']} |",
                f"| learned_binary_cls vs knn_binary | {fmt(result['paired']['learned_binary_cls_vs_knn_binary']['delta'])} | {result['paired']['learned_binary_cls_vs_knn_binary']['wins']} | {result['paired']['learned_binary_cls_vs_knn_binary']['losses']} | {result['paired']['learned_binary_cls_vs_knn_binary']['ties']} |",
                f"| learned_binary_reg vs knn_binary | {fmt(result['paired']['learned_binary_reg_vs_knn_binary']['delta'])} | {result['paired']['learned_binary_reg_vs_knn_binary']['wins']} | {result['paired']['learned_binary_reg_vs_knn_binary']['losses']} | {result['paired']['learned_binary_reg_vs_knn_binary']['ties']} |",
                f"| oracle_trimode vs fixed_edge | {fmt(result['paired']['oracle_trimode_vs_fixed_edge']['delta'])} | {result['paired']['oracle_trimode_vs_fixed_edge']['wins']} | {result['paired']['oracle_trimode_vs_fixed_edge']['losses']} | {result['paired']['oracle_trimode_vs_fixed_edge']['ties']} |",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
