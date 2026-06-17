#!/usr/bin/env python3
"""Closing run-level edge-vs-finger target on the fast-path regime.

This is an experiment memo, not paper text.

Move one level above row-level support targets:
- train on short temporal runs from fast-path val rows
- choose one repair family per run (`edge` vs `finger`)
- optionally append cached top4 support either as singleton rows or relaxed runs
"""

from __future__ import annotations

import json
from collections import Counter

from tools.build_interaction_realized_closing_selector_support_scaling import (
    CACHE_DIR,
    TASK_TARGET,
    all_sequence_labels,
    collect_rows_fast,
    subset_to_closing_sequences,
)
from tools.build_interaction_realized_feasible_left_temporal_window_knn import build_frames
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)


OTHER_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown"]
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown"]
REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
GAIN_MARGIN_LEVELS = (0.0, 0.05, 0.1, 0.2)
GRID = []
for k in (1, 3, 5, 7):
    for subtype_weight in (0.25, 0.5, 1.0):
        for agreement_weight in (0.5, 1.0):
            for length_weight in (0.05, 0.1, 0.25):
                GRID.append(
                    {
                        "k": k,
                        "subtype_weight": subtype_weight,
                        "agreement_weight": agreement_weight,
                        "length_weight": length_weight,
                    }
                )


def fmt(x: float) -> str:
    return f"{x:.4f}"


VAL_RUN_GAP = 12
EXTERNAL_RELAXED_RUN_GAP = 18


def build_runs(rows, gap_threshold):
    rows = sorted(rows, key=lambda r: (r["seq_name"], r["frame_idx"]))
    runs = []
    cur = []
    prev = None
    for row in rows:
        ok = False
        if prev is not None:
            ok = row["seq_name"] == prev["seq_name"] and int(row["frame_idx"]) - int(prev["frame_idx"]) <= gap_threshold
        if prev is None or ok:
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
    none_mean = sum(float(row["left_none_joint_score"]) for row in run) / len(run)
    edge_mean = sum(float(row["left_edge_transition_snap_joint_score"]) for row in run) / len(run)
    finger_mean = sum(float(row["left_finger_profile_snap_joint_score"]) for row in run) / len(run)
    mode_means = {
        "none": none_mean,
        "edge_transition_snap": edge_mean,
        "finger_profile_snap": finger_mean,
    }
    return {
        "rows": run,
        "num_frames": len(run),
        "subtype_counts": subtype_counts,
        "left_state_mean": sum(float(row["left_none_left_state_agreement"]) for row in run) / len(run),
        "left_transition_mean": sum(float(row["left_none_left_transition_agreement"]) for row in run) / len(run),
        "right_state_mean": sum(float(row["left_none_right_state_agreement"]) for row in run) / len(run),
        "right_transition_mean": sum(float(row["left_none_right_transition_agreement"]) for row in run) / len(run),
        "none_mean": none_mean,
        "edge_mean": edge_mean,
        "finger_mean": finger_mean,
        "mode_means": mode_means,
        "finger_gain_vs_edge": finger_mean - edge_mean,
        "none_gain_vs_edge": none_mean - edge_mean,
        "none_gain_vs_finger": none_mean - finger_mean,
        "best_mode": "finger_profile_snap" if finger_mean > edge_mean else "edge_transition_snap",
        "best_trimode": max(REPAIR_MODES, key=lambda mode: mode_means[mode]),
    }


def best_gain_binary(summary, tau):
    return "finger_profile_snap" if summary["finger_gain_vs_edge"] > tau else "edge_transition_snap"


def best_gain_trimode(summary, tau_none, tau_finger):
    if summary["none_gain_vs_edge"] > tau_none and summary["none_gain_vs_finger"] > tau_none:
        return "none"
    if summary["finger_gain_vs_edge"] > tau_finger and summary["finger_mean"] > summary["none_mean"]:
        return "finger_profile_snap"
    return "edge_transition_snap"


def dist_counts(a, b, vocab, axis):
    total_a = sum(a.values()) or 1
    total_b = sum(b.values()) or 1
    out = 0.0
    for value in vocab:
        pa = sum(freq for key, freq in a.items() if key[axis] == value) / total_a
        pb = sum(freq for key, freq in b.items() if key[axis] == value) / total_b
        out += abs(pa - pb)
    return out


def run_distance(a, b, cfg):
    return (
        cfg["length_weight"] * abs(a["num_frames"] - b["num_frames"])
        + cfg["subtype_weight"] * dist_counts(a["subtype_counts"], b["subtype_counts"], OTHER_VALUES, axis=0)
        + cfg["subtype_weight"] * dist_counts(a["subtype_counts"], b["subtype_counts"], INTERACTION_VALUES, axis=1)
        + cfg["agreement_weight"] * abs(a["left_state_mean"] - b["left_state_mean"])
        + cfg["agreement_weight"] * abs(a["left_transition_mean"] - b["left_transition_mean"])
        + cfg["agreement_weight"] * abs(a["right_state_mean"] - b["right_state_mean"])
        + cfg["agreement_weight"] * abs(a["right_transition_mean"] - b["right_transition_mean"])
    )


def predict_mode(train_runs, query, cfg, target_key):
    neighbors = sorted(((run_distance(item, query, cfg), item) for item in train_runs), key=lambda x: x[0])[: cfg["k"]]
    votes = Counter(item[target_key] for _, item in neighbors)
    mode_sum = {mode: sum(item["mode_means"][mode] for _, item in neighbors) for mode in REPAIR_MODES}
    if target_key == "best_mode":
        candidate_modes = ("edge_transition_snap", "finger_profile_snap")
    else:
        candidate_modes = REPAIR_MODES
    return max(candidate_modes, key=lambda mode: (votes[mode], mode_sum[mode]))


def evaluate_run_modes(run_summaries, pred_modes):
    total = 0.0
    n = 0
    for summary, mode in zip(run_summaries, pred_modes):
        total += summary["mode_means"][mode] * summary["num_frames"]
        n += summary["num_frames"]
    return total / n


def choose_cfg(train_runs, target_key):
    best = None
    for cfg in GRID:
        pred_modes = []
        for idx, run in enumerate(train_runs):
            pred_modes.append(predict_mode(train_runs[:idx] + train_runs[idx + 1 :], run, cfg, target_key))
        score = evaluate_run_modes(train_runs, pred_modes)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(pred_modes)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def choose_cfg_binary_gain(train_runs):
    best = None
    for tau in GAIN_MARGIN_LEVELS:
        labeled_runs = [{**run, "best_gain_binary": best_gain_binary(run, tau)} for run in train_runs]
        candidate = choose_cfg(labeled_runs, "best_gain_binary")
        candidate["tau"] = tau
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def choose_cfg_trimode_gain(train_runs):
    best = None
    for tau_none in GAIN_MARGIN_LEVELS:
        for tau_finger in GAIN_MARGIN_LEVELS:
            labeled_runs = [{**run, "best_gain_trimode": best_gain_trimode(run, tau_none, tau_finger)} for run in train_runs]
            candidate = choose_cfg(labeled_runs, "best_gain_trimode")
            candidate["tau_none"] = tau_none
            candidate["tau_finger"] = tau_finger
            if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
                best = candidate
    return best


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


def evaluate_support(name, train_runs, test_runs):
    binary_best = choose_cfg(train_runs, "best_mode")
    trimode_best = choose_cfg(train_runs, "best_trimode")
    binary_gain_best = choose_cfg_binary_gain(train_runs)
    trimode_gain_best = choose_cfg_trimode_gain(train_runs)

    train_runs_binary_gain = [{**run, "best_gain_binary": best_gain_binary(run, binary_gain_best["tau"])} for run in train_runs]
    test_runs_binary_gain = [{**run, "best_gain_binary": best_gain_binary(run, binary_gain_best["tau"])} for run in test_runs]
    train_runs_trimode_gain = [
        {**run, "best_gain_trimode": best_gain_trimode(run, trimode_gain_best["tau_none"], trimode_gain_best["tau_finger"])}
        for run in train_runs
    ]
    test_runs_trimode_gain = [
        {**run, "best_gain_trimode": best_gain_trimode(run, trimode_gain_best["tau_none"], trimode_gain_best["tau_finger"])}
        for run in test_runs
    ]

    pred_modes = [predict_mode(train_runs, run, binary_best["cfg"], "best_mode") for run in test_runs]
    pred_trimodes = [predict_mode(train_runs, run, trimode_best["cfg"], "best_trimode") for run in test_runs]
    pred_gain_binary = [predict_mode(train_runs_binary_gain, run, binary_gain_best["cfg"], "best_gain_binary") for run in test_runs_binary_gain]
    pred_gain_trimode = [predict_mode(train_runs_trimode_gain, run, trimode_gain_best["cfg"], "best_gain_trimode") for run in test_runs_trimode_gain]
    output_rows = []
    for run, mode, trimode, gain_binary_mode, gain_trimode_mode in zip(
        test_runs, pred_modes, pred_trimodes, pred_gain_binary, pred_gain_trimode
    ):
        for row in run["rows"]:
            rec = dict(row)
            oracle_mode = "finger_profile_snap" if float(row["left_finger_profile_snap_joint_score"]) > float(row["left_edge_transition_snap_joint_score"]) else "edge_transition_snap"
            oracle_trimode = max(REPAIR_MODES, key=lambda candidate_mode: float(row[f"left_{candidate_mode}_joint_score"]))
            for out_name, out_mode in [
                ("fixed_edge", "edge_transition_snap"),
                ("run_target", mode),
                ("run_target_trimode", trimode),
                ("run_target_gain_binary", gain_binary_mode),
                ("run_target_gain_trimode", gain_trimode_mode),
                ("oracle_edge_finger", oracle_mode),
                ("oracle_trimode", oracle_trimode),
            ]:
                prefix = f"left_{out_mode}"
                for field in ("right_grouped_match", "left_preserve", "joint_score"):
                    rec[f"{out_name}_{field}"] = rec[f"{prefix}_{field}"]
            output_rows.append(rec)
    return {
        "support_name": name,
        "num_support_runs": len(train_runs),
        "threshold_stats": {
            "binary": binary_best,
            "trimode": trimode_best,
            "gain_binary": binary_gain_best,
            "gain_trimode": trimode_gain_best,
        },
        "summary": {
            k: summarize_rows(output_rows, k)
            for k in (
                "fixed_edge",
                "run_target",
                "run_target_trimode",
                "run_target_gain_binary",
                "run_target_gain_trimode",
                "oracle_edge_finger",
                "oracle_trimode",
            )
        },
        "paired": {
            "run_target_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "run_target"),
            "run_target_trimode_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "run_target_trimode"),
            "run_target_gain_binary_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "run_target_gain_binary"),
            "run_target_gain_trimode_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "run_target_gain_trimode"),
            "oracle_edge_finger_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle_edge_finger"),
            "oracle_trimode_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle_trimode"),
        },
        "pred_mode_counts": {
            "binary": dict(Counter(pred_modes)),
            "trimode": dict(Counter(pred_trimodes)),
            "gain_binary": dict(Counter(pred_gain_binary)),
            "gain_trimode": dict(Counter(pred_gain_trimode)),
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
            "goal": "closing run-level edge-vs-finger target on fast-path support",
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

    out_json = GEN / "interaction_realized_closing_run_target.json"
    out_md = SUM / "interaction_realized_closing_run_target.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Run Target",
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
                f"Config stats: `{result['threshold_stats']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in (
            "fixed_edge",
            "run_target",
            "run_target_trimode",
            "run_target_gain_binary",
            "run_target_gain_trimode",
            "oracle_edge_finger",
            "oracle_trimode",
        ):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        paired = result["paired"]
        lines.extend(
            [
                "",
                f"Pred run modes: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| run_target vs fixed_edge | {fmt(paired['run_target_vs_fixed_edge']['delta'])} | {paired['run_target_vs_fixed_edge']['wins']} | {paired['run_target_vs_fixed_edge']['losses']} | {paired['run_target_vs_fixed_edge']['ties']} |",
                f"| run_target_trimode vs fixed_edge | {fmt(paired['run_target_trimode_vs_fixed_edge']['delta'])} | {paired['run_target_trimode_vs_fixed_edge']['wins']} | {paired['run_target_trimode_vs_fixed_edge']['losses']} | {paired['run_target_trimode_vs_fixed_edge']['ties']} |",
                f"| run_target_gain_binary vs fixed_edge | {fmt(paired['run_target_gain_binary_vs_fixed_edge']['delta'])} | {paired['run_target_gain_binary_vs_fixed_edge']['wins']} | {paired['run_target_gain_binary_vs_fixed_edge']['losses']} | {paired['run_target_gain_binary_vs_fixed_edge']['ties']} |",
                f"| run_target_gain_trimode vs fixed_edge | {fmt(paired['run_target_gain_trimode_vs_fixed_edge']['delta'])} | {paired['run_target_gain_trimode_vs_fixed_edge']['wins']} | {paired['run_target_gain_trimode_vs_fixed_edge']['losses']} | {paired['run_target_gain_trimode_vs_fixed_edge']['ties']} |",
                f"| oracle_edge_finger vs fixed_edge | {fmt(paired['oracle_edge_finger_vs_fixed_edge']['delta'])} | {paired['oracle_edge_finger_vs_fixed_edge']['wins']} | {paired['oracle_edge_finger_vs_fixed_edge']['losses']} | {paired['oracle_edge_finger_vs_fixed_edge']['ties']} |",
                f"| oracle_trimode vs fixed_edge | {fmt(paired['oracle_trimode_vs_fixed_edge']['delta'])} | {paired['oracle_trimode_vs_fixed_edge']['wins']} | {paired['oracle_trimode_vs_fixed_edge']['losses']} | {paired['oracle_trimode_vs_fixed_edge']['ties']} |",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
