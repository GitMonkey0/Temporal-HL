#!/usr/bin/env python3
"""Evaluate a learned gate over configurable family-correction candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.audit_family_disagreement_broad import build_query_events_all, family_best_from_row, passes_trigger
from tools.compare_symbolic_slices import slice_membership
from tools.eval_occlusion_late_correction import FAMILY_LABELS, build_gallery, retrieve_family_label
from tools.eval_sequence_symbolic_retrieval import canonical_label, load_json


LABEL_ORDER = [
    "ROM03_LT_No_Occlusion",
    "ROM03_RT_No_Occlusion",
    "ROM04_LT_Occlusion",
    "ROM04_RT_Occlusion",
    "ROM05_LT_Wrist_ROM",
    "ROM05_RT_Wrist_ROM",
    "ROM07_Rt_Finger_Occlusions",
    "ROM08_Lt_Finger_Occlusions",
]
FAMILY_SET = set(FAMILY_LABELS)


def aggregate_slice_accuracy(rows):
    totals = Counter()
    correct = Counter()
    for row in rows:
        for group in slice_membership(row["seq_name"]):
            totals[group] += 1
            if row["correct"]:
                correct[group] += 1
    out = []
    for group in sorted(totals):
        out.append(
            {
                "slice": group,
                "total": totals[group],
                "correct": correct[group],
                "accuracy": correct[group] / max(totals[group], 1),
            }
        )
    return out


def compare_slices(old_rows, new_rows):
    old = {row["slice"]: row for row in aggregate_slice_accuracy(old_rows)}
    new = {row["slice"]: row for row in aggregate_slice_accuracy(new_rows)}
    out = []
    for key in sorted(set(old) | set(new)):
        old_acc = old.get(key, {}).get("accuracy", 0.0)
        new_acc = new.get(key, {}).get("accuracy", 0.0)
        out.append(
            {
                "slice": key,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
            }
        )
    out.sort(key=lambda row: (-row["delta"], row["slice"]))
    return out


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_thresholds(raw: str):
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("No thresholds provided")
    return sorted(set(vals))


def select_feature_order(all_feature_order, preset: str):
    if preset == "all":
        return list(all_feature_order)
    if preset == "changed_only":
        return ["changed_prediction"]
    if preset == "margins_only":
        keep = {
            "changed_prediction",
            "mainline_margin",
            "mainline_top1_score",
            "mainline_top2_score",
            "retrieval_margin",
            "retrieval_top1_score",
            "retrieval_top2_score",
            "retrieval_in_main_rank",
            "retrieval_in_main_missing",
            "retrieval_main_score",
            "original_main_score",
            "retrieval_minus_original_main_score",
        }
        return [name for name in all_feature_order if name in keep]
    if preset == "no_changed":
        return [name for name in all_feature_order if name != "changed_prediction"]
    if preset == "topk_only":
        keep_prefixes = (
            "retrieval_vote_",
            "retrieval_topk_",
            "retrieval_label_",
            "retrieval_seq_",
        )
        return [name for name in all_feature_order if name.startswith(keep_prefixes)]
    if preset == "non_margin":
        remove = {
            "changed_prediction",
            "mainline_margin",
            "mainline_top1_score",
            "mainline_top2_score",
            "retrieval_margin",
            "retrieval_top1_score",
            "retrieval_top2_score",
            "retrieval_in_main_rank",
            "retrieval_in_main_missing",
            "retrieval_main_score",
            "original_main_score",
            "retrieval_minus_original_main_score",
        }
        return [name for name in all_feature_order if name not in remove]
    if preset == "structure_only":
        keep_prefixes = (
            "main_has_",
            "ret_has_",
            "orig_is_",
            "ret_is_",
        )
        return [name for name in all_feature_order if name.startswith(keep_prefixes)]
    raise ValueError(f"Unknown feature preset: {preset}")


def load_detailed_rows(paths):
    rows = []
    for path in paths:
        data = load_json(path)
        seed = data["seed"]
        for row in data["details"]["pretrained"]["sequence_rows"]:
            family_best_label, family_best_score = family_best_from_row(row)
            rows.append(
                {
                    "seed": seed,
                    "seq_name": row["seq_name"],
                    "target": canonical_label(row["target"]),
                    "prediction": canonical_label(row["prediction"]),
                    "correct": bool(row["correct"]),
                    "mainline_top3": row["top3"],
                    "prediction_in_family": canonical_label(row["prediction"]) in FAMILY_SET,
                    "top3_family_labels": [canonical_label(label) for label, _ in row["top3"] if canonical_label(label) in FAMILY_SET],
                    "family_best_label": family_best_label,
                    "family_best_score": family_best_score,
                    "top1_score": float(row["top3"][0][1]),
                    "mainline_margin": float(row["top3"][0][1] - row["top3"][1][1]) if len(row["top3"]) > 1 else 0.0,
                    "family_gap_to_top1": None if family_best_score is None else float(row["top3"][0][1]) - family_best_score,
                }
            )
            rows[-1]["target_in_family"] = rows[-1]["target"] in FAMILY_SET
    return rows


def label_family_features(label: str):
    return {
        "is_left": 1.0 if "LT_" in label or "Lt_" in label else 0.0,
        "is_right": 1.0 if "RT_" in label or "Rt_" in label else 0.0,
        "is_no_occ": 1.0 if "No_Occlusion" in label else 0.0,
        "is_occ": 1.0 if "Occlusion" in label and "Finger_Occlusions" not in label else 0.0,
        "is_finger_occ": 1.0 if "Finger_Occlusions" in label else 0.0,
        "is_wrist": 1.0 if "Wrist_ROM" in label else 0.0,
    }


def top3_to_map(top3):
    return {canonical_label(label): float(score) for label, score in top3}


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def build_topk_consensus_features(retrieval_ranking, original_prediction, retrieval_prediction):
    feats = {}
    for k in (3, 5, 8):
        topk = retrieval_ranking[: min(k, len(retrieval_ranking))]
        if not topk:
            feats[f"retrieval_topk{k}_available"] = 0.0
            continue
        feats[f"retrieval_topk{k}_available"] = 1.0
        label_counts = Counter(canonical_label(label) for label, _, _ in topk)
        unique_labels = len(label_counts)
        top1_votes = label_counts.get(retrieval_prediction, 0)
        orig_votes = label_counts.get(original_prediction, 0)
        vote_values = list(label_counts.values())
        vote_mean, vote_std = mean_std([float(v) for v in vote_values])
        top_scores = [float(score) for _, _, score in topk]
        score_mean, score_std = mean_std(top_scores)
        feats[f"retrieval_topk{k}_unique_labels"] = float(unique_labels)
        feats[f"retrieval_topk{k}_top1_vote_ratio"] = top1_votes / len(topk)
        feats[f"retrieval_topk{k}_orig_vote_ratio"] = orig_votes / len(topk)
        feats[f"retrieval_topk{k}_top1_vote_count"] = float(top1_votes)
        feats[f"retrieval_topk{k}_orig_vote_count"] = float(orig_votes)
        feats[f"retrieval_topk{k}_vote_mean"] = vote_mean
        feats[f"retrieval_topk{k}_vote_std"] = vote_std
        feats[f"retrieval_topk{k}_score_mean"] = score_mean
        feats[f"retrieval_topk{k}_score_std"] = score_std
        feats[f"retrieval_topk{k}_score_range"] = max(top_scores) - min(top_scores)
        sorted_counts = sorted(label_counts.values(), reverse=True)
        top_vote = float(sorted_counts[0])
        second_vote = float(sorted_counts[1]) if len(sorted_counts) > 1 else 0.0
        feats[f"retrieval_topk{k}_vote_margin"] = top_vote - second_vote
        for label in LABEL_ORDER:
            feats[f"retrieval_vote_{label}_top{k}"] = label_counts.get(label, 0) / len(topk)
    return feats


def parse_scope(raw: str):
    if raw not in {"all", "target_family_only"}:
        raise ValueError(f"Unknown scope: {raw}")
    return raw


def row_in_scope(row, scope):
    if scope == "target_family_only":
        return row["target_in_family"]
    return row["target_in_family"] or row["prediction_in_family"] or bool(row["top3_family_labels"]) or row["family_gap_to_top1"] is not None


def build_family_dataset(base_rows, query_events, gallery, candidate_trigger, candidate_scope, candidate_margin_max, candidate_family_gap_max):
    dataset = []
    retrieval_cache = {}
    for row in base_rows:
        if not row_in_scope(row, candidate_scope):
            continue
        if not passes_trigger(row, candidate_trigger, candidate_margin_max, candidate_family_gap_max):
            continue
        if row["seq_name"] not in query_events:
            continue
        if row["seq_name"] not in retrieval_cache:
            ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
            retrieval_cache[row["seq_name"]] = [[label, seq_name, float(score)] for label, seq_name, score in ranking]
        retrieval_ranking = retrieval_cache[row["seq_name"]]
        retrieval_top3 = retrieval_ranking[:3]
        retrieval_prediction = canonical_label(retrieval_top3[0][0])
        retrieval_map = {canonical_label(label): float(score) for label, _, score in retrieval_top3}
        mainline_map = top3_to_map(row["mainline_top3"])
        main_top1 = float(row["mainline_top3"][0][1])
        main_top2 = float(row["mainline_top3"][1][1]) if len(row["mainline_top3"]) > 1 else main_top1
        retrieval_top1 = float(retrieval_top3[0][2])
        retrieval_top2 = float(retrieval_top3[1][2]) if len(retrieval_top3) > 1 else retrieval_top1
        retrieval_in_main_rank = 3.0
        for idx, (label, _) in enumerate(row["mainline_top3"]):
            if canonical_label(label) == retrieval_prediction:
                retrieval_in_main_rank = float(idx)
                break
        retrieval_main_score = mainline_map.get(retrieval_prediction, -10.0)
        original_main_score = mainline_map.get(row["prediction"], -10.0)

        feats = {
            "changed_prediction": 1.0 if retrieval_prediction != row["prediction"] else 0.0,
            "mainline_margin": row["mainline_margin"],
            "mainline_top1_score": main_top1,
            "mainline_top2_score": main_top2,
            "retrieval_margin": retrieval_top1 - retrieval_top2,
            "retrieval_top1_score": retrieval_top1,
            "retrieval_top2_score": retrieval_top2,
            "retrieval_in_main_rank": retrieval_in_main_rank,
            "retrieval_in_main_missing": 1.0 if retrieval_in_main_rank >= 3.0 else 0.0,
            "retrieval_main_score": retrieval_main_score,
            "original_main_score": original_main_score,
            "retrieval_minus_original_main_score": retrieval_main_score - original_main_score,
        }
        for prefix, label in [("orig", row["prediction"]), ("ret", retrieval_prediction)]:
            for key, value in label_family_features(label).items():
                feats[f"{prefix}_{key}"] = value
        for label in LABEL_ORDER:
            feats[f"main_has_{label}"] = 1.0 if label in mainline_map else 0.0
            feats[f"ret_has_{label}"] = 1.0 if label in retrieval_map else 0.0
        feats.update(
            build_topk_consensus_features(
                retrieval_ranking,
                row["prediction"],
                retrieval_prediction,
            )
        )

        would_fix = (retrieval_prediction != row["prediction"]) and (retrieval_prediction == row["target"])
        would_harm = row["correct"] and (retrieval_prediction != row["target"])
        dataset.append(
            {
                "seed": row["seed"],
                "seq_name": row["seq_name"],
                "target": row["target"],
                "original_prediction": row["prediction"],
                "original_correct": row["correct"],
                "retrieval_prediction": retrieval_prediction,
                "retrieval_top3": retrieval_top3,
                "retrieval_ranking": retrieval_ranking[:8],
                "features": feats,
                "label_apply": 1 if would_fix else 0,
                "would_fix": would_fix,
                "would_harm": would_harm,
                "candidate_trigger": candidate_trigger,
            }
        )
    return dataset


def feature_matrix(rows, feature_order):
    x = np.asarray([[row["features"][name] for name in feature_order] for row in rows], dtype=np.float64)
    return x


def run_loso(dataset, feature_order, threshold):
    seeds = sorted({row["seed"] for row in dataset})
    family_outcomes = []
    for test_seed in seeds:
        train_rows = [row for row in dataset if row["seed"] != test_seed]
        test_rows = [row for row in dataset if row["seed"] == test_seed]
        if not train_rows or not test_rows:
            continue
        y_train = np.asarray([row["label_apply"] for row in train_rows], dtype=np.int64)
        if len(set(y_train.tolist())) < 2:
            # If the train fold is single-class, back off to the deterministic baseline.
            for row in test_rows:
                family_outcomes.append(
                    {
                        "seed": row["seed"],
                        "seq_name": row["seq_name"],
                        "target": row["target"],
                        "original_prediction": row["original_prediction"],
                        "new_prediction": row["original_prediction"],
                        "original_correct": row["original_correct"],
                        "new_correct": row["original_correct"],
                        "apply_probability": 0.0,
                        "apply_gate": False,
                        "would_fix": row["would_fix"],
                        "would_harm": row["would_harm"],
                    }
                )
            continue
        x_train = feature_matrix(train_rows, feature_order)
        x_test = feature_matrix(test_rows, feature_order)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=0,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        for row, prob in zip(test_rows, probs):
            apply_gate = (
                row["retrieval_prediction"] != row["original_prediction"]
                and prob >= threshold
            )
            new_prediction = row["retrieval_prediction"] if apply_gate else row["original_prediction"]
            new_correct = new_prediction == row["target"]
            family_outcomes.append(
                {
                    "seed": row["seed"],
                    "seq_name": row["seq_name"],
                    "target": row["target"],
                    "original_prediction": row["original_prediction"],
                    "new_prediction": new_prediction,
                    "original_correct": row["original_correct"],
                    "new_correct": new_correct,
                    "apply_probability": float(prob),
                    "apply_gate": apply_gate,
                    "would_fix": row["would_fix"],
                    "would_harm": row["would_harm"],
                }
            )
    return family_outcomes


def evaluate_threshold(dataset, feature_order, threshold):
    family_outcomes = run_loso(dataset, feature_order, threshold)
    totals = Counter()
    correct = Counter()
    improved = harmed = applied = 0
    for row in family_outcomes:
        totals["family"] += 1
        correct["old_family"] += int(row["original_correct"])
        correct["new_family"] += int(row["new_correct"])
        applied += int(row["apply_gate"])
        improved += int((not row["original_correct"]) and row["new_correct"])
        harmed += int(row["original_correct"] and (not row["new_correct"]))
    return {
        "threshold": threshold,
        "candidate_accuracy": {
            "old": correct["old_family"] / max(totals["family"], 1),
            "new": correct["new_family"] / max(totals["family"], 1),
            "delta": correct["new_family"] / max(totals["family"], 1) - correct["old_family"] / max(totals["family"], 1),
        },
        "counts": {
            "num_candidate_rows": totals["family"],
            "num_applied": applied,
            "num_improved": improved,
            "num_harmed": harmed,
        },
        "applied_cases": [row for row in family_outcomes if row["apply_gate"]],
        "all_family_outcomes": family_outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--feature-preset", type=str, default="all", choices=["all", "changed_only", "margins_only", "no_changed", "topk_only", "non_margin", "structure_only"])
    parser.add_argument("--candidate-trigger", type=str, default="pred_in_family")
    parser.add_argument("--candidate-scope", type=str, default="target_family_only")
    parser.add_argument("--candidate-margin-max", type=float, default=0.0)
    parser.add_argument("--candidate-family-gap-max", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_learned_gate.json"),
    )
    args = parser.parse_args()

    base_rows = load_detailed_rows(args.detailed_analysis)
    gallery = build_gallery(load_json(args.gallery_json), FAMILY_LABELS)
    query_events = build_query_events_all(load_json(args.query_json))
    candidate_scope = parse_scope(args.candidate_scope)
    dataset = build_family_dataset(
        base_rows,
        query_events,
        gallery,
        args.candidate_trigger,
        candidate_scope,
        args.candidate_margin_max,
        args.candidate_family_gap_max,
    )
    if not dataset:
        raise SystemExit("No candidate rows matched the requested benchmark.")
    all_feature_order = sorted(dataset[0]["features"].keys())
    feature_order = select_feature_order(all_feature_order, args.feature_preset)
    thresholds = parse_thresholds(args.thresholds)

    sweeps = [evaluate_threshold(dataset, feature_order, thr) for thr in thresholds]
    sweeps.sort(
        key=lambda row: (
            -row["candidate_accuracy"]["delta"],
            row["counts"]["num_harmed"],
            -row["counts"]["num_improved"],
            row["threshold"],
        )
    )

    payload = {
        "detailed_analysis": [str(path) for path in args.detailed_analysis],
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "candidate_trigger": args.candidate_trigger,
        "candidate_scope": candidate_scope,
        "candidate_margin_max": args.candidate_margin_max,
        "candidate_family_gap_max": args.candidate_family_gap_max,
        "feature_preset": args.feature_preset,
        "all_feature_order": all_feature_order,
        "feature_order": feature_order,
        "thresholds": thresholds,
        "family_dataset_size": len(dataset),
        "sweeps": sweeps,
        "fixed_threshold_0.5": next(row for row in sweeps if abs(row["threshold"] - 0.5) < 1e-9),
        "best_zero_harm": [row for row in sweeps if row["counts"]["num_harmed"] == 0][:10],
    }
    payload = to_builtin(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    fixed = payload["fixed_threshold_0.5"]
    print(
        "fixed@0.5 candidate",
        fixed["candidate_accuracy"]["old"],
        "->",
        fixed["candidate_accuracy"]["new"],
        "delta",
        fixed["candidate_accuracy"]["delta"],
        "counts",
        fixed["counts"],
    )
    for row in payload["best_zero_harm"][:5]:
        print(
            "zero_harm",
            row["threshold"],
            "candidate",
            row["candidate_accuracy"]["old"],
            "->",
            row["candidate_accuracy"]["new"],
            "delta",
            row["candidate_accuracy"]["delta"],
            "counts",
            row["counts"],
        )


if __name__ == "__main__":
    main()
