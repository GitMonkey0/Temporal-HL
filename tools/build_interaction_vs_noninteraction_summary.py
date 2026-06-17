#!/usr/bin/env python3
"""Build a concise interaction-vs-noninteraction summary bundle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load(name: str):
    return json.loads((GEN / name).read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def weighted_mean(rows, value_key: str, weight_key: str) -> float:
    total_w = sum(float(row[weight_key]) for row in rows)
    if total_w <= 0:
        return 0.0
    return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / total_w


def main():
    editor = load("transition_conditioned_symbolic_editor.json")
    compact = load("pairguided_reranker_multislice.json")

    source_rollups = []
    source_task_rows = []
    for src in editor["sources"]:
        source_name = src["source"]
        summary_rows = src["summary"]
        enriched = []
        for row in summary_rows:
            num_frames = float(row["num_frames"])
            interaction_frames = num_frames * float(row["interaction_slice_rate"])
            noninteraction_frames = num_frames - interaction_frames
            enriched_row = {
                "task": row["task"],
                "num_frames": num_frames,
                "interaction_frames": interaction_frames,
                "noninteraction_frames": noninteraction_frames,
                "symbolic_interaction": float(row["symbolic_grouped_motif_match_interaction"]),
                "proxy_interaction": float(row["proxy_grouped_motif_match_interaction"]),
                "interaction_delta": float(row["symbolic_grouped_motif_match_interaction"])
                - float(row["proxy_grouped_motif_match_interaction"]),
                "symbolic_noninteraction": float(row["symbolic_grouped_motif_match_noninteraction"]),
                "proxy_noninteraction": float(row["proxy_grouped_motif_match_noninteraction"]),
                "noninteraction_delta": float(row["symbolic_grouped_motif_match_noninteraction"])
                - float(row["proxy_grouped_motif_match_noninteraction"]),
                "symbolic_all": float(row["symbolic_grouped_motif_match"]),
                "proxy_all": float(row["proxy_grouped_motif_match"]),
                "all_delta": float(row["symbolic_grouped_motif_match"])
                - float(row["proxy_grouped_motif_match"]),
            }
            enriched.append(enriched_row)
            source_task_rows.append({"source": source_name, **enriched_row})

        source_rollups.append(
            {
                "source": source_name,
                "weighted_symbolic_interaction": weighted_mean(enriched, "symbolic_interaction", "interaction_frames"),
                "weighted_proxy_interaction": weighted_mean(enriched, "proxy_interaction", "interaction_frames"),
                "weighted_interaction_delta": weighted_mean(enriched, "interaction_delta", "interaction_frames"),
                "weighted_symbolic_noninteraction": weighted_mean(enriched, "symbolic_noninteraction", "noninteraction_frames"),
                "weighted_proxy_noninteraction": weighted_mean(enriched, "proxy_noninteraction", "noninteraction_frames"),
                "weighted_noninteraction_delta": weighted_mean(enriched, "noninteraction_delta", "noninteraction_frames"),
                "weighted_symbolic_all": weighted_mean(enriched, "symbolic_all", "num_frames"),
                "weighted_proxy_all": weighted_mean(enriched, "proxy_all", "num_frames"),
                "weighted_all_delta": weighted_mean(enriched, "all_delta", "num_frames"),
            }
        )

    hard_rows = []
    for task, bundle in compact["task_results"].items():
        summary = bundle["summary"]
        hard_rows.append(
            {
                "task": task,
                "base_top5": float(summary["base_top5"]["joint_hit_rate_overall"]),
                "base_top10": float(summary["base_top10"]["joint_hit_rate_overall"]),
                "base_top20": float(summary["base_top20"]["joint_hit_rate_overall"]),
                "pairguided_top5": float(summary["pairguided_top5"]["joint_hit_rate_overall"]),
                "pairguided_top10": float(summary["pairguided_top10"]["joint_hit_rate_overall"]),
                "pairguided_minus_base_top5": float(summary["pairguided_top5"]["joint_hit_rate_overall"])
                - float(summary["base_top5"]["joint_hit_rate_overall"]),
                "pairguided_minus_base_top10": float(summary["pairguided_top10"]["joint_hit_rate_overall"])
                - float(summary["base_top10"]["joint_hit_rate_overall"]),
            }
        )

    payload = {
        "focus": {
            "goal": "concise top-level comparison between interaction and noninteraction symbolic editing behavior",
            "decision_boundary": "interaction remains the main weakness, but symbolic grouped-motif edits and hard-slice compact search still define the strongest temporal-control evidence",
        },
        "editor_rollups": source_rollups,
        "editor_task_rows": source_task_rows,
        "hard_right_hand_compact_search": hard_rows,
        "source_artifacts": {
            "transition_conditioned_symbolic_editor": str(GEN / "transition_conditioned_symbolic_editor.json"),
            "pairguided_reranker_multislice": str(GEN / "pairguided_reranker_multislice.json"),
        },
    }

    out_json = GEN / "interaction_vs_noninteraction_summary.json"
    out_md = SUM / "interaction_vs_noninteraction_summary.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction vs Noninteraction Summary",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "A concise top-level summary of where symbolic temporal editing is strong,",
        "where it is weaker, and how the compact search mechanism fits that gap.",
        "",
        "## Editor Rollups",
        "",
        "| source | symbolic interaction | proxy interaction | delta | symbolic noninteraction | proxy noninteraction | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in source_rollups:
        lines.append(
            f"| {row['source']} | {fmt(row['weighted_symbolic_interaction'])} | {fmt(row['weighted_proxy_interaction'])} | {fmt(row['weighted_interaction_delta'])} | "
            f"{fmt(row['weighted_symbolic_noninteraction'])} | {fmt(row['weighted_proxy_noninteraction'])} | {fmt(row['weighted_noninteraction_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Per Task Editor Detail",
            "",
            "| source | task | interaction delta | noninteraction delta | all delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in source_task_rows:
        lines.append(
            f"| {row['source']} | {row['task']} | {fmt(row['interaction_delta'])} | {fmt(row['noninteraction_delta'])} | {fmt(row['all_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Hard Right-Hand Compact Search",
            "",
            "| task | base top-5 | base top-10 | base top-20 | pair-guided top-5 | pair-guided top-10 | top-5 gain | top-10 gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hard_rows:
        lines.append(
            f"| {row['task']} | {fmt(row['base_top5'])} | {fmt(row['base_top10'])} | {fmt(row['base_top20'])} | "
            f"{fmt(row['pairguided_top5'])} | {fmt(row['pairguided_top10'])} | {fmt(row['pairguided_minus_base_top5'])} | {fmt(row['pairguided_minus_base_top10'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Noninteraction is easier overall, but symbolic grouped-motif edits remain better than opaque proxies on both interaction and noninteraction slices.",
            "- Interaction is still the concentrated weakness because compact-search recovery is only needed on the hard right-hand interaction tasks.",
            "- The right current claim is therefore not universal interaction mastery, but symbolic editability plus compact hard-slice recovery.",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
