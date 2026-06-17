#!/usr/bin/env python3
"""Build a hard-case event alignment report comparing state-event vs temporal-event."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools.compare_symbolic_slices import slice_membership


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    intrinsic = load_json(GEN / "symbolic_representation_intrinsic_val_test_plus.json")
    slice_frontier = load_json(GEN / "slice_frontier_report.json")

    state = next(r for r in intrinsic["results"] if r["name"] == "state_event")
    temporal = next(r for r in intrinsic["results"] if r["name"] == "temporal_event")
    state_map = {row["seq_name"]: row for row in state["rows"]}
    temp_map = {row["seq_name"]: row for row in temporal["rows"]}

    per_query = []
    by_slice = defaultdict(list)
    for seq_name in sorted(state_map):
        s = state_map[seq_name]
        t = temp_map[seq_name]
        row = {
            "seq_name": seq_name,
            "label": s["label"],
            "state_top1_correct": s["top1_correct"],
            "temporal_top1_correct": t["top1_correct"],
            "state_margin": s["positive_margin"],
            "temporal_margin": t["positive_margin"],
            "margin_delta": t["positive_margin"] - s["positive_margin"],
            "state_rank": s["correct_rank"],
            "temporal_rank": t["correct_rank"],
            "rank_delta": t["correct_rank"] - s["correct_rank"],
            "state_best_wrong": s["best_wrong_label"],
            "temporal_best_wrong": t["best_wrong_label"],
            "wrong_neighbor_changed": s["best_wrong_label"] != t["best_wrong_label"],
            "slices": sorted(slice_membership(seq_name)),
        }
        per_query.append(row)
        for group in row["slices"]:
            by_slice[group].append(row)

    slice_rows = []
    for group, rows in by_slice.items():
        slice_rows.append(
            {
                "slice": group,
                "num_queries": len(rows),
                "mean_state_margin": mean([r["state_margin"] for r in rows]),
                "mean_temporal_margin": mean([r["temporal_margin"] for r in rows]),
                "mean_margin_delta": mean([r["margin_delta"] for r in rows]),
                "mean_state_rank": mean([r["state_rank"] for r in rows]),
                "mean_temporal_rank": mean([r["temporal_rank"] for r in rows]),
                "mean_rank_delta": mean([r["rank_delta"] for r in rows]),
                "wrong_neighbor_changed_count": sum(r["wrong_neighbor_changed"] for r in rows),
                "temporal_margin_better_count": sum(r["margin_delta"] > 0 for r in rows),
            }
        )
    slice_rows.sort(key=lambda r: (-r["mean_margin_delta"], r["slice"]))

    notable = [
        r
        for r in per_query
        if r["margin_delta"] > 0.01 or r["rank_delta"] != 0 or r["wrong_neighbor_changed"]
    ]
    notable.sort(key=lambda r: (-r["margin_delta"], r["seq_name"]))

    report = {
        "artifacts": {
            "intrinsic_report": str(GEN / "symbolic_representation_intrinsic_val_test_plus.json"),
            "slice_frontier_report": str(GEN / "slice_frontier_report.json"),
        },
        "summary": {
            "num_queries": len(per_query),
            "temporal_margin_better_count": sum(r["margin_delta"] > 0 for r in per_query),
            "temporal_rank_better_count": sum(r["rank_delta"] < 0 for r in per_query),
            "wrong_neighbor_changed_count": sum(r["wrong_neighbor_changed"] for r in per_query),
            "mean_margin_delta_all": mean([r["margin_delta"] for r in per_query]),
        },
        "slice_alignment": slice_rows,
        "notable_queries": notable,
        "cross_checks": {
            "lowdata_major_slice_from_frontier": next(
                r for r in slice_frontier["fraction_0.5"]["rows"] if r["slice"] == "wrist_rom"
            ),
            "fulldata_finger_slice_from_frontier": next(
                r for r in slice_frontier["fraction_1.0"]["rows"] if r["slice"] == "finger_occlusion"
            ),
        },
        "derived_claim_checks": {
            "temporal_margin_better_on_majority_of_queries": sum(r["margin_delta"] > 0 for r in per_query) > len(per_query) / 2,
            "temporal_changes_wrong_neighbor_for_multiple_hard_cases": sum(r["wrong_neighbor_changed"] for r in per_query) >= 6,
            "interaction_slice_margin_improves": next(r for r in slice_rows if r["slice"] == "interaction")["mean_margin_delta"] > 0,
            "wrist_rom_slice_changes_wrong_neighbor": next(r for r in slice_rows if r["slice"] == "wrist_rom")["wrong_neighbor_changed_count"] > 0,
            "finger_occlusion_slice_margin_improves": next(r for r in slice_rows if r["slice"] == "finger_occlusion")["mean_margin_delta"] > 0,
        },
        "takeaways": [
            "Temporal transition information helps mostly by increasing positive-vs-negative separation, not by changing top-1 on many queries.",
            "The nearest wrong neighbor changes on multiple hard cases, indicating that temporal events reshape local symbolic neighborhoods instead of only rescoring the same confusions.",
            "Interaction and finger-occlusion slices show positive mean margin deltas under temporal events.",
            "Wrist-ROM is more nuanced: its event-level mean margin does not improve, but the nearest wrong neighbor changes, so temporal transitions still alter the local confusion structure there.",
            "This is consistent with the slice frontier: transition-aware symbols act as a hard-case separator rather than a universal easy-case booster.",
        ],
    }

    out_json = GEN / "hardcase_event_alignment_report.json"
    out_md = SUM / "hardcase_event_alignment_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Hard-Case Event Alignment Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Summary",
        "",
        f"- `num_queries`: `{report['summary']['num_queries']}`",
        f"- `temporal_margin_better_count`: `{report['summary']['temporal_margin_better_count']}`",
        f"- `temporal_rank_better_count`: `{report['summary']['temporal_rank_better_count']}`",
        f"- `wrong_neighbor_changed_count`: `{report['summary']['wrong_neighbor_changed_count']}`",
        f"- `mean_margin_delta_all`: `{fmt(report['summary']['mean_margin_delta_all'])}`",
        "",
        "## Slice Alignment",
        "",
        "| slice | queries | mean state margin | mean temporal margin | mean margin delta | wrong-neighbor changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in slice_rows:
        lines.append(
            f"| {row['slice']} | {row['num_queries']} | {fmt(row['mean_state_margin'])} | {fmt(row['mean_temporal_margin'])} | {fmt(row['mean_margin_delta'])} | {row['wrong_neighbor_changed_count']} |"
        )

    lines.extend(["", "## Notable Queries", "", "| seq | state margin | temporal margin | delta | state wrong | temporal wrong |", "| --- | ---: | ---: | ---: | --- | --- |"])
    for row in notable[:12]:
        lines.append(
            f"| {row['seq_name']} | {fmt(row['state_margin'])} | {fmt(row['temporal_margin'])} | {fmt(row['margin_delta'])} | {row['state_best_wrong']} | {row['temporal_best_wrong']} |"
        )

    lines.extend(["", "## Claim Checks", ""])
    for key, value in report["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Takeaways", ""])
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
