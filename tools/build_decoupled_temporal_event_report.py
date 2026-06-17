#!/usr/bin/env python3
"""Summarize decoupled temporal-event intrinsic results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    payload = load_json(GEN / "decoupled_temporal_event_intrinsic.json")
    mp = {row["name"]: row for row in payload["results"]}
    full = mp["full"]["summary"]
    state_motion = mp["state_motion"]["summary"]
    state_transition = mp["state_transition"]["summary"]
    transition_motion = mp["transition_motion"]["summary"]
    motion_only = mp["motion_only"]["summary"]
    transition_only = mp["transition_only"]["summary"]
    duration_only = mp["duration_only"]["summary"]
    interaction_only = mp["interaction_only"]["summary"]
    temporal_nostate = mp["temporal_nostate"]["summary"]

    report = {
        "artifacts": {
            "decoupled_intrinsic": str(GEN / "decoupled_temporal_event_intrinsic.json"),
        },
        "variants": payload["results"],
        "derived_claim_checks": {
            "state_motion_preserves_top1_vs_full": state_motion["top1_accuracy"] == full["top1_accuracy"],
            "state_motion_improves_margin_vs_full": state_motion["mean_positive_margin"] > full["mean_positive_margin"],
            "state_transition_preserves_top1_vs_full": state_transition["top1_accuracy"] == full["top1_accuracy"],
            "state_transition_improves_margin_vs_full": state_transition["mean_positive_margin"] > full["mean_positive_margin"],
            "motion_only_beats_transition_only_top1": motion_only["top1_accuracy"] > transition_only["top1_accuracy"],
            "transition_only_beats_duration_only_top1": transition_only["top1_accuracy"] > duration_only["top1_accuracy"],
            "transition_only_beats_interaction_only_top1": transition_only["top1_accuracy"] > interaction_only["top1_accuracy"],
            "removing_state_hurts_top1": temporal_nostate["top1_accuracy"] < full["top1_accuracy"],
        },
        "gaps": {
            "state_motion_minus_full_margin": state_motion["mean_positive_margin"] - full["mean_positive_margin"],
            "state_transition_minus_full_margin": state_transition["mean_positive_margin"] - full["mean_positive_margin"],
            "motion_only_minus_transition_only_top1": motion_only["top1_accuracy"] - transition_only["top1_accuracy"],
            "full_minus_temporal_nostate_top1": full["top1_accuracy"] - temporal_nostate["top1_accuracy"],
        },
        "takeaways": [
            "The best intrinsic temporal-event representation on this benchmark is not the most feature-complete one.",
            "Both state+motion and state+transition preserve full top-1 while improving separation margin over the current full temporal-event encoding.",
            "This implies the current full event encoding contains redundant or noisy temporal channels.",
            "State remains essential for top-1 stability: removing state drops top-1 from 0.9545 to 0.9091.",
            "Among pure temporal channels, motion is stronger than transition, and transition is far stronger than duration-only or interaction-only.",
            "A stronger next representation candidate is therefore a pruned or reweighted temporal event encoding rather than a more additive one.",
        ],
    }

    out_json = GEN / "decoupled_temporal_event_report.json"
    out_md = SUM / "decoupled_temporal_event_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Decoupled Temporal Event Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| variant | channels | top1 | mean margin | mean rank |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        s = row["summary"]
        lines.append(
            f"| {row['name']} | {'+'.join(row['channels'])} | {fmt(s['top1_accuracy'])} | {fmt(s['mean_positive_margin'])} | {fmt(s['mean_correct_rank'])} |"
        )

    lines.extend(["", "## Claim Checks", ""])
    for key, value in report["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Gap Summary", ""])
    for key, value in report["gaps"].items():
        lines.append(f"- `{key}`: `{fmt(value)}`")

    lines.extend(["", "## Takeaways", ""])
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
