#!/usr/bin/env python3
"""Summarize classifier-level results for symbolic channel variants."""

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
    payload = load_json(GEN / "symbolic_channel_variants_classifier.json")
    variants = payload["variants"]
    state_only = variants["state_only"]["summary"]
    full_temporal = variants["full_temporal"]["summary"]
    state_transition = variants["state_transition"]["summary"]
    state_motion = variants["state_motion"]["summary"]

    report = {
        "artifacts": {
            "classifier_variants": str(GEN / "symbolic_channel_variants_classifier.json"),
        },
        "variants": variants,
        "derived_claim_checks": {
            "state_transition_beats_full_temporal_at_fraction_1.0": state_transition["1.0"]["sequence_accuracy_mean"] > full_temporal["1.0"]["sequence_accuracy_mean"],
            "state_motion_beats_full_temporal_at_fraction_1.0": state_motion["1.0"]["sequence_accuracy_mean"] > full_temporal["1.0"]["sequence_accuracy_mean"],
            "state_transition_beats_full_temporal_at_fraction_0.5": state_transition["0.5"]["sequence_accuracy_mean"] > full_temporal["0.5"]["sequence_accuracy_mean"],
            "state_motion_beats_full_temporal_at_fraction_0.5": state_motion["0.5"]["sequence_accuracy_mean"] > full_temporal["0.5"]["sequence_accuracy_mean"],
            "best_lowdata_variant_is_state_motion": (
                state_motion["0.5"]["sequence_accuracy_mean"] >= state_transition["0.5"]["sequence_accuracy_mean"]
                and state_motion["0.5"]["sequence_accuracy_mean"] >= full_temporal["0.5"]["sequence_accuracy_mean"]
            ),
            "best_fulldata_variants_tie": state_motion["1.0"]["sequence_accuracy_mean"] == state_transition["1.0"]["sequence_accuracy_mean"],
        },
        "gaps": {
            "fraction_0.5_state_motion_minus_full_temporal": state_motion["0.5"]["sequence_accuracy_mean"] - full_temporal["0.5"]["sequence_accuracy_mean"],
            "fraction_0.5_state_transition_minus_full_temporal": state_transition["0.5"]["sequence_accuracy_mean"] - full_temporal["0.5"]["sequence_accuracy_mean"],
            "fraction_1.0_state_motion_minus_full_temporal": state_motion["1.0"]["sequence_accuracy_mean"] - full_temporal["1.0"]["sequence_accuracy_mean"],
            "fraction_1.0_state_transition_minus_full_temporal": state_transition["1.0"]["sequence_accuracy_mean"] - full_temporal["1.0"]["sequence_accuracy_mean"],
            "fraction_1.0_full_temporal_minus_state_only": full_temporal["1.0"]["sequence_accuracy_mean"] - state_only["1.0"]["sequence_accuracy_mean"],
        },
        "takeaways": [
            "The intrinsic channel-pruning gains do transfer to the lightweight classifier level.",
            "Both state+motion and state+transition beat the current full temporal feature mixture at full and low data.",
            "At low data, state+motion is the strongest classifier variant on this protocol.",
            "At full data, state+motion and state+transition tie and both outperform full temporal.",
            "This strengthens the case that the current full temporal feature mixture is over-complete and that pruning temporal channels is a real upgrade direction rather than a purely intrinsic artifact.",
        ],
    }

    out_json = GEN / "symbolic_channel_variants_classifier_report.json"
    out_md = SUM / "symbolic_channel_variants_classifier_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Symbolic Channel Variants Classifier Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| variant | fraction | seq acc | win acc |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in ["state_only", "full_temporal", "state_transition", "state_motion"]:
        summary = variants[name]["summary"]
        for frac in ["0.5", "1.0"]:
            lines.append(
                f"| {name} | {frac} | {fmt(summary[frac]['sequence_accuracy_mean'])} | {fmt(summary[frac]['window_accuracy_mean'])} |"
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
