#!/usr/bin/env python3
"""Build a consolidated evidence bundle for the sequence-native symbolic representation."""

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


def pick_ablation_row(rows, name: str, fraction: str):
    for row in rows:
        if row["name"] == name and str(row["fraction"]) == str(fraction):
            return row
    raise KeyError((name, fraction))


def pick_intrinsic(results, name: str):
    for row in results:
        if row["name"] == name:
            return row
    raise KeyError(name)


def main():
    ablation = load_json(GEN / "temporal_representation_ablation_report.json")
    intrinsic = load_json(GEN / "symbolic_representation_intrinsic_val_test_plus.json")
    mainline = load_json(GEN / "mainline_evidence_report.json")

    rows = ablation["rows"]
    intrinsic_rows = intrinsic["results"]

    symbolic_state_only = pick_ablation_row(rows, "symbolic_state_only", "1.0")
    symbolic_state_transition = pick_ablation_row(rows, "symbolic_state_transition", "1.0")
    symbolic_state_transition_duration = pick_ablation_row(rows, "symbolic_state_transition_duration", "1.0")
    symbolic_state_transition_coord = pick_ablation_row(rows, "symbolic_state_transition_coord", "1.0")

    state_frame = pick_intrinsic(intrinsic_rows, "state_frame")
    temporal_frame = pick_intrinsic(intrinsic_rows, "temporal_frame")
    state_rle = pick_intrinsic(intrinsic_rows, "state_rle")
    temporal_rle = pick_intrinsic(intrinsic_rows, "temporal_rle")
    state_event = pick_intrinsic(intrinsic_rows, "state_event")
    temporal_event = pick_intrinsic(intrinsic_rows, "temporal_event")
    temporal_event_persist = pick_intrinsic(intrinsic_rows, "temporal_event_persist")
    temporal_event_segdur = pick_intrinsic(intrinsic_rows, "temporal_event_segdur")

    bundle = {
        "artifacts": {
            "ablation_report": str(GEN / "temporal_representation_ablation_report.json"),
            "intrinsic_report": str(GEN / "symbolic_representation_intrinsic_val_test_plus.json"),
            "mainline_evidence_report": str(GEN / "mainline_evidence_report.json"),
        },
        "classifier_evidence": {
            "state_only_fraction_1.0": symbolic_state_only,
            "state_transition_fraction_1.0": symbolic_state_transition,
            "state_transition_duration_fraction_1.0": symbolic_state_transition_duration,
            "state_transition_coord_fraction_1.0": symbolic_state_transition_coord,
        },
        "intrinsic_evidence": {
            "state_frame": state_frame,
            "temporal_frame": temporal_frame,
            "state_rle": state_rle,
            "temporal_rle": temporal_rle,
            "state_event": state_event,
            "temporal_event": temporal_event,
            "temporal_event_persist": temporal_event_persist,
            "temporal_event_segdur": temporal_event_segdur,
        },
        "derived_claim_checks": {
            "classifier_state_transition_beats_state_only": (
                symbolic_state_transition["pretrained_sequence_accuracy_mean"]
                > symbolic_state_only["pretrained_sequence_accuracy_mean"]
            ),
            "classifier_state_transition_beats_duration": (
                symbolic_state_transition["pretrained_sequence_accuracy_mean"]
                > symbolic_state_transition_duration["pretrained_sequence_accuracy_mean"]
            ),
            "classifier_state_transition_beats_coord": (
                symbolic_state_transition["pretrained_sequence_accuracy_mean"]
                > symbolic_state_transition_coord["pretrained_sequence_accuracy_mean"]
            ),
            "intrinsic_event_beats_frame_top1": (
                temporal_event["summary"]["top1_accuracy"]
                > temporal_frame["summary"]["top1_accuracy"]
            ),
            "intrinsic_event_beats_rle_top1": (
                temporal_event["summary"]["top1_accuracy"]
                > temporal_rle["summary"]["top1_accuracy"]
            ),
            "intrinsic_temporal_event_margin_beats_state_event": (
                temporal_event["summary"]["mean_positive_margin"]
                > state_event["summary"]["mean_positive_margin"]
            ),
            "intrinsic_temporal_event_rank_beats_state_event": (
                temporal_event["summary"]["mean_correct_rank"]
                < state_event["summary"]["mean_correct_rank"]
            ),
            "intrinsic_temporal_event_margin_beats_persist": (
                temporal_event["summary"]["mean_positive_margin"]
                > temporal_event_persist["summary"]["mean_positive_margin"]
            ),
            "intrinsic_temporal_event_margin_beats_segdur": (
                temporal_event["summary"]["mean_positive_margin"]
                > temporal_event_segdur["summary"]["mean_positive_margin"]
            ),
        },
        "mainline_context": {
            "plain_summary": mainline["plain_summary"],
            "retrieval_evidence": mainline["retrieval_evidence"],
        },
        "takeaways": [
            "At the classifier level, state+transition is stronger than state-only, duration, and coordination variants.",
            "At the intrinsic level, eventized symbolic comparison is dramatically stronger than frame-token or RLE comparison.",
            "State-event and temporal-event tie on top-1 intrinsic retrieval, but temporal-event has higher separation margin and better mean correct rank.",
            "Persistence and segment-duration variants match event top-1 but do not exceed plain temporal-event on separation margin.",
            "Current evidence supports transition-aware eventized symbolic representation as the cleanest mainline formulation.",
        ],
    }

    out_json = GEN / "representation_evidence_bundle.json"
    out_md = SUM / "representation_evidence_bundle.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2))

    lines = [
        "# Representation Evidence Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Classifier Evidence",
        "",
        "| representation | pretrained seq | scratch seq | pretrained win | scratch win |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| state_only | {fmt(symbolic_state_only['pretrained_sequence_accuracy_mean'])} | {fmt(symbolic_state_only['scratch_sequence_accuracy_mean'])} | {fmt(symbolic_state_only['pretrained_window_accuracy_mean'])} | {fmt(symbolic_state_only['scratch_window_accuracy_mean'])} |",
        f"| state_transition | {fmt(symbolic_state_transition['pretrained_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition['scratch_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition['pretrained_window_accuracy_mean'])} | {fmt(symbolic_state_transition['scratch_window_accuracy_mean'])} |",
        f"| state_transition_duration | {fmt(symbolic_state_transition_duration['pretrained_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition_duration['scratch_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition_duration['pretrained_window_accuracy_mean'])} | {fmt(symbolic_state_transition_duration['scratch_window_accuracy_mean'])} |",
        f"| state_transition_coord | {fmt(symbolic_state_transition_coord['pretrained_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition_coord['scratch_sequence_accuracy_mean'])} | {fmt(symbolic_state_transition_coord['pretrained_window_accuracy_mean'])} | {fmt(symbolic_state_transition_coord['scratch_window_accuracy_mean'])} |",
        "",
        "## Intrinsic Evidence",
        "",
        "| representation | top1 | mean positive margin | mean correct rank |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in [state_frame, temporal_frame, state_rle, temporal_rle, state_event, temporal_event, temporal_event_persist, temporal_event_segdur]:
        s = row["summary"]
        lines.append(
            f"| {row['name']} | {fmt(s['top1_accuracy'])} | {fmt(s['mean_positive_margin'])} | {fmt(s['mean_correct_rank'])} |"
        )

    lines.extend(
        [
            "",
            "## Claim Checks",
            "",
        ]
    )
    for key, value in bundle["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
        ]
    )
    for item in bundle["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
