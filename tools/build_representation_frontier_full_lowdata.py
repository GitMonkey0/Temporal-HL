#!/usr/bin/env python3
"""Build a unified full-data / low-data frontier report for the symbolic representation line."""

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
    joint = load_json(GEN / "joint_token_control_summary.json")
    evidence = load_json(GEN / "representation_evidence_bundle.json")
    mainline = load_json(GEN / "mainline_evidence_report.json")
    intrinsic = load_json(GEN / "symbolic_representation_intrinsic_val_test_plus.json")

    rows = ablation["rows"]
    intrinsic_rows = intrinsic["results"]

    state_only_full = pick_ablation_row(rows, "symbolic_state_only", "1.0")
    duration_full = pick_ablation_row(rows, "symbolic_state_transition_duration", "1.0")
    duration_low = pick_ablation_row(rows, "symbolic_state_transition_duration", "0.5")
    temporal_encoder_full = pick_ablation_row(rows, "temporal_sequence_state_transition", "1.0")
    temporal_encoder_low = pick_ablation_row(rows, "temporal_sequence_state_transition", "0.5")

    main_full = mainline["mainline"]["fraction_1.0"]
    main_low = mainline["mainline"]["fraction_0.5"]

    joint_seq_full = joint["joint_sequence_fraction_1.0"]
    joint_seq_low = joint["joint_sequence_fraction_0.5"]
    joint64_full = joint["joint_token64"]["1.0"]
    joint64_low = joint["joint_token64"]["0.5"]
    joint32_full = joint["joint_token32"]["1.0"]
    joint32_low = joint["joint_token32"]["0.5"]

    state_frame = pick_intrinsic(intrinsic_rows, "state_frame")
    temporal_frame = pick_intrinsic(intrinsic_rows, "temporal_frame")
    state_rle = pick_intrinsic(intrinsic_rows, "state_rle")
    temporal_rle = pick_intrinsic(intrinsic_rows, "temporal_rle")
    state_event = pick_intrinsic(intrinsic_rows, "state_event")
    temporal_event = pick_intrinsic(intrinsic_rows, "temporal_event")
    temporal_event_persist = pick_intrinsic(intrinsic_rows, "temporal_event_persist")
    temporal_event_segdur = pick_intrinsic(intrinsic_rows, "temporal_event_segdur")

    full_low = {
        "artifacts": {
            "ablation_report": str(GEN / "temporal_representation_ablation_report.json"),
            "joint_token_control_summary": str(GEN / "joint_token_control_summary.json"),
            "representation_evidence_bundle": str(GEN / "representation_evidence_bundle.json"),
            "mainline_evidence_report": str(GEN / "mainline_evidence_report.json"),
            "intrinsic_report": str(GEN / "symbolic_representation_intrinsic_val_test_plus.json"),
        },
        "frontier": {
            "fraction_1.0": {
                "symbolic_mainline_state_transition": main_full,
                "symbolic_state_only": state_only_full,
                "symbolic_state_transition_duration": duration_full,
                "temporal_sequence_state_transition": temporal_encoder_full,
                "joint_sequence_best": joint_seq_full,
                "joint_token64": joint64_full,
                "joint_token32": joint32_full,
            },
            "fraction_0.5": {
                "symbolic_mainline_state_transition": main_low,
                "symbolic_state_transition_duration": duration_low,
                "temporal_sequence_state_transition": temporal_encoder_low,
                "joint_sequence_best": joint_seq_low,
                "joint_token64": joint64_low,
                "joint_token32": joint32_low,
            },
        },
        "intrinsic": {
            "state_frame": state_frame["summary"],
            "temporal_frame": temporal_frame["summary"],
            "state_rle": state_rle["summary"],
            "temporal_rle": temporal_rle["summary"],
            "state_event": state_event["summary"],
            "temporal_event": temporal_event["summary"],
            "temporal_event_persist": temporal_event_persist["summary"],
            "temporal_event_segdur": temporal_event_segdur["summary"],
        },
        "derived_claim_checks": {
            "full_mainline_beats_state_only": (
                main_full["pretrained_seq"] > state_only_full["pretrained_sequence_accuracy_mean"]
            ),
            "full_mainline_beats_duration": (
                main_full["pretrained_seq"] > duration_full["pretrained_sequence_accuracy_mean"]
            ),
            "lowdata_mainline_beats_duration": (
                main_low["pretrained_seq"] > duration_low["pretrained_sequence_accuracy_mean"]
            ),
            "full_mainline_beats_temporal_encoder": (
                main_full["pretrained_seq"] > temporal_encoder_full["pretrained_sequence_accuracy_mean"]
            ),
            "lowdata_mainline_beats_temporal_encoder": (
                main_low["pretrained_seq"] > temporal_encoder_low["pretrained_sequence_accuracy_mean"]
            ),
            "full_mainline_beats_joint_sequence_best": (
                main_full["pretrained_seq"] > joint_seq_full["pretrained_sequence_accuracy_mean"]
            ),
            "lowdata_mainline_beats_joint_sequence_best": (
                main_low["pretrained_seq"] > joint_seq_low["pretrained_sequence_accuracy_mean"]
            ),
            "full_mainline_beats_joint_token64": (
                main_full["pretrained_seq"] > joint64_full["pretrained_sequence_accuracy_mean"]
            ),
            "lowdata_mainline_beats_joint_token64": (
                main_low["pretrained_seq"] > joint64_low["pretrained_sequence_accuracy_mean"]
            ),
            "intrinsic_event_beats_frame_top1": (
                temporal_event["summary"]["top1_accuracy"] > temporal_frame["summary"]["top1_accuracy"]
            ),
            "intrinsic_event_beats_rle_top1": (
                temporal_event["summary"]["top1_accuracy"] > temporal_rle["summary"]["top1_accuracy"]
            ),
            "intrinsic_temporal_event_margin_beats_state_event": (
                temporal_event["summary"]["mean_positive_margin"]
                > state_event["summary"]["mean_positive_margin"]
            ),
            "intrinsic_temporal_event_rank_beats_state_event": (
                temporal_event["summary"]["mean_correct_rank"]
                < state_event["summary"]["mean_correct_rank"]
            ),
        },
        "gaps": {
            "fraction_1.0": {
                "mainline_minus_state_only": main_full["pretrained_seq"] - state_only_full["pretrained_sequence_accuracy_mean"],
                "mainline_minus_duration": main_full["pretrained_seq"] - duration_full["pretrained_sequence_accuracy_mean"],
                "mainline_minus_temporal_encoder": main_full["pretrained_seq"] - temporal_encoder_full["pretrained_sequence_accuracy_mean"],
                "mainline_minus_joint_sequence_best": main_full["pretrained_seq"] - joint_seq_full["pretrained_sequence_accuracy_mean"],
                "mainline_minus_joint_token64": main_full["pretrained_seq"] - joint64_full["pretrained_sequence_accuracy_mean"],
            },
            "fraction_0.5": {
                "mainline_minus_duration": main_low["pretrained_seq"] - duration_low["pretrained_sequence_accuracy_mean"],
                "mainline_minus_temporal_encoder": main_low["pretrained_seq"] - temporal_encoder_low["pretrained_sequence_accuracy_mean"],
                "mainline_minus_joint_sequence_best": main_low["pretrained_seq"] - joint_seq_low["pretrained_sequence_accuracy_mean"],
                "mainline_minus_joint_token64": main_low["pretrained_seq"] - joint64_low["pretrained_sequence_accuracy_mean"],
            },
        },
        "carry_over_context": {
            "existing_claim_checks": evidence["derived_claim_checks"],
            "plain_summary": mainline["plain_summary"],
            "slice_gains": mainline["slice_gains"],
        },
        "takeaways": [
            "The strongest sequence-native symbolic representation remains the state+transition mainline under the pretrain-only normalized protocol.",
            "This mainline wins in both full-data and low-data regimes, so the result is not a single-regime artifact.",
            "At full data, the symbolic mainline beats state-only, duration-augmented, temporal-sequence-encoder, and learned-token controls.",
            "At low data, the symbolic mainline remains ahead of duration, temporal-sequence encoder, and the best joint learned-token control.",
            "Intrinsic evaluation still shows that eventized symbolic comparison is the real source of discriminability, while frame-token and RLE variants are weak controls.",
            "Transition-aware symbolic structure is the current clean frontier; duration and opaque learned tokens remain secondary or negative controls.",
        ],
    }

    out_json = GEN / "representation_frontier_full_lowdata.json"
    out_md = SUM / "representation_frontier_full_lowdata.md"
    out_json.write_text(json.dumps(full_low, indent=2))

    lines = [
        "# Representation Frontier: Full Data and Low Data",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Sequence Frontier",
        "",
        "| representation | fraction | pretrained seq | scratch seq | pretrained win | scratch win |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| symbolic_mainline_state_transition | 1.0 | {fmt(main_full['pretrained_seq'])} | {fmt(main_full['scratch_seq'])} | {fmt(main_full['pretrained_win'])} | {fmt(main_full['scratch_win'])} |",
        f"| symbolic_state_only | 1.0 | {fmt(state_only_full['pretrained_sequence_accuracy_mean'])} | {fmt(state_only_full['scratch_sequence_accuracy_mean'])} | {fmt(state_only_full['pretrained_window_accuracy_mean'])} | {fmt(state_only_full['scratch_window_accuracy_mean'])} |",
        f"| symbolic_state_transition_duration | 1.0 | {fmt(duration_full['pretrained_sequence_accuracy_mean'])} | {fmt(duration_full['scratch_sequence_accuracy_mean'])} | {fmt(duration_full['pretrained_window_accuracy_mean'])} | {fmt(duration_full['scratch_window_accuracy_mean'])} |",
        f"| temporal_sequence_state_transition | 1.0 | {fmt(temporal_encoder_full['pretrained_sequence_accuracy_mean'])} | {fmt(temporal_encoder_full['scratch_sequence_accuracy_mean'])} | {fmt(temporal_encoder_full['pretrained_window_accuracy_mean'])} | {fmt(temporal_encoder_full['scratch_window_accuracy_mean'])} |",
        f"| joint_sequence_best | 1.0 | {fmt(joint_seq_full['pretrained_sequence_accuracy_mean'])} | {fmt(joint_seq_full['scratch_sequence_accuracy_mean'])} | {fmt(joint_seq_full['pretrained_window_accuracy_mean'])} | {fmt(joint_seq_full['scratch_window_accuracy_mean'])} |",
        f"| joint_token64 | 1.0 | {fmt(joint64_full['pretrained_sequence_accuracy_mean'])} | {fmt(joint64_full['scratch_sequence_accuracy_mean'])} | {fmt(joint64_full['pretrained_window_accuracy_mean'])} | {fmt(joint64_full['scratch_window_accuracy_mean'])} |",
        f"| joint_token32 | 1.0 | {fmt(joint32_full['pretrained_sequence_accuracy_mean'])} | {fmt(joint32_full['scratch_sequence_accuracy_mean'])} | {fmt(joint32_full['pretrained_window_accuracy_mean'])} | {fmt(joint32_full['scratch_window_accuracy_mean'])} |",
        f"| symbolic_mainline_state_transition | 0.5 | {fmt(main_low['pretrained_seq'])} | {fmt(main_low['scratch_seq'])} | {fmt(main_low['pretrained_win'])} | {fmt(main_low['scratch_win'])} |",
        f"| symbolic_state_transition_duration | 0.5 | {fmt(duration_low['pretrained_sequence_accuracy_mean'])} | {fmt(duration_low['scratch_sequence_accuracy_mean'])} | {fmt(duration_low['pretrained_window_accuracy_mean'])} | {fmt(duration_low['scratch_window_accuracy_mean'])} |",
        f"| temporal_sequence_state_transition | 0.5 | {fmt(temporal_encoder_low['pretrained_sequence_accuracy_mean'])} | {fmt(temporal_encoder_low['scratch_sequence_accuracy_mean'])} | {fmt(temporal_encoder_low['pretrained_window_accuracy_mean'])} | {fmt(temporal_encoder_low['scratch_window_accuracy_mean'])} |",
        f"| joint_sequence_best | 0.5 | {fmt(joint_seq_low['pretrained_sequence_accuracy_mean'])} | {fmt(joint_seq_low['scratch_sequence_accuracy_mean'])} | {fmt(joint_seq_low['pretrained_window_accuracy_mean'])} | {fmt(joint_seq_low['scratch_window_accuracy_mean'])} |",
        f"| joint_token64 | 0.5 | {fmt(joint64_low['pretrained_sequence_accuracy_mean'])} | {fmt(joint64_low['scratch_sequence_accuracy_mean'])} | {fmt(joint64_low['pretrained_window_accuracy_mean'])} | {fmt(joint64_low['scratch_window_accuracy_mean'])} |",
        f"| joint_token32 | 0.5 | {fmt(joint32_low['pretrained_sequence_accuracy_mean'])} | {fmt(joint32_low['scratch_sequence_accuracy_mean'])} | {fmt(joint32_low['pretrained_window_accuracy_mean'])} | {fmt(joint32_low['scratch_window_accuracy_mean'])} |",
        "",
        "## Gap Summary",
        "",
        "| gap | fraction 1.0 | fraction 0.5 |",
        "| --- | ---: | ---: |",
        f"| mainline - duration | {fmt(full_low['gaps']['fraction_1.0']['mainline_minus_duration'])} | {fmt(full_low['gaps']['fraction_0.5']['mainline_minus_duration'])} |",
        f"| mainline - temporal encoder | {fmt(full_low['gaps']['fraction_1.0']['mainline_minus_temporal_encoder'])} | {fmt(full_low['gaps']['fraction_0.5']['mainline_minus_temporal_encoder'])} |",
        f"| mainline - joint sequence best | {fmt(full_low['gaps']['fraction_1.0']['mainline_minus_joint_sequence_best'])} | {fmt(full_low['gaps']['fraction_0.5']['mainline_minus_joint_sequence_best'])} |",
        f"| mainline - joint token64 | {fmt(full_low['gaps']['fraction_1.0']['mainline_minus_joint_token64'])} | {fmt(full_low['gaps']['fraction_0.5']['mainline_minus_joint_token64'])} |",
        "",
        "## Intrinsic Representation Evidence",
        "",
        "| representation | top1 | mean positive margin | mean correct rank |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in [
        state_frame,
        temporal_frame,
        state_rle,
        temporal_rle,
        state_event,
        temporal_event,
        temporal_event_persist,
        temporal_event_segdur,
    ]:
        s = row["summary"]
        lines.append(
            f"| {row['name']} | {fmt(s['top1_accuracy'])} | {fmt(s['mean_positive_margin'])} | {fmt(s['mean_correct_rank'])} |"
        )

    lines.extend(["", "## Claim Checks", ""])
    for key, value in full_low["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Takeaways", ""])
    for item in full_low["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
