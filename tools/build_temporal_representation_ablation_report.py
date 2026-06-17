#!/usr/bin/env python3
"""Build a compact comparison report for temporal representation ablations.

This is an experiment-report utility, not paper text.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def fetch_summary(data: dict, fraction: str) -> dict | None:
    summary = data.get("summary", data)
    return summary.get(fraction)


def row_from_summary(
    name: str,
    family: str,
    formulation: str,
    data: dict,
    fraction: str,
) -> dict | None:
    item = fetch_summary(data, fraction)
    if item is None:
        return None
    row = {
        "name": name,
        "family": family,
        "formulation": formulation,
        "fraction": fraction,
        "scratch_sequence_accuracy_mean": item.get("scratch_sequence_accuracy_mean"),
        "pretrained_sequence_accuracy_mean": item.get("pretrained_sequence_accuracy_mean"),
        "scratch_window_accuracy_mean": item.get("scratch_window_accuracy_mean"),
        "pretrained_window_accuracy_mean": item.get("pretrained_window_accuracy_mean"),
    }
    for key in ("avg_feature_dim", "avg_frame_dim", "avg_finetune_train_windows"):
        if key in item:
            row[key] = item[key]
    return row


def error_counter(path: str) -> list[list[object]]:
    data = load_json(path)
    counts: dict[tuple[str, str], int] = {}
    for run in data.get("runs", []):
        for err in run.get("pretrained_errors", []):
            key = (err["target"], err["prediction"])
            counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[target, pred, count] for (target, pred), count in items]


def main() -> None:
    artifacts = {
        "state_only": load_json("experiments/generated/symbolic_pretrain_state.json"),
        "temporal_main": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05.json"),
        "temporal_duration": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_duration.json"),
        "temporal_event": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event.json"),
        "temporal_event_w01": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w01.json"),
        "temporal_event_w005": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w005.json"),
        "temporal_coord": load_json("experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord.json"),
        "temporal_seq": load_json("experiments/generated/temporal_hl_sequence_pre168_84_v1.json"),
        "temporal_seq_duration": load_json("experiments/generated/temporal_hl_sequence_pre168_84_duration_v1.json"),
        "joint_token32": load_json("experiments/generated/joint_token_baseline_pre186_96_v1.json"),
        "joint_token64": load_json("experiments/generated/joint_token_baseline_pre186_96_tok64_v1.json"),
    }

    rows: list[dict] = []
    rows.extend(
        filter(
            None,
            [
                row_from_summary("symbolic_state_only", "symbolic_window_mlp", "state", artifacts["state_only"], "1.0"),
                row_from_summary("symbolic_state_transition", "symbolic_window_mlp", "state+transition", artifacts["temporal_main"], "1.0"),
                row_from_summary("symbolic_state_transition_duration", "symbolic_window_mlp", "state+transition+duration", artifacts["temporal_duration"], "1.0"),
                row_from_summary("symbolic_state_transition_duration", "symbolic_window_mlp", "state+transition+duration", artifacts["temporal_duration"], "0.5"),
                row_from_summary("symbolic_state_transition_event_w03", "symbolic_window_mlp", "state+transition+event[w=0.3]", artifacts["temporal_event"], "1.0"),
                row_from_summary("symbolic_state_transition_event_w03", "symbolic_window_mlp", "state+transition+event[w=0.3]", artifacts["temporal_event"], "0.5"),
                row_from_summary("symbolic_state_transition_event_w01", "symbolic_window_mlp", "state+transition+event[w=0.1]", artifacts["temporal_event_w01"], "1.0"),
                row_from_summary("symbolic_state_transition_event_w005", "symbolic_window_mlp", "state+transition+event[w=0.05]", artifacts["temporal_event_w005"], "1.0"),
                row_from_summary("symbolic_state_transition_coord", "symbolic_window_mlp", "state+transition+coordination", artifacts["temporal_coord"], "1.0"),
                row_from_summary("temporal_sequence_state_transition", "temporal_sequence_encoder", "state+transition", artifacts["temporal_seq"], "1.0"),
                row_from_summary("temporal_sequence_state_transition", "temporal_sequence_encoder", "state+transition", artifacts["temporal_seq"], "0.5"),
                row_from_summary("temporal_sequence_state_transition_duration", "temporal_sequence_encoder", "state+transition+duration", artifacts["temporal_seq_duration"], "1.0"),
                row_from_summary("temporal_sequence_state_transition_duration", "temporal_sequence_encoder", "state+transition+duration", artifacts["temporal_seq_duration"], "0.5"),
                row_from_summary("joint_token32", "joint_token_control", "learned_token_32", artifacts["joint_token32"], "1.0"),
                row_from_summary("joint_token32", "joint_token_control", "learned_token_32", artifacts["joint_token32"], "0.5"),
                row_from_summary("joint_token64", "joint_token_control", "learned_token_64", artifacts["joint_token64"], "1.0"),
                row_from_summary("joint_token64", "joint_token_control", "learned_token_64", artifacts["joint_token64"], "0.5"),
            ],
        )
    )

    best_pretrained = max(
        rows,
        key=lambda item: (
            float(item["pretrained_sequence_accuracy_mean"] or -1.0),
            float(item["pretrained_window_accuracy_mean"] or -1.0),
        ),
    )

    report = {
        "best_pretrained_sequence_model": best_pretrained,
        "rows": rows,
        "sequence_encoder_error_shift": {
            "state_transition": error_counter("experiments/generated/temporal_hl_sequence_pre168_84_analysis_v1.json"),
            "state_transition_duration": error_counter("experiments/generated/temporal_hl_sequence_pre168_84_duration_analysis_v1.json"),
        },
        "takeaways": [
            "The strongest symbolic mainline remains the pretrain-only normalized state+transition formulation.",
            "Adding handcrafted duration features hurts the symbolic window-MLP branch relative to state+transition.",
            "Exact-state event augmentation is a negative control in its current form; reducing event weight helps but still does not beat the base state+transition mainline.",
            "The current coordination / cross-hand relation augmentation is stronger than exact-state event augmentation, but still remains below the unchanged state+transition mainline.",
            "Adding duration to the temporal sequence encoder does not improve the pretrained mean at fraction 1.0 and hurts at fraction 0.5.",
            "Learned joint-token controls are substantially weaker than the strongest symbolic state+transition mainline.",
            "Current evidence supports transition as the effective temporal factor; duration is not yet a reliable additive gain.",
        ],
    }

    out_json = ROOT / "experiments/generated/temporal_representation_ablation_report.json"
    out_md = ROOT / "experiments/generated/summary_tables/temporal_representation_ablation_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Temporal Representation Ablation Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Best pretrained sequence result",
        "",
        f"- `{best_pretrained['name']}` at fraction `{best_pretrained['fraction']}`",
        f"- pretrained sequence accuracy mean: `{best_pretrained['pretrained_sequence_accuracy_mean']:.4f}`",
        f"- pretrained window accuracy mean: `{best_pretrained['pretrained_window_accuracy_mean']:.4f}`",
        "",
        "## Comparison",
        "",
        "| name | family | formulation | fraction | pretrained seq | scratch seq | pretrained win | scratch win |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (item["family"], item["formulation"], item["fraction"]),
    ):
        lines.append(
            f"| {row['name']} | {row['family']} | {row['formulation']} | {row['fraction']} "
            f"| {row['pretrained_sequence_accuracy_mean']:.4f} | {row['scratch_sequence_accuracy_mean']:.4f} "
            f"| {row['pretrained_window_accuracy_mean']:.4f} | {row['scratch_window_accuracy_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
        ]
    )
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Sequence encoder error shift",
            "",
            "- `state+transition` pretrained recurring errors:",
        ]
    )
    for target, pred, count in report["sequence_encoder_error_shift"]["state_transition"]:
        lines.append(f"  - `{target} -> {pred}`: `{count}`")
    lines.append("- `state+transition+duration` pretrained recurring errors:")
    for target, pred, count in report["sequence_encoder_error_shift"]["state_transition_duration"]:
        lines.append(f"  - `{target} -> {pred}`: `{count}`")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
