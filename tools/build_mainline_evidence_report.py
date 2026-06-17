#!/usr/bin/env python3
"""Build a consolidated evidence report for the strongest temporal-HL mainline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, headers: list[str], rows: list[list[object]], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pick_row(rows, representation: str, fraction: float, protocol: str):
    for row in rows:
        if (
            row["representation"] == representation
            and abs(float(row["fraction"]) - fraction) < 1e-9
            and row["protocol"] == protocol
        ):
            return row
    raise KeyError((representation, fraction, protocol))


def flatten_control_summaries():
    joint_token = load_json(GEN / "joint_token_control_summary.json")
    refined_phase = load_json(GEN / "refined_phase_sequence_summary.json")
    temporal_hl_seq = load_json(GEN / "temporal_hl_sequence_summary.json")
    rows = []

    for fraction in (0.5, 1.0):
        frac_key = str(fraction)
        rows.append(
            {
                "control": "joint_sequence_best",
                "fraction": fraction,
                "pretrained_sequence_accuracy": joint_token[f"joint_sequence_fraction_{frac_key}"][
                    "pretrained_sequence_accuracy_mean"
                ],
                "scratch_sequence_accuracy": joint_token[f"joint_sequence_fraction_{frac_key}"][
                    "scratch_sequence_accuracy_mean"
                ],
                "source": str(GEN / "joint_token_control_summary.json"),
            }
        )
        for codebook in ("joint_token32", "joint_token64"):
            rows.append(
                {
                    "control": codebook,
                    "fraction": fraction,
                    "pretrained_sequence_accuracy": joint_token[codebook][frac_key][
                        "pretrained_sequence_accuracy_mean"
                    ],
                    "scratch_sequence_accuracy": joint_token[codebook][frac_key][
                        "scratch_sequence_accuracy_mean"
                    ],
                    "source": str(GEN / "joint_token_control_summary.json"),
                }
            )

        if fraction == 0.5:
            refined_key = "0.5"
        else:
            refined_key = "1.0"

        rows.append(
            {
                "control": "refined_phase_sequence_best",
                "fraction": fraction,
                "pretrained_sequence_accuracy": refined_phase["pre186_96"][refined_key][
                    "pretrained_sequence_accuracy_mean"
                ],
                "scratch_sequence_accuracy": refined_phase["pre186_96"][refined_key][
                    "scratch_sequence_accuracy_mean"
                ],
                "source": str(GEN / "refined_phase_sequence_summary.json"),
            }
        )
        rows.append(
            {
                "control": "temporal_hl_sequence_best",
                "fraction": fraction,
                "pretrained_sequence_accuracy": temporal_hl_seq["pre168_84"][refined_key][
                    "pretrained_sequence_accuracy_mean"
                ],
                "scratch_sequence_accuracy": temporal_hl_seq["pre168_84"][refined_key][
                    "scratch_sequence_accuracy_mean"
                ],
                "source": str(GEN / "temporal_hl_sequence_summary.json"),
            }
        )
    return rows


def best_control_rows(control_rows: list[dict[str, object]], fraction: float):
    rows = [row for row in control_rows if abs(float(row["fraction"]) - fraction) < 1e-9]
    rows.sort(key=lambda row: (-float(row["pretrained_sequence_accuracy"]), row["control"]))
    return rows


def slice_gap_rows(slice_compare_path: Path, top_k: int = 12):
    payload = load_json(slice_compare_path)
    return payload["pretrained_diff"][:top_k]


def sequence_slice_table(analysis: dict[str, object], mode: str = "pretrained"):
    totals = defaultdict(int)
    correct = defaultdict(int)
    seq_entries = analysis["sequence_consistency"]
    key_name = f"{mode}_predictions"
    for seq_name, item in seq_entries.items():
        groups = {"all"}
        if "Interaction" in seq_name:
            groups.add("interaction")
        if "No_Interaction" in seq_name:
            groups.add("no_interaction")
        if "LT_" in seq_name or "Lt_" in seq_name:
            groups.add("left")
        if "RT_" in seq_name or "Rt_" in seq_name:
            groups.add("right")
        if "Wrist_ROM" in seq_name:
            groups.add("wrist_rom")
        if "Occlusion" in seq_name:
            groups.add("occlusion")
        if "Finger_Occlusions" in seq_name:
            groups.add("finger_occlusion")
        if "No_Occlusion" in seq_name:
            groups.add("no_occlusion")
        if "Touching" in seq_name:
            groups.add("touching")
        for pred in item[key_name]:
            for group in groups:
                totals[group] += 1
                if pred["correct"]:
                    correct[group] += 1
    rows = []
    for group in sorted(totals):
        rows.append(
            {
                "slice": group,
                "total": totals[group],
                "correct": correct[group],
                "accuracy": correct[group] / max(totals[group], 1),
            }
        )
    rows.sort(key=lambda row: row["slice"])
    return rows


def compare_errors(old_errors: list[dict[str, object]], new_errors: list[dict[str, object]], limit: int = 10):
    old_map = {(row["target"], row["prediction"]): row["count"] for row in old_errors}
    new_map = {(row["target"], row["prediction"]): row["count"] for row in new_errors}
    keys = sorted(set(old_map) | set(new_map))
    rows = []
    for key in keys:
        old_count = old_map.get(key, 0)
        new_count = new_map.get(key, 0)
        rows.append(
            {
                "target": key[0],
                "prediction": key[1],
                "old_count": old_count,
                "new_count": new_count,
                "delta": new_count - old_count,
            }
        )
    rows.sort(key=lambda row: (row["delta"], -row["old_count"], row["target"], row["prediction"]))
    return rows[:limit]


def build_retrieval_rows():
    event = load_json(GEN / "sequence_symbolic_retrieval_event_summary.json")
    segdur = load_json(GEN / "sequence_symbolic_retrieval_segdur_event_summary.json")
    phase = load_json(GEN / "sequence_symbolic_retrieval_phase_event_summary.json")
    refined = load_json(GEN / "sequence_symbolic_retrieval_refined_phase_event_summary.json")
    wanted = [
        ("event", "state_event_dtw", event["state_event_dtw"]),
        ("event", "temporal_event_dtw", event["temporal_event_dtw"]),
        ("event", "temporal_persist_event_dtw", event["temporal_persist_event_dtw"]),
        ("segdur", "temporal_segdur_event_dtw", segdur["temporal_segdur_event_dtw"]),
        ("phase", "temporal_phase_event_dtw", phase["temporal_phase_event_dtw"]),
        ("refined_phase", "temporal_refined_phase_segdur_event_dtw", refined["temporal_refined_phase_segdur_event_dtw"]),
    ]
    rows = []
    for family, name, metrics in wanted:
        rows.append(
            {
                "family": family,
                "representation": name,
                "top1_accuracy": metrics["top1_accuracy"],
                "avg_similarity_margin": metrics["avg_similarity_margin"],
                "avg_num_gallery_events": metrics["avg_num_gallery_events"],
                "avg_num_query_events": metrics["avg_num_query_events"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=GEN / "mainline_evidence_report.json",
    )
    args = parser.parse_args()

    protocol_matrix = load_json(GEN / "protocol_matrix.json")
    rows = protocol_matrix["rows"]
    symbolic_old = load_json(GEN / "symbolic_pretrain_temporal_boost_rom05_analysis_seedfixed.json")
    symbolic_new = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json")
    symbolic_new_05 = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json")
    symbolic_old_05 = load_json(GEN / "symbolic_pretrain_temporal_boost_rom05_fraction05_analysis.json")
    symbolic_hybrid = load_json(
        GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_changeonly_analysis.json"
    )
    symbolic_hybrid_05 = load_json(
        GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_changeonly_analysis.json"
    )

    control_rows = flatten_control_summaries()
    symbolic_mainline = {
        1.0: pick_row(rows, "symbolic", 1.0, "pretrain_only_norm_168_84"),
        0.5: pick_row(rows, "symbolic", 0.5, "pretrain_only_norm_168_84"),
    }
    symbolic_default = {
        1.0: pick_row(rows, "symbolic", 1.0, "default"),
        0.5: pick_row(rows, "symbolic", 0.5, "default"),
    }
    hybrid_seq_mean = {
        1.0: sum(row["pretrained_sequence_accuracy"] for row in symbolic_hybrid["runs"])
        / max(len(symbolic_hybrid["runs"]), 1),
        0.5: sum(row["pretrained_sequence_accuracy"] for row in symbolic_hybrid_05["runs"])
        / max(len(symbolic_hybrid_05["runs"]), 1),
    }
    hybrid_scratch_mean = {
        1.0: sum(row["scratch_sequence_accuracy"] for row in symbolic_hybrid["runs"])
        / max(len(symbolic_hybrid["runs"]), 1),
        0.5: sum(row["scratch_sequence_accuracy"] for row in symbolic_hybrid_05["runs"])
        / max(len(symbolic_hybrid_05["runs"]), 1),
    }

    against_controls = []
    for fraction in (0.5, 1.0):
        baseline = symbolic_mainline[fraction]
        against_controls.append(
            {
                "control": "symbolic_default",
                "fraction": fraction,
                "control_pretrained_sequence_accuracy": symbolic_default[fraction]["pretrained_seq"],
                "mainline_pretrained_sequence_accuracy": baseline["pretrained_seq"],
                "gap_vs_control": baseline["pretrained_seq"] - symbolic_default[fraction]["pretrained_seq"],
                "source": symbolic_default[fraction]["path"],
            }
        )
        for control in best_control_rows(control_rows, fraction):
            against_controls.append(
                {
                    "control": control["control"],
                    "fraction": fraction,
                    "control_pretrained_sequence_accuracy": control["pretrained_sequence_accuracy"],
                    "mainline_pretrained_sequence_accuracy": baseline["pretrained_seq"],
                    "gap_vs_control": baseline["pretrained_seq"] - control["pretrained_sequence_accuracy"],
                    "source": control["source"],
                }
            )

    payload = {
        "artifacts": {
            "protocol_matrix": str(GEN / "protocol_matrix.json"),
            "symbolic_mainline_fraction_1_analysis": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json"),
            "symbolic_mainline_fraction_05_analysis": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json"),
            "symbolic_hybrid_fraction_1_analysis": str(
                GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_changeonly_analysis.json"
            ),
            "symbolic_hybrid_fraction_05_analysis": str(
                GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_changeonly_analysis.json"
            ),
            "symbolic_default_fraction_1_analysis": str(GEN / "symbolic_pretrain_temporal_boost_rom05_analysis_seedfixed.json"),
            "symbolic_default_fraction_05_analysis": str(GEN / "symbolic_pretrain_temporal_boost_rom05_fraction05_analysis.json"),
            "slice_compare_fraction_1": str(GEN / "symbolic_slice_compare_old_vs_new.json"),
            "slice_compare_fraction_05": str(GEN / "symbolic_slice_compare_old_vs_new_fraction05.json"),
            "joint_token_control_summary": str(GEN / "joint_token_control_summary.json"),
            "refined_phase_sequence_summary": str(GEN / "refined_phase_sequence_summary.json"),
            "temporal_hl_sequence_summary": str(GEN / "temporal_hl_sequence_summary.json"),
            "retrieval_event_summary": str(GEN / "sequence_symbolic_retrieval_event_summary.json"),
            "retrieval_refined_phase_summary": str(GEN / "sequence_symbolic_retrieval_refined_phase_event_summary.json"),
        },
        "mainline": {
            "fraction_0.5": symbolic_mainline[0.5],
            "fraction_1.0": symbolic_mainline[1.0],
        },
        "hybrid_mainline": {
            "fraction_0.5": {
                "pretrained_seq_mean": hybrid_seq_mean[0.5],
                "scratch_seq_mean": hybrid_scratch_mean[0.5],
                "patch": symbolic_hybrid_05["hybrid_patch"],
            },
            "fraction_1.0": {
                "pretrained_seq_mean": hybrid_seq_mean[1.0],
                "scratch_seq_mean": hybrid_scratch_mean[1.0],
                "patch": symbolic_hybrid["hybrid_patch"],
            },
        },
        "control_comparison_rows": against_controls,
        "control_ranking": {
            "fraction_0.5": best_control_rows(control_rows, 0.5),
            "fraction_1.0": best_control_rows(control_rows, 1.0),
        },
        "slice_gains": {
            "fraction_0.5": slice_gap_rows(GEN / "symbolic_slice_compare_old_vs_new_fraction05.json"),
            "fraction_1.0": slice_gap_rows(GEN / "symbolic_slice_compare_old_vs_new.json"),
        },
        "mainline_sequence_slices": {
            "fraction_0.5": sequence_slice_table(symbolic_new_05),
            "fraction_1.0": sequence_slice_table(symbolic_new),
        },
        "hybrid_sequence_slices": {
            "fraction_0.5": sequence_slice_table(symbolic_hybrid_05),
            "fraction_1.0": sequence_slice_table(symbolic_hybrid),
        },
        "default_sequence_slices": {
            "fraction_0.5": sequence_slice_table(symbolic_old_05),
            "fraction_1.0": sequence_slice_table(symbolic_old),
        },
        "error_repair_frontier": {
            "fraction_0.5": compare_errors(
                symbolic_old_05["pretrained_error_counts"],
                symbolic_new_05["pretrained_error_counts"],
            ),
            "fraction_1.0": compare_errors(
                symbolic_old["pretrained_error_counts"],
                symbolic_new["pretrained_error_counts"],
            ),
        },
        "hybrid_vs_mainline_error_repair_frontier": {
            "fraction_0.5": compare_errors(
                symbolic_new_05["pretrained_error_counts"],
                symbolic_hybrid_05["pretrained_error_counts"],
            ),
            "fraction_1.0": compare_errors(
                symbolic_new["pretrained_error_counts"],
                symbolic_hybrid["pretrained_error_counts"],
            ),
        },
        "retrieval_evidence": build_retrieval_rows(),
        "plain_summary": {
            "fraction_0.5_mainline_vs_default_gain": symbolic_mainline[0.5]["pretrained_seq"]
            - symbolic_default[0.5]["pretrained_seq"],
            "fraction_1.0_mainline_vs_default_gain": symbolic_mainline[1.0]["pretrained_seq"]
            - symbolic_default[1.0]["pretrained_seq"],
            "fraction_0.5_mainline_vs_best_joint_sequence_gap": symbolic_mainline[0.5]["pretrained_seq"]
            - best_control_rows(control_rows, 0.5)[0]["pretrained_sequence_accuracy"],
            "fraction_1.0_mainline_vs_best_joint_sequence_gap": symbolic_mainline[1.0]["pretrained_seq"]
            - best_control_rows(control_rows, 1.0)[0]["pretrained_sequence_accuracy"],
            "fraction_0.5_hybrid_vs_mainline_gain": hybrid_seq_mean[0.5] - symbolic_mainline[0.5]["pretrained_seq"],
            "fraction_1.0_hybrid_vs_mainline_gain": hybrid_seq_mean[1.0] - symbolic_mainline[1.0]["pretrained_seq"],
        },
    }

    write_json(args.output, payload)

    summary_dir = GEN / "summary_tables"
    control_csv_rows = []
    for row in against_controls:
        control_csv_rows.append(
            {
                "fraction": row["fraction"],
                "control": row["control"],
                "control_pretrained_seq": fmt(float(row["control_pretrained_sequence_accuracy"])),
                "mainline_pretrained_seq": fmt(float(row["mainline_pretrained_sequence_accuracy"])),
                "gap_vs_control": fmt(float(row["gap_vs_control"])),
            }
        )
    write_csv(
        summary_dir / "mainline_vs_controls.csv",
        control_csv_rows,
        ["fraction", "control", "control_pretrained_seq", "mainline_pretrained_seq", "gap_vs_control"],
    )
    write_md_table(
        summary_dir / "mainline_vs_controls.md",
        ["fraction", "control", "control_pretrained_seq", "mainline_pretrained_seq", "gap_vs_control"],
        [
            [
                row["fraction"],
                row["control"],
                row["control_pretrained_seq"],
                row["mainline_pretrained_seq"],
                row["gap_vs_control"],
            ]
            for row in control_csv_rows
        ],
        "Mainline vs Controls",
    )

    retrieval_rows = payload["retrieval_evidence"]
    write_csv(
        summary_dir / "retrieval_evidence.csv",
        retrieval_rows,
        ["family", "representation", "top1_accuracy", "avg_similarity_margin", "avg_num_gallery_events", "avg_num_query_events"],
    )
    write_md_table(
        summary_dir / "retrieval_evidence.md",
        ["family", "representation", "top1_accuracy", "avg_similarity_margin", "avg_num_gallery_events", "avg_num_query_events"],
        [
            [
                row["family"],
                row["representation"],
                fmt(float(row["top1_accuracy"])),
                fmt(float(row["avg_similarity_margin"])),
                "None" if row["avg_num_gallery_events"] is None else fmt(float(row["avg_num_gallery_events"])),
                "None" if row["avg_num_query_events"] is None else fmt(float(row["avg_num_query_events"])),
            ]
            for row in retrieval_rows
        ],
        "Retrieval Evidence",
    )

    slice_rows = []
    for fraction_key, rows_ in payload["slice_gains"].items():
        for row in rows_:
            slice_rows.append(
                {
                    "fraction": fraction_key,
                    "slice": row["slice"],
                    "old_accuracy": fmt(float(row["old_accuracy"])),
                    "new_accuracy": fmt(float(row["new_accuracy"])),
                    "delta": fmt(float(row["delta"])),
                }
            )
    write_csv(
        summary_dir / "mainline_slice_gains.csv",
        slice_rows,
        ["fraction", "slice", "old_accuracy", "new_accuracy", "delta"],
    )
    write_md_table(
        summary_dir / "mainline_slice_gains.md",
        ["fraction", "slice", "old_accuracy", "new_accuracy", "delta"],
        [
            [row["fraction"], row["slice"], row["old_accuracy"], row["new_accuracy"], row["delta"]]
            for row in slice_rows
        ],
        "Mainline Slice Gains",
    )

    error_rows = []
    for fraction_key, rows_ in payload["error_repair_frontier"].items():
        for row in rows_:
            error_rows.append(
                {
                    "fraction": fraction_key,
                    "target": row["target"],
                    "prediction": row["prediction"],
                    "old_count": row["old_count"],
                    "new_count": row["new_count"],
                    "delta": row["delta"],
                }
            )
    write_csv(
        summary_dir / "mainline_error_repair_frontier.csv",
        error_rows,
        ["fraction", "target", "prediction", "old_count", "new_count", "delta"],
    )
    write_md_table(
        summary_dir / "mainline_error_repair_frontier.md",
        ["fraction", "target", "prediction", "old_count", "new_count", "delta"],
        [
            [row["fraction"], row["target"], row["prediction"], row["old_count"], row["new_count"], row["delta"]]
            for row in error_rows
        ],
        "Mainline Error Repair Frontier",
    )

    hybrid_rows = []
    for fraction in (0.5, 1.0):
        fraction_key = "fraction_0.5" if fraction == 0.5 else "fraction_1.0"
        hybrid_rows.append(
            {
                "fraction": fraction,
                "mainline_pretrained_seq": fmt(float(symbolic_mainline[fraction]["pretrained_seq"])),
                "hybrid_pretrained_seq": fmt(float(hybrid_seq_mean[fraction])),
                "hybrid_minus_mainline": fmt(float(hybrid_seq_mean[fraction] - symbolic_mainline[fraction]["pretrained_seq"])),
                "num_patch_applied": payload["hybrid_mainline"][fraction_key]["patch"]["num_applied"],
                "num_patch_improved": payload["hybrid_mainline"][fraction_key]["patch"]["num_improved"],
                "num_patch_harmed": payload["hybrid_mainline"][fraction_key]["patch"]["num_harmed"],
            }
        )
    write_csv(
        summary_dir / "hybrid_vs_mainline.csv",
        hybrid_rows,
        [
            "fraction",
            "mainline_pretrained_seq",
            "hybrid_pretrained_seq",
            "hybrid_minus_mainline",
            "num_patch_applied",
            "num_patch_improved",
            "num_patch_harmed",
        ],
    )
    write_md_table(
        summary_dir / "hybrid_vs_mainline.md",
        [
            "fraction",
            "mainline_pretrained_seq",
            "hybrid_pretrained_seq",
            "hybrid_minus_mainline",
            "num_patch_applied",
            "num_patch_improved",
            "num_patch_harmed",
        ],
        [
            [
                row["fraction"],
                row["mainline_pretrained_seq"],
                row["hybrid_pretrained_seq"],
                row["hybrid_minus_mainline"],
                row["num_patch_applied"],
                row["num_patch_improved"],
                row["num_patch_harmed"],
            ]
            for row in hybrid_rows
        ],
        "Hybrid vs Mainline",
    )

    print(f"output: {args.output}")
    print(
        "mainline gains vs default:",
        fmt(payload["plain_summary"]["fraction_0.5_mainline_vs_default_gain"]),
        fmt(payload["plain_summary"]["fraction_1.0_mainline_vs_default_gain"]),
    )
    print(
        "mainline gaps vs best non-symbolic control:",
        fmt(payload["plain_summary"]["fraction_0.5_mainline_vs_best_joint_sequence_gap"]),
        fmt(payload["plain_summary"]["fraction_1.0_mainline_vs_best_joint_sequence_gap"]),
    )


if __name__ == "__main__":
    main()
