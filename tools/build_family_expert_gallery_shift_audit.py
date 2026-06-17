#!/usr/bin/env python3
"""Audit family-expert stability under degraded gallery variants."""

from __future__ import annotations

import json
from pathlib import Path

from tools.eval_occlusion_late_correction import FAMILY_LABELS, build_query_events, retrieve_family_label
from tools.eval_symbolic_family_expert import (
    build_dataset,
    evaluate_threshold,
    load_family_rows,
    parse_thresholds,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def sequence_events_from_data(data: dict[str, object]):
    from tools.eval_occlusion_late_correction import build_gallery

    return build_gallery(data, set(FAMILY_LABELS))


def transform_frames(frames: list[dict[str, object]], variant: str):
    n = len(frames)
    if n <= 2 or variant == "full":
        return frames
    if variant == "first_half":
        keep = max(2, n // 2)
        return frames[:keep]
    if variant == "second_half":
        keep = max(2, n // 2)
        return frames[-keep:]
    if variant == "stride2":
        out = frames[::2]
        return out if len(out) >= 2 else frames
    if variant == "center_half":
        keep = max(2, n // 2)
        start = max(0, (n - keep) // 2)
        end = min(n, start + keep)
        out = frames[start:end]
        return out if len(out) >= 2 else frames
    raise ValueError(f"Unknown variant: {variant}")


def transform_gallery_data(base: dict[str, object], variant: str):
    data = {
        "direction_codebook": base["direction_codebook"],
        "sequences": [],
    }
    for seq in base["sequences"]:
        new_seq = dict(seq)
        new_seq["frames"] = transform_frames(seq["frames"], variant)
        new_seq["num_frames"] = len(new_seq["frames"])
        data["sequences"].append(new_seq)
    return data


def evaluate_variant(base_rows, query_events, gallery_data, thresholds):
    gallery = sequence_events_from_data(gallery_data)
    dataset = build_dataset(base_rows, query_events, gallery)
    feature_order = sorted(dataset[0]["features"].keys())
    sweeps = [evaluate_threshold(dataset, feature_order, thr) for thr in thresholds]
    zero_harm = [row for row in sweeps if row["counts"]["num_harmed"] == 0]
    best = max(
        sweeps,
        key=lambda row: (
            row["family_accuracy"]["delta"],
            -row["counts"]["num_harmed"],
            row["counts"]["num_improved"],
            -row["threshold"],
        ),
    )
    return {
        "family_dataset_size": len(dataset),
        "feature_order": feature_order,
        "sweeps": sweeps,
        "zero_harm_thresholds": [row["threshold"] for row in zero_harm],
        "best_zero_harm": zero_harm[0] if zero_harm else None,
        "best_overall": best,
    }


def main():
    thresholds = parse_thresholds("0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    gallery_base = load_json(GEN / "temporal_hl_val.json")
    query_data = load_json(GEN / "temporal_hl_test.json")
    query_events = build_query_events(query_data, set(FAMILY_LABELS))

    fraction_inputs = {
        "0.25": [
            GEN / "symbolic_pretrain_grouped_concat_fraction025_seed0_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_fraction025_seed1_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_fraction025_seed2_familyvec_analysis.json",
        ],
        "0.5": [
            GEN / "symbolic_pretrain_grouped_concat_fraction05_seed0_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_fraction05_seed1_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_fraction05_seed2_familyvec_analysis.json",
        ],
        "1.0": [
            GEN / "symbolic_pretrain_grouped_concat_seed0_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_seed1_familyvec_analysis.json",
            GEN / "symbolic_pretrain_grouped_concat_seed2_familyvec_analysis.json",
        ],
    }
    variants = ["full", "first_half", "second_half", "center_half", "stride2"]

    rows = []
    for fraction, paths in fraction_inputs.items():
        base_rows = load_family_rows(paths)
        for variant in variants:
            gallery_variant = transform_gallery_data(gallery_base, variant)
            result = evaluate_variant(base_rows, query_events, gallery_variant, thresholds)
            best_zero = result["best_zero_harm"]
            best = best_zero if best_zero is not None else result["best_overall"]
            rows.append(
                {
                    "fraction": fraction,
                    "variant": variant,
                    "family_dataset_size": result["family_dataset_size"],
                    "zero_harm_thresholds": result["zero_harm_thresholds"],
                    "has_zero_harm": bool(result["zero_harm_thresholds"]),
                    "best_threshold": best["threshold"],
                    "family_old": best["family_accuracy"]["old"],
                    "family_new": best["family_accuracy"]["new"],
                    "family_delta": best["family_accuracy"]["delta"],
                    "num_improved": best["counts"]["num_improved"],
                    "num_harmed": best["counts"]["num_harmed"],
                }
            )

    payload = {
        "artifacts": {
            "gallery_base": str(GEN / "temporal_hl_val.json"),
            "query_json": str(GEN / "temporal_hl_test.json"),
            "fraction_025_analyses": [str(p) for p in fraction_inputs["0.25"]],
            "fraction_05_analyses": [str(p) for p in fraction_inputs["0.5"]],
            "fraction_10_analyses": [str(p) for p in fraction_inputs["1.0"]],
        },
        "variants": variants,
        "thresholds": thresholds,
        "rows": rows,
        "takeaways": [],
    }

    full_zero_harm = [
        row for row in rows
        if row["variant"] == "full" and row["has_zero_harm"]
    ]
    degraded_zero_harm = [
        row for row in rows
        if row["variant"] != "full" and row["has_zero_harm"]
    ]
    payload["takeaways"] = [
        f"Full gallery remains zero-harm for {len(full_zero_harm)} / 3 fractions under the shifted-audit script.",
        f"Degraded gallery variants retain at least one zero-harm threshold in {len(degraded_zero_harm)} / {len([r for r in rows if r['variant'] != 'full'])} fraction-variant settings.",
        "This audit tests whether the family-expert repair depends on having the complete gallery template sequence or still works when gallery templates are temporally degraded.",
    ]

    out_json = GEN / "grouped_concat_family_expert_gallery_shift_audit.json"
    out_md = SUM / "grouped_concat_family_expert_gallery_shift_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Grouped Concat + Family Expert Gallery-Shift Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| fraction | variant | zero-harm | zero-harm thresholds | best thr | family old | family new | delta | improved | harmed |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fraction']} | {row['variant']} | {row['has_zero_harm']} | {row['zero_harm_thresholds']} | "
            f"{row['best_threshold']:.1f} | {fmt(row['family_old'])} | {fmt(row['family_new'])} | "
            f"{fmt(row['family_delta'])} | {row['num_improved']} | {row['num_harmed']} |"
        )
    lines.extend(["", "## Takeaways", ""])
    for item in payload["takeaways"]:
        lines.append(f"- {item}")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
