#!/usr/bin/env python3
"""Build a small casebook of hard-case symbolic event diffs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tools.eval_sequence_symbolic_retrieval import dtw_similarity, sequence_event_sets


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def flatten_event_counts(events):
    counter = Counter()
    for event in events:
        counter.update(event)
    return counter


def informative_token(token: str) -> bool:
    return (
        ":trans:" in token
        or ":motion:" in token
        or token.startswith("interaction=")
        or token.startswith("event::duration::")
    )


def top_tokens(counter: Counter, limit: int = 8):
    return [{"token": token, "count": count} for token, count in counter.most_common(limit)]


def main():
    intrinsic = load_json(GEN / "symbolic_representation_intrinsic_val_test_plus.json")
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    align = load_json(GEN / "hardcase_event_alignment_report.json")

    state_rows = {row["seq_name"]: row for row in next(r for r in intrinsic["results"] if r["name"] == "state_event")["rows"]}
    temporal_rows = {row["seq_name"]: row for row in next(r for r in intrinsic["results"] if r["name"] == "temporal_event")["rows"]}

    val_map = {seq["seq_name"]: seq for seq in val_data["sequences"]}
    test_map = {seq["seq_name"]: seq for seq in test_data["sequences"]}

    chosen = [
        "ROM04_RT_Occlusion",
        "ROM04_LT_Occlusion",
        "ROM07_RT_Finger_Occlusions",
        "ROM05_RT_Wrist_ROM",
        "ROM02_Interaction_2_Hand",
    ]

    cases = []
    for seq_name in chosen:
        if seq_name not in state_rows or seq_name not in temporal_rows or seq_name not in test_map:
            continue
        srow = state_rows[seq_name]
        trow = temporal_rows[seq_name]
        query = test_map[seq_name]
        correct_name = trow["best_correct_seq"]
        state_wrong_name = srow["best_wrong_seq"]
        temporal_wrong_name = trow["best_wrong_seq"]
        if correct_name not in val_map or state_wrong_name not in val_map or temporal_wrong_name not in val_map:
            continue

        correct = val_map[correct_name]
        state_wrong = val_map[state_wrong_name]
        temporal_wrong = val_map[temporal_wrong_name]

        q_state = sequence_event_sets(query, "state")
        q_temp = sequence_event_sets(query, "temporal")
        c_state = sequence_event_sets(correct, "state")
        c_temp = sequence_event_sets(correct, "temporal")
        sw_state = sequence_event_sets(state_wrong, "state")
        tw_temp = sequence_event_sets(temporal_wrong, "temporal")

        q_temp_counts = flatten_event_counts(q_temp)
        c_temp_counts = flatten_event_counts(c_temp)
        tw_temp_counts = flatten_event_counts(tw_temp)

        supportive = Counter()
        conflicting = Counter()
        for token, count in q_temp_counts.items():
            if not informative_token(token):
                continue
            if c_temp_counts[token] > 0 and tw_temp_counts[token] == 0:
                supportive[token] = count
            if tw_temp_counts[token] > 0 and c_temp_counts[token] == 0:
                conflicting[token] = count

        cases.append(
            {
                "seq_name": seq_name,
                "correct_seq": correct_name,
                "state_wrong_seq": state_wrong_name,
                "temporal_wrong_seq": temporal_wrong_name,
                "wrong_neighbor_changed": state_wrong_name != temporal_wrong_name,
                "state": {
                    "query_num_events": len(q_state),
                    "correct_num_events": len(c_state),
                    "wrong_num_events": len(sw_state),
                    "sim_to_correct": dtw_similarity(q_state, c_state),
                    "sim_to_wrong": dtw_similarity(q_state, sw_state),
                    "margin": srow["positive_margin"],
                },
                "temporal": {
                    "query_num_events": len(q_temp),
                    "correct_num_events": len(c_temp),
                    "wrong_num_events": len(tw_temp),
                    "sim_to_correct": dtw_similarity(q_temp, c_temp),
                    "sim_to_wrong": dtw_similarity(q_temp, tw_temp),
                    "margin": trow["positive_margin"],
                },
                "query_temporal_profile": top_tokens(
                    Counter(
                        {
                            token: count
                            for token, count in q_temp_counts.items()
                            if informative_token(token)
                        }
                    )
                ),
                "temporal_supportive_tokens": top_tokens(supportive),
                "temporal_conflicting_tokens": top_tokens(conflicting),
            }
        )

    report = {
        "artifacts": {
            "intrinsic_report": str(GEN / "symbolic_representation_intrinsic_val_test_plus.json"),
            "hardcase_alignment_report": str(GEN / "hardcase_event_alignment_report.json"),
            "gallery_json": str(GEN / "temporal_hl_val.json"),
            "query_json": str(GEN / "temporal_hl_test.json"),
        },
        "chosen_cases": chosen,
        "cases": cases,
        "takeaways": [
            "The casebook makes the event-level differences concrete by comparing each query against the correct gallery neighbor and the nearest wrong neighbors under state and temporal representations.",
            "When temporal changes the nearest wrong neighbor, it usually corresponds to a shift toward a harder but more structurally related competitor rather than a generic no-occlusion baseline.",
            "The query-side temporal profiles are dominated by motion, interaction, and event-duration tokens, which are exactly the channels absent from frame-only symbolic state matching.",
            "Wrist-ROM remains mixed: the wrong neighbor changes, but the temporal margin does not necessarily increase.",
        ],
    }

    out_json = GEN / "hardcase_event_casebook.json"
    out_md = SUM / "hardcase_event_casebook.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Hard-Case Event Casebook",
        "",
        "This is an experiment memo, not paper text.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['seq_name']}",
                "",
                f"- `correct_seq`: `{case['correct_seq']}`",
                f"- `state_wrong_seq`: `{case['state_wrong_seq']}`",
                f"- `temporal_wrong_seq`: `{case['temporal_wrong_seq']}`",
                f"- `wrong_neighbor_changed`: `{case['wrong_neighbor_changed']}`",
                "",
                "| mode | query events | correct events | wrong events | sim to correct | sim to wrong | margin |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| state | {case['state']['query_num_events']} | {case['state']['correct_num_events']} | {case['state']['wrong_num_events']} | {fmt(case['state']['sim_to_correct'])} | {fmt(case['state']['sim_to_wrong'])} | {fmt(case['state']['margin'])} |",
                f"| temporal | {case['temporal']['query_num_events']} | {case['temporal']['correct_num_events']} | {case['temporal']['wrong_num_events']} | {fmt(case['temporal']['sim_to_correct'])} | {fmt(case['temporal']['sim_to_wrong'])} | {fmt(case['temporal']['margin'])} |",
                "",
                "Query temporal token profile",
                "",
            ]
        )
        if case["query_temporal_profile"]:
            for item in case["query_temporal_profile"]:
                lines.append(f"- `{item['token']}` x `{item['count']}`")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Supportive temporal tokens",
                "",
            ]
        )
        if case["temporal_supportive_tokens"]:
            for item in case["temporal_supportive_tokens"]:
                lines.append(f"- `{item['token']}` x `{item['count']}`")
        else:
            lines.append("- none")
        lines.extend(["", "Conflicting temporal tokens", ""])
        if case["temporal_conflicting_tokens"]:
            for item in case["temporal_conflicting_tokens"]:
                lines.append(f"- `{item['token']}` x `{item['count']}`")
        else:
            lines.append("- none")
        lines.append("")

    lines.extend(["## Takeaways", ""])
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
