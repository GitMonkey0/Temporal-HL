#!/usr/bin/env python3
"""Sequence-native symbolic retrieval for state-only vs temporal HL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.train_symbolic_classifier import canonical_label, overlap_labels


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def iter_sequences(dataset: dict[str, object], allowed_labels: set[str]):
    for sequence in dataset["sequences"]:
        label = canonical_label(sequence["seq_name"])
        if label in allowed_labels:
            yield sequence, label


def pack_list(values):
    if values is None:
        return "none"
    return "|".join(str(v) for v in values)


def frame_signature(
    frame: dict[str, object], mode: str, include_persistence: bool = False
) -> str:
    parts = [f"hand_type={frame.get('hand_type', 'unknown')}"]
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is None:
            parts.append(f"{hand_name}=none")
            continue
        parts.append(f"{hand_name}:state={pack_list(hand_record.get('token_labels'))}")
        if mode == "temporal":
            parts.append(
                f"{hand_name}:trans={pack_list(hand_record.get('transition_labels'))}"
            )
            parts.append(f"{hand_name}:motion={hand_record.get('hand_motion', 'unknown')}")
            if include_persistence:
                parts.append(
                    f"{hand_name}:state_persist={hand_record.get('state_persistence_label', 'missing')}"
                )
                parts.append(
                    f"{hand_name}:activity_persist={hand_record.get('activity_persistence_label', 'missing')}"
                )
    if mode == "temporal":
        parts.append(f"interaction={frame.get('interaction_motion', 'unknown')}")
        if include_persistence:
            parts.append(
                f"interaction_persist={frame.get('interaction_persistence_label', 'missing')}"
            )
            parts.append(
                "interaction_activity_persist="
                f"{frame.get('interaction_activity_persistence_label', 'missing')}"
            )
    return ";".join(parts)


def frame_token_set(
    frame: dict[str, object], mode: str, include_persistence: bool = False
) -> set[str]:
    tokens = {f"hand_type={frame.get('hand_type', 'unknown')}"}
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is None:
            tokens.add(f"{hand_name}=none")
            continue
        for token in hand_record.get("token_labels", []):
            tokens.add(f"{hand_name}:state:{token}")
        if mode == "temporal":
            for token in hand_record.get("transition_labels", []):
                tokens.add(f"{hand_name}:trans:{token}")
            tokens.add(f"{hand_name}:motion:{hand_record.get('hand_motion', 'unknown')}")
            if include_persistence:
                tokens.add(
                    f"{hand_name}:state_persist:{hand_record.get('state_persistence_label', 'missing')}"
                )
                tokens.add(
                    f"{hand_name}:activity_persist:{hand_record.get('activity_persistence_label', 'missing')}"
                )
    if mode == "temporal":
        tokens.add(f"interaction={frame.get('interaction_motion', 'unknown')}")
        if include_persistence:
            tokens.add(
                f"interaction_persist:{frame.get('interaction_persistence_label', 'missing')}"
            )
            tokens.add(
                "interaction_activity_persist:"
                f"{frame.get('interaction_activity_persistence_label', 'missing')}"
            )
    return tokens


def sequence_tokens(
    sequence: dict[str, object],
    mode: str,
    use_rle: bool,
    include_persistence: bool = False,
) -> list[str]:
    tokens = [
        frame_signature(frame, mode, include_persistence=include_persistence)
        for frame in sequence["frames"]
    ]
    if not use_rle:
        return tokens
    packed = []
    prev = None
    count = 0
    for token in tokens:
        if prev is None or token == prev:
            count += 1
        else:
            packed.append(f"{prev}#run={count}")
            count = 1
        prev = token
    if prev is not None:
        packed.append(f"{prev}#run={count}")
    return packed


def sequence_frame_sets(
    sequence: dict[str, object], mode: str, include_persistence: bool = False
) -> list[set[str]]:
    return [
        frame_token_set(frame, mode, include_persistence=include_persistence)
        for frame in sequence["frames"]
    ]


def segment_duration_bucket(num_frames: int) -> str:
    if num_frames <= 1:
        return "instant"
    if num_frames <= 3:
        return "short"
    if num_frames <= 7:
        return "medium"
    return "long"


def classify_hand_phase(hand_record: dict[str, object] | None) -> str:
    if hand_record is None:
        return "absent"
    transitions = hand_record.get("transition_labels", [])
    if not transitions:
        return "unknown"
    non_stay = sum(token not in ("start", "stay") for token in transitions)
    major = sum(token == "major_shift" for token in transitions)
    motion = hand_record.get("hand_motion", "unknown")
    if all(token == "start" for token in transitions):
        return "onset"
    if non_stay == 0 and motion in ("steady", "start", "unknown"):
        return "hold"
    if major >= 6:
        return "reconfigure"
    if motion in ("opening", "closing"):
        return f"{motion}_phase"
    if motion == "mixed" or non_stay >= 10:
        return "articulate"
    if non_stay > 0:
        return "adjust"
    return "hold"


def classify_interaction_phase(frame: dict[str, object]) -> str:
    motion = frame.get("interaction_motion", "unknown")
    if motion == "unknown":
        return "unknown"
    if motion == "steady":
        return "steady_pair"
    if motion == "approach":
        return "approach_phase"
    if motion == "separate":
        return "separate_phase"
    return str(motion)


PROXIMAL_INDICES = [3, 7, 11, 15, 19]


def classify_hand_phase_refined(hand_record: dict[str, object] | None) -> str:
    if hand_record is None:
        return "absent"
    transitions = hand_record.get("transition_labels", [])
    if not transitions:
        return "unknown"
    if all(token == "start" for token in transitions):
        return "onset"
    motion = hand_record.get("hand_motion", "unknown")
    non_stay = sum(token not in ("start", "stay") for token in transitions)
    major = sum(token == "major_shift" for token in transitions)
    prox_nonstay = sum(
        transitions[idx] not in ("start", "stay")
        for idx in PROXIMAL_INDICES
        if idx < len(transitions)
    )
    prox_major = sum(
        transitions[idx] == "major_shift"
        for idx in PROXIMAL_INDICES
        if idx < len(transitions)
    )
    dist_nonstay = non_stay - prox_nonstay
    dist_major = major - prox_major

    if non_stay == 0 and motion in ("steady", "start", "unknown"):
        return "hold"
    if motion in ("opening", "closing"):
        if dist_nonstay >= prox_nonstay + 2:
            return f"distal_{motion}"
        if prox_nonstay >= dist_nonstay + 1:
            return f"proximal_{motion}"
        return f"coordinated_{motion}"
    if major >= 8:
        if prox_major >= dist_major + 2:
            return "wrist_reconfigure"
        if dist_major >= prox_major + 2:
            return "distal_reconfigure"
        return "global_reconfigure"
    if motion == "mixed":
        if dist_nonstay >= prox_nonstay + 2:
            return "distal_articulate"
        if prox_nonstay >= dist_nonstay + 2:
            return "proximal_articulate"
        return "coordinated_articulate"
    if dist_nonstay >= 8 and prox_nonstay <= 2:
        return "distal_articulate"
    if prox_nonstay >= 4 and prox_nonstay >= dist_nonstay:
        return "proximal_adjust"
    if non_stay >= 10:
        return "coordinated_articulate"
    if non_stay >= 3:
        return "micro_adjust"
    return "fine_adjust"


def classify_interaction_phase_refined(frame: dict[str, object]) -> str:
    motion = frame.get("interaction_motion", "unknown")
    distance = frame.get("cross_hand_distance")
    if motion == "unknown":
        return "unknown_pair"
    if motion == "steady":
        if distance is None:
            return "steady_pair"
        if distance < 60:
            return "near_hold"
        if distance < 120:
            return "mid_hold"
        return "far_hold"
    if motion == "approach":
        if distance is not None and distance < 60:
            return "contact_approach"
        return "approach_phase"
    if motion == "separate":
        if distance is not None and distance < 80:
            return "release_phase"
        return "separate_phase"
    return str(motion)


def event_group_key(frame: dict[str, object], mode: str) -> str:
    parts = [f"hand_type={frame.get('hand_type', 'unknown')}"]
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is None:
            parts.append(f"{hand_name}=none")
            continue
        parts.append(f"{hand_name}:state={pack_list(hand_record.get('token_labels'))}")
        if mode == "temporal":
            parts.append(f"{hand_name}:motion={hand_record.get('hand_motion', 'unknown')}")
    if mode == "temporal":
        parts.append(f"interaction={frame.get('interaction_motion', 'unknown')}")
    return ";".join(parts)


def phase_group_key(frame: dict[str, object], mode: str) -> str:
    parts = [f"hand_type={frame.get('hand_type', 'unknown')}"]
    if mode == "state":
        for hand_name in ("right", "left"):
            hand_record = frame.get(hand_name)
            if hand_record is None:
                parts.append(f"{hand_name}=none")
            else:
                parts.append(
                    f"{hand_name}:state_bin={pack_list(hand_record.get('token_labels'))}"
                )
        return ";".join(parts)
    for hand_name in ("right", "left"):
        parts.append(f"{hand_name}:phase={classify_hand_phase(frame.get(hand_name))}")
    parts.append(f"interaction_phase={classify_interaction_phase(frame)}")
    return ";".join(parts)


def refined_phase_group_key(frame: dict[str, object], mode: str) -> str:
    parts = [f"hand_type={frame.get('hand_type', 'unknown')}"]
    if mode == "state":
        return phase_group_key(frame, mode)
    for hand_name in ("right", "left"):
        parts.append(
            f"{hand_name}:phase={classify_hand_phase_refined(frame.get(hand_name))}"
        )
    parts.append(
        f"interaction_phase={classify_interaction_phase_refined(frame)}"
    )
    return ";".join(parts)


def segment_to_event_set(
    frames: list[dict[str, object]],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> set[str]:
    if not frames:
        return set()
    anchor = frames[-1]
    event = frame_token_set(anchor, mode, include_persistence=False)
    event.add(f"event::duration::{segment_duration_bucket(len(frames))}")
    if include_persistence and mode == "temporal":
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_persist:{hand_record.get('state_persistence_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_persist:{hand_record.get('activity_persistence_label', 'missing')}"
            )
        event.add(
            f"interaction_persist:{anchor.get('interaction_persistence_label', 'missing')}"
        )
        event.add(
            "interaction_activity_persist:"
            f"{anchor.get('interaction_activity_persistence_label', 'missing')}"
        )
    if include_segment_duration and mode == "temporal":
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_segdur:{hand_record.get('state_segment_duration_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_segdur:{hand_record.get('activity_segment_duration_label', 'missing')}"
            )
        event.add(
            f"interaction_segdur:{anchor.get('interaction_segment_duration_label', 'missing')}"
        )
        event.add(
            "interaction_activity_segdur:"
            f"{anchor.get('interaction_activity_segment_duration_label', 'missing')}"
        )
    return event


def segment_to_phase_event_set(
    frames: list[dict[str, object]],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> set[str]:
    if not frames:
        return set()
    anchor = frames[-1]
    event = {f"hand_type={anchor.get('hand_type', 'unknown')}"}
    if mode == "state":
        return segment_to_event_set(
            frames,
            mode,
            include_persistence=include_persistence,
            include_segment_duration=include_segment_duration,
        )
    for hand_name in ("right", "left"):
        event.add(f"{hand_name}:phase:{classify_hand_phase(anchor.get(hand_name))}")
    event.add(f"interaction_phase:{classify_interaction_phase(anchor)}")
    event.add(f"phase_event::duration::{segment_duration_bucket(len(frames))}")
    if include_persistence:
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_persist:{hand_record.get('state_persistence_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_persist:{hand_record.get('activity_persistence_label', 'missing')}"
            )
        event.add(
            f"interaction_persist:{anchor.get('interaction_persistence_label', 'missing')}"
        )
        event.add(
            "interaction_activity_persist:"
            f"{anchor.get('interaction_activity_persistence_label', 'missing')}"
        )
    if include_segment_duration:
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_segdur:{hand_record.get('state_segment_duration_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_segdur:{hand_record.get('activity_segment_duration_label', 'missing')}"
            )
        event.add(
            f"interaction_segdur:{anchor.get('interaction_segment_duration_label', 'missing')}"
        )
        event.add(
            "interaction_activity_segdur:"
            f"{anchor.get('interaction_activity_segment_duration_label', 'missing')}"
        )
    return event


def segment_to_refined_phase_event_set(
    frames: list[dict[str, object]],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> set[str]:
    if not frames:
        return set()
    anchor = frames[-1]
    event = {f"hand_type={anchor.get('hand_type', 'unknown')}"}
    if mode == "state":
        return segment_to_phase_event_set(
            frames,
            mode,
            include_persistence=include_persistence,
            include_segment_duration=include_segment_duration,
        )
    for hand_name in ("right", "left"):
        event.add(
            f"{hand_name}:phase:{classify_hand_phase_refined(anchor.get(hand_name))}"
        )
    event.add(
        f"interaction_phase:{classify_interaction_phase_refined(anchor)}"
    )
    event.add(f"refined_phase_event::duration::{segment_duration_bucket(len(frames))}")
    if include_persistence:
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_persist:{hand_record.get('state_persistence_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_persist:{hand_record.get('activity_persistence_label', 'missing')}"
            )
        event.add(
            f"interaction_persist:{anchor.get('interaction_persistence_label', 'missing')}"
        )
        event.add(
            "interaction_activity_persist:"
            f"{anchor.get('interaction_activity_persistence_label', 'missing')}"
        )
    if include_segment_duration:
        for hand_name in ("right", "left"):
            hand_record = anchor.get(hand_name)
            if hand_record is None:
                continue
            event.add(
                f"{hand_name}:state_segdur:{hand_record.get('state_segment_duration_label', 'missing')}"
            )
            event.add(
                f"{hand_name}:activity_segdur:{hand_record.get('activity_segment_duration_label', 'missing')}"
            )
        event.add(
            f"interaction_segdur:{anchor.get('interaction_segment_duration_label', 'missing')}"
        )
        event.add(
            "interaction_activity_segdur:"
            f"{anchor.get('interaction_activity_segment_duration_label', 'missing')}"
        )
    return event


def sequence_event_sets(
    sequence: dict[str, object],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> list[set[str]]:
    events = []
    segment = []
    prev_key = None
    for frame in sequence["frames"]:
        key = event_group_key(frame, mode)
        if prev_key is None or key == prev_key:
            segment.append(frame)
        else:
            events.append(
                segment_to_event_set(
                    segment,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                )
            )
            segment = [frame]
        prev_key = key
    if segment:
        events.append(
            segment_to_event_set(
                segment,
                mode,
                include_persistence=include_persistence,
                include_segment_duration=include_segment_duration,
            )
        )
    return events


def sequence_phase_event_sets(
    sequence: dict[str, object],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> list[set[str]]:
    events = []
    segment = []
    prev_key = None
    for frame in sequence["frames"]:
        key = phase_group_key(frame, mode)
        if prev_key is None or key == prev_key:
            segment.append(frame)
        else:
            events.append(
                segment_to_phase_event_set(
                    segment,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                )
            )
            segment = [frame]
        prev_key = key
    if segment:
        events.append(
            segment_to_phase_event_set(
                segment,
                mode,
                include_persistence=include_persistence,
                include_segment_duration=include_segment_duration,
            )
        )
    return events


def sequence_refined_phase_event_sets(
    sequence: dict[str, object],
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
) -> list[set[str]]:
    events = []
    segment = []
    prev_key = None
    for frame in sequence["frames"]:
        key = refined_phase_group_key(frame, mode)
        if prev_key is None or key == prev_key:
            segment.append(frame)
        else:
            events.append(
                segment_to_refined_phase_event_set(
                    segment,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                )
            )
            segment = [frame]
        prev_key = key
    if segment:
        events.append(
            segment_to_refined_phase_event_set(
                segment,
                mode,
                include_persistence=include_persistence,
                include_segment_duration=include_segment_duration,
            )
        )
    return events


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def normalized_similarity(a: list[str], b: list[str]) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - (levenshtein_distance(a, b) / denom)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def dtw_similarity(a: list[set[str]], b: list[set[str]]) -> float:
    n, m = len(a), len(b)
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 1.0 - jaccard_similarity(a[i - 1], b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    norm = max(n + m, 1)
    return 1.0 - (dp[n][m] / norm)


def retrieval_margin_stats(gallery, queries, scorer):
    same_scores = []
    impostor_scores = []
    margins = []
    for query in queries:
        same = []
        diff = []
        for item in gallery:
            score = scorer(query, item)
            if item["label"] == query["label"]:
                same.append(score)
            else:
                diff.append(score)
        if same:
            best_same = max(same)
            same_scores.append(best_same)
        if diff:
            best_diff = max(diff)
            impostor_scores.append(best_diff)
        if same and diff:
            margins.append(best_same - best_diff)
    return {
        "avg_best_same_label_similarity": sum(same_scores) / max(len(same_scores), 1),
        "avg_best_impostor_similarity": sum(impostor_scores) / max(len(impostor_scores), 1),
        "avg_similarity_margin": sum(margins) / max(len(margins), 1),
    }


def retrieve(
    gallery_data,
    query_data,
    mode: str,
    use_rle: bool,
    include_persistence: bool = False,
):
    allowed = overlap_labels(gallery_data, query_data)
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "tokens": sequence_tokens(
                    sequence,
                    mode,
                    use_rle,
                    include_persistence=include_persistence,
                ),
            }
        )
    queries = []
    for sequence, label in iter_sequences(query_data, allowed):
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "tokens": sequence_tokens(
                    sequence,
                    mode,
                    use_rle,
                    include_persistence=include_persistence,
                ),
            }
        )

    results = []
    per_class = defaultdict(list)
    for query in queries:
        ranking = []
        for item in gallery:
            score = normalized_similarity(query["tokens"], item["tokens"])
            ranking.append((item["label"], item["seq_name"], score))
        ranking.sort(key=lambda x: x[2], reverse=True)
        pred = ranking[0][0]
        correct = pred == query["label"]
        per_class[query["label"]].append(correct)
        results.append(
            {
                "seq_name": query["seq_name"],
                "target": query["label"],
                "prediction": pred,
                "top3": [[label, seq_name, float(score)] for label, seq_name, score in ranking[:3]],
                "correct": correct,
            }
        )
    top1 = sum(item["correct"] for item in results) / max(len(results), 1)
    per_class_acc = {
        label: sum(items) / max(len(items), 1) for label, items in sorted(per_class.items())
    }
    confusion = Counter()
    for item in results:
        if not item["correct"]:
            confusion[(item["target"], item["prediction"])] += 1
    return {
        "mode": mode,
        "use_rle": use_rle,
        "include_persistence": include_persistence,
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "top1_accuracy": top1,
        **retrieval_margin_stats(
            gallery,
            queries,
            scorer=lambda query, item: normalized_similarity(
                query["tokens"], item["tokens"]
            ),
        ),
        "per_class_accuracy": per_class_acc,
        "error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in confusion.most_common()
        ],
        "results": results,
    }


def retrieve_dtw(gallery_data, query_data, mode: str, include_persistence: bool = False):
    allowed = overlap_labels(gallery_data, query_data)
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "frames": sequence_frame_sets(
                    sequence, mode, include_persistence=include_persistence
                ),
            }
        )
    queries = []
    for sequence, label in iter_sequences(query_data, allowed):
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "frames": sequence_frame_sets(
                    sequence, mode, include_persistence=include_persistence
                ),
            }
        )

    results = []
    per_class = defaultdict(list)
    for query in queries:
        ranking = []
        for item in gallery:
            score = dtw_similarity(query["frames"], item["frames"])
            ranking.append((item["label"], item["seq_name"], score))
        ranking.sort(key=lambda x: x[2], reverse=True)
        pred = ranking[0][0]
        correct = pred == query["label"]
        per_class[query["label"]].append(correct)
        results.append(
            {
                "seq_name": query["seq_name"],
                "target": query["label"],
                "prediction": pred,
                "top3": [[label, seq_name, float(score)] for label, seq_name, score in ranking[:3]],
                "correct": correct,
            }
        )
    top1 = sum(item["correct"] for item in results) / max(len(results), 1)
    per_class_acc = {
        label: sum(items) / max(len(items), 1) for label, items in sorted(per_class.items())
    }
    confusion = Counter()
    for item in results:
        if not item["correct"]:
            confusion[(item["target"], item["prediction"])] += 1
    return {
        "mode": mode,
        "distance": "dtw_jaccard",
        "include_persistence": include_persistence,
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "top1_accuracy": top1,
        **retrieval_margin_stats(
            gallery,
            queries,
            scorer=lambda query, item: dtw_similarity(
                query["frames"], item["frames"]
            ),
        ),
        "per_class_accuracy": per_class_acc,
        "error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in confusion.most_common()
        ],
        "results": results,
    }


def retrieve_event_dtw(
    gallery_data,
    query_data,
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
):
    allowed = overlap_labels(gallery_data, query_data)
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )
    queries = []
    for sequence, label in iter_sequences(query_data, allowed):
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )

    results = []
    per_class = defaultdict(list)
    for query in queries:
        ranking = []
        for item in gallery:
            score = dtw_similarity(query["events"], item["events"])
            ranking.append((item["label"], item["seq_name"], score, len(item["events"])))
        ranking.sort(key=lambda x: x[2], reverse=True)
        pred = ranking[0][0]
        correct = pred == query["label"]
        per_class[query["label"]].append(correct)
        results.append(
            {
                "seq_name": query["seq_name"],
                "target": query["label"],
                "prediction": pred,
                "num_query_events": len(query["events"]),
                "top3": [
                    [label, seq_name, float(score), int(num_events)]
                    for label, seq_name, score, num_events in ranking[:3]
                ],
                "correct": correct,
            }
        )
    top1 = sum(item["correct"] for item in results) / max(len(results), 1)
    per_class_acc = {
        label: sum(items) / max(len(items), 1) for label, items in sorted(per_class.items())
    }
    confusion = Counter()
    for item in results:
        if not item["correct"]:
            confusion[(item["target"], item["prediction"])] += 1
    return {
        "mode": mode,
        "distance": "event_dtw_jaccard",
        "include_persistence": include_persistence,
        "include_segment_duration": include_segment_duration,
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "top1_accuracy": top1,
        **retrieval_margin_stats(
            gallery,
            queries,
            scorer=lambda query, item: dtw_similarity(
                query["events"], item["events"]
            ),
        ),
        "avg_num_gallery_events": sum(len(item["events"]) for item in gallery) / max(len(gallery), 1),
        "avg_num_query_events": sum(len(item["events"]) for item in queries) / max(len(queries), 1),
        "per_class_accuracy": per_class_acc,
        "error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in confusion.most_common()
        ],
        "results": results,
    }


def retrieve_phase_event_dtw(
    gallery_data,
    query_data,
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
):
    allowed = overlap_labels(gallery_data, query_data)
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_phase_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )
    queries = []
    for sequence, label in iter_sequences(query_data, allowed):
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_phase_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )

    results = []
    per_class = defaultdict(list)
    for query in queries:
        ranking = []
        for item in gallery:
            score = dtw_similarity(query["events"], item["events"])
            ranking.append((item["label"], item["seq_name"], score, len(item["events"])))
        ranking.sort(key=lambda x: x[2], reverse=True)
        pred = ranking[0][0]
        correct = pred == query["label"]
        per_class[query["label"]].append(correct)
        results.append(
            {
                "seq_name": query["seq_name"],
                "target": query["label"],
                "prediction": pred,
                "num_query_events": len(query["events"]),
                "top3": [
                    [label, seq_name, float(score), int(num_events)]
                    for label, seq_name, score, num_events in ranking[:3]
                ],
                "correct": correct,
            }
        )
    top1 = sum(item["correct"] for item in results) / max(len(results), 1)
    per_class_acc = {
        label: sum(items) / max(len(items), 1) for label, items in sorted(per_class.items())
    }
    confusion = Counter()
    for item in results:
        if not item["correct"]:
            confusion[(item["target"], item["prediction"])] += 1
    return {
        "mode": mode,
        "distance": "phase_event_dtw_jaccard",
        "include_persistence": include_persistence,
        "include_segment_duration": include_segment_duration,
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "top1_accuracy": top1,
        **retrieval_margin_stats(
            gallery,
            queries,
            scorer=lambda query, item: dtw_similarity(
                query["events"], item["events"]
            ),
        ),
        "avg_num_gallery_events": sum(len(item["events"]) for item in gallery) / max(len(gallery), 1),
        "avg_num_query_events": sum(len(item["events"]) for item in queries) / max(len(queries), 1),
        "per_class_accuracy": per_class_acc,
        "error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in confusion.most_common()
        ],
        "results": results,
    }


def retrieve_refined_phase_event_dtw(
    gallery_data,
    query_data,
    mode: str,
    include_persistence: bool = False,
    include_segment_duration: bool = False,
):
    allowed = overlap_labels(gallery_data, query_data)
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_refined_phase_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )
    queries = []
    for sequence, label in iter_sequences(query_data, allowed):
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_refined_phase_event_sets(
                    sequence,
                    mode,
                    include_persistence=include_persistence,
                    include_segment_duration=include_segment_duration,
                ),
            }
        )

    results = []
    per_class = defaultdict(list)
    for query in queries:
        ranking = []
        for item in gallery:
            score = dtw_similarity(query["events"], item["events"])
            ranking.append((item["label"], item["seq_name"], score, len(item["events"])))
        ranking.sort(key=lambda x: x[2], reverse=True)
        pred = ranking[0][0]
        correct = pred == query["label"]
        per_class[query["label"]].append(correct)
        results.append(
            {
                "seq_name": query["seq_name"],
                "target": query["label"],
                "prediction": pred,
                "num_query_events": len(query["events"]),
                "top3": [
                    [label, seq_name, float(score), int(num_events)]
                    for label, seq_name, score, num_events in ranking[:3]
                ],
                "correct": correct,
            }
        )
    top1 = sum(item["correct"] for item in results) / max(len(results), 1)
    per_class_acc = {
        label: sum(items) / max(len(items), 1) for label, items in sorted(per_class.items())
    }
    confusion = Counter()
    for item in results:
        if not item["correct"]:
            confusion[(item["target"], item["prediction"])] += 1
    return {
        "mode": mode,
        "distance": "refined_phase_event_dtw_jaccard",
        "include_persistence": include_persistence,
        "include_segment_duration": include_segment_duration,
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "top1_accuracy": top1,
        **retrieval_margin_stats(
            gallery,
            queries,
            scorer=lambda query, item: dtw_similarity(
                query["events"], item["events"]
            ),
        ),
        "avg_num_gallery_events": sum(len(item["events"]) for item in gallery) / max(len(gallery), 1),
        "avg_num_query_events": sum(len(item["events"]) for item in queries) / max(len(queries), 1),
        "per_class_accuracy": per_class_acc,
        "error_counts": [
            {"target": tgt, "prediction": pred, "count": count}
            for (tgt, pred), count in confusion.most_common()
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/sequence_symbolic_retrieval.json"),
    )
    args = parser.parse_args()

    gallery_data = load_json(args.gallery_json)
    query_data = load_json(args.query_json)
    payload = {
        "state_frames": retrieve(gallery_data, query_data, mode="state", use_rle=False),
        "state_rle": retrieve(gallery_data, query_data, mode="state", use_rle=True),
        "temporal_frames": retrieve(gallery_data, query_data, mode="temporal", use_rle=False),
        "temporal_rle": retrieve(gallery_data, query_data, mode="temporal", use_rle=True),
        "temporal_persist_frames": retrieve(
            gallery_data,
            query_data,
            mode="temporal",
            use_rle=False,
            include_persistence=True,
        ),
        "temporal_persist_rle": retrieve(
            gallery_data,
            query_data,
            mode="temporal",
            use_rle=True,
            include_persistence=True,
        ),
        "state_dtw": retrieve_dtw(gallery_data, query_data, mode="state"),
        "temporal_dtw": retrieve_dtw(gallery_data, query_data, mode="temporal"),
        "temporal_persist_dtw": retrieve_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_persistence=True,
        ),
        "state_event_dtw": retrieve_event_dtw(
            gallery_data, query_data, mode="state"
        ),
        "temporal_event_dtw": retrieve_event_dtw(
            gallery_data, query_data, mode="temporal"
        ),
        "temporal_persist_event_dtw": retrieve_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_persistence=True,
        ),
        "temporal_segdur_event_dtw": retrieve_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_segment_duration=True,
        ),
        "temporal_persist_segdur_event_dtw": retrieve_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_persistence=True,
            include_segment_duration=True,
        ),
        "temporal_phase_event_dtw": retrieve_phase_event_dtw(
            gallery_data, query_data, mode="temporal"
        ),
        "temporal_phase_persist_event_dtw": retrieve_phase_event_dtw(
            gallery_data, query_data, mode="temporal", include_persistence=True
        ),
        "temporal_phase_segdur_event_dtw": retrieve_phase_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_segment_duration=True,
        ),
        "temporal_phase_persist_segdur_event_dtw": retrieve_phase_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_persistence=True,
            include_segment_duration=True,
        ),
        "temporal_refined_phase_event_dtw": retrieve_refined_phase_event_dtw(
            gallery_data, query_data, mode="temporal"
        ),
        "temporal_refined_phase_persist_event_dtw": retrieve_refined_phase_event_dtw(
            gallery_data, query_data, mode="temporal", include_persistence=True
        ),
        "temporal_refined_phase_segdur_event_dtw": retrieve_refined_phase_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_segment_duration=True,
        ),
        "temporal_refined_phase_persist_segdur_event_dtw": retrieve_refined_phase_event_dtw(
            gallery_data,
            query_data,
            mode="temporal",
            include_persistence=True,
            include_segment_duration=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    for key in [
        "state_frames",
        "state_rle",
        "temporal_frames",
        "temporal_rle",
        "temporal_persist_frames",
        "temporal_persist_rle",
        "state_dtw",
        "temporal_dtw",
        "temporal_persist_dtw",
        "state_event_dtw",
        "temporal_event_dtw",
        "temporal_persist_event_dtw",
        "temporal_segdur_event_dtw",
        "temporal_persist_segdur_event_dtw",
        "temporal_phase_event_dtw",
        "temporal_phase_persist_event_dtw",
        "temporal_phase_segdur_event_dtw",
        "temporal_phase_persist_segdur_event_dtw",
        "temporal_refined_phase_event_dtw",
        "temporal_refined_phase_persist_event_dtw",
        "temporal_refined_phase_segdur_event_dtw",
        "temporal_refined_phase_persist_segdur_event_dtw",
    ]:
        print(f"{key} top1={payload[key]['top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()
