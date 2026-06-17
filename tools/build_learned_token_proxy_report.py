#!/usr/bin/env python3
"""Build a learned-token proxy baseline report on current HL data."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tools.eval_sequence_symbolic_retrieval import (
    canonical_label,
    dtw_similarity,
    frame_token_set,
    normalized_similarity,
    overlap_labels,
    segment_duration_bucket,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"

HAND_TYPES = ["right", "left", "interacting"]
HAND_MOTIONS = ["start", "steady", "opening", "closing", "mixed", "unknown"]
INTERACTION_MOTIONS = ["unknown", "steady", "approach", "separate"]
PERSISTENCE_LABELS = ["start", "instant", "short", "medium", "long", "unknown"]


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def one_hot(value: str, vocab: list[str], prefix: str) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def hand_vector(frame: dict[str, object], hand_name: str) -> list[float]:
    hand = frame.get(hand_name)
    if hand is None:
        return [0.0] * (1 + 60 + 5 + 20 + len(HAND_MOTIONS) + 2 * len(PERSISTENCE_LABELS))
    out = [1.0]
    local_vectors = hand.get("local_vectors", [])
    for vec in local_vectors:
        out.extend(float(v) for v in vec)
    while len(out) < 1 + 60:
        out.append(0.0)
    flexion = hand.get("flexion_scores", {})
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        out.append(float(flexion.get(finger, 0.0)) / math.pi)
    angles = hand.get("transition_angles_deg", [])
    out.extend(float(v) / 90.0 for v in angles[:20])
    while len(out) < 1 + 60 + 5 + 20:
        out.append(0.0)
    out.extend(one_hot(str(hand.get("hand_motion", "unknown")), HAND_MOTIONS, f"{hand_name}_motion"))
    out.extend(one_hot(str(hand.get("state_persistence_label", "unknown")), PERSISTENCE_LABELS, f"{hand_name}_sp"))
    out.extend(one_hot(str(hand.get("activity_persistence_label", "unknown")), PERSISTENCE_LABELS, f"{hand_name}_ap"))
    return out


def frame_vector(frame: dict[str, object]) -> list[float]:
    out = []
    out.extend(one_hot(str(frame.get("hand_type", "unknown")), HAND_TYPES, "hand_type"))
    out.extend(hand_vector(frame, "right"))
    out.extend(hand_vector(frame, "left"))
    out.extend(one_hot(str(frame.get("interaction_motion", "unknown")), INTERACTION_MOTIONS, "interaction_motion"))
    out.extend(
        one_hot(
            str(frame.get("interaction_persistence_label", "unknown")),
            PERSISTENCE_LABELS,
            "interaction_persistence",
        )
    )
    out.extend(
        one_hot(
            str(frame.get("interaction_activity_persistence_label", "unknown")),
            PERSISTENCE_LABELS,
            "interaction_activity_persistence",
        )
    )
    distance = frame.get("cross_hand_distance")
    out.append((0.0 if distance is None else float(distance) / 300.0))
    return out


def iter_sequences(dataset: dict[str, object], allowed_labels: set[str]):
    for sequence in dataset["sequences"]:
        label = canonical_label(sequence["seq_name"])
        if label in allowed_labels:
            yield sequence, label


def collect_frame_matrix(dataset: dict[str, object], allowed_labels: set[str]) -> np.ndarray:
    rows = []
    for sequence, _ in iter_sequences(dataset, allowed_labels):
        for frame in sequence["frames"]:
            rows.append(frame_vector(frame))
    return np.asarray(rows, dtype=np.float32)


def collect_semantic_frame_matrix(
    dataset: dict[str, object], allowed_labels: set[str], vocab: list[str]
) -> np.ndarray:
    token_to_idx = {tok: idx for idx, tok in enumerate(vocab)}
    rows = []
    for sequence, _ in iter_sequences(dataset, allowed_labels):
        for frame in sequence["frames"]:
            vec = np.zeros(len(vocab), dtype=np.float32)
            for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
                idx = token_to_idx.get(tok)
                if idx is not None:
                    vec[idx] = 1.0
            rows.append(vec)
    return np.asarray(rows, dtype=np.float32)


def build_semantic_frame_vocab(dataset: dict[str, object], allowed_labels: set[str]) -> list[str]:
    vocab = set()
    for sequence, _ in iter_sequences(dataset, allowed_labels):
        for frame in sequence["frames"]:
            vocab.update(frame_token_set(frame, mode="temporal", include_persistence=True))
    return sorted(vocab)


def encode_sequence(sequence: dict[str, object], kmeans: KMeans) -> list[int]:
    matrix = np.asarray([frame_vector(frame) for frame in sequence["frames"]], dtype=np.float32)
    return list(map(int, kmeans.predict(matrix)))


def encode_sequence_semantic(sequence: dict[str, object], kmeans: KMeans, vocab: list[str]) -> list[int]:
    token_to_idx = {tok: idx for idx, tok in enumerate(vocab)}
    rows = []
    for frame in sequence["frames"]:
        vec = np.zeros(len(vocab), dtype=np.float32)
        for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
            idx = token_to_idx.get(tok)
            if idx is not None:
                vec[idx] = 1.0
        rows.append(vec)
    matrix = np.asarray(rows, dtype=np.float32)
    return list(map(int, kmeans.predict(matrix)))


def tokens_to_rle(tokens: list[int]) -> list[str]:
    out = []
    prev = None
    count = 0
    for tok in tokens:
        if prev is None or tok == prev:
            count += 1
        else:
            out.append(f"c{prev}#run={count}")
            count = 1
        prev = tok
    if prev is not None:
        out.append(f"c{prev}#run={count}")
    return out


def tokens_to_events(tokens: list[int]) -> list[set[str]]:
    out = []
    prev = None
    count = 0
    for tok in tokens:
        if prev is None or tok == prev:
            count += 1
        else:
            out.append({f"cluster::{prev}", f"duration::{segment_duration_bucket(count)}"})
            count = 1
        prev = tok
    if prev is not None:
        out.append({f"cluster::{prev}", f"duration::{segment_duration_bucket(count)}"})
    return out


def score_token_rep(query_rep, gallery_rep, token_mode: str) -> float:
    if token_mode in {"frame_tokens", "rle_tokens"}:
        return normalized_similarity(query_rep, gallery_rep)
    if token_mode == "event_dtw":
        return dtw_similarity(query_rep, gallery_rep)
    raise ValueError(token_mode)


def evaluate_retrieval(gallery_data, query_data, allowed_labels, encoder):
    configs = {
        "proxy_frame": "frame_tokens",
        "proxy_rle": "rle_tokens",
        "proxy_event": "event_dtw",
    }
    encoded_gallery = []
    encoded_queries = []
    for sequence, label in iter_sequences(gallery_data, allowed_labels):
        toks = encoder(sequence)
        encoded_gallery.append({"seq_name": sequence["seq_name"], "label": label, "tokens": toks})
    for sequence, label in iter_sequences(query_data, allowed_labels):
        toks = encoder(sequence)
        encoded_queries.append({"seq_name": sequence["seq_name"], "label": label, "tokens": toks})

    results = []
    for name, token_mode in configs.items():
        gallery = []
        queries = []
        for item in encoded_gallery:
            rep = item["tokens"]
            if token_mode == "frame_tokens":
                rep = [f"c{tok}" for tok in item["tokens"]]
            elif token_mode == "rle_tokens":
                rep = tokens_to_rle(item["tokens"])
            else:
                rep = tokens_to_events(item["tokens"])
            gallery.append({**item, "rep": rep})
        for item in encoded_queries:
            rep = item["tokens"]
            if token_mode == "frame_tokens":
                rep = [f"c{tok}" for tok in item["tokens"]]
            elif token_mode == "rle_tokens":
                rep = tokens_to_rle(item["tokens"])
            else:
                rep = tokens_to_events(item["tokens"])
            queries.append({**item, "rep": rep})

        rows = []
        top1 = 0
        margins = []
        ranks = []
        for query in queries:
            ranking = []
            for item in gallery:
                score = score_token_rep(query["rep"], item["rep"], token_mode)
                ranking.append((item["label"], item["seq_name"], float(score)))
            ranking.sort(key=lambda x: x[2], reverse=True)
            if canonical_label(ranking[0][0]) == canonical_label(query["label"]):
                top1 += 1
            correct_items = [x for x in ranking if canonical_label(x[0]) == canonical_label(query["label"])]
            wrong_items = [x for x in ranking if canonical_label(x[0]) != canonical_label(query["label"])]
            best_correct = correct_items[0]
            best_wrong = wrong_items[0] if wrong_items else None
            margin = best_correct[2] - (best_wrong[2] if best_wrong else best_correct[2])
            rank = next(i for i, x in enumerate(ranking, start=1) if canonical_label(x[0]) == canonical_label(query["label"]))
            margins.append(margin)
            ranks.append(rank)
            rows.append(
                {
                    "seq_name": query["seq_name"],
                    "label": query["label"],
                    "top1_label": ranking[0][0],
                    "top1_correct": canonical_label(ranking[0][0]) == canonical_label(query["label"]),
                    "positive_margin": margin,
                    "correct_rank": rank,
                }
            )
        results.append(
            {
                "name": name,
                "token_mode": token_mode,
                "summary": {
                    "num_queries": len(queries),
                    "top1_accuracy": top1 / max(len(queries), 1),
                    "mean_positive_margin": sum(margins) / max(len(margins), 1),
                    "mean_correct_rank": sum(ranks) / max(len(ranks), 1),
                },
                "rows": rows,
            }
        )
    results.sort(
        key=lambda x: (
            -x["summary"]["top1_accuracy"],
            -x["summary"]["mean_positive_margin"],
            x["summary"]["mean_correct_rank"],
            x["name"],
        )
    )
    return results


def window_features_from_tokens(tokens: list[int]) -> dict[str, float]:
    tok_hist = Counter(tokens)
    bigrams = Counter((a, b) for a, b in zip(tokens, tokens[1:]))
    runs = []
    prev = None
    count = 0
    for tok in tokens:
        if prev is None or tok == prev:
            count += 1
        else:
            runs.append((prev, count))
            count = 1
        prev = tok
    if prev is not None:
        runs.append((prev, count))

    features = {}
    n = max(len(tokens), 1)
    for tok, c in tok_hist.items():
        features[f"tok::{tok}"] = c / n
    for (a, b), c in bigrams.items():
        features[f"bigram::{a}->{b}"] = c / max(len(tokens) - 1, 1)
    for tok, run_len in runs:
        bucket = segment_duration_bucket(run_len)
        features[f"run::{tok}::{bucket}"] = features.get(f"run::{tok}::{bucket}", 0.0) + 1.0 / max(len(runs), 1)
    denom = math.sqrt(sum(v * v for v in features.values()))
    if denom > 0:
        for key in list(features):
            features[key] /= denom
    return features


def make_window_samples(dataset: dict[str, object], allowed_labels: set[str], encoder, window_size: int, stride: int):
    samples = []
    for seq_idx, (sequence, label) in enumerate(iter_sequences(dataset, allowed_labels)):
        toks = encoder(sequence)
        if len(toks) < window_size:
            starts = [0]
        else:
            starts = list(range(0, len(toks) - window_size + 1, stride))
            if starts[-1] != len(toks) - window_size:
                starts.append(len(toks) - window_size)
        for window_idx, start in enumerate(starts):
            window = toks[start : start + window_size]
            samples.append(
                {
                    "sample_id": f"{seq_idx}:{window_idx}",
                    "seq_name": sequence["seq_name"],
                    "label": label,
                    "features": window_features_from_tokens(window),
                }
            )
    return samples


def build_vocabulary(samples):
    vocab = set()
    for sample in samples:
        vocab.update(sample["features"].keys())
    return sorted(vocab)


def vectorize(samples, vocab):
    feat_index = {key: idx for idx, key in enumerate(vocab)}
    matrix = np.zeros((len(samples), len(vocab)), dtype=np.float32)
    for row_idx, sample in enumerate(samples):
        for key, value in sample["features"].items():
            matrix[row_idx, feat_index[key]] = value
    return matrix


def subsample_by_fraction(samples, fraction: float, seed: int):
    if fraction >= 0.999:
        return list(samples)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    subset = []
    for label, rows in grouped.items():
        rows = list(rows)
        rng.shuffle(rows)
        keep = max(1, int(round(len(rows) * fraction)))
        subset.extend(rows[:keep])
    return subset


def aggregate_sequence_predictions(probs, pred_labels, labels, samples, method="mean_log_prob"):
    by_seq_probs = defaultdict(list)
    by_seq_votes = defaultdict(list)
    gt = {}
    for prob, pred, sample in zip(probs, pred_labels, samples):
        by_seq_probs[sample["seq_name"]].append(prob)
        by_seq_votes[sample["seq_name"]].append(int(pred))
        gt[sample["seq_name"]] = sample["label"]
    y_true, y_pred = [], []
    for seq_name in sorted(by_seq_probs):
        probs_arr = np.asarray(by_seq_probs[seq_name], dtype=np.float64)
        if method == "mean_log_prob":
            aggregated = np.mean(np.log(np.clip(probs_arr, 1e-8, 1.0)), axis=0)
            pred_idx = int(np.argmax(aggregated))
        else:
            vote_counter = Counter(by_seq_votes[seq_name])
            pred_idx = vote_counter.most_common(1)[0][0]
        y_true.append(gt[seq_name])
        y_pred.append(labels[pred_idx])
    return y_true, y_pred


def run_classifier(train_samples, test_samples, fraction: float, seed: int, c_value: float):
    train_subset = subsample_by_fraction(train_samples, fraction, seed)
    vocab = build_vocabulary(train_subset + test_samples)
    x_train = vectorize(train_subset, vocab)
    x_test = vectorize(test_samples, vocab)
    label_names = sorted({sample["label"] for sample in train_subset})
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    y_train = np.array([label_to_idx[sample["label"]] for sample in train_subset], dtype=int)
    y_test = np.array([label_to_idx[sample["label"]] for sample in test_samples], dtype=int)
    clf = LogisticRegression(max_iter=2000, C=c_value, solver="lbfgs", random_state=seed)
    clf.fit(x_train, y_train)
    window_pred = clf.predict(x_test)
    window_acc = accuracy_score(y_test, window_pred)
    seq_true, seq_pred = aggregate_sequence_predictions(
        clf.predict_proba(x_test), window_pred, label_names, test_samples, method="mean_log_prob"
    )
    seq_acc = accuracy_score(seq_true, seq_pred)
    return {
        "fraction": fraction,
        "seed": seed,
        "num_train_windows": len(train_subset),
        "num_test_windows": len(test_samples),
        "window_accuracy": float(window_acc),
        "sequence_accuracy": float(seq_acc),
    }


def summarize_classifier(results):
    by_fraction = defaultdict(list)
    for result in results:
        by_fraction[result["fraction"]].append(result)
    out = {}
    for fraction, items in sorted(by_fraction.items()):
        out[str(fraction)] = {
            "num_runs": len(items),
            "avg_train_windows": statistics.mean(item["num_train_windows"] for item in items),
            "window_accuracy_mean": statistics.mean(item["window_accuracy"] for item in items),
            "sequence_accuracy_mean": statistics.mean(item["sequence_accuracy"] for item in items),
        }
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    train_matrix = collect_frame_matrix(train_data, labels)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    semantic_train_matrix = collect_semantic_frame_matrix(train_data, labels, semantic_vocab)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "codebook_results": [],
    }

    sources = [
        ("semantic_frame", semantic_train_matrix, lambda km: (lambda seq: encode_sequence_semantic(seq, km, semantic_vocab))),
        ("continuous_frame", train_matrix, lambda km: (lambda seq: encode_sequence(seq, km))),
    ]

    for source_name, source_matrix, encoder_factory in sources:
        for n_clusters in [32, 64, 128]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            kmeans.fit(source_matrix)
            encoder = encoder_factory(kmeans)
            retrieval = evaluate_retrieval(train_data, test_data, labels, encoder)
            train_samples = make_window_samples(train_data, labels, encoder, window_size=32, stride=16)
            test_samples = make_window_samples(test_data, labels, encoder, window_size=32, stride=16)
            clf_results = []
            for fraction in [0.25, 0.5, 1.0]:
                for seed in [0, 1, 2, 3, 4]:
                    clf_results.append(run_classifier(train_samples, test_samples, fraction, seed, c_value=16.0))
            payload["codebook_results"].append(
                {
                    "source": source_name,
                    "num_clusters": n_clusters,
                    "retrieval": retrieval,
                    "classifier": {
                        "window_size": 32,
                        "stride": 16,
                        "c_value": 16.0,
                        "results": clf_results,
                        "summary": summarize_classifier(clf_results),
                    },
                }
            )

    payload["codebook_results"].sort(
        key=lambda row: (
            row["source"],
            -row["retrieval"][0]["summary"]["top1_accuracy"],
            -row["retrieval"][0]["summary"]["mean_positive_margin"],
            -row["classifier"]["summary"]["1.0"]["sequence_accuracy_mean"],
            row["num_clusters"],
        )
    )

    out_json = GEN / "learned_token_proxy_report.json"
    out_md = SUM / "learned_token_proxy_report.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Learned-Token Proxy Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Retrieval",
        "",
        "| source | codebook | best retrieval view | top1 | mean margin | mean rank |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in payload["codebook_results"]:
        best_ret = row["retrieval"][0]
        s = best_ret["summary"]
        lines.append(
            f"| {row['source']} | {row['num_clusters']} | {best_ret['name']} | {fmt(s['top1_accuracy'])} | {fmt(s['mean_positive_margin'])} | {fmt(s['mean_correct_rank'])} |"
        )

    lines.extend([
        "",
        "## Classifier",
        "",
        "| source | codebook | frac | seq acc | win acc | avg train windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["codebook_results"]:
        for fraction in ["0.25", "0.5", "1.0"]:
            s = row["classifier"]["summary"][fraction]
            lines.append(
                f"| {row['source']} | {row['num_clusters']} | {fraction} | {fmt(s['sequence_accuracy_mean'])} | {fmt(s['window_accuracy_mean'])} | {s['avg_train_windows']:.1f} |"
            )

    grouped = defaultdict(list)
    for row in payload["codebook_results"]:
        grouped[row["source"]].append(row)
    lines.extend([
        "",
        "## Takeaways",
        "",
    ])
    for source_name, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                -row["retrieval"][0]["summary"]["top1_accuracy"],
                -row["retrieval"][0]["summary"]["mean_positive_margin"],
                -row["classifier"]["summary"]["1.0"]["sequence_accuracy_mean"],
                row["num_clusters"],
            )
        )
        best_codebook = rows[0]
        best_ret = best_codebook["retrieval"][0]["summary"]
        best_clf = best_codebook["classifier"]["summary"]
        lines.append(
            f"- Best `{source_name}` proxy: `{best_codebook['num_clusters']}` clusters; retrieval top1 `{fmt(best_ret['top1_accuracy'])}`, margin `{fmt(best_ret['mean_positive_margin'])}`, classifier seq accs `{fmt(best_clf['0.25']['sequence_accuracy_mean'])} / {fmt(best_clf['0.5']['sequence_accuracy_mean'])} / {fmt(best_clf['1.0']['sequence_accuracy_mean'])}` for fractions `0.25 / 0.5 / 1.0`."
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
