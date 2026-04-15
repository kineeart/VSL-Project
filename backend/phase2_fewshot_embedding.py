"""Phase 2: Large-scale few-shot benchmark using embedding + cosine NN.

This script keeps many classes and evaluates a low-data setting by:
- extracting keypoint sequences
- building per-video embeddings from engineered features
- one/few-shot support set per class
- cosine nearest-prototype evaluation (Top-1/Top-5)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import train_gpu as tg
from keypoint_variants import KeypointType


def configure_variant(variant_name: str) -> None:
    variant = KeypointType(variant_name)
    tg.KEYPOINT_VARIANT = variant
    tg.VARIANT_CONFIG = tg.VARIANTS[variant]
    tg.HAS_HANDS_POSE = (
        tg.VARIANT_CONFIG.include_pose
        and tg.VARIANT_CONFIG.include_left_hand
        and tg.VARIANT_CONFIG.include_right_hand
    )
    tg.NRF = tg.VARIANT_CONFIG.feature_size
    tg.EXTRA_FEATURES = 9 if tg.HAS_HANDS_POSE else 0
    tg.NF = tg.NRF * 3 + tg.EXTRA_FEATURES


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def sequence_to_embedding(seq: np.ndarray) -> np.ndarray:
    feats = tg.eng_feat(seq, tg.KEYPOINT_VARIANT)  # (seq, nf)
    emb = feats.mean(axis=0).astype(np.float32)
    return l2_normalize(emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 few-shot embedding benchmark")
    parser.add_argument("--variant", type=str, default=KeypointType.HOLISTIC_NO_FACE.value)
    parser.add_argument("--seq", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-classes", type=int, default=0, help="0 means keep all classes")
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        type=str,
        default="benchmark/reports/phase2_fewshot_embedding.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    configure_variant(args.variant)
    tg.SEQ = args.seq
    tg.NW = max(1, args.workers)

    print("=" * 70)
    print("PHASE 2 - FEW-SHOT EMBEDDING BENCHMARK")
    print("=" * 70)
    print(f"Variant: {tg.KEYPOINT_VARIANT.value} ({tg.VARIANT_CONFIG.name})")
    print(f"Features: raw={tg.NRF}, engineered={tg.NF}")
    print(f"Support per class: {args.support_per_class}")

    mapping = tg.load_data_mapping()
    if not mapping:
        raise RuntimeError("No mapping loaded from Data.xlsx")

    counts = Counter(mapping.values())
    ranked = [(lb, cnt) for lb, cnt in counts.items() if cnt >= args.min_samples]
    ranked.sort(key=lambda x: (-x[1], str(x[0])))

    if args.max_classes > 0:
        selected_labels = {lb for lb, _ in ranked[: args.max_classes]}
    else:
        selected_labels = {lb for lb, _ in ranked}

    mapping = {fn: lb for fn, lb in mapping.items() if lb in selected_labels}
    print(f"[PHASE2] Labels retained: {len(selected_labels)}")
    print(f"[PHASE2] Videos retained: {len(mapping)}")

    tg.extract_all(mapping)

    vectors_by_class = defaultdict(list)
    skipped = 0
    for fn, lb in mapping.items():
        cp = tg.landmark_cache_path(fn, tg.KEYPOINT_VARIANT)
        if not cp.exists():
            skipped += 1
            continue
        try:
            seq = np.load(str(cp))
            if seq.ndim != 2 or seq.shape[1] != tg.NRF:
                skipped += 1
                continue
            if seq.shape[0] != tg.SEQ:
                seq = tg.random_choose_seq(seq, tg.SEQ, auto_pad=True)
            emb = sequence_to_embedding(seq)
            vectors_by_class[lb].append((fn, emb))
        except Exception:
            skipped += 1

    eligible_classes = [lb for lb, items in vectors_by_class.items() if len(items) >= (args.support_per_class + 1)]
    eligible_classes.sort(key=str)

    if not eligible_classes:
        raise RuntimeError(
            "No class has enough vectors for support+query split. "
            "Reduce --support-per-class or increase available videos."
        )

    class_to_idx = {lb: i for i, lb in enumerate(eligible_classes)}

    proto_list = []
    query_vecs = []
    query_labels = []

    for lb in eligible_classes:
        items = vectors_by_class[lb]
        order = rng.permutation(len(items))
        items = [items[i] for i in order]

        support = items[: args.support_per_class]
        query = items[args.support_per_class :]

        support_mat = np.stack([v for _, v in support], axis=0)
        proto = l2_normalize(support_mat.mean(axis=0).astype(np.float32))
        proto_list.append(proto)

        for _, qv in query:
            query_vecs.append(qv)
            query_labels.append(class_to_idx[lb])

    proto_mat = np.stack(proto_list, axis=0)  # (C, D)

    top1 = 0
    top5 = 0
    for qv, yi in zip(query_vecs, query_labels):
        scores = proto_mat @ qv
        order = np.argsort(-scores)
        if yi == order[0]:
            top1 += 1
        if yi in order[: min(5, len(order))]:
            top5 += 1

    total_q = len(query_labels)
    top1_acc = top1 / total_q if total_q else 0.0
    top5_acc = top5 / total_q if total_q else 0.0

    report = {
        "phase": "phase2_fewshot_embedding",
        "keypoint_variant": tg.KEYPOINT_VARIANT.value,
        "raw_features": int(tg.NRF),
        "engineered_features": int(tg.NF),
        "seq_len": int(tg.SEQ),
        "support_per_class": int(args.support_per_class),
        "num_classes_requested": int(args.max_classes),
        "num_classes_evaluated": int(len(eligible_classes)),
        "num_queries": int(total_q),
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "num_videos_skipped": int(skipped),
        "notes": "Cosine nearest-prototype evaluation on per-video embeddings",
    }

    out_path = Path(args.report_path)
    if not out_path.is_absolute():
        out_path = tg.BASE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS")
    print("=" * 70)
    print(f"Classes evaluated: {len(eligible_classes)}")
    print(f"Queries: {total_q}")
    print(f"Top-1: {top1_acc * 100:.2f}%")
    print(f"Top-5: {top5_acc * 100:.2f}%")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
