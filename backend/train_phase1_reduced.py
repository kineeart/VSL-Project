"""Phase 1: Reduced-class classification benchmark.

This script reuses train_gpu.py but applies a stable reduced setting:
- top-K classes only
- minimum real videos per class
- lower oversampling target
- lightweight keypoint variant by default (no face)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

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


def make_filtered_loader(max_classes: int, min_samples: int):
    original_loader = tg.load_data_mapping

    def _filtered_loader():
        mapping = original_loader()
        counts = Counter(mapping.values())
        ranked = [(lb, cnt) for lb, cnt in counts.items() if cnt >= min_samples]
        ranked.sort(key=lambda x: (-x[1], str(x[0])))

        if not ranked:
            raise RuntimeError(
                f"No class has >= {min_samples} videos. "
                "Reduce --min-samples or inspect Data.xlsx labels."
            )

        selected = ranked[:max_classes]
        selected_labels = {lb for lb, _ in selected}
        filtered = {fn: lb for fn, lb in mapping.items() if lb in selected_labels}

        print("\n[PHASE1] Reduced-class filtering")
        print(f"[PHASE1] Requested max classes: {max_classes}")
        print(f"[PHASE1] Min real videos/class: {min_samples}")
        print(f"[PHASE1] Selected classes: {len(selected_labels)}")
        print(f"[PHASE1] Retained videos: {len(filtered)}")

        preview = selected[:10]
        if preview:
            print("[PHASE1] Top class counts (first 10):")
            for lb, cnt in preview:
                print(f"  - {lb}: {cnt}")

        return filtered

    return _filtered_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 reduced-class benchmark")
    parser.add_argument("--max-classes", type=int, default=40)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--target-per-class", type=int, default=20)
    parser.add_argument("--variant", type=str, default=KeypointType.HOLISTIC_NO_FACE.value)
    parser.add_argument("--seq", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-subdir", type=str, default="phase1_reduced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_variant(args.variant)
    tg.SEQ = args.seq
    tg.EPOCHS = args.epochs
    tg.BS = args.batch_size
    tg.NW = max(1, args.workers)

    tg.MIN_SAMPLES = args.min_samples
    tg.TARGET_SAMPLES_PER_CLASS = args.target_per_class

    tg.MODEL_DIR = tg.MODEL_DIR / args.output_subdir
    tg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tg.load_data_mapping = make_filtered_loader(args.max_classes, args.min_samples)

    print("=" * 70)
    print("PHASE 1 - REDUCED CLASS BENCHMARK")
    print("=" * 70)
    print(f"Variant: {tg.KEYPOINT_VARIANT.value} ({tg.VARIANT_CONFIG.name})")
    print(f"Features: raw={tg.NRF}, engineered={tg.NF}")
    print(f"SEQ={tg.SEQ}, BS={tg.BS}, Epochs={tg.EPOCHS}")
    print(f"Output: {tg.MODEL_DIR}")

    tg.train()


if __name__ == "__main__":
    main()
