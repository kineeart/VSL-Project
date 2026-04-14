import argparse
import shutil
from pathlib import Path

import train_gpu as tg


def _cache_filename(video_filename: str, variant_value: str) -> str:
    return f"{Path(video_filename).stem}__{variant_value}.npy"


def _reuse_existing_cache(mapping, src_dir: Path, dst_dir: Path, variant_value: str) -> int:
    copied = 0
    for fn in mapping:
        name = _cache_filename(fn, variant_value)
        src = src_dir / name
        dst = dst_dir / name
        if src.exists() and (not dst.exists()):
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Extract landmarks only (no training), optionally reusing existing cache"
    )
    parser.add_argument(
        "--out-dir",
        default="backend/landmarks_extract_only",
        help="Output landmark directory",
    )
    parser.add_argument(
        "--reuse-from",
        default="backend/landmarks",
        help="Existing landmark cache directory to copy from before extraction",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Disable copying from existing cache",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Point train pipeline extraction to a separate landmark folder.
    tg.LANDMARKS_DIR = out_dir

    mapping = tg.load_data_mapping()
    if not mapping:
        print("[ERROR] No video-label mapping found.")
        return

    variant_value = tg.KEYPOINT_VARIANT.value
    print(f"[LM-ONLY] Variant: {variant_value}")
    print(f"[LM-ONLY] Output dir: {out_dir}")

    if not args.no_reuse:
        src_dir = Path(args.reuse_from)
        if src_dir.exists():
            copied = _reuse_existing_cache(mapping, src_dir, out_dir, variant_value)
            print(f"[LM-ONLY] Reused existing cache: {copied} files")
        else:
            print(f"[LM-ONLY] Reuse dir not found: {src_dir}")

    # Extract only missing files thanks to internal cache checks.
    tg.extract_all(mapping)
    print("[LM-ONLY] Done.")


if __name__ == "__main__":
    main()
