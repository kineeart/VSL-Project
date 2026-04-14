import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl


def slugify_label(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unk"


def infer_signer_id(video_name: str) -> str:
    stem = Path(video_name).stem
    if stem and stem[-1] in ("B", "N", "T"):
        return stem[-1]
    return "U"


def dialect_from_signer(signer_id: str) -> str:
    if signer_id == "B":
        return "north"
    if signer_id == "N":
        return "central"
    if signer_id == "T":
        return "south"
    return "unknown"


def build_manifest(data_xlsx: Path, videos_dir: Path):
    if not data_xlsx.exists():
        raise FileNotFoundError(f"Data.xlsx not found: {data_xlsx}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    workbook = openpyxl.load_workbook(data_xlsx, read_only=True)
    sheet = workbook.active

    samples = []
    label_counts = Counter()
    signer_counts = Counter()
    dialect_counts = Counter()
    missing_files = []

    for index, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=1):
        if not row or not row[0] or not row[1]:
            continue

        raw_name = str(row[0]).strip()
        label = str(row[1]).strip()
        video_name = raw_name.replace(".webm", ".mp4")
        video_path = videos_dir / video_name
        signer_id = infer_signer_id(video_name)
        dialect = dialect_from_signer(signer_id)

        sample = {
            "sample_id": f"ROW_{index:05d}",
            "video_file": video_name,
            "video_path": str(video_path).replace("\\", "/"),
            "label": label,
            "label_slug": slugify_label(label),
            "signer_id": signer_id,
            "dialect": dialect,
            "exists": video_path.exists(),
        }
        if not sample["exists"]:
            missing_files.append(video_name)

        samples.append(sample)
        label_counts[label] += 1
        signer_counts[signer_id] += 1
        dialect_counts[dialect] += 1

    workbook.close()

    manifest = {
        "version": "1.0",
        "description": "Canonical VSL dataset manifest built from Data.xlsx and Videos/",
        "source": {
            "data_xlsx": str(data_xlsx).replace("\\", "/"),
            "videos_dir": str(videos_dir).replace("\\", "/"),
        },
        "summary": {
            "num_samples": len(samples),
            "num_labels": len(label_counts),
            "num_signers": len(signer_counts),
            "label_distribution": dict(label_counts),
            "signer_distribution": dict(signer_counts),
            "dialect_distribution": dict(dialect_counts),
            "missing_video_count": len(missing_files),
            "missing_video_files": missing_files,
        },
        "samples": samples,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build canonical VSL dataset manifest from Data.xlsx and Videos/")
    parser.add_argument("--data-xlsx", default="Data.xlsx", help="Path to Data.xlsx")
    parser.add_argument("--videos-dir", default="Videos", help="Directory containing source videos")
    parser.add_argument("--output", default="benchmark/sentence_level/data/full_dataset_manifest.json", help="Output manifest JSON path")
    args = parser.parse_args()

    data_xlsx = Path(args.data_xlsx)
    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)

    manifest = build_manifest(data_xlsx, videos_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    summary = manifest["summary"]
    print("=" * 72)
    print("Canonical VSL Dataset Manifest")
    print("=" * 72)
    print(f"output              : {output_path}")
    print(f"num_samples         : {summary['num_samples']}")
    print(f"num_labels          : {summary['num_labels']}")
    print(f"num_signers         : {summary['num_signers']}")
    print(f"dialect_distribution: {summary['dialect_distribution']}")
    print(f"missing_video_count : {summary['missing_video_count']}")


if __name__ == "__main__":
    main()
