import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import openpyxl

from detect_active_span import estimate_active_span, trim_video


def slugify_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def infer_signer_id_from_name(video_name: str) -> str:
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


def read_label_mapping(data_xlsx: Path, videos_dir: Path):
    wb = openpyxl.load_workbook(data_xlsx, read_only=True)
    ws = wb.active
    mapping = defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0] or not row[1]:
            continue
        file_name = str(row[0]).replace(".webm", ".mp4")
        label = str(row[1]).strip()
        video_path = videos_dir / file_name
        if video_path.exists():
            mapping[label].append(video_path)
    wb.close()
    return mapping


def get_video_meta(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"duration_ms": 0, "fps": 25.0}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"duration_ms": int(round(frame_count / fps * 1000.0)), "fps": float(fps)}


def concatenate_videos(source_paths, output_path: Path):
    metas = [get_video_meta(p) for p in source_paths]
    if not metas:
        return []
    fps = metas[0]["fps"] or 25.0
    first_cap = cv2.VideoCapture(str(source_paths[0]))
    if not first_cap.isOpened():
        return []
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    durations = []

    for path in source_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            durations.append(0)
            continue
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_count += 1
        cap.release()
        durations.append(int(round(frame_count / fps * 1000.0)))

    writer.release()
    return durations


def build_dataset(
    label_to_videos,
    num_samples,
    min_units,
    max_units,
    rng,
    render_videos,
    output_videos_dir: Path,
    trim_padding_ms=120,
):
    labels = [label for label, videos in label_to_videos.items() if videos]
    if not labels:
        raise ValueError("No labels with available videos")

    samples = []
    for idx in range(num_samples):
        unit_count = rng.randint(min_units, max_units)
        chosen_labels = [rng.choice(labels) for _ in range(unit_count)]
        chosen_paths = [rng.choice(label_to_videos[label]) for label in chosen_labels]

        source_segments = []
        trimmed_paths = []
        for unit_idx, (label, path) in enumerate(zip(chosen_labels, chosen_paths)):
            active = estimate_active_span(path, pad_ms=trim_padding_ms)
            signer_id = infer_signer_id_from_name(path.name)
            source_segments.append(
                {
                    "source_video": str(path).replace("\\", "/"),
                    "label": label,
                    "signer_id": signer_id,
                    "dialect": dialect_from_signer(signer_id),
                    "unit_index": unit_idx,
                    "segment_index": unit_idx,
                    "trim_start_ms": int(active["start_ms"]),
                    "trim_end_ms": int(active["end_ms"]),
                    "source_duration_ms": int(active["end_ms"] - active["start_ms"]),
                    "active_confidence": float(active.get("confidence", 0.0)),
                }
            )

        if render_videos:
            sample_id = f"CONT_{idx + 1:05d}"
            out_video = output_videos_dir / f"{sample_id}.mp4"
            for seg in source_segments:
                tmp_path = output_videos_dir / "_tmp" / f"{sample_id}_{seg['unit_index']}.mp4"
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                trim_video(Path(seg["source_video"]), tmp_path, seg["trim_start_ms"], seg["trim_end_ms"])
                trimmed_paths.append(tmp_path)
            concatenate_videos(trimmed_paths, out_video)
            video_path = str(out_video).replace("\\", "/")
        else:
            video_path = "|".join(seg["source_video"] for seg in source_segments)

        signer_votes = Counter(seg.get("signer_id", "U") for seg in source_segments)
        signer_id = signer_votes.most_common(1)[0][0] if signer_votes else "U"
        dialect = dialect_from_signer(signer_id)

        sample = {
            "sample_id": f"CONT_{idx + 1:05d}",
            "split": "train",
            "video_path": video_path,
            "sequence_type": "continuous",
            "sentence_text": " ".join(chosen_labels),
            "sentence_gloss": [slugify_label(label) for label in chosen_labels],
            "source_segments": source_segments,
            "signer_id": signer_id,
            "dialect": dialect,
            "notes": "synthetic continuous sample generated from trimmed isolated clips",
        }
        samples.append(sample)

    split_order = ["train", "val", "test"]
    for i, sample in enumerate(samples):
        sample["split"] = split_order[i % len(split_order)] if len(samples) >= 3 else "train"

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build a continuous-recognition dataset from isolated VSL videos")
    parser.add_argument("--data-xlsx", required=True, help="Path to Data.xlsx")
    parser.add_argument("--videos-dir", required=True, help="Directory containing source videos")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-videos-dir", default="benchmark/continuous/data/rendered", help="Where rendered videos are written")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of continuous samples to generate")
    parser.add_argument("--min-units", type=int, default=2, help="Minimum number of isolated units per sample")
    parser.add_argument("--max-units", type=int, default=4, help="Maximum number of isolated units per sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-videos", action="store_true", help="Physically render concatenated videos")
    parser.add_argument("--trim-padding-ms", type=int, default=120, help="Padding applied when trimming active spans")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_xlsx = Path(args.data_xlsx)
    videos_dir = Path(args.videos_dir)
    output_json = Path(args.output_json)
    output_videos_dir = Path(args.output_videos_dir)

    if not data_xlsx.exists():
        raise FileNotFoundError(f"Data.xlsx not found: {data_xlsx}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    label_to_videos = read_label_mapping(data_xlsx, videos_dir)
    samples = build_dataset(
        label_to_videos=label_to_videos,
        num_samples=args.num_samples,
        min_units=args.min_units,
        max_units=args.max_units,
        rng=rng,
        render_videos=args.render_videos,
        output_videos_dir=output_videos_dir,
        trim_padding_ms=args.trim_padding_ms,
    )

    dataset = {
        "version": "1.0",
        "description": "Continuous VSL dataset built from trimmed isolated videos",
        "generation": {
            "data_xlsx": str(data_xlsx).replace("\\", "/"),
            "videos_dir": str(videos_dir).replace("\\", "/"),
            "num_samples": args.num_samples,
            "min_units": args.min_units,
            "max_units": args.max_units,
            "seed": args.seed,
            "render_videos": args.render_videos,
            "trim_padding_ms": args.trim_padding_ms,
        },
        "samples": samples,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    split_counts = Counter(sample["split"] for sample in samples)
    print("=" * 72)
    print("Continuous Dataset Generated")
    print("=" * 72)
    print(f"output_json    : {output_json}")
    print(f"num_samples    : {len(samples)}")
    print(f"split_counts   : {dict(split_counts)}")
    print(f"render_videos   : {args.render_videos}")
    if args.render_videos:
        print(f"render_dir     : {output_videos_dir}")


if __name__ == "__main__":
    main()