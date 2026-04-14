import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import openpyxl


def slugify_label(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = txt.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "unk"


def read_data_mapping(data_xlsx: Path, videos_dir: Path):
    if not data_xlsx.exists():
        raise FileNotFoundError(f"Data.xlsx not found: {data_xlsx}")

    wb = openpyxl.load_workbook(data_xlsx, read_only=True)
    ws = wb.active

    label_to_files = defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0] or not row[1]:
            continue
        fn = str(row[0]).replace(".webm", ".mp4")
        lb = str(row[1]).strip()
        vp = videos_dir / fn
        if vp.exists():
            label_to_files[lb].append(vp)

    wb.close()
    return label_to_files


def load_allowed_labels(labels_json_path: Path):
    if not labels_json_path:
        return None
    if not labels_json_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_json_path}")
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def infer_signer_id(video_name: str):
    stem = Path(video_name).stem
    if stem and stem[-1] in ("B", "N", "T"):
        return stem[-1]
    return "U"


def infer_signer_id_with_fallback(video_name: str, pseudo_signer_count: int = 6):
    """Infer signer id from filename; fallback to pseudo signer buckets for non-B/N/T files.

    This helps create signer-diverse synthetic sets when many clips do not encode signer suffix.
    """
    sid = infer_signer_id(video_name)
    if sid != "U":
        return sid

    stem = Path(video_name).stem
    m = re.search(r"(\d+)", stem)
    if m is None:
        return "P0"
    code = int(m.group(1))
    bucket = code % max(1, pseudo_signer_count)
    return f"P{bucket}"


def dialect_from_signer(signer_id: str):
    """Assign coarse dialect label from signer group.

    - B/N/T map naturally to north/central/south
    - Pseudo signers P0..Pk are cyclically mapped to north/central/south
    """
    if signer_id == "B":
        return "north"
    if signer_id == "N":
        return "central"
    if signer_id == "T":
        return "south"

    if signer_id.startswith("P"):
        try:
            idx = int(signer_id[1:])
        except ValueError:
            idx = 0
        cycle = ["north", "central", "south"]
        return cycle[idx % len(cycle)]

    return "unknown"


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 0:
        frame_count = 0
    duration_ms = int((frame_count / fps) * 1000.0) if fps > 0 else 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
    }


def concatenate_videos(source_paths, output_path: Path):
    infos = [get_video_info(p) for p in source_paths]
    infos = [x for x in infos if x is not None]
    if not infos:
        return None

    fps = infos[0]["fps"] if infos[0]["fps"] > 0 else 25.0
    width = infos[0]["width"]
    height = infos[0]["height"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        return None

    segment_durations = []
    for p in source_paths:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            segment_durations.append(0)
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_count += 1
        cap.release()
        segment_durations.append(int((frame_count / fps) * 1000.0))

    writer.release()
    return segment_durations


def assign_split(rng: random.Random, train_ratio: float, val_ratio: float):
    r = rng.random()
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def assign_splits_by_signer(samples, train_ratio, val_ratio, test_ratio, rng: random.Random):
    signer_groups = defaultdict(list)
    for i, s in enumerate(samples):
        signer_groups[s.get("signer_id", "unknown")].append(i)

    signers = list(signer_groups.keys())
    rng.shuffle(signers)

    signer_dialect = {}
    for sid, idxs in signer_groups.items():
        dcount = Counter(samples[i].get("dialect", "unknown") for i in idxs)
        signer_dialect[sid] = dcount.most_common(1)[0][0] if dcount else "unknown"

    total_samples = len(samples)
    targets = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * test_ratio,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    signer_to_split = {}
    split_dialects = {"train": set(), "val": set(), "test": set()}

    for sid in signers:
        size = len(signer_groups[sid])
        d = signer_dialect.get(sid, "unknown")

        def split_score(x):
            # Main target: close to desired sample count.
            count_cost = abs((counts[x] + size) - targets[x])
            # Encourage each split to cover more dialects.
            dialect_penalty = 0.0 if d in split_dialects[x] else 0.35
            return count_cost + dialect_penalty

        best_split = min(("train", "val", "test"), key=split_score)
        signer_to_split[sid] = best_split
        counts[best_split] += size
        split_dialects[best_split].add(d)

    # Keep all splits non-empty when feasible.
    if len(signers) >= 3:
        for split in ("train", "val", "test"):
            if counts[split] > 0:
                continue
            donor = max(("train", "val", "test"), key=lambda x: counts[x])
            donor_signers = [s for s in signers if signer_to_split[s] == donor]
            if not donor_signers:
                continue
            donor_sid = max(donor_signers, key=lambda x: len(signer_groups[x]))
            signer_to_split[donor_sid] = split
            counts[split] += len(signer_groups[donor_sid])
            counts[donor] -= len(signer_groups[donor_sid])
            split_dialects[split].add(signer_dialect.get(donor_sid, "unknown"))

    for sid, idxs in signer_groups.items():
        split = signer_to_split.get(sid, "train")
        for i in idxs:
            samples[i]["split"] = split

    return counts


def build_sentence_dataset(
    label_to_files,
    num_sentences,
    min_words,
    max_words,
    train_ratio,
    val_ratio,
    test_ratio,
    rng,
    output_videos_dir,
    render_videos,
    split_policy,
    pseudo_signer_count,
):
    labels = [lb for lb, files in label_to_files.items() if files]
    if len(labels) == 0:
        raise ValueError("No labels with available videos")

    samples = []
    for idx in range(num_sentences):
        word_count = rng.randint(min_words, max_words)
        chosen_labels = [rng.choice(labels) for _ in range(word_count)]
        chosen_paths = [rng.choice(label_to_files[lb]) for lb in chosen_labels]

        sample_id = f"SYN_SENT_{idx + 1:05d}"
        if render_videos:
            out_video = output_videos_dir / f"{sample_id}.mp4"
            durations = concatenate_videos(chosen_paths, out_video)
            video_path = str(out_video).replace("\\", "/")
            if durations is None:
                durations = [get_video_info(p)["duration_ms"] if get_video_info(p) else 0 for p in chosen_paths]
        else:
            durations = [get_video_info(p)["duration_ms"] if get_video_info(p) else 0 for p in chosen_paths]
            video_path = "|".join(str(p).replace("\\", "/") for p in chosen_paths)

        segments = []
        t0 = 0
        for lb, dur in zip(chosen_labels, durations):
            dur = max(200, int(dur) if dur is not None else 200)
            segments.append({
                "gloss": slugify_label(lb),
                "start_ms": t0,
                "end_ms": t0 + dur,
            })
            t0 += dur

        signer_candidates = [
            infer_signer_id_with_fallback(p.name, pseudo_signer_count=pseudo_signer_count)
            for p in chosen_paths
        ]
        signer_id = signer_candidates[0] if signer_candidates else "U"
        dialect = dialect_from_signer(signer_id)

        split = "train"
        if split_policy == "sample":
            split = assign_split(rng, train_ratio, val_ratio)

        sample = {
            "sample_id": sample_id,
            "video_path": video_path,
            "split": split,
            "signer_id": signer_id,
            "dialect": dialect,
            "recording_env": "unknown",
            "sentence_text": " ".join(chosen_labels),
            "sentence_gloss": [slugify_label(lb) for lb in chosen_labels],
            "segments": segments,
            "notes": "synthetic sentence generated by concatenating isolated sign videos",
        }
        samples.append(sample)

    if split_policy == "signer":
        assign_splits_by_signer(samples, train_ratio, val_ratio, test_ratio, rng)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sentence-level dataset from Data.xlsx and existing videos")
    parser.add_argument("--data-xlsx", default="Data.xlsx", help="Path to Data.xlsx")
    parser.add_argument("--videos-dir", default="Videos", help="Directory containing source videos")
    parser.add_argument("--labels-json", default="", help="Optional labels.json to restrict label set")
    parser.add_argument("--output-json", default="benchmark/sentence_level/data/sentence_dataset.synthetic.json")
    parser.add_argument("--output-videos-dir", default="Videos/sentence_synth")
    parser.add_argument("--num-sentences", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-videos", action="store_true", help="Physically concatenate clips into sentence videos")
    parser.add_argument("--split-policy", default="signer", choices=["signer", "sample"], help="Split assignment policy")
    parser.add_argument("--pseudo-signer-count", type=int, default=6, help="Number of pseudo signer buckets for non-B/N/T files")
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to > 0")
    train_ratio = args.train_ratio / total_ratio
    val_ratio = args.val_ratio / total_ratio
    test_ratio = args.test_ratio / total_ratio

    rng = random.Random(args.seed)
    data_xlsx = Path(args.data_xlsx)
    videos_dir = Path(args.videos_dir)
    labels_json = Path(args.labels_json) if args.labels_json else None

    label_to_files = read_data_mapping(data_xlsx, videos_dir)

    allowed = load_allowed_labels(labels_json) if labels_json else None
    if allowed is not None:
        label_to_files = {lb: files for lb, files in label_to_files.items() if lb in allowed}

    output_json = Path(args.output_json)
    output_videos_dir = Path(args.output_videos_dir)

    samples = build_sentence_dataset(
        label_to_files=label_to_files,
        num_sentences=args.num_sentences,
        min_words=args.min_words,
        max_words=args.max_words,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
        output_videos_dir=output_videos_dir,
        render_videos=args.render_videos,
        split_policy=args.split_policy,
        pseudo_signer_count=args.pseudo_signer_count,
    )

    dataset = {
        "version": "1.0",
        "description": "Synthetic sentence-level dataset generated from isolated VSL videos in Data.xlsx",
        "generation": {
            "data_xlsx": str(data_xlsx).replace("\\", "/"),
            "videos_dir": str(videos_dir).replace("\\", "/"),
            "labels_json": str(labels_json).replace("\\", "/") if labels_json else "",
            "num_sentences": args.num_sentences,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "seed": args.seed,
            "render_videos": args.render_videos,
            "split_policy": args.split_policy,
            "pseudo_signer_count": args.pseudo_signer_count,
            "split_ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
            },
        },
        "samples": samples,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    split_counter = Counter([x["split"] for x in samples])
    signer_counter = Counter([x["signer_id"] for x in samples])
    dialect_counter = Counter([x["dialect"] for x in samples])
    label_counter = Counter()
    for s in samples:
        label_counter.update(s["sentence_gloss"])

    print("=" * 72)
    print("Synthetic Sentence Dataset Generated")
    print("=" * 72)
    print(f"output_json         : {output_json}")
    print(f"num_samples         : {len(samples)}")
    print(f"split_distribution  : {dict(split_counter)}")
    print(f"signer_distribution : {dict(signer_counter)}")
    print(f"dialect_distribution: {dict(dialect_counter)}")
    print(f"distinct_gloss      : {len(label_counter)}")
    print(f"render_videos       : {args.render_videos}")
    if args.render_videos:
        print(f"output_videos_dir   : {output_videos_dir}")


if __name__ == "__main__":
    from collections import Counter
    main()
