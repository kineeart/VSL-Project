import argparse
import json
from collections import deque
from pathlib import Path

import cv2


def load_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    return frames, fps


def estimate_active_span(video_path: Path, min_energy_ratio=0.25, smoothing=3, pad_ms=120):
    frames, fps = load_video_frames(video_path)
    if not frames:
        return {
            "start_ms": 0,
            "end_ms": 0,
            "confidence": 0.0,
            "frame_count": 0,
            "reason": "empty_video",
        }

    energies = [0.0]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        energies.append(float(diff.mean()))
        prev_gray = gray

    window = deque(maxlen=max(1, smoothing))
    smooth = []
    for value in energies:
        window.append(value)
        smooth.append(sum(window) / len(window))

    max_energy = max(smooth) if smooth else 0.0
    if max_energy <= 1e-6:
        start_ms = 0
        end_ms = int(len(frames) / fps * 1000.0)
        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": 0.0,
            "frame_count": len(frames),
            "reason": "zero_motion",
        }

    threshold = max_energy * float(min_energy_ratio)
    active_indices = [i for i, value in enumerate(smooth) if value >= threshold]
    if not active_indices:
        start_ms = 0
        end_ms = int(len(frames) / fps * 1000.0)
        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": 0.0,
            "frame_count": len(frames),
            "reason": "no_active_region",
        }

    start_idx = active_indices[0]
    end_idx = active_indices[-1]

    pad_frames = max(1, int(round(pad_ms / 1000.0 * fps)))
    start_idx = max(0, start_idx - pad_frames)
    end_idx = min(len(frames) - 1, end_idx + pad_frames)

    start_ms = int(round(start_idx / fps * 1000.0))
    end_ms = int(round((end_idx + 1) / fps * 1000.0))
    confidence = len(active_indices) / max(1, len(frames))

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": round(float(confidence), 4),
        "frame_count": len(frames),
        "max_energy": float(max_energy),
        "threshold": float(threshold),
        "reason": "motion_window",
    }


def trim_video(video_path: Path, out_path: Path, start_ms: int, end_ms: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False

    start_frame = max(0, int(round(start_ms / 1000.0 * fps)))
    end_frame = max(start_frame, int(round(end_ms / 1000.0 * fps)))
    current = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start_frame <= current < end_frame:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        current += 1

    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Detect active sign span in a video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--trimmed-video", default="", help="Optional trimmed video output path")
    parser.add_argument("--min-energy-ratio", type=float, default=0.25, help="Threshold ratio relative to peak motion")
    parser.add_argument("--smoothing", type=int, default=3, help="Moving average window for motion energy")
    parser.add_argument("--pad-ms", type=int, default=120, help="Padding before/after active region")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    span = estimate_active_span(
        input_path,
        min_energy_ratio=args.min_energy_ratio,
        smoothing=args.smoothing,
        pad_ms=args.pad_ms,
    )

    result = {
        "input": str(input_path).replace("\\", "/"),
        "output": str(Path(args.trimmed_video)).replace("\\", "/") if args.trimmed_video else "",
        "active_span": span,
    }

    if args.trimmed_video:
        trim_ok = trim_video(input_path, Path(args.trimmed_video), span["start_ms"], span["end_ms"])
        result["trimmed_video_written"] = bool(trim_ok)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()