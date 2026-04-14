import argparse
from pathlib import Path

import cv2
import numpy as np

from keypoint_variants import KeypointType, VARIANTS
from spatial_augmentation import apply_spatial_augmentation


def parse_variant(name: str) -> KeypointType:
    try:
        return KeypointType(name)
    except ValueError as exc:
        allowed = ", ".join([v.value for v in KeypointType])
        raise ValueError(f"Unknown variant '{name}'. Allowed: {allowed}") from exc


def component_offsets(variant: KeypointType):
    """Return component ranges as (name, start_idx, end_idx_exclusive, channels)."""
    config = VARIANTS[variant]
    ranges = []
    offset = 0

    if config.include_pose:
        pose_channels = 4 if config.include_pose_visibility else 3
        pose_size = config.pose_points * pose_channels
        ranges.append(("pose", offset, offset + pose_size, pose_channels))
        offset += pose_size

    if config.include_left_hand:
        size = 21 * 3
        ranges.append(("left_hand", offset, offset + size, 3))
        offset += size

    if config.include_right_hand:
        size = 21 * 3
        ranges.append(("right_hand", offset, offset + size, 3))
        offset += size

    if config.include_face:
        size = 468 * 3
        ranges.append(("face", offset, offset + size, 3))
        offset += size

    return ranges


def draw_points(canvas, frame_vec, variant: KeypointType):
    h, w = canvas.shape[:2]

    colors = {
        "pose": (60, 180, 255),
        "left_hand": (50, 200, 80),
        "right_hand": (255, 140, 70),
        "face": (190, 120, 255),
    }

    for name, st, ed, channels in component_offsets(variant):
        block = frame_vec[st:ed]
        if block.size == 0:
            continue

        pts = block.reshape(-1, channels)
        xs = pts[:, 0]
        ys = pts[:, 1]

        valid = np.isfinite(xs) & np.isfinite(ys)
        valid = valid & ((np.abs(xs) > 1e-8) | (np.abs(ys) > 1e-8))
        if not np.any(valid):
            continue

        xs = np.clip(xs[valid], 0.0, 1.0)
        ys = np.clip(ys[valid], 0.0, 1.0)

        px = (xs * (w - 1)).astype(np.int32)
        py = (ys * (h - 1)).astype(np.int32)

        radius = 1 if name == "face" else 3
        color = colors.get(name, (50, 50, 50))
        for x, y in zip(px, py):
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def augment_sequence(seq, variant: KeypointType, mode: str, rng: np.random.Generator):
    out = seq.copy()

    if mode == "spatial":
        return apply_spatial_augmentation(out, variant=variant, scale_min=0.75, scale_max=1.45, rng=rng)

    if mode == "full":
        # 1) light random noise
        sigma = float(rng.uniform(0.004, 0.018))
        out = out + rng.normal(0.0, sigma, out.shape)

        # 2) sequence-level spatial transform
        out = apply_spatial_augmentation(out, variant=variant, scale_min=0.75, scale_max=1.45, rng=rng)

        # 3) temporal shift
        shift = int(rng.integers(-5, 6))
        out = np.roll(out, shift=shift, axis=0)

        # 4) speed variation then resample back to original length
        sp = float(rng.uniform(0.82, 1.18))
        n_new = max(3, int(round(out.shape[0] * sp)))
        ix = np.linspace(0, out.shape[0] - 1, n_new)
        warped = np.array([out[min(int(round(i)), out.shape[0] - 1)] for i in ix], dtype=np.float32)
        back_ix = np.linspace(0, warped.shape[0] - 1, out.shape[0])
        out = np.array([warped[min(int(round(i)), warped.shape[0] - 1)] for i in back_ix], dtype=np.float32)

        # 5) random frame dropout
        n_drop = int(rng.integers(1, 4))
        di = rng.choice(out.shape[0], size=n_drop, replace=False)
        out[di] = 0.0

        # 6) temporal reverse (sometimes)
        if rng.random() < 0.35:
            out = out[::-1].copy()

        return out.astype(np.float32)

    raise ValueError(f"Unknown mode: {mode}")


def render_comparison_video(seq_orig, seq_aug, variant, out_path: Path, fps: int):
    t = min(seq_orig.shape[0], seq_aug.shape[0])
    w, h = 1280, 720
    panel_w = w // 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for: {out_path}")

    for i in range(t):
        frame = np.full((h, w, 3), 248, dtype=np.uint8)

        left = frame[:, :panel_w]
        right = frame[:, panel_w:]

        draw_points(left, seq_orig[i], variant)
        draw_points(right, seq_aug[i], variant)

        cv2.putText(frame, "Original", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(frame, "Augmented", (panel_w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Frame {i+1}/{t}",
            (w // 2 - 90, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )

        cv2.line(frame, (panel_w, 0), (panel_w, h), (200, 200, 200), 2)
        writer.write(frame)

    writer.release()


def collect_landmark_files(landmarks_dir: Path, variant: KeypointType):
    pattern = f"*__{variant.value}.npy"
    files = sorted(landmarks_dir.glob(pattern))
    if files:
        return files

    # Backward compatibility: old cache naming without variant suffix
    return sorted(landmarks_dir.glob("*.mp4.npy"))


def main():
    parser = argparse.ArgumentParser(description="Export original vs augmented keypoint preview videos")
    parser.add_argument("--landmarks-dir", default="backend/landmarks")
    parser.add_argument("--out-dir", default="backend/keypoint_previews")
    parser.add_argument("--variant", default="holistic")
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mode", choices=["spatial", "full"], default="full")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    variant = parse_variant(args.variant)
    landmarks_dir = Path(args.landmarks_dir)
    out_dir = Path(args.out_dir)

    if not landmarks_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {landmarks_dir}")

    files = collect_landmark_files(landmarks_dir, variant)
    if not files:
        raise FileNotFoundError(f"No landmark .npy files found in {landmarks_dir}")

    rng = np.random.default_rng(args.seed)
    selected = files[: max(1, args.max_videos)]

    print(f"Variant: {variant.value}")
    print(f"Mode: {args.mode}")
    print(f"Input files: {len(selected)}")

    exported = []
    for p in selected:
        seq = np.load(p)
        if seq.ndim != 2:
            print(f"[SKIP] {p.name}: invalid shape {seq.shape}")
            continue

        aug = augment_sequence(seq, variant, args.mode, rng)
        out_name = f"{p.stem}__{args.mode}_preview.mp4"
        out_path = out_dir / out_name
        render_comparison_video(seq, aug, variant, out_path, fps=args.fps)
        exported.append(out_path)
        print(f"[OK] {out_path}")

    if not exported:
        print("No preview videos were exported.")
        return

    print("\nExported preview videos:")
    for p in exported:
        print(f" - {p}")


if __name__ == "__main__":
    main()
