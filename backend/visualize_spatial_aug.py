import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from keypoint_variants import KeypointType
from spatial_augmentation import apply_spatial_augmentation, get_landmarks_bbox, get_xy_indices


def _parse_variant(name):
    try:
        return KeypointType(name)
    except ValueError as exc:
        allowed = ", ".join([v.value for v in KeypointType])
        raise ValueError(f"Unknown variant '{name}'. Allowed: {allowed}") from exc


def _plot_frame(ax, frame, variant, title):
    x_idx, y_idx = get_xy_indices(variant)
    x = frame[x_idx]
    y = frame[y_idx]
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=5, alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _draw_bbox(ax, bbox, color):
    if bbox is None:
        return
    rect = plt.Rectangle(
        (bbox["x_min"], bbox["y_min"]),
        bbox["width"],
        bbox["height"],
        fill=False,
        linewidth=2,
        edgecolor=color,
    )
    ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(description="Visualize spatial keypoint augmentation")
    parser.add_argument("--input", required=True, help="Path to input .npy sequence (shape [T, F])")
    parser.add_argument("--variant", default="holistic", help="Keypoint variant id")
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.35)
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--out", default="backend/spatial_aug_preview.png", help="Output image path")
    args = parser.parse_args()

    variant = _parse_variant(args.variant)
    seq = np.load(args.input)
    if seq.ndim != 2:
        raise ValueError(f"Expected [T, F], got shape {seq.shape}")

    aug, params = apply_spatial_augmentation(
        seq,
        variant=variant,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        return_params=True,
    )

    frame_idx = max(0, min(args.frame, seq.shape[0] - 1))
    bbox_before = get_landmarks_bbox(seq, variant)
    bbox_after = get_landmarks_bbox(aug, variant)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _plot_frame(axes[0], seq[frame_idx], variant, "Before")
    _draw_bbox(axes[0], bbox_before, "tab:red")
    _plot_frame(axes[1], aug[frame_idx], variant, "After")
    _draw_bbox(axes[1], bbox_after, "tab:green")

    subtitle = (
        f"scale={params.get('scale', 1.0):.3f}, dx={params.get('dx', 0.0):.3f}, "
        f"dy={params.get('dy', 0.0):.3f}"
    )
    fig.suptitle(subtitle)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()
