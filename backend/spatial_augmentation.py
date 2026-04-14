import numpy as np

from keypoint_variants import KeypointType, VARIANTS


def get_xy_indices(variant):
    """Return flattened x/y indices for a keypoint variant layout."""
    config = VARIANTS[variant]
    x_idx = []
    y_idx = []
    offset = 0

    if config.include_pose:
        pose_channels = 4 if config.include_pose_visibility else 3
        for i in range(config.pose_points):
            base = offset + i * pose_channels
            x_idx.append(base)
            y_idx.append(base + 1)
        offset += config.pose_points * pose_channels

    if config.include_left_hand:
        for i in range(21):
            base = offset + i * 3
            x_idx.append(base)
            y_idx.append(base + 1)
        offset += 21 * 3

    if config.include_right_hand:
        for i in range(21):
            base = offset + i * 3
            x_idx.append(base)
            y_idx.append(base + 1)
        offset += 21 * 3

    if config.include_face:
        for i in range(468):
            base = offset + i * 3
            x_idx.append(base)
            y_idx.append(base + 1)

    return np.array(x_idx, dtype=np.int64), np.array(y_idx, dtype=np.int64)


def get_landmarks_bbox(landmarks, variant, ignore_zeros=True, eps=1e-8):
    """Compute bbox over all frames and all points in sequence coordinates [0, 1]."""
    x_idx, y_idx = get_xy_indices(variant)
    if x_idx.size == 0:
        return None

    x = landmarks[:, x_idx]
    y = landmarks[:, y_idx]
    finite = np.isfinite(x) & np.isfinite(y)

    if ignore_zeros:
        non_zero = (np.abs(x) > eps) | (np.abs(y) > eps)
        valid = finite & non_zero
    else:
        valid = finite

    if not np.any(valid):
        valid = finite
    if not np.any(valid):
        return None

    x_valid = x[valid]
    y_valid = y[valid]
    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": max(x_max - x_min, 1e-8),
        "height": max(y_max - y_min, 1e-8),
        "cx": (x_min + x_max) * 0.5,
        "cy": (y_min + y_max) * 0.5,
    }


def normalize_landmarks_to_bbox(landmarks, variant, bbox=None):
    """Normalize x/y landmarks into [0,1] in the local bbox space."""
    if bbox is None:
        bbox = get_landmarks_bbox(landmarks, variant)
    if bbox is None:
        return landmarks.copy(), None

    x_idx, y_idx = get_xy_indices(variant)
    out = landmarks.copy()
    out[:, x_idx] = (out[:, x_idx] - bbox["x_min"]) / bbox["width"]
    out[:, y_idx] = (out[:, y_idx] - bbox["y_min"]) / bbox["height"]
    return out, bbox


def scale_landmarks(landmarks, variant, scale, center=None):
    """Uniformly scale all x/y points around center while preserving relative shape."""
    out = landmarks.copy()
    x_idx, y_idx = get_xy_indices(variant)
    if x_idx.size == 0:
        return out

    if center is None:
        bbox = get_landmarks_bbox(out, variant, ignore_zeros=False)
        if bbox is None:
            return out
        cx = bbox["cx"]
        cy = bbox["cy"]
    else:
        cx, cy = center

    out[:, x_idx] = cx + (out[:, x_idx] - cx) * scale
    out[:, y_idx] = cy + (out[:, y_idx] - cy) * scale
    return out


def translate_landmarks(landmarks, variant, dx, dy):
    """Translate all x/y points by (dx, dy)."""
    out = landmarks.copy()
    x_idx, y_idx = get_xy_indices(variant)
    if x_idx.size == 0:
        return out
    out[:, x_idx] = out[:, x_idx] + dx
    out[:, y_idx] = out[:, y_idx] + dy
    return out


def clip_landmarks(landmarks, variant, lo=0.0, hi=1.0):
    """Clip x/y points to visible image range."""
    out = landmarks.copy()
    x_idx, y_idx = get_xy_indices(variant)
    if x_idx.size == 0:
        return out
    out[:, x_idx] = np.clip(out[:, x_idx], lo, hi)
    out[:, y_idx] = np.clip(out[:, y_idx], lo, hi)
    return out


def _sample_scale_translation(bbox, scale_min, scale_max, rng):
    width = bbox["width"]
    height = bbox["height"]
    cx = bbox["cx"]
    cy = bbox["cy"]

    max_fit = min(1.0 / width, 1.0 / height)
    max_allowed = max(1e-6, min(scale_max, max_fit))
    min_allowed = max(1e-6, min(scale_min, max_allowed))

    scale = float(rng.uniform(min_allowed, max_allowed))
    scaled_w = width * scale
    scaled_h = height * scale

    scaled_x_min = cx - scaled_w * 0.5
    scaled_x_max = cx + scaled_w * 0.5
    scaled_y_min = cy - scaled_h * 0.5
    scaled_y_max = cy + scaled_h * 0.5

    tx_min = -scaled_x_min
    tx_max = 1.0 - scaled_x_max
    ty_min = -scaled_y_min
    ty_max = 1.0 - scaled_y_max

    dx = float(rng.uniform(tx_min, tx_max)) if tx_max >= tx_min else 0.0
    dy = float(rng.uniform(ty_min, ty_max)) if ty_max >= ty_min else 0.0

    return scale, dx, dy


def apply_spatial_augmentation(
    seq,
    variant=KeypointType.HOLISTIC,
    scale_min=0.75,
    scale_max=1.35,
    rng=None,
    return_params=False,
):
    """
    Spatial augmentation with global bbox scaling + translation.

    The same transform is applied to the entire sequence so temporal motion remains valid.
    Output is guaranteed to stay inside [0,1] for x/y coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = seq.copy()
    bbox = get_landmarks_bbox(out, variant)
    if bbox is None:
        if return_params:
            return out.astype(np.float32), {
                "applied": False,
                "reason": "empty_bbox",
                "scale": 1.0,
                "dx": 0.0,
                "dy": 0.0,
            }
        return out.astype(np.float32)

    scale, dx, dy = _sample_scale_translation(bbox, scale_min, scale_max, rng)
    out = scale_landmarks(out, variant, scale, center=(bbox["cx"], bbox["cy"]))
    out = translate_landmarks(out, variant, dx, dy)
    out = clip_landmarks(out, variant)

    if return_params:
        return out.astype(np.float32), {
            "applied": True,
            "scale": float(scale),
            "dx": float(dx),
            "dy": float(dy),
            "bbox_before": bbox,
            "bbox_after": get_landmarks_bbox(out, variant, ignore_zeros=False),
        }

    return out.astype(np.float32)
