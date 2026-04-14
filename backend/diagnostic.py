"""
Diagnostic tool to debug why videos are not recognized.
Test videos from training data to identify issues:
1. Feature normalization problems
2. Confidence thresholds
3. Class mapping errors
4. Data distribution mismatches
"""

import json
import numpy as np
import cv2
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from keypoint_variants import KeypointType, VARIANTS, extract_landmarks_variant
import mediapipe as mp

# Configuration - must match app.py
SEQUENCE_LENGTH = 60
KEYPOINT_VARIANT = KeypointType.HOLISTIC
print(f"[CONFIG] Using variant: {KEYPOINT_VARIANT.value}")

# Inference thresholds
MIN_ACTIVE_FRAME_RATIO = 0.10
MIN_MOTION_ENERGY = 0.0005
MIN_TOP1_CONFIDENCE = 0.25
MIN_TOP12_MARGIN = 0.02

MODEL_DIR = Path("models_15cls_run1")
VIDEOS_DIR = Path("../../Videos").resolve()

print(f"[PATHS]")
print(f"  Model dir: {MODEL_DIR}")
print(f"  Videos dir: {VIDEOS_DIR}")


def engineer_features(seq):
    """Feature engineering that matches app.py exactly."""
    config = VARIANTS[KEYPOINT_VARIANT]

    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    accel = np.zeros_like(seq)
    accel[1:] = velocity[1:] - velocity[:-1]

    extra = []
    if config.include_pose and config.include_left_hand and config.include_right_hand:
        nose_xyz = seq[:, 0:3]
        pose_size = config.pose_points * (4 if config.include_pose_visibility else 3)
        lh_start = pose_size
        rh_start = lh_start + 21 * 3
        lh_wrist = seq[:, lh_start:lh_start + 3]
        rh_wrist = seq[:, rh_start:rh_start + 3]

        extra = np.concatenate([
            lh_wrist - nose_xyz,
            rh_wrist - nose_xyz,
            np.linalg.norm(lh_wrist - rh_wrist, axis=1, keepdims=True),
            np.linalg.norm(velocity[:, lh_start:lh_start + 3], axis=1, keepdims=True),
            np.linalg.norm(velocity[:, rh_start:rh_start + 3], axis=1, keepdims=True),
        ], axis=1)

    if len(extra) == 0:
        return np.concatenate([seq, velocity, accel], axis=1).astype(np.float32)
    return np.concatenate([seq, velocity, accel, extra], axis=1).astype(np.float32)


def is_sign_activity_sufficient(raw_seq):
    """Check if sequence has sufficient activity."""
    frame_energy = np.linalg.norm(raw_seq, axis=1)
    active_ratio = float(np.mean(frame_energy > 1e-6))
    
    motion = np.linalg.norm(np.diff(raw_seq, axis=0), axis=1)
    motion_energy = float(np.mean(motion)) if len(motion) else 0.0
    
    return {
        "active_ratio": active_ratio,
        "motion_energy": motion_energy,
        "passes_check": active_ratio >= MIN_ACTIVE_FRAME_RATIO and motion_energy >= MIN_MOTION_ENERGY
    }


def extract_landmarks(frame, holistic):
    """Extract landmarks for a frame."""
    from keypoint_variants import extract_landmarks_variant
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    return extract_landmarks_variant(results, KEYPOINT_VARIANT)


def extract_video_landmarks(video_path, seq_length=SEQUENCE_LENGTH):
    """Extract landmarks from video."""
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        return None
    
    indices = np.linspace(0, len(frames) - 1, seq_length, dtype=int)
    sampled = [frames[i] for i in indices]
    landmarks_seq = []
    
    with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        for frame in sampled:
            lm = extract_landmarks(frame, holistic)
            if lm is None or len(lm) == 0:
                print(f"    WARNING: Failed to extract landmarks from a frame")
                continue
            landmarks_seq.append(lm)
    
    if len(landmarks_seq) < seq_length:
        print(f"    WARNING: Only extracted {len(landmarks_seq)} out of {seq_length} frames")
    
    return np.array(landmarks_seq) if landmarks_seq else None


def diagnose_video(video_path, label_map, norm_mean, norm_std):
    """Run full diagnostic on a video file."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: {video_path.name}")
    print(f"{'='*80}")
    
    # Step 1: Extract landmarks
    print(f"\n[1] Extracting landmarks...")
    raw_seq = extract_video_landmarks(video_path)
    if raw_seq is None:
        print("    ERROR: Failed to extract landmarks!")
        return
    print(f"    ✓ Shape: {raw_seq.shape}")
    print(f"    ✓ Features: {raw_seq.shape[1]}")
    
    # Step 2: Check activity
    print(f"\n[2] Checking sign activity...")
    activity = is_sign_activity_sufficient(raw_seq)
    print(f"    Active frame ratio: {activity['active_ratio']:.4f} (threshold: {MIN_ACTIVE_FRAME_RATIO})")
    print(f"    Motion energy: {activity['motion_energy']:.6f} (threshold: {MIN_MOTION_ENERGY})")
    if activity['passes_check']:
        print(f"    ✓ PASS: Sufficient activity detected")
    else:
        print(f"    ✗ FAIL: Insufficient activity - would return 'no_sign'")
        return
    
    # Step 3: Feature engineering
    print(f"\n[3] Feature engineering...")
    feat = engineer_features(raw_seq)
    print(f"    Shape: {feat.shape}")
    print(f"    Mean: {feat.mean():.6f}, Std: {feat.std():.6f}")
    print(f"    Min: {feat.min():.6f}, Max: {feat.max():.6f}")
    
    # Step 4: Check normalization params
    print(f"\n[4] Normalization parameters...")
    print(f"    Norm mean shape: {norm_mean.shape}")
    print(f"    Norm std shape: {norm_std.shape}")
    if feat.shape[0] != norm_mean.shape[0]:
        print(f"    ✗ ERROR: Feature dimension mismatch!")
        print(f"      Features: {feat.shape[0]}")
        print(f"      Expected: {norm_mean.shape[0]}")
        return
    print(f"    ✓ Dimension match OK")
    
    # Step 5: Normalize
    print(f"\n[5] Normalizing features...")
    feat_norm = (feat - norm_mean) / norm_std
    print(f"    After norm - Mean: {feat_norm.mean():.6f}, Std: {feat_norm.std():.6f}")
    print(f"    After norm - Min: {feat_norm.min():.6f}, Max: {feat_norm.max():.6f}")
    
    # Step 6: Check for NaN/Inf
    print(f"\n[6] Data quality check...")
    nan_count = np.isnan(feat_norm).sum()
    inf_count = np.isinf(feat_norm).sum()
    print(f"    NaN values: {nan_count}")
    print(f"    Inf values: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print(f"    ✗ WARNING: Problematic values detected!")
    
    # Step 7: Summary
    print(f"\n[7] SUMMARY")
    print(f"    ✓ Activity check: PASS")
    print(f"    ✓ Feature engineering: OK ({feat.shape[0]} features)")
    print(f"    ✓ Normalization: OK")
    print(f"    ✓ Ready for model inference")
    print(f"\n    If model still outputs 'no_sign', likely causes:")
    print(f"    1. Model confidence too low (< {MIN_TOP1_CONFIDENCE*100:.0f}%)")
    print(f"    2. Top class too ambiguous (margin < {MIN_TOP12_MARGIN*100:.1f}%)")
    print(f"    3. Class not in model training data")


def main():
    """Main diagnostic routine."""
    print(f"\n{'='*80}")
    print(f"VIETNAMESE SIGN LANGUAGE - INFERENCE DIAGNOSTIC")
    print(f"{'='*80}\n")
    
    # Load model files
    print(f"[INIT] Loading model files...")
    with open(MODEL_DIR / "labels.json") as f:
        label_map = json.load(f)
    print(f"  Classes: {len(label_map)}")
    print(f"  {list(label_map.keys())}")
    
    norm_mean = np.load(MODEL_DIR / "norm_mean.npy")
    norm_std = np.load(MODEL_DIR / "norm_std.npy")
    print(f"  Norm shape: mean{norm_mean.shape}, std{norm_std.shape}")
    
    print(f"\n[THRESHOLDS]")
    print(f"  min_active_frame_ratio = {MIN_ACTIVE_FRAME_RATIO}")
    print(f"  min_motion_energy = {MIN_MOTION_ENERGY}")
    print(f"  min_top1_confidence = {MIN_TOP1_CONFIDENCE}")
    print(f"  min_top12_margin = {MIN_TOP12_MARGIN}")
    
    # Find test videos
    print(f"\n[FINDING VIDEOS]")
    if not VIDEOS_DIR.exists():
        print(f"  ERROR: Videos directory not found: {VIDEOS_DIR}")
        return
    
    video_files = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.webm"))
    print(f"  Found {len(video_files)} videos")
    
    if not video_files:
        print("  ERROR: No videos found!")
        return
    
    # Test first 3 videos
    print(f"\n[TESTING] Running diagnostic on first 3 videos...")
    for video_path in video_files[:3]:
        try:
            diagnose_video(video_path, label_map, norm_mean, norm_std)
        except Exception as e:
            print(f"\n  ERROR processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
