# Keypoint Variants Documentation

## Overview

This project supports **9 different keypoint variants** to optimize for different use cases and hardware constraints. Switch between variants easily by changing a single parameter.

## Available Variants

| Variant | Size | Components | Use Case |
|---------|------|------------|----------|
| **HOLISTIC** | 1662 | Pose(33×4) + Hands(42×3) + Face(468×3) | **Default - Best accuracy** |
| **HOLISTIC_NO_FACE** | 198 | Pose(33×4) + Hands(42×3) | Faster, less VRAM |
| **HOLISTIC_NO_VIZ** | 1659 | Pose(33×3) + Hands(42×3) + Face(468×3) | Remove visibility channel |
| **HANDS_ONLY** | 126 | Hands(42×3) | Minimal features, very fast |
| **POSE_ONLY** | 132 | Pose(33×4) | Body movement focus |
| **HANDS_POSE** | 198 | Pose(33×4) + Hands(42×3) | Balanced: hands + body |
| **UPPER_BODY** | 1524 | Pose(16×4) + Hands(42×3) + Face(468×3) | Upper body only |
| **HANDS_UPPER** | 130 | Pose(16×4) + Hands(42×3) | Compact upper body |
| **FACE_ONLY** | 1404 | Face(468×3) | Expression/lips focus |

## How to Switch Variants

### Method 1: In Backend Code

**train_gpu.py (line ~27):**
```python
KEYPOINT_VARIANT = KeypointType.HOLISTIC  # Change to any variant
```

**app.py (line ~28):**
```python
KEYPOINT_VARIANT = KeypointType.HOLISTIC  # Change to any variant
```

Both files MUST use the same variant.

### Method 2: Via API

```bash
# Get current variant
curl http://localhost:8000/api/keypoint/current

# List all variants
curl http://localhost:8000/api/keypoint/variants

# Switch variant (requires model restart)
curl -X POST http://localhost:8000/api/keypoint/set \
  -H "Content-Type: application/json" \
  -d '{"variant": "hands_pose"}'
```

## Feature Engineering

Each variant goes through feature engineering:
```
Raw Landmarks (NRF features)
    ↓
+ Velocity (NRF features)
+ Acceleration (NRF features)
+ Engineered Features (9 features if hands+pose available)
    ↓
Total Features (NF) = 3*NRF + 9
```

### Engineered Features (when applicable)
- Left hand relative to nose (3)
- Right hand relative to nose (3)
- Hand distance (1)
- Left hand speed (1)
- Right hand speed (1)

## When to Use Each Variant

### HOLISTIC (Default)
- **Pros**: Best accuracy, includes facial expressions
- **Cons**: Largest model, highest VRAM usage
- **Use**: When accuracy is priority

### HANDS_POSE
- **Pros**: Balanced, ~11% feature reduction vs HOLISTIC
- **Cons**: Loses facial expression context
- **Use**: When VRAM is limited

### HANDS_ONLY
- **Pros**: Minimal features (126), very fast
- **Cons**: No body/face context, lower accuracy
- **Use**: When speed is critical

### UPPER_BODY
- **Pros**: Omits lower body (legs), 8% smaller than HOLISTIC
- **Cons**: Loses leg information if relevant
- **Use**: When filming upper body only

### POSE_ONLY
- **Pros**: Smallest temporal model
- **Cons**: No hand movement
- **Use**: For pose-only gesture recognition

## Performance Comparison

Approximate metrics (relative to HOLISTIC=100%):

| Variant | Model Size | VRAM | Speed | Accuracy | Disk |
|---------|-----------|------|-------|----------|------|
| HOLISTIC | 100% | 100% | 100% | 100% | 100% |
| HOLISTIC_NO_FACE | 20% | 85% | 110% | 92% | 20% |
| HANDS_POSE | 20% | 85% | 110% | 90% | 20% |
| UPPER_BODY | 95% | 98% | 101% | 97% | 95% |
| HANDS_ONLY | 13% | 70% | 130% | 75% | 13% |
| POSE_ONLY | 14% | 72% | 125% | 70% | 14% |

## Step-by-Step: Switch and Retrain

### 1. Update Code
Edit both `train_gpu.py` and `app.py`:
```python
from keypoint_variants import KeypointType
KEYPOINT_VARIANT = KeypointType.HANDS_POSE  # Choose variant
```

### 2. Delete Old Landmarks (Optional)
If switching from high-dim to low-dim variant, recompute:
```bash
rm backend/landmarks/*.npy
```

### 3. Train Model
```bash
# Old landmarks will be re-extracted with new variant
python backend/train_gpu.py
```

### 4. Verify in API
```bash
curl http://localhost:8000/api/status
# Should show new feature count
```

## Important Notes

⚠️ **Models are variant-specific**
- A model trained with HOLISTIC cannot be used with HANDS_POSE
- Always retrain when changing variants
- Landmarks are cached but with variant-specific naming

✅ **Best Practices**
1. Start with **HOLISTIC** for baseline accuracy
2. Try **HANDS_POSE** to reduce VRAM
3. Try **UPPER_BODY** if legs aren't in frame
4. Only use minimal variants (HANDS_ONLY, POSE_ONLY) if speed is critical

## Augmentation Strategy by Variant

All variants support:
- ✅ Noise injection
- ✅ Sequence-level spatial box transform (uniform scale + translation)
- ✅ Time shift
- ✅ Speed variation
- ✅ Frame dropout
- ✅ Temporal reverse
- ✅ Hand-specific noise (if hands available)
- ✅ Mirror augmentation (if both hands available)

Augmentation adapts to variant automatically.

## Spatial Augmentation (Camera Distance Simulation)

This project now includes global sequence-level spatial augmentation in [backend/spatial_augmentation.py](backend/spatial_augmentation.py).

Pipeline per sequence:
1. Compute one global bbox from all x/y keypoints in all frames.
2. Uniformly scale that box (keep keypoint geometry ratios unchanged).
3. Translate the scaled box inside frame bounds.
4. Ensure all transformed x/y remain in [0, 1].

This simulates a signer moving closer/farther from camera and shifting position, while preserving gesture shape.

Main APIs:
- `get_landmarks_bbox`
- `normalize_landmarks_to_bbox`
- `scale_landmarks`
- `translate_landmarks`
- `clip_landmarks`
- `apply_spatial_augmentation`

Training knobs in [backend/train_gpu.py](backend/train_gpu.py):
- `SPATIAL_AUG_PROB`
- `SPATIAL_SCALE_MIN`
- `SPATIAL_SCALE_MAX`

Visual preview tool:

```bash
python backend/visualize_spatial_aug.py \
  --input backend/landmarks/D0001B.mp4.npy \
  --variant holistic \
  --scale-min 0.75 \
  --scale-max 1.45 \
  --frame 0 \
  --out backend/spatial_aug_preview.png
```

This command draws before/after keypoints and bbox in one image.

## Example: Custom Comparison

Train 3 models to compare:

```bash
# Model 1: Full Holistic
# train_gpu.py: KEYPOINT_VARIANT = KeypointType.HOLISTIC
python backend/train_gpu.py
mv backend/models/sign_model.pt backend/models/holistic.pt

# Model 2: Hands + Pose (faster)
# train_gpu.py: KEYPOINT_VARIANT = KeypointType.HANDS_POSE  
python backend/train_gpu.py
mv backend/models/sign_model.pt backend/models/hands_pose.pt

# Model 3: Upper Body (good compromise)
# train_gpu.py: KEYPOINT_VARIANT = KeypointType.UPPER_BODY
python backend/train_gpu.py
mv backend/models/sign_model.pt backend/models/upper_body.pt
```

Then compare metrics from training logs.

## Troubleshooting

**Q: Model crashes with shape mismatch?**
- Ensure train_gpu.py and app.py use the same KEYPOINT_VARIANT

**Q: Accuracy dropped after switching?**
- Different variants have different signal. Retrain with enough epochs (200+)

**Q: Landmarks not re-extracted?**
- Cached landmarks use old variant. Delete backend/landmarks/*.npy

**Q: Feature engineering not working?**
- Some variants don't have hands+pose combo. Check config.components in API
