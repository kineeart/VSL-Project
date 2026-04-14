"""
Keypoint Variants - Support multiple ways to extract and combine landmarks.
Inspired by oneshotsViSL's modular approach for maximum flexibility.
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass


class KeypointType(Enum):
    """Define all keypoint variant types."""
    HOLISTIC = "holistic"                          # Full: Pose(33×4) + Hands(42×3) + Face(468×3) = 1662
    HOLISTIC_NO_FACE = "holistic_no_face"          # Pose(33×4) + Hands(42×3) = 198
    HOLISTIC_NO_VIZ = "holistic_no_viz"            # Pose(33×3) + Hands(42×3) + Face(468×3) = 1659
    HANDS_ONLY = "hands_only"                      # Hands(42×3) = 126
    POSE_ONLY = "pose_only"                        # Pose(33×4) = 132
    HANDS_POSE = "hands_pose"                      # Hands(42×3) + Pose(33×4) = 198
    UPPER_BODY = "upper_body"                      # Pose(16×4) + Hands(42×3) + Face(468×3) = 1524
    HANDS_UPPER = "hands_upper"                    # Hands(42×3) + Pose(16×4) = 130
    FACE_ONLY = "face_only"                        # Face(468×3) = 1404


@dataclass
class KeypointConfig:
    """Configuration for each keypoint variant."""
    name: str
    include_pose: bool = False
    include_pose_visibility: bool = False
    pose_points: int = 33  # 0-32 for full, 0-15 for upper body
    include_left_hand: bool = False
    include_right_hand: bool = False
    include_face: bool = False
    
    @property
    def feature_size(self) -> int:
        """Calculate total feature count."""
        size = 0
        if self.include_pose:
            pose_channels = 4 if self.include_pose_visibility else 3
            size += self.pose_points * pose_channels
        if self.include_left_hand:
            size += 21 * 3
        if self.include_right_hand:
            size += 21 * 3
        if self.include_face:
            size += 468 * 3
        return size
    
    @property
    def components(self) -> list:
        """List components in order of concatenation."""
        components = []
        if self.include_pose:
            components.append(f"pose({self.pose_points}×{'4' if self.include_pose_visibility else '3'})")
        if self.include_left_hand:
            components.append("left_hand(21×3)")
        if self.include_right_hand:
            components.append("right_hand(21×3)")
        if self.include_face:
            components.append("face(468×3)")
        return components


# Define all variants
VARIANTS = {
    KeypointType.HOLISTIC: KeypointConfig(
        name="Holistic Full",
        include_pose=True, include_pose_visibility=True, pose_points=33,
        include_left_hand=True, include_right_hand=True, include_face=True
    ),
    KeypointType.HOLISTIC_NO_FACE: KeypointConfig(
        name="Holistic (No Face)",
        include_pose=True, include_pose_visibility=True, pose_points=33,
        include_left_hand=True, include_right_hand=True, include_face=False
    ),
    KeypointType.HOLISTIC_NO_VIZ: KeypointConfig(
        name="Holistic (No Visibility)",
        include_pose=True, include_pose_visibility=False, pose_points=33,
        include_left_hand=True, include_right_hand=True, include_face=True
    ),
    KeypointType.HANDS_ONLY: KeypointConfig(
        name="Hands Only",
        include_pose=False,
        include_left_hand=True, include_right_hand=True, include_face=False
    ),
    KeypointType.POSE_ONLY: KeypointConfig(
        name="Pose Only",
        include_pose=True, include_pose_visibility=True, pose_points=33,
        include_left_hand=False, include_right_hand=False, include_face=False
    ),
    KeypointType.HANDS_POSE: KeypointConfig(
        name="Hands + Pose",
        include_pose=True, include_pose_visibility=True, pose_points=33,
        include_left_hand=True, include_right_hand=True, include_face=False
    ),
    KeypointType.UPPER_BODY: KeypointConfig(
        name="Upper Body (Hands + Upper Pose + Face)",
        include_pose=True, include_pose_visibility=True, pose_points=16,
        include_left_hand=True, include_right_hand=True, include_face=True
    ),
    KeypointType.HANDS_UPPER: KeypointConfig(
        name="Hands + Upper Pose",
        include_pose=True, include_pose_visibility=True, pose_points=16,
        include_left_hand=True, include_right_hand=True, include_face=False
    ),
    KeypointType.FACE_ONLY: KeypointConfig(
        name="Face Only",
        include_pose=False,
        include_left_hand=False, include_right_hand=False, include_face=True
    ),
}


def get_variant_config(variant: KeypointType) -> KeypointConfig:
    """Get configuration for a keypoint variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
    return VARIANTS[variant]


def extract_landmarks_variant(holistic_results, variant: KeypointType = KeypointType.HOLISTIC):
    """
    Extract landmarks based on variant type.
    
    Args:
        holistic_results: MediaPipe Holistic results
        variant: KeypointType enum value
        
    Returns:
        np.array of shape (variant_feature_size,)
    """
    config = get_variant_config(variant)
    parts = []
    
    # Extract Pose
    if config.include_pose:
        if holistic_results.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in holistic_results.pose_landmarks.landmark])
            pose = pose[:config.pose_points]  # Take only required points
            if config.include_pose_visibility:
                # Keep x,y,z,visibility
                parts.append(pose.flatten())
            else:
                # Keep only x,y,z (remove visibility)
                parts.append(pose[:, :3].flatten())
        else:
            pose_size = config.pose_points * (4 if config.include_pose_visibility else 3)
            parts.append(np.zeros(pose_size))
    
    # Extract Left Hand
    if config.include_left_hand:
        if holistic_results.left_hand_landmarks:
            lh = np.array([[l.x, l.y, l.z] for l in holistic_results.left_hand_landmarks.landmark])
            parts.append(lh.flatten())
        else:
            parts.append(np.zeros(21 * 3))
    
    # Extract Right Hand
    if config.include_right_hand:
        if holistic_results.right_hand_landmarks:
            rh = np.array([[l.x, l.y, l.z] for l in holistic_results.right_hand_landmarks.landmark])
            parts.append(rh.flatten())
        else:
            parts.append(np.zeros(21 * 3))
    
    # Extract Face
    if config.include_face:
        if holistic_results.face_landmarks:
            face = np.array([[l.x, l.y, l.z] for l in holistic_results.face_landmarks.landmark])
            parts.append(face.flatten())
        else:
            parts.append(np.zeros(468 * 3))
    
    return np.concatenate(parts).astype(np.float32)


def print_variant_info(variant: KeypointType):
    """Print detailed info about a variant."""
    config = get_variant_config(variant)
    print(f"\n[Variant: {variant.value}]")
    print(f"  Name: {config.name}")
    print(f"  Components: {', '.join(config.components)}")
    print(f"  Total features: {config.feature_size}")


def print_all_variants():
    """Print info about all available variants."""
    print("\n" + "="*70)
    print("KEYPOINT VARIANTS SUMMARY")
    print("="*70)
    for variant in KeypointType:
        print_variant_info(variant)
    print("="*70)


if __name__ == "__main__":
    print_all_variants()
