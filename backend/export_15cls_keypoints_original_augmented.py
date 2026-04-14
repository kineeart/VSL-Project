"""
Export 15 original keypoint classes and augmented versions for comparison.
Generates side-by-side visualizations showing original vs augmented keypoints for each class.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from keypoint_variants import KeypointType, VARIANTS
from spatial_augmentation import apply_spatial_augmentation, get_landmarks_bbox, get_xy_indices


class KeypointExporter:
    def __init__(self, 
                 model_dir: str = "models_15cls_run1",
                 landmarks_dir: str = "landmarks",
                 output_dir: str = "keypoint_previews/15cls_comparison"):
        self.model_dir = Path(model_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load labels
        with open(self.model_dir / "labels.json", "r") as f:
            self.labels = json.load(f)
        
        # Load normalization parameters
        self.norm_mean = np.load(self.model_dir / "norm_mean.npy")
        self.norm_std = np.load(self.model_dir / "norm_std.npy")
        
        print(f"Loaded {len(self.labels)} classes from {self.model_dir / 'labels.json'}")
        print(f"Labels: {list(self.labels.keys())}")
        
    def find_sample_for_class(self, class_name: str) -> Tuple[np.ndarray, str]:
        """Find and load a sample landmarks sequence for a given class."""
        # Try to find matching landmark files
        for landmark_file in self.landmarks_dir.glob("*.npy"):
            filename = landmark_file.stem
            # Skip holistic variants which are full features
            if filename.endswith("__holistic"):
                file_base = filename[:-10]  # Remove __holistic
            else:
                file_base = filename
            
            # Simple heuristic: if class appears in filename
            if file_base in self.landmarks_dir.glob("*.npy"):
                try:
                    data = np.load(landmark_file)
                    if data.ndim == 2 and data.shape[0] > 1:
                        return data, str(landmark_file.name)
                except:
                    continue
        
        # If no match found, return first available landmarks
        landmark_files = list(self.landmarks_dir.glob("*.npy"))
        if landmark_files:
            data = np.load(landmark_files[0])
            return data, landmark_files[0].name
        
        raise FileNotFoundError(f"No landmark files found in {self.landmarks_dir}")
    
    def plot_keypoints(self, 
                      ax, 
                      landmarks: np.ndarray, 
                      variant: KeypointType,
                      title: str,
                      show_bbox: bool = True):
        """Plot keypoints on an axis."""
        x_idx, y_idx = get_xy_indices(variant)
        
        if len(landmarks.shape) == 2:
            # Multiple frames - use mean
            frame = landmarks.mean(axis=0)
        else:
            frame = landmarks
        
        x = frame[x_idx]
        y = frame[y_idx]
        
        # Check for valid points
        mask = np.isfinite(x) & np.isfinite(y)
        valid_x = x[mask]
        valid_y = y[mask]
        
        if len(valid_x) == 0:
            ax.text(0.5, 0.5, "No valid landmarks", ha="center", va="center",
                   transform=ax.transAxes)
        else:
            # Plot points with different colors for different body parts
            # Pose points - blue
            if "include_pose" in VARIANTS[variant].__dataclass_fields__:
                config = VARIANTS[variant]
                if config.include_pose:
                    pose_end = config.pose_points * (4 if config.include_pose_visibility else 3)
                    pose_x_idx = x_idx[x_idx < pose_end]
                    pose_y_idx = y_idx[y_idx < pose_end]
                    if len(pose_x_idx) > 0:
                        pose_mask = np.isfinite(x[pose_x_idx]) & np.isfinite(y[pose_y_idx])
                        ax.scatter(x[pose_x_idx][pose_mask], y[pose_y_idx][pose_mask], 
                                 s=30, alpha=0.7, c='blue', label='Pose')
                
            # Plot all points
            ax.scatter(valid_x, valid_y, s=20, alpha=0.6, c='red')
        
        # Set limits and aspect
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    def export_class_comparisons(self, 
                                variant: KeypointType = KeypointType.HOLISTIC,
                                augmentation_configs: List[Dict] = None):
        """Export original and augmented keypoints for each class."""
        
        if augmentation_configs is None:
            # Default augmentation configurations to compare
            augmentation_configs = [
                {"name": "Original", "scale_min": 1.0, "scale_max": 1.0, "shift_range": (0, 0)},
                {"name": "Scale×0.8", "scale_min": 0.8, "scale_max": 0.8, "shift_range": (0, 0)},
                {"name": "Scale×1.2", "scale_min": 1.2, "scale_max": 1.2, "shift_range": (0, 0)},
                {"name": "Shift+X", "scale_min": 1.0, "scale_max": 1.0, "shift_range": (0.1, 0)},
                {"name": "Shift+Y", "scale_min": 1.0, "scale_max": 1.0, "shift_range": (0, 0.1)},
                {"name": "Random Mix", "scale_min": 0.75, "scale_max": 1.35, "shift_range": (0.1, 0.1)},
            ]
        
        class_names = sorted(self.labels.keys())
        num_classes = len(class_names)
        
        # Create summary figure with all classes
        fig_summary = plt.figure(figsize=(20, 12))
        fig_summary.suptitle("15 Original Keypoint Classes - Models_15cls_run1", 
                            fontsize=16, fontweight='bold')
        
        for class_idx, class_name in enumerate(class_names):
            print(f"Processing class {class_idx + 1}/{num_classes}: {class_name}")
            
            try:
                # Load sample landmarks
                landmarks, source_file = self.find_sample_for_class(class_name)
                print(f"  Using sample: {source_file}")
                
                # Create detailed comparison figure for this class
                fig, axes = plt.subplots(2, len(augmentation_configs), figsize=(16, 8))
                fig.suptitle(f"Class {class_idx}: {class_name} - Original vs Augmented Keypoints", 
                            fontsize=12, fontweight='bold')
                
                # First row: original landmarks
                # Second row: augmented landmarks
                
                for aug_idx, config in enumerate(augmentation_configs):
                    if config["name"] == "Original":
                        aug_landmarks = landmarks
                    else:
                        # Apply augmentation
                        try:
                            aug_landmarks = apply_spatial_augmentation(
                                landmarks,
                                variant=variant,
                                scale_min=config.get("scale_min", 0.75),
                                scale_max=config.get("scale_max", 1.35)
                            )
                        except Exception as e:
                            print(f"  Warning: Augmentation failed for {config['name']}: {e}")
                            aug_landmarks = landmarks
                    
                    # Plot original on first row
                    if aug_idx < len(axes[0]):
                        self.plot_keypoints(axes[0, aug_idx], landmarks, variant, 
                                          f"Original\n({config['name']})")
                    
                    # Plot augmented on second row
                    if aug_idx < len(axes[1]):
                        self.plot_keypoints(axes[1, aug_idx], aug_landmarks, variant,
                                          config["name"])
                
                fig.tight_layout()
                output_file = self.output_dir / f"class_{class_idx:02d}_{class_name.replace(' ', '_')}.png"
                fig.savefig(output_file, dpi=100, bbox_inches='tight')
                print(f"  Saved: {output_file}")
                plt.close(fig)
                
                # Add to summary figure
                if class_idx < len(fig_summary.axes):
                    ax = fig_summary.add_subplot(4, 4, class_idx + 1)
                    ax.set_title(f"{class_idx}: {class_name[:20]}", fontsize=8)
                    self.plot_keypoints(ax, landmarks, variant, "")
            
            except Exception as e:
                print(f"  ERROR processing class {class_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary figure
        summary_file = self.output_dir / "00_summary_15classes.png"
        fig_summary.savefig(summary_file, dpi=100, bbox_inches='tight')
        print(f"\nSaved summary: {summary_file}")
        plt.close('all')
    
    def export_augmentation_comparison(self,
                                      variant: KeypointType = KeypointType.HOLISTIC):
        """Create a detailed report comparing multiple augmentations."""
        
        output_file = self.output_dir / "AUGMENTATION_GUIDE.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("15-CLASS KEYPOINT AUGMENTATION GUIDE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Model: models_15cls_run1\n")
            f.write(f"Variant: {variant.value}\n")
            f.write(f"Normalization Mean shape: {self.norm_mean.shape}\n")
            f.write(f"Normalization Std shape: {self.norm_std.shape}\n\n")
            
            f.write("CLASSES (15 total):\n")
            f.write("-" * 80 + "\n")
            for class_idx, class_name in enumerate(sorted(self.labels.keys())):
                f.write(f"{class_idx:2d}. {class_name}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("AUGMENTATION STRATEGIES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. SCALE AUGMENTATION:\n")
            f.write("   - Original: scale = 1.0\n")
            f.write("   - Scale×0.8: scale = 0.8 (shrink keypoints)\n")
            f.write("   - Scale×1.2: scale = 1.2 (expand keypoints)\n")
            f.write("   - Random Mix: scale in [0.75, 1.35]\n\n")
            
            f.write("2. POSITIONAL AUGMENTATION:\n")
            f.write("   - Shift+X: horizontal offset\n")
            f.write("   - Shift+Y: vertical offset\n\n")
            
            f.write("3. COMBINED AUGMENTATION:\n")
            f.write("   - Random Mix: combines scale and shift\n\n")
            
            f.write("OUTPUT FILES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Location: {self.output_dir}\n")
            f.write("- 00_summary_15classes.png: Overview of all 15 classes\n")
            f.write("- class_XX_*.png: Detailed comparison for each class\n")
            f.write("- AUGMENTATION_GUIDE.txt: This file\n\n")
            
            f.write("USAGE:\n")
            f.write("-" * 80 + "\n")
            f.write("View the PNG files to compare:\n")
            f.write("  - Top row: Original keypoints labeled with augmentation type\n")
            f.write("  - Bottom row: Augmented keypoints with same scale\n")
            f.write("  - Each column represents one augmentation variant\n\n")
        
        print(f"Saved guide: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export 15-class keypoint originals and augmented variants"
    )
    parser.add_argument("--model-dir", default="models_15cls_run1",
                       help="Model directory containing labels.json, norm_mean.npy, norm_std.npy")
    parser.add_argument("--landmarks-dir", default="landmarks",
                       help="Directory containing landmark .npy files")
    parser.add_argument("--output-dir", default="keypoint_previews/15cls_comparison",
                       help="Output directory for generated previews")
    parser.add_argument("--variant", default="holistic",
                       help="Keypoint variant to use (holistic, hands_only, pose_only, etc.)")
    
    args = parser.parse_args()
    
    # Parse variant
    try:
        variant = KeypointType(args.variant.lower())
    except ValueError:
        print(f"Invalid variant '{args.variant}'. Available: {[v.value for v in KeypointType]}")
        return
    
    # Export
    exporter = KeypointExporter(
        model_dir=args.model_dir,
        landmarks_dir=args.landmarks_dir,
        output_dir=args.output_dir
    )
    
    exporter.export_class_comparisons(variant=variant)
    exporter.export_augmentation_comparison(variant=variant)
    
    print("\n" + "=" * 80)
    print(f"Export complete! Check: {exporter.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
