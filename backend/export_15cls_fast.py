"""
Fast export of 15 original keypoint classes and augmented variants.
Generates an efficient HTML report and summary statistics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import base64
from io import BytesIO
import argparse

from keypoint_variants import KeypointType, VARIANTS
from spatial_augmentation import apply_spatial_augmentation, get_xy_indices


class FastKeyPointExporter:
    def __init__(self, 
                 model_dir: str = "models_15cls_run1",
                 landmarks_dir: str = "landmarks",
                 output_dir: str = "keypoint_previews/15cls_comparison"):
        self.model_dir = Path(model_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_dir / "labels.json", "r") as f:
            self.labels = json.load(f)
        
        self.norm_mean = np.load(self.model_dir / "norm_mean.npy")
        self.norm_std = np.load(self.model_dir / "norm_std.npy")
    
    def get_sample_data(self) -> Dict[str, np.ndarray]:
        """Get one sample landmarks file - used for all classes for speed."""
        landmark_files = list(self.landmarks_dir.glob("*.mp4.npy"))
        if not landmark_files:
            landmark_files = list(self.landmarks_dir.glob("*.npy"))
        
        if landmark_files:
            data = np.load(landmark_files[0])
            return data
        return None
    
    def plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64
    
    def create_keypoint_preview(self, 
                               landmarks: np.ndarray,
                               variant: KeypointType,
                               title: str = "Keypoints") -> str:
        """Create a single keypoint visualization and return as base64."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        x_idx, y_idx = get_xy_indices(variant)
        
        if len(landmarks.shape) == 2:
            frame = landmarks.mean(axis=0)
        else:
            frame = landmarks
        
        x = frame[x_idx]
        y = frame[y_idx]
        mask = np.isfinite(x) & np.isfinite(y)
        
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=30, alpha=0.7, c='#1f77b4', edgecolors='black', linewidth=0.5)
        
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        return self.plot_to_base64(fig)
    
    def export_html_report(self, variant: KeypointType = KeypointType.HOLISTIC):
        """Generate a comprehensive HTML report."""
        
        print("Loading sample data...")
        sample_data = self.get_sample_data()
        
        if sample_data is None:
            print("ERROR: No landmark data found!")
            return
        
        class_names = sorted(self.labels.keys())
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model {self.model_dir.name} - 15 Classes Keypoint Comparison</title>
    <style>
        * {{ font-family: Arial, sans-serif; }}
        body {{ margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #333; font-size: 28px; margin: 0; }}
        .header p {{ color: #666; margin: 5px 0; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .class-section {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .class-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #007bff;
        }}
        .preview-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .preview-item {{
            text-align: center;
        }}
        .preview-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }}
        .preview-label {{
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            font-weight: 500;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .toc {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .toc h2 {{ margin-top: 0; color: #333; }}
        .toc ol {{ columns: 2; }}
        .toc a {{ color: #007bff; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Model {self.model_dir.name}</h1>
            <h2>15-Class Keypoint Analysis</h2>
            <p>Variant: <strong>{variant.value}</strong></p>
            <p>Comparing Original vs Augmented Keypoints</p>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(class_names)}</div>
                <div class="stat-label">Total Classes</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{sample_data.shape[1]}</div>
                <div class="stat-label">Features per Frame</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{sample_data.shape[0]}</div>
                <div class="stat-label">Sample Frames</div>
            </div>
        </div>
        
        <div class="toc">
            <h2>Table of Contents</h2>
            <ol>
"""
        
        for class_idx, class_name in enumerate(class_names):
            anchor = f"class_{class_idx}"
            html_content += f'                <li><a href="#{anchor}">{class_name}</a></li>\n'
        
        html_content += """
            </ol>
        </div>
"""
        
        print("\nGenerating previews for each class...")
        for class_idx, class_name in enumerate(class_names):
            print(f"  {class_idx + 1}/15: {class_name}", end=" ... ")
            
            class_name_id = f"class_{class_idx}"
            
            # Original preview
            print("original", end=" ")
            original_b64 = self.create_keypoint_preview(
                sample_data, 
                variant,
                f"Original Keypoints"
            )
            
            # Augmented preview (scaled)
            print("augmented", end="")
            try:
                augmented_data = apply_spatial_augmentation(
                    sample_data,
                    variant=variant,
                    scale_min=0.8,
                    scale_max=1.2,
                )
                augmented_b64 = self.create_keypoint_preview(
                    augmented_data,
                    variant,
                    f"Augmented Keypoints (Scale: 0.8-1.2)"
                )
            except Exception as e:
                print(f" [warning: {e}]", end="")
                augmented_b64 = original_b64
            
            html_content += f"""
        <div class="class-section" id="{class_name_id}">
            <div class="class-title">
                Class {class_idx}: {class_name}
            </div>
            <div class="preview-grid">
                <div class="preview-item">
                    <img src="data:image/png;base64,{original_b64}" alt="Original">
                    <div class="preview-label">Original Keypoints</div>
                </div>
                <div class="preview-item">
                    <img src="data:image/png;base64,{augmented_b64}" alt="Augmented">
                    <div class="preview-label">Augmented (Scale 0.8-1.2)</div>
                </div>
            </div>
        </div>
"""
            print(" done")
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save HTML
        html_file = self.output_dir / "comparison_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"\n✓ Saved HTML report: {html_file}")
        return html_file
    
    def export_summary_txt(self):
        """Generate a text summary."""
        
        output_file = self.output_dir / "SUMMARY.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"MODEL {self.model_dir.name} - 15-CLASS KEYPOINT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CLASSES (15 total):\n")
            f.write("-" * 80 + "\n")
            for class_idx, class_name in enumerate(sorted(self.labels.keys())):
                f.write(f"{class_idx:2d}. {class_name}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("NORMALIZATION PARAMETERS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Mean values shape: {self.norm_mean.shape}\n")
            f.write(f"Std values shape: {self.norm_std.shape}\n")
            f.write(f"Mean range: [{self.norm_mean.min():.6f}, {self.norm_mean.max():.6f}]\n")
            f.write(f"Std range: [{self.norm_std.min():.6f}, {self.norm_std.max():.6f}]\n\n")
            
            f.write("Mean values (first 10):\n")
            for i, val in enumerate(self.norm_mean[:10]):
                f.write(f"  [{i}]: {val:.6f}\n")
            f.write("  ...\n\n")
            
            f.write("Std values (first 10):\n")
            for i, val in enumerate(self.norm_std[:10]):
                f.write(f"  [{i}]: {val:.6f}\n")
            f.write("  ...\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("AUGMENTATION STRATEGIES APPLIED\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. SCALE AUGMENTATION:\n")
            f.write("   - Original: scale = 1.0 (no change)\n")
            f.write("   - Augmented: scale in range [0.8, 1.2]\n")
            f.write("     * Shrinks or expands keypoint coordinates uniformly\n")
            f.write("     * Preserves relative spatial relationships\n\n")
            
            f.write("2. PURPOSE OF AUGMENTATION:\n")
            f.write("   - Increase robustness against different hand sizes\n")
            f.write("   - Simulate zooming in/out during sign language capture\n")
            f.write("   - Reduce overfitting to specific scale\n")
            f.write("   - Improve model generalization\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Location: {self.output_dir}\n")
            f.write("- comparison_report.html: Interactive HTML report with all visualizations\n")
            f.write("- SUMMARY.txt: This file with analysis details\n\n")
            
            f.write("HOW TO VIEW:\n")
            f.write("1. Open 'comparison_report.html' in a web browser\n")
            f.write("2. Scroll through each of the 15 classes\n")
            f.write("3. Compare original keypoints (left) vs augmented (right)\n")
            f.write("4. Use Table of Contents to jump to specific classes\n\n")
        
        print(f"✓ Saved summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fast export of 15-class keypoints with augmentation comparison")
    parser.add_argument("--model-dir", default="models_15cls_run1", help="Model directory")
    parser.add_argument("--landmarks-dir", default="landmarks", help="Landmarks directory")
    parser.add_argument("--output-dir", default="keypoint_previews/15cls_comparison_fast", help="Output directory")
    parser.add_argument("--variant", default="holistic", help="Keypoint variant")
    
    args = parser.parse_args()
    
    try:
        variant = KeypointType(args.variant.lower())
    except ValueError:
        print(f"ERROR: Invalid variant. Available: {[v.value for v in KeypointType]}")
        return
    
    exporter = FastKeyPointExporter(
        model_dir=args.model_dir,
        landmarks_dir=args.landmarks_dir,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"KEYPOINT EXPORT FOR {args.model_dir}")
    print(f"{'='*60}\n")
    
    exporter.export_html_report(variant=variant)
    exporter.export_summary_txt()
    
    print(f"\n{'='*60}")
    print("✓ Export complete!")
    print(f"Please open: {exporter.output_dir}/comparison_report.html")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
