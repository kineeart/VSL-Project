"""
Vietnamese Sign Language - Realtime Camera Demo
Capture frames from camera, extract landmarks, predict sign language class in realtime.
"""
import argparse
import os
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

import mediapipe as mp
from keypoint_variants import KeypointType, VARIANTS, extract_landmarks_variant
from app import SignModel, engineer_features, device, load_data_mapping


def load_labels_from_dir(model_dir: Path):
    """Load label mappings from model directory."""
    import json
    labels_file = model_dir / "labels.json"
    if not labels_file.exists():
        return None, None
    with open(labels_file, encoding='utf-8') as f:
        label_map = json.load(f)
    reverse_label_map = {int(v): k for k, v in label_map.items()}
    return label_map, reverse_label_map


def load_model(model_dir: Path, keypoint_variant: KeypointType):
    """Load trained model from given directory."""
    pt_path = model_dir / "sign_model.pt"
    mean_path = model_dir / "norm_mean.npy"
    std_path = model_dir / "norm_std.npy"
    
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")
    
    checkpoint = torch.load(str(pt_path), map_location=device, weights_only=True)
    num_features = checkpoint.get('num_features')
    num_classes = checkpoint.get('num_classes')
    
    model = SignModel(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    norm_mean = None
    norm_std = None
    if mean_path.exists() and std_path.exists():
        norm_mean = np.load(str(mean_path))
        norm_std = np.load(str(std_path))
    
    return model, norm_mean, norm_std, num_classes


def predict_from_seq(raw_seq, model, norm_mean, norm_std, reverse_label_map, variant: KeypointType):
    """Run prediction on a sequence of landmarks."""
    feat = engineer_features(raw_seq, variant)
    if norm_mean is not None and norm_std is not None:
        feat = (feat - norm_mean) / norm_std
    
    tensor = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    
    top_indices = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top_indices:
        label = reverse_label_map.get(int(idx), "unknown")
        results.append((label, float(probs[idx])))
    
    return results


def extract_landmarks_frame(frame, holistic):
    """Extract holistic landmarks from single frame."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    return extract_landmarks_variant(results, KeypointType.HOLISTIC)


def draw_results(frame, results, seq_len=60):
    """Draw prediction results on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Top prediction
    if results:
        top_label, top_conf = results[0]
        cv2.putText(frame, f"Top: {top_label} ({top_conf:.2%})", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Recent predictions
    y_offset = 90
    for i, (label, conf) in enumerate(results[:4]):
        cv2.putText(frame, f"{i+1}. {label} {conf:.2%}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_offset += 30
    
    # Sequence buffer indicator
    cv2.putText(frame, f"Buffer: {len(landmark_buffer)}/{seq_len}", 
               (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Realtime camera demo for sign language recognition")
    parser.add_argument("--model-dir", default="backend/models_15cls_run1", 
                       help="Path to model directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0=default)")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length for model")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"[DEMO] Loading model from: {model_dir}")
    model, norm_mean, norm_std, num_classes = load_model(model_dir, KeypointType.HOLISTIC)
    label_map, reverse_label_map = load_labels_from_dir(model_dir)
    
    if not label_map:
        raise FileNotFoundError(f"Labels not found in {model_dir}")
    
    print(f"[DEMO] Model loaded: {num_classes} classes")
    print(f"[DEMO] Labels: {list(label_map.keys())}")
    
    # Setup camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"[DEMO] Camera opened. Streaming...")
    print("[DEMO] Press 'q' to quit, 's' to save screenshot")
    
    # Landmark buffer
    landmark_buffer = deque(maxlen=args.seq_len)
    
    frame_count = 0
    pred_interval = 10  # Predict every N frames
    
    with mp.solutions.holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            try:
                lm = extract_landmarks_frame(frame, holistic)
                landmark_buffer.append(lm)
            except:
                pass
            
            # Predict when buffer is full and interval reached
            results = []
            if frame_count % pred_interval == 0 and len(landmark_buffer) == args.seq_len:
                seq = np.array(list(landmark_buffer))
                results = predict_from_seq(seq, model, norm_mean, norm_std, reverse_label_map, KeypointType.HOLISTIC)
            
            # Draw
            frame = draw_results(frame, results, args.seq_len)
            cv2.imshow("Sign Language Recognition - Press 'q' to quit", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.png"
                cv2.imwrite(filename, frame)
                print(f"[DEMO] Screenshot saved: {filename}")
            
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("[DEMO] Done.")


if __name__ == "__main__":
    main()
