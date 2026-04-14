import os, json, cv2, numpy as np, mediapipe as mp, openpyxl, base64, time, math
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from keypoint_variants import KeypointType, VARIANTS, extract_landmarks_variant

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "Videos"
DATA_FILE = BASE_DIR / "Data.xlsx"
MODEL_DIR = Path(
    os.environ.get("VSL_MODEL_DIR", str(BASE_DIR / "backend" / "models")).strip()
)
LANDMARKS_DIR = BASE_DIR / "backend" / "landmarks"
CUSTOM_VIDEOS_DIR = BASE_DIR / "backend" / "custom_videos"

for d in [MODEL_DIR, LANDMARKS_DIR, CUSTOM_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[APP] MODEL_DIR (from VSL_MODEL_DIR={os.environ.get('VSL_MODEL_DIR', 'NOT SET')}): {MODEL_DIR}")

mp_holistic = mp.solutions.holistic

# Global
KEYPOINT_VARIANT = KeypointType.HOLISTIC  # Change this to use different keypoint types
model = None
model_members = []
label_map = {}
reverse_label_map = {}
norm_mean = None
norm_std = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LENGTH = 60
NUM_RAW_FEATURES = 33 * 4 + 21 * 3 * 2 + 468 * 3  # 1662

# Inference guardrails to avoid forced predictions on idle/no-sign frames.
NO_SIGN_LABEL = "no_sign"
MIN_ACTIVE_FRAME_RATIO = 0.10      # Reduced from 0.20 - allow more inactive frames
MIN_MOTION_ENERGY = 0.0005          # Reduced from 0.003 - detect subtle movements
MIN_TOP1_CONFIDENCE = 0.25          # Reduced from 0.45 - more lenient confidence threshold
MIN_TOP12_MARGIN = 0.02             # Reduced from 0.08 - allow closer competition

# Test-Time Augmentation (TTA) and Temporal Smoothing
USE_TTA = os.environ.get('USE_TTA', 'true').lower() == 'true'  # Enable TTA for better accuracy
USE_TEMPORAL_SMOOTH = os.environ.get('USE_TEMPORAL_SMOOTH', 'true').lower() == 'true'
TTA_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]  # Scale factors for TTA
TEMPORAL_WINDOW = 3  # Number of frames to smooth across
prediction_history = []  # For temporal smoothing
last_activity_sufficient = None  # Track activity state to reset history on sign boundary
DEBUG_PREDICTION_LOG = os.environ.get('DEBUG_PREDICTION_LOG', 'true').lower() == 'true'
DEBUG_TOPK = int(os.environ.get('DEBUG_TOPK', '5'))
DEBUG_INCLUDE_RESPONSE = os.environ.get('DEBUG_INCLUDE_RESPONSE', 'true').lower() == 'true'
debug_predict_counter = 0

# Ensemble + LM settings
ENSEMBLE_MODEL_DIRS = [
    p.strip() for p in os.environ.get("VSL_ENSEMBLE_MODEL_DIRS", "").split(";") if p.strip()
]
ENABLE_LM_DECODE = os.environ.get("ENABLE_LM_DECODE", "true").lower() == "true"
LM_WEIGHT = float(os.environ.get("LM_WEIGHT", "0.35"))
LM_TOPK = int(os.environ.get("LM_TOPK", "3"))
LM_PATH = Path(
    os.environ.get(
        "VSL_LM_PATH",
        str(BASE_DIR / "benchmark" / "sentence_level" / "data" / "vsl_bigram_lm.json")
    ).strip()
)


class BigramLM:
    """Simple add-k smoothed bigram LM for token re-ranking in continuous decode."""

    def __init__(self, unigram_counts=None, bigram_counts=None, smoothing=0.5):
        self.unigram_counts = unigram_counts or {}
        self.bigram_counts = bigram_counts or {}
        self.smoothing = float(smoothing)
        self.vocab = set(self.unigram_counts.keys())
        for prev, nxt in self.bigram_counts.items():
            self.vocab.add(prev)
            self.vocab.update(nxt.keys())
        self.vocab = sorted(self.vocab)
        self.vocab_size = max(1, len(self.vocab))

    @classmethod
    def from_json(cls, lm_path: Path):
        if not lm_path.exists():
            return None
        with open(lm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            unigram_counts=payload.get("unigram_counts", {}),
            bigram_counts=payload.get("bigram_counts", {}),
            smoothing=payload.get("smoothing", 0.5),
        )

    def log_prob(self, prev_token, next_token):
        prev = str(prev_token or "<s>")
        nxt = str(next_token)
        row = self.bigram_counts.get(prev, {})
        row_total = float(sum(float(v) for v in row.values()))
        count = float(row.get(nxt, 0.0))
        denom = row_total + self.smoothing * self.vocab_size
        prob = (count + self.smoothing) / max(1e-12, denom)
        return math.log(max(1e-12, prob))


sequence_lm = BigramLM.from_json(LM_PATH) if ENABLE_LM_DECODE else None


# ============================================================
# Model Architecture (must match train_gpu.py exactly)
# ============================================================
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for temporal sequences."""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out).mean(dim=1)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier - much better for few-shot learning."""
    def __init__(self, in_features, num_classes, temperature=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.temperature * (x_norm @ w_norm.t())


class SignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.attention = MultiHeadAttention(768, num_heads=4)
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = CosineClassifier(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.conv_drop(c)
        c = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm1(c)
        attn_out = self.attention(lstm_out)
        x = F.gelu(self.bn3(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


# ============================================================
# Feature Engineering (must match train_gpu.py exactly)
# ============================================================
def engineer_features(seq):
    """Feature engineering that matches train_gpu.py for the active variant."""
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


def _is_sign_activity_sufficient(raw_seq):
    """Reject idle/no-sign windows before classification."""
    frame_energy = np.linalg.norm(raw_seq, axis=1)
    active_ratio = float(np.mean(frame_energy > 1e-6))
    if active_ratio < MIN_ACTIVE_FRAME_RATIO:
        return False

    motion = np.linalg.norm(np.diff(raw_seq, axis=0), axis=1)
    motion_energy = float(np.mean(motion)) if len(motion) else 0.0
    return motion_energy >= MIN_MOTION_ENERGY


def apply_spatial_scale(seq, scale):
    """Apply spatial scaling to keypoint sequence."""
    from spatial_augmentation import get_landmarks_bbox, scale_landmarks, clip_landmarks
    try:
        bbox = get_landmarks_bbox(seq, KEYPOINT_VARIANT)
        if bbox is None:
            return seq
        scaled = scale_landmarks(seq, KEYPOINT_VARIANT, scale, center=(bbox['cx'], bbox['cy']))
        return clip_landmarks(scaled, KEYPOINT_VARIANT)
    except Exception:
        return seq


def _predict_probs_single_member(member, feat):
    """Single forward pass for one model member with member-specific normalization."""
    local_feat = feat
    member_mean = member.get("norm_mean")
    member_std = member.get("norm_std")
    if member_mean is not None and member_std is not None:
        local_feat = (local_feat - member_mean) / member_std
    tensor = torch.from_numpy(local_feat.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = member["model"](tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs


def predict_probs_from_seq(raw_seq):
    """Compute class probabilities from one sequence using ensemble if available."""
    feat = engineer_features(raw_seq)
    members = model_members if model_members else []
    if not members and model is not None:
        members = [{"name": "primary", "model": model, "norm_mean": norm_mean, "norm_std": norm_std}]
    if not members:
        return np.array([])
    probs_list = [_predict_probs_single_member(member, feat) for member in members]
    return np.mean(probs_list, axis=0)


def _format_topk(probs, k=5):
    top_sorted = np.argsort(probs)[::-1][:k]
    items = []
    for idx in top_sorted:
        label = reverse_label_map.get(int(idx), "unknown")
        items.append(f"{label}:{float(probs[idx]):.3f}")
    return " | ".join(items)


def _debug_log_probs(source, stage, probs, extra=""):
    if not DEBUG_PREDICTION_LOG:
        return
    global debug_predict_counter
    debug_predict_counter += 1
    top1_idx = int(np.argmax(probs))
    top1_label = reverse_label_map.get(top1_idx, "unknown")
    top1 = float(probs[top1_idx])
    top2 = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
    margin = top1 - top2
    topk_str = _format_topk(probs, max(1, DEBUG_TOPK))
    print(
        f"[DEBUG-PRED][{debug_predict_counter:05d}][{source}][{stage}] "
        f"top1={top1_label}:{top1:.3f} margin={margin:.3f} {extra} topk={topk_str}"
    )


def _topk_payload(probs, k=5):
    top_sorted = np.argsort(probs)[::-1][:k]
    payload = []
    for idx in top_sorted:
        payload.append({
            "label": reverse_label_map.get(int(idx), "unknown"),
            "confidence": float(probs[idx]),
        })
    return payload


def predict_from_seq_tta(raw_seq):
    """Prediction with Test-Time Augmentation (TTA)."""
    probs_list = []
    
    # Forward pass with multiple scales
    for scale in TTA_SCALES:
        scaled_seq = apply_spatial_scale(raw_seq, scale) if scale != 1.0 else raw_seq
        probs_list.append(predict_probs_from_seq(scaled_seq))
    
    # Average probabilities across TTA passes (more stable than averaging features)
    return np.mean(probs_list, axis=0)


def apply_temporal_smoothing(prediction, reset=False):
    """Apply temporal smoothing to smooth out prediction jitter.
    
    Args:
        prediction: probability vector from model
        reset: if True, clear history before adding new prediction
    """
    global prediction_history
    
    # Reset history on sign boundary (activity state change)
    if reset:
        prediction_history = []
    
    prediction_history.append(prediction)
    prediction_history = prediction_history[-TEMPORAL_WINDOW:]
    
    if len(prediction_history) == 0:
        return prediction
    
    return np.mean(prediction_history, axis=0)


def predict_from_seq(raw_seq, source="unknown", include_debug=False):
    """Run prediction: raw landmarks → feature engineering → normalize → model → top-5.
    
    With optional Test-Time Augmentation (TTA) and Temporal Smoothing.
    Automatically resets temporal history when sign activity ends.
    """
    global last_activity_sufficient
    debug_payload = {
        "source": source,
        "tta": USE_TTA,
        "temporal_smooth": USE_TEMPORAL_SMOOTH,
        "history_len": len(prediction_history),
    }
    
    activity_sufficient = _is_sign_activity_sufficient(raw_seq)
    
    # Reset temporal history when transitioning from sign → no-sign
    if last_activity_sufficient is True and activity_sufficient is False:
        if USE_TEMPORAL_SMOOTH:
            globals()['prediction_history'] = []
    
    last_activity_sufficient = activity_sufficient
    
    if not activity_sufficient:
        if DEBUG_PREDICTION_LOG:
            print(f"[DEBUG-PRED][{source}][activity] rejected: insufficient motion/activity")
        preds = [{"label": NO_SIGN_LABEL, "confidence": 1.0}]
        debug_payload.update({
            "rejected": True,
            "reason": "insufficient_activity",
        })
        if include_debug:
            return preds, debug_payload
        return preds

    # Get base prediction or TTA prediction
    if USE_TTA:
        probs = predict_from_seq_tta(raw_seq)
    else:
        probs = predict_probs_from_seq(raw_seq)

    _debug_log_probs(source, "raw", probs, extra=f"tta={USE_TTA}")
    debug_payload["raw_topk"] = _topk_payload(probs, max(1, DEBUG_TOPK))
    
    # Apply temporal smoothing for streaming
    if USE_TEMPORAL_SMOOTH:
        probs = apply_temporal_smoothing(probs, reset=False)
        _debug_log_probs(source, "smooth", probs, extra=f"hist={len(prediction_history)}")
        debug_payload["smooth_topk"] = _topk_payload(probs, max(1, DEBUG_TOPK))
        debug_payload["history_len"] = len(prediction_history)

    top_sorted = np.argsort(probs)[::-1]
    top1 = float(probs[top_sorted[0]])
    top2 = float(probs[top_sorted[1]]) if len(top_sorted) > 1 else 0.0
    if top1 < MIN_TOP1_CONFIDENCE or (top1 - top2) < MIN_TOP12_MARGIN:
        if USE_TEMPORAL_SMOOTH:
            globals()['prediction_history'] = []
        if DEBUG_PREDICTION_LOG:
            print(
                f"[DEBUG-PRED][{source}][gate] rejected: conf={top1:.3f} "
                f"margin={(top1 - top2):.3f}"
            )
        preds = [{"label": NO_SIGN_LABEL, "confidence": 1.0 - top1}]
        debug_payload.update({
            "rejected": True,
            "reason": "confidence_gate",
            "top1": top1,
            "top2": top2,
            "margin": top1 - top2,
        })
        if include_debug:
            return preds, debug_payload
        return preds

    top_indices = top_sorted[:5]
    results = []
    for idx in top_indices:
        label = reverse_label_map.get(idx, "unknown")
        results.append({"label": label, "confidence": float(probs[idx])})
    debug_payload.update({
        "rejected": False,
        "reason": "accepted",
        "top1": top1,
        "top2": top2,
        "margin": top1 - top2,
    })
    if include_debug:
        return results, debug_payload
    return results


class ContinuousDecoder:
    """Heuristic streaming decoder for continuous sign recognition.

    The current repo still uses an isolated-sign classifier, so this class
    provides a stable phrase-building layer on top of frame streaming.
    """

    def __init__(self, commit_hits=3, no_sign_patience=4, max_tokens=32, lm=None, lm_weight=0.35, lm_topk=3):
        self.commit_hits = commit_hits
        self.no_sign_patience = no_sign_patience
        self.max_tokens = max_tokens
        self.lm = lm
        self.lm_weight = float(lm_weight)
        self.lm_topk = int(max(1, lm_topk))
        self.reset()

    def reset(self):
        self.current_label = None
        self.current_hits = 0
        self.no_sign_hits = 0
        self.boundary_seen = True
        self.committed_tokens = []
        self.last_prediction = None
        self.last_debug = None

    def _commit_current(self):
        if self.current_label and self.current_label != NO_SIGN_LABEL:
            if self.boundary_seen or not self.committed_tokens or self.committed_tokens[-1] != self.current_label:
                self.committed_tokens.append(self.current_label)
                self.committed_tokens = self.committed_tokens[-self.max_tokens:]
            self.boundary_seen = False
        self.current_label = None
        self.current_hits = 0

    def _select_top_label(self, predictions):
        if not predictions:
            return NO_SIGN_LABEL, 0.0

        top_label = predictions[0]["label"]
        top_conf = float(predictions[0]["confidence"])
        if not self.lm or not ENABLE_LM_DECODE:
            return top_label, top_conf

        prev_token = self.committed_tokens[-1] if self.committed_tokens else "<s>"
        candidates = predictions[:self.lm_topk]
        best = (top_label, top_conf)
        best_score = -1e9
        for cand in candidates:
            cand_label = cand["label"]
            cand_conf = float(cand["confidence"])
            acoustic = math.log(max(1e-9, cand_conf))
            lm_bonus = 0.0 if cand_label == NO_SIGN_LABEL else self.lm.log_prob(prev_token, cand_label)
            score = acoustic + self.lm_weight * lm_bonus
            if score > best_score:
                best_score = score
                best = (cand_label, cand_conf)
        return best

    def update(self, raw_seq, source="ws/continuous", predictions=None, debug=None):
        if predictions is None or debug is None:
            predictions, debug = predict_from_seq(raw_seq, source=source, include_debug=True)
        self.last_prediction = predictions
        self.last_debug = debug

        top_label, top_conf = self._select_top_label(predictions)

        if top_label == NO_SIGN_LABEL or top_conf < MIN_TOP1_CONFIDENCE:
            self.no_sign_hits += 1
            self.current_label = None
            self.current_hits = 0
            if self.no_sign_hits >= self.no_sign_patience:
                self.boundary_seen = True
                self._commit_current()
        else:
            self.no_sign_hits = 0
            if top_label == self.current_label:
                self.current_hits += 1
            else:
                self.current_label = top_label
                self.current_hits = 1

            if self.current_hits >= self.commit_hits:
                self._commit_current()

        live_tokens = list(self.committed_tokens)
        if self.current_label and self.current_label != NO_SIGN_LABEL:
            live_tokens.append(self.current_label)

        return {
            "committed_tokens": list(self.committed_tokens),
            "committed_text": " ".join(self.committed_tokens).strip(),
            "live_tokens": live_tokens,
            "live_text": " ".join(live_tokens).strip(),
            "current_label": self.current_label,
            "current_hits": self.current_hits,
            "no_sign_hits": self.no_sign_hits,
            "last_prediction": predictions,
            "lm_enabled": bool(self.lm and ENABLE_LM_DECODE),
            "lm_weight": self.lm_weight,
            "debug": debug,
        }


# ============================================================
# Landmarks
# ============================================================
def load_data_mapping():
    mapping = {}
    if DATA_FILE.exists():
        wb = openpyxl.load_workbook(DATA_FILE, read_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and row[1]:
                fname = str(row[0]).replace('.webm', '.mp4')
                mapping[fname] = str(row[1])
        wb.close()
    custom_file = CUSTOM_VIDEOS_DIR / "custom_labels.json"
    if custom_file.exists():
        with open(custom_file) as f:
            mapping.update(json.load(f))
    return mapping


def extract_landmarks(frame, holistic):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    return extract_landmarks_variant(results, KEYPOINT_VARIANT)


def extract_video_landmarks(video_path, seq_length=SEQUENCE_LENGTH):
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
            landmarks_seq.append(extract_landmarks(frame, holistic))
    return np.array(landmarks_seq)


# ============================================================
# Model Loading
# ============================================================
def load_model_if_exists():
    global model, model_members, label_map, reverse_label_map, norm_mean, norm_std, KEYPOINT_VARIANT, sequence_lm

    def _load_member_from_dir(member_dir: Path, expected_labels=None):
        pt_path = member_dir / "sign_model.pt"
        labels_path = member_dir / "labels.json"
        mean_path = member_dir / "norm_mean.npy"
        std_path = member_dir / "norm_std.npy"
        if not pt_path.exists() or not labels_path.exists():
            return None

        with open(labels_path, encoding="utf-8") as f:
            loaded_labels = json.load(f)
        if expected_labels is not None and loaded_labels != expected_labels:
            print(f"[ENSEMBLE] Skip {member_dir}: label map mismatch")
            return None

        checkpoint = torch.load(str(pt_path), map_location=device, weights_only=True)
        num_classes = len(loaded_labels)
        num_features = checkpoint.get("num_features", NUM_RAW_FEATURES * 3 + 9)
        variant_id = checkpoint.get("keypoint_variant")

        m = SignModel(num_features, checkpoint.get("num_classes", num_classes))
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(device)
        m.eval()

        member = {
            "name": str(member_dir),
            "model": m,
            "labels": loaded_labels,
            "num_features": num_features,
            "variant_id": variant_id,
            "norm_mean": np.load(str(mean_path)) if mean_path.exists() and std_path.exists() else None,
            "norm_std": np.load(str(std_path)) if mean_path.exists() and std_path.exists() else None,
        }
        return member

    primary_member = _load_member_from_dir(MODEL_DIR)
    if primary_member is None:
        print("No trained model found. Run train.bat first.")
        return

    label_map = primary_member["labels"]
    reverse_label_map = {int(v): k for k, v in label_map.items()}
    model = primary_member["model"]
    norm_mean = primary_member["norm_mean"]
    norm_std = primary_member["norm_std"]
    variant_id = primary_member.get("variant_id")
    if variant_id:
        try:
            KEYPOINT_VARIANT = KeypointType(variant_id)
        except ValueError:
            pass

    model_members = [
        {
            "name": "primary",
            "model": primary_member["model"],
            "norm_mean": primary_member["norm_mean"],
            "norm_std": primary_member["norm_std"],
        }
    ]

    for extra_dir in ENSEMBLE_MODEL_DIRS:
        member = _load_member_from_dir(Path(extra_dir), expected_labels=label_map)
        if member is None:
            continue
        model_members.append(
            {
                "name": member["name"],
                "model": member["model"],
                "norm_mean": member["norm_mean"],
                "norm_std": member["norm_std"],
            }
        )

    sequence_lm = BigramLM.from_json(LM_PATH) if ENABLE_LM_DECODE else None
    if ENABLE_LM_DECODE and sequence_lm is None:
        print(f"[LM] LM file not found: {LM_PATH}. Using acoustic-only decode.")

    print(f"\n[MODEL] Loaded: {len(label_map)} classes, {primary_member['num_features']} features")
    print(f"[MODEL] Variant: {KEYPOINT_VARIANT.value}")
    print(f"[MODEL] Device: {device}")
    print(f"[ENSEMBLE] Members: {len(model_members)}")
    print(f"[LM] Enabled: {ENABLE_LM_DECODE and sequence_lm is not None} (weight={LM_WEIGHT})")
    print(f"[INFERENCE] TTA (Test-Time Aug): {'ENABLED' if USE_TTA else 'DISABLED'}")
    print(f"[INFERENCE] Temporal Smoothing: {'ENABLED' if USE_TEMPORAL_SMOOTH else 'DISABLED'}")
    print(f"[INFERENCE] Debug prediction log: {'ENABLED' if DEBUG_PREDICTION_LOG else 'DISABLED'}")
    print(f"[INFERENCE] Debug top-k: {DEBUG_TOPK}")
    print(f"[INFERENCE] Debug in API response: {'ENABLED' if DEBUG_INCLUDE_RESPONSE else 'DISABLED'}")
    print(f"[INFERENCE] Confidence threshold: {MIN_TOP1_CONFIDENCE}")
    print(f"[INFERENCE] Margin threshold: {MIN_TOP12_MARGIN}\n")


# ============================================================
# FastAPI
# ============================================================
@asynccontextmanager
async def lifespan(app):
    load_model_if_exists()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def status():
    current_config = VARIANTS[KEYPOINT_VARIANT]
    return {
        "model_loaded": model is not None,
        "num_classes": len(label_map),
        "labels": list(label_map.keys()) if label_map else [],
        "keypoint_variant": KEYPOINT_VARIANT.value,
        "keypoint_features": current_config.feature_size,
        "inference_config": {
            "use_tta": USE_TTA,
            "tta_scales": TTA_SCALES if USE_TTA else None,
            "use_temporal_smooth": USE_TEMPORAL_SMOOTH,
            "temporal_window": TEMPORAL_WINDOW if USE_TEMPORAL_SMOOTH else None,
            "ensemble_members": len(model_members),
            "lm_decode_enabled": bool(sequence_lm is not None and ENABLE_LM_DECODE),
            "lm_weight": LM_WEIGHT,
            "min_confidence": MIN_TOP1_CONFIDENCE,
            "min_margin": MIN_TOP12_MARGIN,
        }
    }

@app.get("/api/labels")
async def get_labels():
    mapping = load_data_mapping()
    unique_labels = sorted(set(mapping.values()), key=str)
    return {"labels": unique_labels, "total_videos": len(mapping)}

@app.get("/api/keypoint/variants")
async def get_keypoint_variants():
    """Get list of all available keypoint variants."""
    variants = []
    for variant_type in KeypointType:
        config = VARIANTS[variant_type]
        variants.append({
            "id": variant_type.value,
            "name": config.name,
            "components": config.components,
            "feature_size": config.feature_size,
        })
    return {"variants": variants}

@app.get("/api/keypoint/current")
async def get_current_keypoint():
    """Get current keypoint variant."""
    config = VARIANTS[KEYPOINT_VARIANT]
    return {
        "variant": KEYPOINT_VARIANT.value,
        "name": config.name,
        "components": config.components,
        "feature_size": config.feature_size,
    }

@app.post("/api/keypoint/set")
async def set_keypoint_variant(data: dict):
    """Set keypoint variant (requires model restart)."""
    global KEYPOINT_VARIANT
    variant_id = data.get("variant")
    try:
        KEYPOINT_VARIANT = KeypointType(variant_id)
        config = VARIANTS[KEYPOINT_VARIANT]
        return {
            "message": f"Switched to {config.name}",
            "variant": KEYPOINT_VARIANT.value,
            "feature_size": config.feature_size,
            "note": "Model needs retraining with this keypoint variant"
        }
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown variant: {variant_id}"}
        )

@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=400, content={"error": "Model not trained yet"})
    temp_path = CUSTOM_VIDEOS_DIR / f"temp_{int(time.time())}.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        seq = extract_video_landmarks(temp_path)
        if seq is None:
            return JSONResponse(status_code=400, content={"error": "Could not process video"})
        predictions, debug = predict_from_seq(seq, source="api/predict/video", include_debug=True)
        response = {"predictions": predictions}
        if DEBUG_INCLUDE_RESPONSE:
            response["debug"] = debug
        return response
    finally:
        temp_path.unlink(missing_ok=True)


def infer_continuous_from_video(video_path: Path, stride=3):
    """Offline continuous decoding over one full video for benchmark evaluation."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        decoder = ContinuousDecoder(
            commit_hits=3,
            no_sign_patience=4,
            max_tokens=64,
            lm=sequence_lm,
            lm_weight=LM_WEIGHT,
            lm_topk=LM_TOPK,
        )
        frame_idx = 0
        last_state = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            lm = extract_landmarks(frame, holistic)
            frame_buffer.append(lm)

            if len(frame_buffer) == SEQUENCE_LENGTH and (frame_idx % max(1, int(stride)) == 0):
                seq = np.array(frame_buffer)
                predictions, debug = predict_from_seq(seq, source="api/predict/continuous_video", include_debug=True)
                last_state = decoder.update(
                    seq,
                    source="api/predict/continuous_video",
                    predictions=predictions,
                    debug=debug,
                )

    cap.release()
    if last_state is None:
        return {
            "committed_tokens": [],
            "committed_text": "",
            "live_tokens": [],
            "live_text": "",
            "current_label": None,
            "current_hits": 0,
            "no_sign_hits": 0,
            "last_prediction": [],
            "lm_enabled": bool(sequence_lm is not None and ENABLE_LM_DECODE),
            "lm_weight": LM_WEIGHT,
            "debug": {"reason": "insufficient_frames"},
        }
    return last_state


@app.post("/api/predict/continuous_video")
async def predict_continuous_video(file: UploadFile = File(...), stride: int = Form(3)):
    if model is None:
        return JSONResponse(status_code=400, content={"error": "Model not trained yet"})

    temp_path = CUSTOM_VIDEOS_DIR / f"temp_cont_{int(time.time())}.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        continuous = infer_continuous_from_video(temp_path, stride=max(1, int(stride)))
        response = {
            "continuous": continuous,
            "pred_sentence_gloss": continuous.get("committed_tokens", []),
            "pred_sentence_text": continuous.get("committed_text", ""),
        }
        if DEBUG_INCLUDE_RESPONSE:
            response["debug"] = continuous.get("debug", {})
        return response
    finally:
        temp_path.unlink(missing_ok=True)

@app.post("/api/predict/frames")
async def predict_frames(data: dict):
    if model is None:
        return JSONResponse(status_code=400, content={"error": "Model not trained yet"})
    frames_b64 = data.get("frames", [])
    if len(frames_b64) < 5:
        return {"predictions": []}
    frames = []
    for b64 in frames_b64:
        img_data = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    if len(frames) < 5:
        return {"predictions": []}
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    sampled = [frames[i] for i in indices]
    landmarks_seq = []
    with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        for frame in sampled:
            landmarks_seq.append(extract_landmarks(frame, holistic))
    predictions, debug = predict_from_seq(np.array(landmarks_seq), source="api/predict/frames", include_debug=True)
    response = {"predictions": predictions}
    if DEBUG_INCLUDE_RESPONSE:
        response["debug"] = debug
    return response

@app.post("/api/train/custom")
async def add_custom_training(file: UploadFile = File(...), label: str = Form(...)):
    fname = f"custom_{int(time.time())}_{file.filename}"
    save_path = CUSTOM_VIDEOS_DIR / fname
    with open(save_path, "wb") as f:
        f.write(await file.read())
    custom_file = CUSTOM_VIDEOS_DIR / "custom_labels.json"
    custom_labels = {}
    if custom_file.exists():
        with open(custom_file) as f:
            custom_labels = json.load(f)
    custom_labels[fname] = label
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_labels, f, ensure_ascii=False, indent=2)
    return {"message": f"Added '{label}' training video", "filename": fname}

@app.get("/api/train/custom/list")
async def list_custom_training():
    custom_file = CUSTOM_VIDEOS_DIR / "custom_labels.json"
    if custom_file.exists():
        with open(custom_file) as f:
            return json.load(f)
    return {}

@app.post("/api/train/start")
async def start_training(data: dict = None):
    return JSONResponse(status_code=400,
        content={"error": "Chay train.bat de huan luyen voi GPU."})

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    holistic = mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    frame_buffer = []
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "frame":
                b64 = msg["data"]
                img_data = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    lm = extract_landmarks(frame, holistic)
                    frame_buffer.append(lm)
                    if len(frame_buffer) > SEQUENCE_LENGTH:
                        frame_buffer.pop(0)
                    if len(frame_buffer) == SEQUENCE_LENGTH and model is not None:
                        seq = np.array(frame_buffer)
                        results, debug = predict_from_seq(seq, source="ws/predict", include_debug=True)
                        payload = {"predictions": results}
                        if DEBUG_INCLUDE_RESPONSE:
                            payload["debug"] = debug
                        await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()


@app.websocket("/ws/continuous")
async def websocket_continuous(websocket: WebSocket):
    await websocket.accept()
    holistic = mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    decoder = ContinuousDecoder(
        commit_hits=3,
        no_sign_patience=4,
        max_tokens=32,
        lm=sequence_lm,
        lm_weight=LM_WEIGHT,
        lm_topk=LM_TOPK,
    )

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            msg_type = msg.get("type", "frame")
            if msg_type == "reset":
                frame_buffer.clear()
                decoder.reset()
                await websocket.send_json({
                    "type": "reset",
                    "predictions": [],
                    "continuous": {
                        "committed_tokens": [],
                        "committed_text": "",
                        "live_tokens": [],
                        "live_text": "",
                        "current_label": None,
                        "current_hits": 0,
                        "no_sign_hits": 0,
                    },
                })
                continue

            if msg_type != "frame":
                continue

            b64 = msg.get("data")
            if not b64:
                continue

            img_data = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            lm = extract_landmarks(frame, holistic)
            frame_buffer.append(lm)

            if len(frame_buffer) == SEQUENCE_LENGTH and model is not None:
                seq = np.array(frame_buffer)
                predictions, debug = predict_from_seq(seq, source="ws/continuous", include_debug=True)
                continuous_state = decoder.update(
                    seq,
                    source="ws/continuous",
                    predictions=predictions,
                    debug=debug,
                )

                payload = {
                    "type": "continuous",
                    "predictions": predictions,
                    "continuous": continuous_state,
                }
                if DEBUG_INCLUDE_RESPONSE:
                    payload["debug"] = debug
                await websocket.send_json(payload)

    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
