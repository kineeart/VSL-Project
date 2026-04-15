"""Vietnamese Sign Language - Max Accuracy GPU Training (PyTorch + CUDA)"""
import os, sys, json, time, cv2, numpy as np, openpyxl, math
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp_proc
import torch
import torch.nn as nn
import torch.nn.functional as F
from keypoint_variants import KeypointType, VARIANTS, extract_landmarks_variant
from spatial_augmentation import apply_spatial_augmentation, get_xy_indices

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "Videos"
DATA_FILE = BASE_DIR / "Data.xlsx"
MODEL_DIR = Path(
    os.environ.get("VSL_MODEL_DIR", str(Path(__file__).resolve().parent / "models"))
)
LANDMARKS_DIR = Path(
    os.environ.get("VSL_LANDMARKS_DIR", str(Path(__file__).resolve().parent / "landmarks"))
)
CUSTOM_VIDEOS_DIR = Path(__file__).resolve().parent / "custom_videos"
for d in [MODEL_DIR, LANDMARKS_DIR, CUSTOM_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# KEYPOINT CONFIGURATION - Change this to use different variants
# ============================================================
KEYPOINT_VARIANT = KeypointType.HOLISTIC  # Change to experiment with different keypoint combinations
VARIANT_CONFIG = VARIANTS[KEYPOINT_VARIANT]
HAS_HANDS_POSE = VARIANT_CONFIG.include_pose and VARIANT_CONFIG.include_left_hand and VARIANT_CONFIG.include_right_hand

SEQ = 60
NRF = VARIANT_CONFIG.feature_size  # Raw features based on selected variant
EXTRA_FEATURES = 9 if HAS_HANDS_POSE else 0
NF = NRF * 3 + EXTRA_FEATURES
BS = 32
EPOCHS = 200
PAT = 30
LR = 0.0005
WARM = 8
SWA_EP = 100
MIXUP_ALPHA = 0.4
MIN_SAMPLES = 3  # Only train classes with >= 3 videos
NW = max(1, mp_proc.cpu_count() - 2)
FAILED_VIDEOS = set()
FAILED_CLASSES = set()


def video_class_name(filename: str) -> str:
    stem = Path(filename).stem
    if stem and stem[-1] in ("B", "N", "T"):
        return stem[:-1]
    return stem

# ============================================================
# Data Limiting (set to 0 or None to load all rows)
# ============================================================
MAX_DATA_ROWS = 0  # Load all rows from Data.xlsx by default (set >0 only for debug runs)

# Fixed 20-class subset mode (requested D0001..D0020 words).
# When enabled, training ignores MAX_DATA_ROWS and loads only these videos,
# while grouping B/N/T variants under the same label.
USE_FIXED_20_CLASSES = False
FIXED_CLASS_LIMIT = 0  # Debug-only limit for fixed subset mode (unused when USE_FIXED_20_CLASSES=False)
FIXED_20_FILES_WEBM = [
    "D0001B.webm", "D0001N.webm", "D0001T.webm",
    "D0002.webm", "D0003.webm", "D0004.webm",
    "D0005B.webm", "D0005N.webm", "D0005T.webm",
    "D0006.webm", "D0007.webm", "D0008.webm", "D0009.webm", "D0010.webm",
    "D0011.webm", "D0012.webm", "D0013.webm", "D0014.webm", "D0015.webm", "D0016.webm",
    "D0017.webm", "D0018.webm",
    "D0019B.webm", "D0019N.webm", "D0019T.webm",
    "D0020B.webm", "D0020N.webm", "D0020T.webm",
]

# Oversampling target (train set only, after val split).
# Set to 0 to disable class balancing by synthetic augmentation.
TARGET_SAMPLES_PER_CLASS = 100

# Spatial augmentation: simulate subject moving closer/farther and shifting in frame.
SPATIAL_AUG_PROB = 0.65
SPATIAL_SCALE_MIN = 0.75
SPATIAL_SCALE_MAX = 1.45

# ST-GCN-inspired temporal-smooth geometric augmentation knobs (minimal integration).
SMOOTH_MOVE_PROB = 0.55
RANDOM_SHIFT_PROB = 0.45


# ============================================================
# GPU Setup
# ============================================================
def setup_gpu():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n[GPU] {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
        return torch.device('cuda')
    print("\n[CPU] Khong tim thay GPU, dung CPU.")
    return torch.device('cpu')



# ============================================================
# Data Loading
# ============================================================
def load_data_mapping():
    def base_code(filename: str) -> str:
        stem = Path(filename).stem
        if stem and stem[-1] in ("B", "N", "T"):
            return stem[:-1]
        return stem

    fixed_mp4_all = {f.replace('.webm', '.mp4') for f in FIXED_20_FILES_WEBM}
    if USE_FIXED_20_CLASSES and FIXED_CLASS_LIMIT and FIXED_CLASS_LIMIT > 0:
        fixed_codes = sorted({base_code(f) for f in fixed_mp4_all})
        selected_codes = set(fixed_codes[:FIXED_CLASS_LIMIT])
        fixed_mp4 = {f for f in fixed_mp4_all if base_code(f) in selected_codes}
    else:
        fixed_mp4 = fixed_mp4_all

    m = {}
    if DATA_FILE.exists():
        wb = openpyxl.load_workbook(DATA_FILE, read_only=True)
        ws = wb.active
        row_count = 0
        for r in ws.iter_rows(min_row=2, values_only=True):
            if (not USE_FIXED_20_CLASSES) and MAX_DATA_ROWS > 0 and row_count >= MAX_DATA_ROWS:
                break
            if r[0] and r[1]:
                fn = str(r[0]).replace('.webm', '.mp4')
                if USE_FIXED_20_CLASSES and fn not in fixed_mp4:
                    continue
                m[fn] = str(r[1]).strip()
                row_count += 1
        wb.close()

    if USE_FIXED_20_CLASSES:
        # Force B/N/T variants to share the same label by base code.
        by_code = {}
        for fn, lb in m.items():
            code = base_code(fn)
            if code not in by_code:
                by_code[code] = lb

        remapped = {}
        for fn in fixed_mp4:
            code = base_code(fn)
            if code in by_code:
                remapped[fn] = by_code[code]
        m = remapped

    cf = CUSTOM_VIDEOS_DIR / "custom_labels.json"
    if cf.exists() and not USE_FIXED_20_CLASSES:
        with open(cf, encoding='utf-8') as f:
            m.update(json.load(f))

    if USE_FIXED_20_CLASSES:
        selected_classes = sorted({base_code(fn) for fn in m})
        print(
            f"[DATA] Fixed-subset mode ON. Selected classes: {len(selected_classes)} "
            f"(limit={FIXED_CLASS_LIMIT or 'all'})  Selected videos: {len(m)}"
        )

    return m


# ============================================================
# Landmark Extraction (parallel)
# ============================================================
def extract_single_video(args):
    try:
        import mediapipe as mp_lib
        from keypoint_variants import KeypointType, VARIANTS, extract_landmarks_variant
        # Note: KEYPOINT_VARIANT must be reimported in each worker process
        fn, vp, cp, sl, variant_type = args
        variant_config = VARIANTS[variant_type]
        expected_shape = (sl, variant_config.feature_size)

        if cp.exists():
            try:
                s = np.load(str(cp))
                if s.shape == expected_shape:
                    return fn, True, None
            except Exception:
                pass

        cap = cv2.VideoCapture(str(vp))
        frames = []
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return fn, False, "Video read returned no frames"

        idx = np.linspace(0, len(frames) - 1, sl, dtype=int)
        samp = [frames[i] for i in idx]
        lms = []
        with mp_lib.solutions.holistic.Holistic(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as h:
            for fr in samp:
                img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                r = h.process(img)
                lm = extract_landmarks_variant(r, variant_type)
                lms.append(lm)
        np.save(str(cp), np.array(lms))
        return fn, True, None
    except Exception as e:
        return args[0], False, f"{type(e).__name__}: {e}"


def extract_single_video_isolated(args):
    try:
        with ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(extract_single_video, args)
            return fut.result()
    except Exception as e:
        return args[0], False, f"{type(e).__name__}: {e}"


def landmark_cache_path(filename: str, variant_type: KeypointType) -> Path:
    """Return variant-specific landmark cache path for one video file."""
    stem = Path(filename).stem
    return LANDMARKS_DIR / f"{stem}__{variant_type.value}.npy"


def extract_all(mapping):
    global FAILED_VIDEOS, FAILED_CLASSES
    FAILED_VIDEOS = set()
    FAILED_CLASSES = set()

    tasks = []
    for fn in mapping:
        vp = None
        for d in [VIDEOS_DIR, CUSTOM_VIDEOS_DIR]:
            if (d / fn).exists():
                vp = d / fn
                break
        if vp:
            tasks.append((fn, vp, landmark_cache_path(fn, KEYPOINT_VARIANT), SEQ, KEYPOINT_VARIANT))
    cached = sum(1 for _, _, c, _, _ in tasks if c.exists())
    tot = len(tasks)
    print(f"\n[LM] Keypoint Variant: {KEYPOINT_VARIANT.value} ({VARIANT_CONFIG.name}, {NRF} features)")
    print(f"[LM] Total: {tot}  Cached: {cached}  Todo: {tot - cached}")
    if cached == tot:
        return
    unc = [t for t in tasks if not t[2].exists()]
    t0 = time.time()
    ok = 0
    fail = 0

    def _mark_failed(fn, reason):
        cls = video_class_name(fn)
        FAILED_VIDEOS.add(fn)
        FAILED_CLASSES.add(cls)
        print(f"  ✗ {fn} ({cls}): {reason}")

    def _run_sequential():
        nonlocal ok, fail
        for t in unc:
            _, s, reason = extract_single_video_isolated(t)
            if s:
                ok += 1
            else:
                fail += 1
                _mark_failed(t[0], reason or "unknown error")
            td = ok + fail
            if td % 100 == 0 or td == len(unc):
                el = time.time() - t0
                sp = td / max(el, 1)
                print(f"  [{ok + cached}/{tot}] {sp:.1f} v/s  ~{(len(unc) - td) / max(sp, .01):.0f}s left")

    if NW <= 1:
        print("[LM] Extracting sequentially in the main process...")
        _run_sequential()
    else:
        print(f"[LM] Extracting with {NW} workers...")
        try:
            with ProcessPoolExecutor(max_workers=NW) as ex:
                futs = {ex.submit(extract_single_video, t): t[0] for t in unc}
                for fut in as_completed(futs):
                    fn = futs[fut]
                    try:
                        res_fn, s, reason = fut.result()
                    except Exception as e:
                        fail += 1
                        _mark_failed(fn, f"{type(e).__name__}: {e}")
                        continue
                    if s:
                        ok += 1
                    else:
                        fail += 1
                        _mark_failed(res_fn, reason or "unknown error")
                    td = ok + fail
                    if td % 100 == 0 or td == len(unc):
                        el = time.time() - t0
                        sp = td / max(el, 1)
                        print(f"  [{ok + cached}/{tot}] {sp:.1f} v/s  ~{(len(unc) - td) / max(sp, .01):.0f}s left")
        except Exception as e:
            print(f"  ⚠ Process pool crashed ({type(e).__name__}: {e}), retrying sequentially...")
            ok = 0
            fail = 0
            _run_sequential()
    print(f"[LM] Done! {ok} ok, {fail} fail ({time.time() - t0:.0f}s)\n")



# ============================================================
# Feature Engineering (must match app.py exactly)
# ============================================================
def eng_feat(s, variant=None):
    """
    Raw landmarks -> engineered features.
    
    Args:
        s: Raw landmarks of shape (seq_len, nrf)
        variant: KeypointType. If None, uses KEYPOINT_VARIANT
    
    Returns:
        Engineered features of shape (seq_len, nf) where nf = 3*nrf + extra
    """
    if variant is None:
        variant = KEYPOINT_VARIANT
    config = VARIANTS[variant]
    
    # Always compute velocity and acceleration
    v = np.zeros_like(s)
    v[1:] = s[1:] - s[:-1]
    a = np.zeros_like(s)
    a[1:] = v[1:] - v[:-1]
    
    # Compute extra engineered features if hands and pose are available
    ex = []
    if config.include_pose and config.include_left_hand and config.include_right_hand:
        # Extract nose position (first 3 values of pose)
        nose = s[:, 0:3]
        
        # Left hand starts after pose
        if config.include_pose_visibility:
            pose_size = config.pose_points * 4
        else:
            pose_size = config.pose_points * 3
        ls = pose_size
        rs = ls + 21 * 3
        
        # Extract wrist positions (first point of each hand)
        lw = s[:, ls:ls + 3]
        rw = s[:, rs:rs + 3]
        
        ex = np.concatenate([
            lw - nose,                                        # left hand relative to nose (3)
            rw - nose,                                        # right hand relative to nose (3)
            np.linalg.norm(lw - rw, axis=1, keepdims=True), # hand distance (1)
            np.linalg.norm(v[:, ls:ls + 3], axis=1, keepdims=True),  # left hand speed (1)
            np.linalg.norm(v[:, rs:rs + 3], axis=1, keepdims=True),  # right hand speed (1)
        ], axis=1)  # total extra = 11
    
    if len(ex) == 0:
        # If we can't compute extra features, just use raw + velocity + acceleration
        return np.concatenate([s, v, a], axis=1).astype(np.float32)
    else:
        return np.concatenate([s, v, a, ex], axis=1).astype(np.float32)


# ============================================================
# Mirror Augmentation (swap left/right hands + flip x)
# ============================================================
def mirror(s, variant=None):
    """
    Mirror augmentation: swap left/right hands and flip x coordinates.
    Only effective if variant includes both hands.
    
    Args:
        s: Raw landmarks of shape (seq_len, nrf)
        variant: KeypointType. If None, uses KEYPOINT_VARIANT
    
    Returns:
        Mirrored landmarks
    """
    if variant is None:
        variant = KEYPOINT_VARIANT
    config = VARIANTS[variant]
    
    m = s.copy()
    
    # Only mirror if we have both hands
    if not (config.include_left_hand and config.include_right_hand):
        return m
    
    # Calculate positions dynamically
    pose_size = 0
    if config.include_pose:
        pose_size = config.pose_points * (4 if config.include_pose_visibility else 3)
    
    ls = pose_size
    le = ls + 21 * 3  # left hand size
    rs = le
    re = rs + 21 * 3  # right hand size
    
    # Swap left and right hand landmarks
    m[:, ls:le], m[:, rs:re] = s[:, rs:re].copy(), s[:, ls:le].copy()
    
    # Flip x coordinates for pose
    if config.include_pose:
        step = 4 if config.include_pose_visibility else 3
        for i in range(config.pose_points):
            m[:, i * step] = 1 - m[:, i * step]
    
    # Flip x coordinates for hands
    for st in [ls, rs]:
        for i in range(21):
            m[:, st + i * 3] = 1 - m[:, st + i * 3]
    
    # Flip x coordinates for face
    if config.include_face:
        fs = re
        for i in range(468):
            m[:, fs + i * 3] = 1 - m[:, fs + i * 3]
    
    return m


def auto_padding_seq(s, size, random_pad=False):
    """Pad [T, F] sequence to target length (ST-GCN style auto padding)."""
    t, f = s.shape
    if t >= size:
        return s
    begin = np.random.randint(0, size - t + 1) if random_pad else 0
    out = np.zeros((size, f), dtype=s.dtype)
    out[begin:begin + t] = s
    return out


def random_choose_seq(s, size, auto_pad=True):
    """Choose random temporal crop to target size (ST-GCN style random choose)."""
    t = s.shape[0]
    if t == size:
        return s
    if t < size:
        return auto_padding_seq(s, size, random_pad=True) if auto_pad else s
    begin = np.random.randint(0, t - size + 1)
    return s[begin:begin + size]


def random_shift_activity_seq(s, eps=1e-8):
    """Shift active temporal window inside the same length with zero padding."""
    out = np.zeros_like(s)
    valid_frame = np.sum(np.abs(s), axis=1) > eps
    if not np.any(valid_frame):
        return s
    begin = int(np.argmax(valid_frame))
    end = int(len(valid_frame) - np.argmax(valid_frame[::-1]))
    size = max(1, end - begin)
    bias = np.random.randint(0, max(1, s.shape[0] - size + 1))
    out[bias:bias + size] = s[begin:end]
    return out


def smooth_random_move_seq(s, variant=None):
    """Smooth frame-wise geometric transform using interpolated angle/scale/translation."""
    if variant is None:
        variant = KEYPOINT_VARIANT

    x_idx, y_idx = get_xy_indices(variant)
    if x_idx.size == 0:
        return s

    out = s.copy()
    t = out.shape[0]
    if t < 2:
        return out

    node = np.array([0, t], dtype=int)
    angle_nodes = np.random.choice(np.array([-8.0, -4.0, 0.0, 4.0, 8.0]), len(node))
    scale_nodes = np.random.choice(np.array([0.94, 0.98, 1.0, 1.02, 1.06]), len(node))
    tx_nodes = np.random.choice(np.array([-0.06, -0.03, 0.0, 0.03, 0.06]), len(node))
    ty_nodes = np.random.choice(np.array([-0.06, -0.03, 0.0, 0.03, 0.06]), len(node))

    a = np.linspace(angle_nodes[0], angle_nodes[1], t) * np.pi / 180.0
    sc = np.linspace(scale_nodes[0], scale_nodes[1], t)
    tx = np.linspace(tx_nodes[0], tx_nodes[1], t)
    ty = np.linspace(ty_nodes[0], ty_nodes[1], t)

    # Apply transform only to valid keypoints (non-zero x/y) to avoid moving missing landmarks.
    for i in range(t):
        x = out[i, x_idx]
        y = out[i, y_idx]
        valid = np.isfinite(x) & np.isfinite(y) & ((np.abs(x) > 1e-8) | (np.abs(y) > 1e-8))
        if not np.any(valid):
            continue

        xv = x[valid] - 0.5
        yv = y[valid] - 0.5
        ca = np.cos(a[i])
        sa = np.sin(a[i])
        x_new = sc[i] * (ca * xv - sa * yv) + 0.5 + tx[i]
        y_new = sc[i] * (sa * xv + ca * yv) + 0.5 + ty[i]

        x[valid] = np.clip(x_new, 0.0, 1.0)
        y[valid] = np.clip(y_new, 0.0, 1.0)
        out[i, x_idx] = x
        out[i, y_idx] = y

    return out


# ============================================================
# Data Augmentation (7 types)
# ============================================================
def aug1(s, variant=None):
    """Apply random augmentations to raw landmark sequence."""
    if variant is None:
        variant = KEYPOINT_VARIANT
    config = VARIANTS[variant]
    
    a = s.copy()
    # 1. Noise injection
    if np.random.random() < 0.85:
        a += np.random.normal(0, np.random.uniform(0.003, 0.025), a.shape)
    # 2. Smooth geometric motion (ST-GCN-inspired random_move).
    if np.random.random() < SMOOTH_MOVE_PROB:
        a = smooth_random_move_seq(a, variant)
    # 2. Global spatial transform on a single sequence-level box (keeps keypoint ratios).
    if np.random.random() < SPATIAL_AUG_PROB:
        a = apply_spatial_augmentation(
            a,
            variant=variant,
            scale_min=SPATIAL_SCALE_MIN,
            scale_max=SPATIAL_SCALE_MAX,
        )
    # 3. Activity-window shift (ST-GCN-inspired random_shift).
    if np.random.random() < RANDOM_SHIFT_PROB:
        a = random_shift_activity_seq(a)
    # 4. Speed variation
    if np.random.random() < 0.6:
        sp = np.random.uniform(0.8, 1.2)
        n = max(3, int(a.shape[0] * sp))
        ix = np.linspace(0, a.shape[0] - 1, n, dtype=float)
        rs = np.array([a[min(int(round(i)), a.shape[0] - 1)] for i in ix])
        a = rs[np.linspace(0, len(rs) - 1, a.shape[0], dtype=int)]
    # 5. Frame dropout
    if np.random.random() < 0.4:
        di = np.random.choice(a.shape[0], np.random.randint(1, 5), replace=False)
        a[di] = 0
    # 6. Temporal reverse
    if np.random.random() < 0.3:
        a = a[::-1].copy()
    # 7. Hand-specific noise (only if variant has hands)
    if config.include_left_hand and config.include_right_hand and np.random.random() < 0.5:
        pose_size = 0
        if config.include_pose:
            pose_size = config.pose_points * (4 if config.include_pose_visibility else 3)
        hs = pose_size
        he = pose_size + 21 * 3 * 2
        if he <= a.shape[1]:  # Make sure hand section exists
            a[:, hs:he] += np.random.normal(0, 0.015, a[:, hs:he].shape)
    return a.astype(np.float32)


def augment(raw, na, variant=None):
    """Generate augmented samples from raw landmarks."""
    if variant is None:
        variant = KEYPOINT_VARIANT
    out = [eng_feat(raw, variant)]
    mi = mirror(raw, variant)
    out.append(eng_feat(mi, variant))
    for _ in range(na - 1):
        out.append(eng_feat(aug1(raw, variant), variant))
        if np.random.random() < 0.5:
            out.append(eng_feat(aug1(mi, variant), variant))
    return out



# ============================================================
# Model Architecture (must match app.py exactly)
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
        return self.out(out).mean(dim=1)  # pool over time


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
        # Multi-scale temporal convolutions (capture different temporal patterns)
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        # Deeper BiLSTM
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        # Multi-head attention
        self.attention = MultiHeadAttention(768, num_heads=4)
        # Dense layers
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        # Cosine classifier for few-shot
        self.fc3 = CosineClassifier(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        # Multi-scale conv
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)  # 256+128+128=512
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
# Focal Loss with label smoothing + class weights
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inp, tgt):
        nc = inp.size(1)
        smooth_tgt = torch.zeros_like(inp).scatter_(1, tgt.unsqueeze(1), 1.0)
        smooth_tgt = smooth_tgt * (1 - self.smoothing) + self.smoothing / nc
        log_prob = F.log_softmax(inp, dim=1)
        focal_weight = (1 - torch.exp(log_prob)) ** self.gamma
        loss = -focal_weight * smooth_tgt * log_prob
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(1).mean()


# ============================================================
# Warmup + Cosine Annealing Scheduler
# ============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr



# ============================================================
# Data Preparation (stratified split, pre-loaded)
# ============================================================
def prep_data(mapping):
    """Prepare data: filter by MIN_SAMPLES, split train/val, augment, normalize."""
    mapping = {
        fn: lb
        for fn, lb in mapping.items()
        if fn not in FAILED_VIDEOS and lb not in FAILED_CLASSES
    }
    label_counts = Counter(mapping.values())
    # In fixed subset mode, allow classes with even 1 source video and rely on augmentation.
    effective_min_samples = 1 if USE_FIXED_20_CLASSES else MIN_SAMPLES
    valid_labels = {lb for lb, cnt in label_counts.items() if cnt >= effective_min_samples}
    filtered = {fn: lb for fn, lb in mapping.items() if lb in valid_labels}

    unique_labels = sorted(valid_labels, key=str)
    label_map = {str(l): i for i, l in enumerate(unique_labels)}
    nc = len(label_map)

    label_to_files = {}
    for fn, lb in filtered.items():
        label_to_files.setdefault(lb, []).append(fn)

    print(f"[DATA] Filter >= {effective_min_samples} samples: {nc} classes, {len(filtered)} videos")
    print(f"  (Skipped {len(mapping) - len(filtered)} videos from {len(label_counts) - nc} rare classes)")
    if FAILED_VIDEOS or FAILED_CLASSES:
        print(f"  (Skipped {len(FAILED_VIDEOS)} failed videos from {len(FAILED_CLASSES)} failed classes)")

    X_train, y_train, X_val, y_val = [], [], [], []
    class_train_counts = {}
    loaded = 0
    skipped = 0

    split_ratio = 0.2

    for lb, fns in label_to_files.items():
        li = label_map[str(lb)]
        cnt = label_counts[lb]
        seqs = []
        for fn in fns:
            cp = landmark_cache_path(fn, KEYPOINT_VARIANT)
            if not cp.exists():
                skipped += 1
                continue
            try:
                s = np.load(str(cp))
                if s.ndim != 2 or s.shape[1] != NRF:
                    skipped += 1
                    continue
                if s.shape[0] != SEQ:
                    # ST-GCN-inspired temporal normalize: random crop / random pad to fixed length.
                    s = random_choose_seq(s, SEQ, auto_pad=True)
                seqs.append(s)
                loaded += 1
            except:
                skipped += 1
                continue

        if len(seqs) < 1:
            continue

        # Clear per-class split: 80% train / 20% val (stratified by class, no overlap).
        np.random.shuffle(seqs)
        n_total = len(seqs)
        if n_total == 1:
            # Keep single-sample classes in train only; avoid train/val leakage.
            train_seqs = seqs
            val_seqs = []
        else:
            n_val = max(1, int(round(n_total * split_ratio)))
            n_val = min(n_val, n_total - 1)  # Always keep at least one training sample
            val_seqs = seqs[:n_val]
            train_seqs = seqs[n_val:]

        for s in val_seqs:
            X_val.append(eng_feat(s, KEYPOINT_VARIANT))
            y_val.append(li)

        # Base augmentation from each real training sequence.
        na = 2
        class_feats = []
        for s in train_seqs:
            for a in augment(s, na, KEYPOINT_VARIANT):
                class_feats.append(a)

        # Auto-balance: oversample classes that have fewer than target samples.
        if TARGET_SAMPLES_PER_CLASS and TARGET_SAMPLES_PER_CLASS > 0:
            while len(class_feats) < TARGET_SAMPLES_PER_CLASS:
                base_seq = train_seqs[np.random.randint(0, len(train_seqs))]
                extra = augment(base_seq, 1, KEYPOINT_VARIANT)
                need = TARGET_SAMPLES_PER_CLASS - len(class_feats)
                class_feats.extend(extra[:need])

        class_train_counts[str(lb)] = len(class_feats)
        X_train.extend(class_feats)
        y_train.extend([li] * len(class_feats))

    print(f"[DATA] Loaded: {loaded}  Skipped: {skipped}  Train: {len(X_train)}  Val: {len(X_val)}")
    if class_train_counts:
        mn = min(class_train_counts.values())
        mx = max(class_train_counts.values())
        print(f"[DATA] Train samples/class after balancing: min={mn}, max={mx}, target={TARGET_SAMPLES_PER_CLASS}")

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)

    # Normalization
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    np.save(str(MODEL_DIR / "norm_mean.npy"), mean)
    np.save(str(MODEL_DIR / "norm_std.npy"), std)
    print(f"[DATA] Normalization saved. RAM: ~{X_train.nbytes / 1024**3:.1f} GB")

    return X_train, y_train, X_val, y_val, label_map, nc



# ============================================================
# Mixup Training (proven to boost generalization)
# ============================================================
def mixup_data(x, y, alpha=0.4):
    """Mixup: interpolate between random pairs of samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted combination of losses for both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Top-K Accuracy
# ============================================================
def topk_accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy safely for any class count."""
    with torch.no_grad():
        num_classes = output.size(1)
        safe_topk = [min(k, num_classes) for k in topk]
        maxk = max(safe_topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in safe_topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item() / batch_size)
        return res


# ============================================================
# Main Training Loop
# ============================================================
def train():
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim.swa_utils import AveragedModel, SWALR

    print("=" * 60)
    print("  Vietnamese Sign Language - Max Accuracy Training")
    print("=" * 60)

    device = setup_gpu()

    # Step 1: Load data mapping
    print("\n[1/5] Loading data mapping...")
    mapping = load_data_mapping()
    if not mapping:
        print("[LOI] Khong tim thay du lieu!")
        return
    print(f"  Found {len(mapping)} video-label pairs.")

    # Step 2: Extract landmarks
    print("\n[2/5] Extracting landmarks...")
    extract_all(mapping)

    # Step 3: Feature engineering + augmentation
    print("\n[3/5] Feature engineering + augmentation...")
    X_train, y_train, X_val, y_val, label_map, nc = prep_data(mapping)
    if len(X_train) == 0:
        print("[LOI] Khong co du lieu train!")
        return

    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    # Class weights
    ct = Counter(y_train.tolist())
    total = len(y_train)
    weights = torch.zeros(nc)
    for i in range(nc):
        weights[i] = min(total / (nc * ct.get(i, 1)), 15.0)
    weights = weights.to(device)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, pin_memory=True)

    # Step 4: Build model
    print(f"\n[4/5] Building model ({nc} classes, {NF} features)...")
    model = SignModel(NF, nc).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = FocalLoss(weight=weights, gamma=2.0, smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARM, EPOCHS)

    # SWA setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    swa_active = False

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Step 5: Training
    print(f"\n[5/5] Training... Device: {device}  Epochs: {EPOCHS}  BS: {BS}  Patience: {PAT}\n")
    best_val_top1 = 0
    best_val_top5 = 0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        # Learning rate scheduling
        if epoch < SWA_EP:
            scheduler.step(epoch)
        else:
            if not swa_active:
                swa_active = True
                print("\n  --- SWA activated ---\n")
            swa_scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        # Train phase with Mixup
        model.train()
        train_correct, train_total = 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # Apply Mixup
            mixed_x, ya, yb_mix, lam = mixup_data(xb, yb, MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    out = model(mixed_x)
                    loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(mixed_x)
                loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_correct += (out.detach().argmax(1) == ya).sum().item()
            train_total += xb.size(0)

        if swa_active:
            swa_model.update_parameters(model)

        train_acc = train_correct / max(train_total, 1)

        # Validation phase with Top-1 and Top-5
        eval_model = swa_model if swa_active else model
        eval_model.eval()
        val_top1_sum, val_top5_sum, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = eval_model(xb)
                t1, t5 = topk_accuracy(out, yb, topk=(1, 5))
                bs = xb.size(0)
                val_top1_sum += t1 * bs
                val_top5_sum += t5 * bs
                val_total += bs

        val_top1 = val_top1_sum / max(val_total, 1)
        val_top5 = val_top5_sum / max(val_total, 1)
        elapsed = time.time() - t0

        print(f"  Ep {epoch+1:3d}/{EPOCHS}  Train: {train_acc*100:.2f}%  Val-Top1: {val_top1*100:.2f}%  Val-Top5: {val_top5*100:.2f}%  LR: {lr:.1e}  {elapsed:.0f}s")

        # Save best model (prioritize top-1; top-5 only as tie-breaker)
        if (val_top1 > best_val_top1) or (val_top1 == best_val_top1 and val_top5 > best_val_top5):
            best_val_top5 = val_top5
            best_val_top1 = val_top1
            patience_counter = 0
            state = swa_model.module.state_dict() if swa_active else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'num_classes': nc,
                'num_features': NF,
                'sequence_length': SEQ,
                'num_raw_features': NRF,
                'keypoint_variant': KEYPOINT_VARIANT.value,
            }, str(MODEL_DIR / "sign_model_best.pt"))
            print(f"    ^ Best! Top1: {val_top1*100:.2f}%  Top5: {val_top5*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PAT and epoch >= SWA_EP:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Final: update SWA batch norm
    if swa_active:
        print("\n  Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # Save final model
    best = torch.load(str(MODEL_DIR / "sign_model_best.pt"), weights_only=True)
    torch.save(best, str(MODEL_DIR / "sign_model.pt"))

    # Save label map
    with open(MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  HOAN TAT!")
    print(f"  Thoi gian: {elapsed/60:.1f} phut")
    print(f"  So lop: {nc}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Best Val Top-1: {best_val_top1*100:.2f}%")
    print(f"  Best Val Top-5: {best_val_top5*100:.2f}%")
    print(f"  Model: {MODEL_DIR / 'sign_model.pt'}")
    print(f"{'=' * 60}")

    # Save training history
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump({
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "minutes": elapsed / 60,
            "num_classes": nc,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "num_features": NF,
        }, f, indent=2)


if __name__ == "__main__":
    train()
