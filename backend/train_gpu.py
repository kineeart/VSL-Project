"""Vietnamese Sign Language - Max Accuracy GPU Training (PyTorch + CUDA)"""
import os, sys, json, time, cv2, numpy as np, openpyxl, math
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp_proc
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "Videos"
DATA_FILE = BASE_DIR / "Data.xlsx"
MODEL_DIR = Path(__file__).resolve().parent / "models"
LANDMARKS_DIR = Path(__file__).resolve().parent / "landmarks"
CUSTOM_VIDEOS_DIR = Path(__file__).resolve().parent / "custom_videos"
for d in [MODEL_DIR, LANDMARKS_DIR, CUSTOM_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEQ = 60
NRF = 33 * 4 + 21 * 3 * 2 + 468 * 3  # 1662 raw features
NF = NRF * 3 + 9  # 4995 total features
BS = 32
EPOCHS = 200
PAT = 30
LR = 0.0005
WARM = 8
SWA_EP = 100
MIXUP_ALPHA = 0.4
MIN_SAMPLES = 3  # Only train classes with >= 3 videos
NW = max(1, mp_proc.cpu_count() - 2)


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
    m = {}
    if DATA_FILE.exists():
        wb = openpyxl.load_workbook(DATA_FILE, read_only=True)
        ws = wb.active
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[0] and r[1]:
                m[str(r[0]).replace('.webm', '.mp4')] = str(r[1]).strip()
        wb.close()
    cf = CUSTOM_VIDEOS_DIR / "custom_labels.json"
    if cf.exists():
        with open(cf, encoding='utf-8') as f:
            m.update(json.load(f))
    return m


# ============================================================
# Landmark Extraction (parallel)
# ============================================================
def extract_single_video(args):
    import mediapipe as mp_lib
    fn, vp, cp, sl = args
    if cp.exists():
        try:
            s = np.load(str(cp))
            if s.shape == (sl, NRF):
                return fn, True
        except:
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
        return fn, False
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
            p = np.array([[l.x, l.y, l.z, l.visibility] for l in r.pose_landmarks.landmark]).flatten() \
                if r.pose_landmarks else np.zeros(33 * 4)
            lh = np.array([[l.x, l.y, l.z] for l in r.left_hand_landmarks.landmark]).flatten() \
                if r.left_hand_landmarks else np.zeros(21 * 3)
            rh = np.array([[l.x, l.y, l.z] for l in r.right_hand_landmarks.landmark]).flatten() \
                if r.right_hand_landmarks else np.zeros(21 * 3)
            fc = np.array([[l.x, l.y, l.z] for l in r.face_landmarks.landmark]).flatten() \
                if r.face_landmarks else np.zeros(468 * 3)
            lms.append(np.concatenate([p, lh, rh, fc]))
    np.save(str(cp), np.array(lms))
    return fn, True


def extract_all(mapping):
    tasks = []
    for fn in mapping:
        vp = None
        for d in [VIDEOS_DIR, CUSTOM_VIDEOS_DIR]:
            if (d / fn).exists():
                vp = d / fn
                break
        if vp:
            tasks.append((fn, vp, LANDMARKS_DIR / f"{fn}.npy", SEQ))
    cached = sum(1 for _, _, c, _ in tasks if c.exists())
    tot = len(tasks)
    print(f"\n[LM] Total: {tot}  Cached: {cached}  Todo: {tot - cached}")
    if cached == tot:
        return
    unc = [t for t in tasks if not t[2].exists()]
    t0 = time.time()
    ok = 0
    fail = 0
    print(f"[LM] Extracting with {NW} workers...")
    with ProcessPoolExecutor(max_workers=NW) as ex:
        futs = {ex.submit(extract_single_video, t): t[0] for t in unc}
        for fut in as_completed(futs):
            _, s = fut.result()
            if s:
                ok += 1
            else:
                fail += 1
            td = ok + fail
            if td % 100 == 0 or td == len(unc):
                el = time.time() - t0
                sp = td / max(el, 1)
                print(f"  [{ok + cached}/{tot}] {sp:.1f} v/s  ~{(len(unc) - td) / max(sp, .01):.0f}s left")
    print(f"[LM] Done! {ok} ok, {fail} fail ({time.time() - t0:.0f}s)\n")



# ============================================================
# Feature Engineering (must match app.py exactly)
# ============================================================
def eng_feat(s):
    """Raw landmarks -> engineered features (1662 -> 4997)."""
    v = np.zeros_like(s)
    v[1:] = s[1:] - s[:-1]
    a = np.zeros_like(s)
    a[1:] = v[1:] - v[:-1]
    nose = s[:, 0:3]
    ls = 33 * 4
    rs = ls + 21 * 3
    lw = s[:, ls:ls + 3]
    rw = s[:, rs:rs + 3]
    ex = np.concatenate([
        lw - nose,                                              # left hand relative to nose (3)
        rw - nose,                                              # right hand relative to nose (3)
        np.linalg.norm(lw - rw, axis=1, keepdims=True),        # hand distance (1)
        np.linalg.norm(v[:, ls:ls + 3], axis=1, keepdims=True),  # left hand speed (1)
        np.linalg.norm(v[:, rs:rs + 3], axis=1, keepdims=True),  # right hand speed (1)
    ], axis=1)  # total extra = 11
    return np.concatenate([s, v, a, ex], axis=1).astype(np.float32)


# ============================================================
# Mirror Augmentation (swap left/right hands + flip x)
# ============================================================
def mirror(s):
    m = s.copy()
    ls, le = 33 * 4, 33 * 4 + 21 * 3
    rs, re = le, le + 21 * 3
    # Swap left and right hand landmarks
    m[:, ls:le], m[:, rs:re] = s[:, rs:re].copy(), s[:, ls:le].copy()
    # Flip x coordinates for pose
    for i in range(33):
        m[:, i * 4] = 1 - m[:, i * 4]
    # Flip x coordinates for hands
    for st in [ls, rs]:
        for i in range(21):
            m[:, st + i * 3] = 1 - m[:, st + i * 3]
    # Flip x coordinates for face
    fs = re
    for i in range(468):
        m[:, fs + i * 3] = 1 - m[:, fs + i * 3]
    return m


# ============================================================
# Data Augmentation (7 types)
# ============================================================
def aug1(s):
    """Apply random augmentations to raw landmark sequence."""
    a = s.copy()
    # 1. Noise injection
    if np.random.random() < 0.85:
        a += np.random.normal(0, np.random.uniform(0.003, 0.025), a.shape)
    # 2. Scale variation
    if np.random.random() < 0.75:
        a *= np.random.uniform(0.90, 1.10, a.shape)
    # 3. Time shift
    if np.random.random() < 0.65:
        a = np.roll(a, np.random.randint(-5, 6), axis=0)
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
    # 7. Hand-specific noise
    if np.random.random() < 0.5:
        hs, he = 33 * 4, 33 * 4 + 21 * 3 * 2
        a[:, hs:he] += np.random.normal(0, 0.015, a[:, hs:he].shape)
    return a.astype(np.float32)


def augment(raw, na):
    """Generate augmented samples from raw landmarks."""
    out = [eng_feat(raw)]
    mi = mirror(raw)
    out.append(eng_feat(mi))
    for _ in range(na - 1):
        out.append(eng_feat(aug1(raw)))
        if np.random.random() < 0.5:
            out.append(eng_feat(aug1(mi)))
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
    label_counts = Counter(mapping.values())
    # Filter: only classes with >= MIN_SAMPLES videos
    valid_labels = {lb for lb, cnt in label_counts.items() if cnt >= MIN_SAMPLES}
    filtered = {fn: lb for fn, lb in mapping.items() if lb in valid_labels}

    unique_labels = sorted(valid_labels, key=str)
    label_map = {str(l): i for i, l in enumerate(unique_labels)}
    nc = len(label_map)

    label_to_files = {}
    for fn, lb in filtered.items():
        label_to_files.setdefault(lb, []).append(fn)

    print(f"[DATA] Filter >= {MIN_SAMPLES} samples: {nc} classes, {len(filtered)} videos")
    print(f"  (Skipped {len(mapping) - len(filtered)} videos from {len(label_counts) - nc} rare classes)")

    X_train, y_train, X_val, y_val = [], [], [], []
    loaded = 0
    skipped = 0

    for lb, fns in label_to_files.items():
        li = label_map[str(lb)]
        cnt = label_counts[lb]
        seqs = []
        for fn in fns:
            cp = LANDMARKS_DIR / f"{fn}.npy"
            if not cp.exists():
                skipped += 1
                continue
            try:
                s = np.load(str(cp))
                if s.shape != (SEQ, NRF):
                    skipped += 1
                    continue
                seqs.append(s)
                loaded += 1
            except:
                skipped += 1
                continue

        if len(seqs) < 2:
            continue

        # Stratified split: 1 for val, rest for train
        np.random.shuffle(seqs)
        X_val.append(eng_feat(seqs[0]))
        y_val.append(li)
        train_seqs = seqs[1:]

        # Augmentation - conservative to fit 32GB RAM
        na = 2
        for s in train_seqs:
            for a in augment(s, na):
                X_train.append(a)
                y_train.append(li)

    print(f"[DATA] Loaded: {loaded}  Skipped: {skipped}  Train: {len(X_train)}  Val: {len(X_val)}")

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
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
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

        # Save best model (based on top-5 accuracy)
        if val_top5 > best_val_top5:
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
