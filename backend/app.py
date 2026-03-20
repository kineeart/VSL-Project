import os, json, cv2, numpy as np, mediapipe as mp, openpyxl, base64, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "Videos"
DATA_FILE = BASE_DIR / "Data.xlsx"
MODEL_DIR = BASE_DIR / "backend" / "models"
LANDMARKS_DIR = BASE_DIR / "backend" / "landmarks"
CUSTOM_VIDEOS_DIR = BASE_DIR / "backend" / "custom_videos"

for d in [MODEL_DIR, LANDMARKS_DIR, CUSTOM_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic

# Global
model = None
label_map = {}
reverse_label_map = {}
norm_mean = None
norm_std = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LENGTH = 60
NUM_RAW_FEATURES = 33 * 4 + 21 * 3 * 2 + 468 * 3  # 1662


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
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    accel = np.zeros_like(seq)
    accel[1:] = velocity[1:] - velocity[:-1]

    nose_xyz = seq[:, 0:3]
    lh_start = 33 * 4
    lh_wrist = seq[:, lh_start:lh_start+3]
    rh_start = lh_start + 21 * 3
    rh_wrist = seq[:, rh_start:rh_start+3]

    lh_rel = lh_wrist - nose_xyz
    rh_rel = rh_wrist - nose_xyz
    hand_dist = np.linalg.norm(lh_wrist - rh_wrist, axis=1, keepdims=True)
    lh_vel = velocity[:, lh_start:lh_start+3]
    rh_vel = velocity[:, rh_start:rh_start+3]
    lh_speed = np.linalg.norm(lh_vel, axis=1, keepdims=True)
    rh_speed = np.linalg.norm(rh_vel, axis=1, keepdims=True)

    extra = np.concatenate([lh_rel, rh_rel, hand_dist, lh_speed, rh_speed], axis=1)
    return np.concatenate([seq, velocity, accel, extra], axis=1).astype(np.float32)


def predict_from_seq(raw_seq):
    """Run prediction: raw landmarks → feature engineering → normalize → model → top-5."""
    feat = engineer_features(raw_seq)
    if norm_mean is not None and norm_std is not None:
        feat = (feat - norm_mean) / norm_std
    tensor = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top_indices:
        label = reverse_label_map.get(idx, "unknown")
        results.append({"label": label, "confidence": float(probs[idx])})
    return results


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
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    return np.concatenate([pose, lh, rh, face])


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
    global model, label_map, reverse_label_map, norm_mean, norm_std
    pt_path = MODEL_DIR / "sign_model.pt"
    labels_path = MODEL_DIR / "labels.json"
    mean_path = MODEL_DIR / "norm_mean.npy"
    std_path = MODEL_DIR / "norm_std.npy"

    if pt_path.exists() and labels_path.exists():
        with open(labels_path, encoding='utf-8') as f:
            label_map = json.load(f)
        reverse_label_map = {int(v): k for k, v in label_map.items()}
        num_classes = len(label_map)

        checkpoint = torch.load(str(pt_path), map_location=device, weights_only=True)
        num_features = checkpoint.get('num_features', NUM_RAW_FEATURES * 3 + 9)

        m = SignModel(num_features, checkpoint.get('num_classes', num_classes))
        m.load_state_dict(checkpoint['model_state_dict'])
        m.to(device)
        m.eval()
        model = m

        # Load normalization params
        if mean_path.exists() and std_path.exists():
            norm_mean = np.load(str(mean_path))
            norm_std = np.load(str(std_path))
            print(f"Normalization params loaded.")

        print(f"Model loaded: {num_classes} classes, {num_features} features, device={device}")
    else:
        print("No trained model found. Run train.bat first.")


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
    return {
        "model_loaded": model is not None,
        "num_classes": len(label_map),
        "labels": list(label_map.keys()) if label_map else []
    }

@app.get("/api/labels")
async def get_labels():
    mapping = load_data_mapping()
    unique_labels = sorted(set(mapping.values()), key=str)
    return {"labels": unique_labels, "total_videos": len(mapping)}

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
        return {"predictions": predict_from_seq(seq)}
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
    return {"predictions": predict_from_seq(np.array(landmarks_seq))}

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
                        results = predict_from_seq(seq)
                        await websocket.send_json({"predictions": results})
    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
