# Vietnamese Sign Language Recognition
# Nhận dạng Ngôn ngữ Ký hiệu Việt Nam

> **Bản quyền © 2026 — Lưu Vĩnh Thắng**
> Thực hiện cho **Kiên Nguyễn**

Hệ thống nhận dạng ngôn ngữ ký hiệu Việt Nam sử dụng AI (PyTorch + CUDA GPU), hỗ trợ nhận dạng realtime qua camera, upload video, và huấn luyện với GPU.

## Tính năng

- **Nhận dạng Realtime** — Webcam + WebSocket, kết quả Top-5 ngay lập tức
- **Upload Video** — Tải video lên để nhận dạng
- **Huấn luyện GPU** — PyTorch + CUDA, mixed precision (FP16)
- **Giao diện đa ngôn ngữ** — Tiếng Việt / English
- **Light / Dark theme**
- **Data Augmentation** — 7 loại + mirror

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Frontend | React 19, Vite 6, i18next, CSS Variables |
| Backend | Python 3.12, FastAPI, Uvicorn |
| AI/ML | PyTorch (CUDA GPU), MediaPipe Holistic (model_complexity=2) |
| Model | Multi-scale CNN + BiLSTM + Multi-head Attention + Cosine Classifier |
| Dữ liệu | OpenCV, NumPy, scikit-learn, openpyxl |

## Kiến trúc Model

```
Video → MediaPipe Holistic (complexity=2, confidence=0.7)
    → 60 frames/video (lấy đều từ video)
    → Landmark Extraction (1662 raw features/frame)
        → Pose (33×4) + Left Hand (21×3) + Right Hand (21×3) + Face (468×3)
    → Feature Engineering (1662 → 4995 features/frame)
        → Raw + Velocity + Acceleration + Relative hand positions + Hand speed
    → Normalization (mean/std)
    → Multi-scale CNN (kernel 3/5/7 song song, 512 channels)
    → BiLSTM (384 hidden, 2 layers, bidirectional → 768)
    → Multi-head Attention (4 heads)
    → Dense (768 → 512 → 256)
    → Cosine Classifier (few-shot optimized)
    → Top-5 Predictions + Confidence
```

## Kỹ thuật tối ưu accuracy

| Kỹ thuật | Mô tả |
|---|---|
| MediaPipe complexity=2 | Model nặng nhất, landmarks chính xác nhất |
| 60 frames/video | Bắt chuyển động chi tiết (gấp đôi mặc định) |
| Cosine Classifier | Cosine similarity — tối ưu cho few-shot |
| Multi-scale CNN | 3 kernel sizes (3,5,7) song song |
| Multi-head Attention | 4 heads |
| Focal Loss | gamma=2.0 + label smoothing 0.1 + class weights (cap 15x) |
| Mixup Training | alpha=0.4 |
| Mirror Augmentation | Swap tay trái/phải + flip x |
| 7 loại Augmentation | Noise, scale, time shift, speed, frame dropout, temporal reverse, hand noise |
| SWA | Stochastic Weight Averaging từ epoch 100 |
| Mixed Precision | FP16 trên GPU |
| Warmup + Cosine LR | 8 epoch warmup → cosine annealing |

## Dữ liệu

- **4362 videos** (.mp4) trong thư mục `Videos/`
- **3319 classes** tổng (nhãn ký hiệu)
- Training chỉ dùng classes có **≥3 mẫu** → **489 classes, ~1475 videos**
- **Data.xlsx** — mapping filename → label
- Landmark cache: ~780KB/video (60 frames × 1662 features)

## Cấu trúc Project

```
├── start.bat                # Khởi động hệ thống (backend + frontend)
├── stop.bat                 # Dừng tất cả servers
├── train.bat                # Huấn luyện model bằng GPU
├── Data.xlsx                # Dữ liệu nhãn (4362 videos, 3319 labels)
├── Videos/                  # Thư mục video huấn luyện (.mp4)
├── README.md
├── backend/
│   ├── app.py               # FastAPI server + inference (PyTorch)
│   ├── train_gpu.py         # Training GPU (PyTorch + CUDA)
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # Model đã train
│   │   ├── sign_model.pt
│   │   ├── sign_model_best.pt
│   │   ├── labels.json
│   │   ├── norm_mean.npy
│   │   ├── norm_std.npy
│   │   └── training_history.json
│   ├── landmarks/           # Cache landmarks (~780KB/file)
│   └── custom_videos/       # Video tự quay
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── i18n.js
        ├── styles.css
        └── pages/
            ├── CameraPage.jsx
            ├── UploadPage.jsx
            ├── TrainingPage.jsx
            └── StatusPage.jsx
```

## Yêu cầu hệ thống

- Python 3.12+
- Node.js 18+
- NVIDIA GPU với CUDA (RTX 3060+ / 8GB VRAM)
- RAM 32GB+
- PyTorch 2.5+ với CUDA 12.1
- Webcam (cho realtime)

## Cách sử dụng

### Bước 1: Huấn luyện model

Double-click `train.bat` — tự động cài dependencies, extract landmarks, train model bằng GPU.

```
Ep   1/200  Train: 12.34%  Val-Top1: 8.50%  Val-Top5: 25.30%  LR: 6.3e-05  45s
Ep   2/200  Train: 18.67%  Val-Top1: 12.10%  Val-Top5: 35.80%  LR: 1.3e-04  90s
...
```

Lần đầu extract landmarks mất 15-30 phút (CPU). Lần sau đã cache → skip thẳng tới training.

### Bước 2: Khởi động hệ thống

Double-click `start.bat` → backend (port 8000) + frontend (port 3000).

### Bước 3: Sử dụng

- http://localhost:3000
- Tab **Camera** — nhận dạng realtime
- Tab **Upload** — tải video nhận dạng
- Tab **Huấn luyện** — thêm mẫu tùy chỉnh
- Tab **Trạng thái** — thông tin model

### Dừng hệ thống

Double-click `stop.bat`.

### Chạy thủ công

```bash
# Backend
cd backend
python -m pip install -r requirements.txt
python app.py

# Frontend
cd frontend
npm install
npm run dev

# Train (cần GPU)
# Cài PyTorch GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
cd backend
python train_gpu.py
```

## Thông số Training

| Thông số | Giá trị |
|---|---|
| Min samples/class | 3 |
| Sequence Length | 60 frames |
| Raw Features | 1662/frame |
| Engineered Features | 4995/frame |
| Batch Size | 32 |
| Epochs | 200 (patience=30) |
| Learning Rate | 0.0005 (AdamW) |
| SWA | Từ epoch 100 |
| Mixup Alpha | 0.4 |
| Model Parameters | ~21.7M |
| Metric | Top-5 Accuracy |

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/status` | Trạng thái model |
| GET | `/api/labels` | Danh sách nhãn |
| POST | `/api/predict/video` | Upload video → Top-5 |
| POST | `/api/predict/frames` | Frames base64 → Top-5 |
| POST | `/api/train/custom` | Thêm video training |
| GET | `/api/train/custom/list` | Danh sách mẫu tự tạo |
| WS | `/ws/predict` | WebSocket realtime |

## Trạng thái

- [x] Backend API (FastAPI + PyTorch)
- [x] Frontend (React 19 + Vite 6)
- [x] Nhận dạng realtime WebSocket (Top-5)
- [x] Upload video nhận dạng
- [x] Training GPU (PyTorch + CUDA)
- [x] Multi-scale CNN + BiLSTM + Multi-head Attention + Cosine Classifier
- [x] Mixup + Focal Loss + SWA + Mixed Precision
- [x] 7 loại augmentation + mirror
- [x] Feature engineering (1662 → 4995)
- [x] MediaPipe Holistic max (complexity=2, confidence=0.7)
- [x] 60 frames/video
- [x] Top-1 + Top-5 accuracy tracking
- [x] Light/Dark theme + i18n (vi/en)
- [x] Landmark caching
- [x] Script tự động: `start.bat`, `stop.bat`, `train.bat`

---

**© 2026 Lưu Vĩnh Thắng — Thực hiện cho Kiên Nguyễn**
