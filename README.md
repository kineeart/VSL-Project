# Vietnamese Sign Language Recognition
## Nhận dạng Ngôn ngữ Ký hiệu Việt Nam

Hệ thống nhận dạng ngôn ngữ ký hiệu Việt Nam sử dụng AI (PyTorch + CUDA GPU), hỗ trợ nhận dạng realtime qua camera, nhận diện liên tục, upload video, và huấn luyện bằng GPU.

## Tính năng

- Nhận dạng Realtime: Webcam + WebSocket, kết quả Top-5 ngay lập tức
- Nhận diện liên tục: ghép chuỗi từ video stream và hiển thị câu đang chốt
- Upload Video: Tải video lên để nhận dạng
- Huấn luyện GPU: PyTorch + CUDA, mixed precision (FP16)
- Giao diện đa ngôn ngữ: Tiếng Việt / English
- Light / Dark theme
- Data Augmentation: 7 loại + mirror

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Frontend | React 19, Vite 6, i18next, CSS Variables |
| Backend | Python 3.10, FastAPI, Uvicorn |
| AI/ML | PyTorch (CUDA GPU), MediaPipe Holistic (model_complexity=2) |
| Model | Multi-scale CNN + BiLSTM + Multi-head Attention + Cosine Classifier |
| Dữ liệu | OpenCV, NumPy, scikit-learn, openpyxl |
| Continuous | WebSocket stream, smoothing, phrase commit |

## Kiến trúc Model

```text
Video -> MediaPipe Holistic (complexity=2, confidence=0.7)
    -> 60 frames/video (lấy đều từ video)
    -> Landmark Extraction (1662 raw features/frame)
        -> Pose (33x4) + Left Hand (21x3) + Right Hand (21x3) + Face (468x3)
    -> Feature Engineering (1662 -> 4995 features/frame)
        -> Raw + Velocity + Acceleration + Relative hand positions + Hand speed
    -> Normalization (mean/std)
    -> Multi-scale CNN (kernel 3/5/7 song song, 512 channels)
    -> BiLSTM (384 hidden, 2 layers, bidirectional -> 768)
    -> Multi-head Attention (4 heads)
    -> Dense (768 -> 512 -> 256)
    -> Cosine Classifier (few-shot optimized)
    -> Top-5 Predictions + Confidence
```

## Kỹ thuật tối ưu accuracy

| Kỹ thuật | Mô tả |
|---|---|
| MediaPipe complexity=2 | Model nặng nhất, landmarks chính xác nhất |
| 60 frames/video | Bắt chuyển động chi tiết (gấp đôi mặc định) |
| Cosine Classifier | Cosine similarity, tối ưu cho few-shot |
| Multi-scale CNN | 3 kernel sizes (3, 5, 7) song song |
| Multi-head Attention | 4 heads |
| Focal Loss | gamma=2.0 + label smoothing 0.1 + class weights (cap 15x) |
| Mixup Training | alpha=0.4 |
| Mirror Augmentation | Swap tay trái/phải + flip x |
| 7 loại Augmentation | Noise, scale, time shift, speed, frame dropout, temporal reverse, hand noise |
| SWA | Stochastic Weight Averaging từ epoch 100 |
| Mixed Precision | FP16 trên GPU |
| Warmup + Cosine LR | 8 epoch warmup -> cosine annealing |
| Continuous commit | Gom kết quả ổn định theo chuỗi frame |

## Dữ liệu

- 4362 videos (.mp4) trong thư mục Videos/
- 3319 classes tổng (nhãn ký hiệu)
- Training chỉ dùng classes có >= 3 mẫu -> 489 classes, ~1475 videos
- Data.xlsx: mapping filename -> label
- Landmark cache: ~780KB/video (60 frames x 1662 features)

## Cấu trúc Project

```text
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
├── benchmark/
│   ├── continuous/          # Cắt đoạn diễn tả và tạo dữ liệu chuỗi
│   └── sentence_level/      # Benchmark CER/WER cho chuỗi
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
            ├── ContinuousPage.jsx
            ├── UploadPage.jsx
            ├── TrainingPage.jsx
            └── StatusPage.jsx
```

## Yêu cầu hệ thống

- Python 3.10+
- Node.js 18+
- NVIDIA GPU với CUDA (RTX 3060+ / 8GB VRAM)
- RAM 32GB+
- PyTorch 2.5+ với CUDA 12.1
- Webcam (cho realtime)

## Cách sử dụng

### 1) Chuẩn bị và sử dụng repo

1. Cài Python 3.10+ và Node.js 18+.
2. Cài dependencies backend và frontend theo hướng dẫn bên dưới.
3. Khởi động backend + frontend bằng `start.bat` để dùng giao diện web.
4. Nếu muốn train lại mô hình, chạy `train.bat`.
5. Nếu muốn đánh giá, chạy các benchmark ở phần bên dưới.

### 2) Huấn luyện model

Double-click `train.bat` để tự động cài dependencies, extract landmarks, train model bằng GPU.

```text
Ep   1/200  Train: 12.34%  Val-Top1: 8.50%  Val-Top5: 25.30%  LR: 6.3e-05  45s
Ep   2/200  Train: 18.67%  Val-Top1: 12.10%  Val-Top5: 35.80%  LR: 1.3e-04  90s
...
```

Lần đầu extract landmarks mất 15-30 phút (CPU). Lần sau đã cache nên sẽ bỏ qua bước này.

### 3) Khởi động hệ thống

Double-click `start.bat` để chạy backend (port 8000) + frontend (port 3000).

### 4) Sử dụng hệ thống

- Truy cập: http://localhost:3000
- Tab Camera: nhận dạng realtime
- Tab Nhận diện liên tục: hiển thị câu đang chốt theo chuỗi frame
- Tab Upload: tải video nhận dạng
- Tab Huấn luyện: thêm mẫu tùy chỉnh
- Tab Trạng thái: thông tin model

### 5) Dừng hệ thống

Double-click `stop.bat`.

## Chạy benchmark

### 1) Benchmark continuous

Benchmark này dùng để đánh giá nhận dạng chuỗi liên tục bằng CER/WER.

Chạy nhanh toàn bộ pipeline trên dữ liệu thực:

```bash
benchmark\continuous\run_pipeline_real.bat benchmark\continuous\data\continuous_dataset.real.json http://127.0.0.1:8000
```

Nếu muốn chạy từng bước:

```bash
# Tìm active span trong video
python benchmark/continuous/scripts/detect_active_span.py \
    --input Videos/D0001B.mp4 \
    --output benchmark/continuous/data/D0001B.active_span.json

# Sinh dự đoán từ backend
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py \
    --dataset benchmark/continuous/data/continuous_dataset.real.json \
    --output benchmark/continuous/data/continuous_predictions.real.json \
    --backend http://127.0.0.1:8000 \
    --split test --stride 3

# Đánh giá CER/WER
python benchmark/continuous/scripts/evaluate_continuous_benchmark.py \
    --gt benchmark/continuous/data/continuous_dataset.real.json \
    --pred benchmark/continuous/data/continuous_predictions.real.json \
    --split test --breakdown \
    --out benchmark/continuous/data/continuous_eval.real.json
```

### 2) Benchmark sentence-level

Benchmark này dùng để đánh giá nhận dạng mức câu, theo gloss/text và có thể tách theo signer/dialect.

Chạy benchmark mẫu nhanh:

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
    --gt benchmark/sentence_level/data/sentence_dataset.sample.json \
    --pred benchmark/sentence_level/data/sentence_predictions.sample.json \
    --split test
```

Chạy benchmark chi tiết hơn với breakdown:

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
    --gt benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
    --pred benchmark/sentence_level/data/sentence_predictions.synthetic.small.json \
    --split test \
    --breakdown
```

Nếu cần tạo và chia dataset sentence:

```bash
python benchmark/sentence_level/scripts/generate_sentence_dataset_from_dataxlsx.py \
    --data-xlsx Data.xlsx \
    --videos-dir Videos \
    --labels-json backend/models_15cls_run1/labels.json \
    --output-json benchmark/sentence_level/data/sentence_dataset.synthetic.json \
    --num-sentences 300 \
    --min-words 2 --max-words 5 \
    --split-policy signer \
    --pseudo-signer-count 6 \
    --seed 42

python benchmark/sentence_level/scripts/split_sentence_dataset.py \
    --input benchmark/sentence_level/data/sentence_dataset.synthetic.json \
    --output benchmark/sentence_level/data/sentence_dataset.split.json \
    --report benchmark/sentence_level/data/sentence_split_report.json \
    --group-by signer \
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
    --seed 42
```

### 3) Chạy trên điện thoại

Repo có sẵn app mobile mini trong `mobile-mini/` để demo camera live và gọi backend.

Chạy môi trường phát triển:

```bash
cd mobile-mini
npm install
npm run start
```

Sau đó quét QR bằng Expo Go, rồi nhập backend host theo dạng:

- `192.168.x.x:8000` nếu chạy trong cùng mạng LAN
- `https://api.your-domain.com` nếu dùng backend public

Build APK để cài trên điện thoại Android:

```bash
cd mobile-mini
npx eas login
npx eas build --platform android --profile preview
```

Lưu ý:

- Mobile app đang gọi websocket backend hiện tại (`/ws/predict`).
- Nếu không nhận được prediction, hãy kiểm tra firewall, địa chỉ host và backend đã chạy chưa.
- Đây là bản MVP để demo NCKH; nếu muốn chạy hoàn toàn on-device thì cần tối ưu thêm model và artifacts.

## Chạy thủ công

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
| GET | /api/status | Trạng thái model |
| GET | /api/labels | Danh sách nhãn |
| POST | /api/predict/video | Upload video -> Top-5 |
| POST | /api/predict/continuous_video | Upload video chuỗi -> decode câu liên tục |
| POST | /api/predict/frames | Frames base64 -> Top-5 |
| POST | /api/train/custom | Thêm video training |
| GET | /api/train/custom/list | Danh sách mẫu tự tạo |
| WS | /ws/predict | WebSocket realtime |
| WS | /ws/continuous | WebSocket nhận diện liên tục |

## Trạng thái

- [x] Backend API (FastAPI + PyTorch)
- [x] Frontend (React 19 + Vite 6)
- [x] Nhận dạng realtime WebSocket (Top-5)
- [x] Nhận diện liên tục dạng phrase commit
- [x] Upload video nhận dạng
- [x] Training GPU (PyTorch + CUDA)
- [x] Multi-scale CNN + BiLSTM + Multi-head Attention + Cosine Classifier
- [x] Mixup + Focal Loss + SWA + Mixed Precision
- [x] 7 loại augmentation + mirror
- [x] Feature engineering (1662 -> 4995)
- [x] MediaPipe Holistic max (complexity=2, confidence=0.7)
- [x] 60 frames/video
- [x] Top-1 + Top-5 accuracy tracking
- [x] Light/Dark theme + i18n (vi/en)
- [x] Landmark caching
- [x] Script tự động: start.bat, stop.bat, train.bat
- [x] Tích hợp Language Model (n-gram/LM decode) cho chuỗi VSL (baseline bigram + rerank)
- [x] Ensemble nhiều mô hình khi suy luận chuỗi (average probability across checkpoints)
- [ ] Huấn luyện và đánh giá trên dataset VSL chuỗi thật (không chỉ synthetic)

## Đối chiếu với mục tiêu đề tài

| Mục tiêu đề tài | Mức độ hiện tại trong repo |
|---|---|
| Trích xuất đặc trưng không gian-thời gian + mô hình học sâu lai cho nhận diện ký hiệu | Đã có: MediaPipe keypoint + feature engineering + Multi-scale CNN + BiLSTM + Attention |
| Chuyển từ nhận diện đơn sang nhận diện chuỗi ký hiệu/từ rời rạc | Đã có bản nền: endpoint `/ws/continuous`, tab Continuous, pipeline benchmark sentence/continuous |
| Tích hợp mô hình ngôn ngữ để tăng độ chính xác chuỗi | Đã có baseline LM bigram + rerank trong decoder liên tục (cấu hình bằng ENV `ENABLE_LM_DECODE`, `VSL_LM_PATH`, `LM_WEIGHT`) |
| Ensemble Learning để tăng độ ổn định và giảm sai số | Đã có baseline ensemble (nhiều checkpoint) bằng trung bình xác suất (`VSL_ENSEMBLE_MODEL_DIRS`) |
| Đánh giá bằng CER/WER và phân tích lỗi | Đã có script CER/WER cho sentence-level và continuous-level; có pipeline sinh dự đoán continuous từ backend |

Ghi chú hiện trạng:
- Chế độ nhận diện liên tục hiện tại đã có LM rerank + ensemble ở mức baseline, nhưng lõi mô hình vẫn là nhận diện đơn lẻ theo cửa sổ thời gian.
- Để bám sát hoàn toàn phần giới thiệu đề tài, trọng tâm còn lại là thu thập và benchmark nghiêm ngặt trên tập chuỗi thật (train/val/test) + phân tích lỗi định tính.

## Kiểm tra hiện trạng trước khi train/benchmark

### 1) Kiểm tra nhanh file và script chính

Các kiểm tra cú pháp hiện tại đã ổn cho các script trọng yếu sau:

- `backend/train_gpu.py`
- `benchmark/continuous/scripts/evaluate_continuous_benchmark.py`
- `benchmark/sentence_level/scripts/validate_sentence_dataset.py`
- `benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py`

### 2) Kiểm tra dữ liệu benchmark đang có

- Continuous:
    - `benchmark/continuous/data/continuous_dataset.real.json`
    - `benchmark/continuous/data/continuous_predictions.real.json`
    - `benchmark/continuous/data/continuous_eval.real.json`
    - `benchmark/continuous/data/charts/continuous_wer_cer_by_signer.png`
    - `benchmark/continuous/data/charts/continuous_wer_cer_by_dialect.png`
- Sentence-level:
    - `benchmark/sentence_level/data/sentence_dataset.sample.json`
    - `benchmark/sentence_level/data/sentence_dataset.synthetic.small.json`
    - `benchmark/sentence_level/data/sentence_dataset.synthetic.300.json`
    - `benchmark/sentence_level/data/sentence_dataset.synthetic.500_ondevice.json`
    - `benchmark/sentence_level/data/sentence_predictions.sample.json`
    - `benchmark/sentence_level/data/sentence_eval.synthetic.small.json`

### 3) Trình tự kiểm tra khuyến nghị

1. Xác nhận `Data.xlsx`, `Videos/`, `backend/landmarks/` còn đầy đủ.
2. Chạy validation dataset sentence-level trước khi split/evaluate.
3. Chạy continuous benchmark trên dataset thật hoặc dataset đã tạo chuẩn.
4. Đối chiếu WER/CER/Exact Match với kết quả cũ.
5. Chạy breakdown theo signer/dialect nếu cần báo cáo nghiên cứu.
6. Chỉ chốt kết luận sau khi đã có số liệu ổn định trên dataset đủ lớn.

## Huấn luyện trên Google Colab

### Cách làm đầy đủ nhưng vẫn nhanh

1. Nén repo hoặc clone repo lên Colab.
2. Mount Google Drive để lưu checkpoint và landmarks.
3. Cài dependencies backend.
4. Đảm bảo `Data.xlsx`, `Videos/`, `backend/landmarks/` nằm đúng chỗ.
5. Chạy `python backend/train_gpu.py` trực tiếp từ root repo.
6. Sau khi train xong, copy `backend/models/` về Drive để lưu kết quả.

### Gợi ý cấu trúc Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/VSL')

!pip install -r backend/requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

!python backend/train_gpu.py
```

### Khuyến nghị để chạy ổn định hơn trên Colab

- Dùng Colab Pro nếu có thể để có GPU mạnh hơn.
- Đặt `VSL_MODEL_DIR` và `VSL_LANDMARKS_DIR` trỏ về Drive nếu muốn giữ checkpoint sau khi session tắt.
- Nếu gặp thiếu RAM, giảm batch size hoặc giới hạn thử nghiệm trước khi chạy full training.
- Nếu muốn kiểm tra nhanh trước, hãy chạy trên một subset nhỏ rồi mới mở full dataset.

### Những gì nên làm trên Colab trước khi train dài

```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra thư mục dữ liệu
ls
ls backend
ls Videos
ls backend/landmarks

# Chạy một bước nhỏ trước nếu muốn test đường đi
python backend/test_model_load.py
```

### Mốc chốt để coi là train Colab thành công

- Model được train xong và lưu checkpoint bình thường.
- `training_history.json` hoặc log train có đủ epoch và validation.
- Không có lỗi dữ liệu đầu vào hoặc mismatch feature size.
- Có thể đem checkpoint đó sang chạy benchmark tiếp.
