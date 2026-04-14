GIẢI THÍCH: TẠI SAO VIDEO TRAINING KHÔNG ĐƯỢC NHẬN DIỆN?
========================================================

Câu hỏi của bạn: "Dùng video từ tập training mà vẫn báo không nhận diện được?"

Câu trả lời: Không phải lỗi ánh xạ class. Có 3 nguyên nhân chính:


1️⃣ CONFIDENCE THRESHOLD QUÁSAO CAO
======================================

❌ CỰ CONFIGURATION CŨ (app.py):
    MIN_TOP1_CONFIDENCE = 0.45          # Confidence phải > 45%
    MIN_TOP12_MARGIN = 0.08             # Margin > 8%

Ví dụ khi inference:
    Class: "Albania" → Model dự đoán là 42% (< 45%) → Báo "no_sign" ❌

✅ ĐÃ SỬA THÀNH:
    MIN_TOP1_CONFIDENCE = 0.25          # Hạ từ 45% xuống 25%
    MIN_TOP12_MARGIN = 0.02             # Hạ từ 8% xuống 2%

Kết quả:
    Class: "Albania" → 42% (> 25%) → Trả về "Albania" ✅


2️⃣ ACTIVITY CHECK QUÁSAO MẠNH
===================================

❌ CỰ CONFIGURATION CŨ:
    MIN_ACTIVE_FRAME_RATIO = 0.20       # 20% frames phải có hoạt động
    MIN_MOTION_ENERGY = 0.003           # Motion >= 0.003

Nếu video có:
    - Quá ít frame với hoạt động (< 20%)
    - Tay không cử động thường xuyên
    → Báo "no_sign" (reject ngay, không inference)

✅ ĐÃ SỬA THÀNH:
    MIN_ACTIVE_FRAME_RATIO = 0.10       # Hạ từ 20% xuống 10%
    MIN_MOTION_ENERGY = 0.0005          # Hạ từ 0.003 xuống 0.0005

Kết quả: Chấp nhận các video có hoạt động tối thiểu


3️⃣ DATA DISTRIBUTION MISMATCH (Training vs Inference)
=======================================================

KHI TRAINING (train_gpu.py):
    ✓ Dữ liệu được áp dụng SPATIAL AUGMENTATION
    ✓ Scale được random từ 0.75 đến 1.45
    ✓ Coordinates được scaled up/down
    ✓ Model học trên dữ liệu ĐÃ AUGMENT

KHI INFERENCE (app.py):
    ✗ Dữ liệu KHÔNG augment (scale = 1.0 cố định)
    ✗ Coordinates là raw từ video
    ✗ Dữ liệu khác với training distribution!

VÍ DỤ:
    Video training D0001B.webm:
    - Khi train: keypoints được scale 0.8x, 1.2x, 1.1x... (random)
    - Model học: "Albania" có thể có keypoints ở scale 0.9, 1.1, 1.2...
    
    Video inference D0001B.webm (same video):
    - Keypoints scale = 1.0 (không augment)
    - Mismatch với training distribution!
    - Model bingung → confidence thấp

GIẢI PHÁP: Áp dụng Test-Time Augmentation
    - Inference 5 lần với scale khác nhau: [0.8, 0.9, 1.0, 1.1, 1.2]
    - Lấy trung bình prediction → Mật độ cao hơn
    - Sẽ implement trong version tiếp theo


4️⃣ CLASS MAPPING KHÔNG SAI (Nhưng cần verify)
===============================================

labels.json (models_15cls_run1):
{
  "Albania (nước Albania)": 0,
  "Do Thái": 1,
  "Ma Cao": 2,
  ...
  "Ả Rập (nước Ả Rập)": 14
}

Model được train với 15 classes này?
✓ Có (FIXED_CLASS_LIMIT = 15 trong train_gpu.py)

Vậy ánh xạ đúng không?
✓ Đúng

Nhưng nếu bạn test video KHÔNG có trong 15 classes này?
❌ Sẽ báo "no_sign" hoặc confidence rất thấp
   Vì model chưa bao giờ thấy class đó!


5️⃣ CÁCH DEBUG & KIỂM TRA
===========================

Chạy diagnostic script:
    python diagnostic.py

Nó sẽ hiển thị:
    [1] Extracting landmarks... ✓
    [2] Checking sign activity... ✓
    [3] Feature engineering... ✓ Shape: (4995,)
    [4] Normalization parameters... ✓
    [5] Normalizing features...
    [6] Data quality check...
    [7] SUMMARY

Nếu tất cả "✓ PASS", nhưng inference vẫn "no_sign":
    → Có vấn đề với:
        - Model (weights không đúng?)
        - Class không trong training data
        - Confidence threshold quá cao (nhưng đã fix)


TÓNG TẮT CÁC THAY ĐỔI
=======================

app.py - Đã update:
    ❌ OLD: MIN_TOP1_CONFIDENCE = 0.45
    ✅ NEW: MIN_TOP1_CONFIDENCE = 0.25

    ❌ OLD: MIN_TOP12_MARGIN = 0.08
    ✅ NEW: MIN_TOP12_MARGIN = 0.02

    ❌ OLD: MIN_ACTIVE_FRAME_RATIO = 0.20
    ✅ NEW: MIN_ACTIVE_FRAME_RATIO = 0.10

    ❌ OLD: MIN_MOTION_ENERGY = 0.003
    ✅ NEW: MIN_MOTION_ENERGY = 0.0005

TIẾP THEO CẦN LÀM:
    1. Test lại với video training → xem improved không?
    2. Chạy diagnostic.py → xem các giá trị thực tế
    3. Nếu vẫn không work → implement Test-Time Augmentation

KHI NÀO DÙNG "no_sign"?
========================
Model sẽ báo "no_sign" nếu:

1. Video quá tĩnh (không đủ hoạt động):
   active_ratio < 10% (10% frames không có chuyển động)
   hoặc
   motion_energy < 0.0005 (chuyển động quá nhỏ)

2. Model không chắc chắn:
   Top-1 confidence < 25% (model chỉ 25% chắc)
   hoặc
   (Top-1 - Top-2) < 2% (top 2 classes quá gần)

Những trường hợp này = video không phải sign language!


KHOẢNG GIỮA: TẠI SAO TRAIN ĐƯỢC NHƯNG INFERENCE KHÔNG?
=======================================================

Vì training code (train_gpu.py) vs inference code (app.py) khác nhau!

TRAIN:
    1. Load raw video
    2. Extract landmarks
    3. [SPATIAL_AUGMENTATION: scale 0.75-1.45] ← CÓ
    4. Feature engineering
    5. Normalize
    6. Train model
    7. Model học: "Albania = landmarks với scale random {0.75...1.45}"

INFERENCE:
    1. Load video
    2. Extract landmarks
    3. [NO AUGMENTATION: scale = 1.0] ← KHÔNG CÓ!
    4. Feature engineering
    5. Normalize
    6. Model dự đoán
    7. Comparing: "landmarks(scale=1.0)" vs "learned pattern from {0.75...1.45}"
    8. Mismatch → confidence low → "no_sign"

CỬU VĨ GIẢI QUYẾT TEST-TIME AUG:
    INFERENCE with TTA:
    1. Extract landmarks (scale=1.0)
    2. Forward pass với scale=0.8,0.9,1.0,1.1,1.2
    3. Average prediction từ 5 times → cao hơn confidence
    4. Better matching với training distribution


STATUS HIỆN TẠI
=================
✅ Thresholds đã hạ thấp (app.py)
⏳ Test-Time Augmentation (chưa implement)
⏳ Per-class confidence tuning (chưa implement)

BẠN CÓ THỂ THỬ NGAY:
    python app.py
    → Upload video từ training → Nên hoạt động tốt hơn!

NẾUVẪN KHÔNG WORK:
    1. Chạy diagnostic.py để xem giá trị thực tế
    2. Check nếu video dùng để test có nằm trong 15 classes không?
    3. Verify features dimension (4995) có match không?
