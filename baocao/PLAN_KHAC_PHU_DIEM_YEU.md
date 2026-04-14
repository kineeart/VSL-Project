# Action Plan: Khắc phục Điểm Yếu của Dự án VSL

**Ngày:** Tháng 4 năm 2026  
**Mục đích:** Chuyển dự án từ "Chưa thành công" → "Thành công" thông qua các cải tiến cụ thể  
**Ưu tiên:** Priority 1 (Critical) → Priority 2 (Important) → Priority 3 (Nice-to-have)

---

## TỔNG QUAN SOLUTION

```
INPUT (Hiện trạng):
  - Training 100% accuracy
  - Inference WER 110%
  - Val/test set quá nhỏ (4/2 mẫu)
  - Không có baseline comparison
  - Gap lớn → nghi ngờ overfitting/bug

OUTPUT (Mục tiêu):
  - WER < 0.5 (từ 1.10)
  - Exact Match > 50%
  - Val/test set ≥200 mẫu
  - Clear baseline comparison
  - Root cause analysis hoàn thành

TIMELINE: 2-4 tuần (tùy khắp phủ)
```

---

## PRIORITY 1: CRITICAL FIXES (TÀI CHÍNH QUAN TRỌNG)

**Timeline:** 1-2 tuần  
**Impact:** Sẽ quyết định liệu dự án có "thành công" hay không

---

### FIX #1: Tái-train Mô hình Với Validation Set Lớn

#### 📋 Vấn đề

```
HIỆN TẠI (SAI):
  Backend/models/
    ├── training_history.json → "val_samples": 4 (!!!)
    └── Accuracy: 100% (nghi ngờ overfitting)

NGUYÊN NHÂN:
  - Split train/val không đúng cách
  - Validation set quá nhỏ để validate generalization
  - Không thể phát hiện overfitting
```

#### ✅ Giải pháp

**Bước 1: Phân tích hiện tại**

```bash
# Xem training history
python -c "
import json
with open('backend/models/training_history.json') as f:
    hist = json.load(f)
    print(f'Epochs: {len(hist)}')
    print(f'Val set size: {hist[-1].get(\"val_samples\", \"?\")}')
    print(f'Train set size: {hist[-1].get(\"train_samples\", \"?\")}')
    print(f'Last val accuracy: {hist[-1].get(\"val_acc\", \"?\")}')
"
```

**Bước 2: Tạo validation set lớn hơn**

Sửa `backend/train_gpu.py`:

```python
# TRƯỚC (SAI):
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
# Result: 400 train, 4 val (ratio 99:1 - SAI!)

# SAU (ĐÚNG):
train_size = int(len(dataset) * 0.7)  # 70% train
val_size = int(len(dataset) * 0.15)   # 15% val
test_size = len(dataset) - train_size - val_size  # 15% test
```

**Bước 3: Tái-train với early stopping**

```bash
# Chạy new training
python backend/train_gpu.py \
  --model-name "sign_model_v2" \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --patience 30 \
  --use-language-model \
  --log-file backend/models/training_v2.log
```

**Kỳ vọng:**
- Training time: ~120 phút (GPU RTX 3060+)
- Val accuracy → 85-92% (thực tế hơn 100%)
- Overfitting signal sẽ rõ (train 95% vs val 85% = 10% gap)

#### 📊 Checklist

- [ ] Sửa train/val split ratio (70/15/15)
- [ ] Thêm early stopping (patience=30)
- [ ] Bật validate every N epoch (N=5)
- [ ] Lưu best model (không phải last model)
- [ ] Log training/val curve
- [ ] Chạy training
- [ ] Kiểm tra kết quả → di chuyển model đến `backend/models/`

---

### FIX #2: Debug Gap Giữa Training 100% vs Inference 110% WER

#### 📋 Vấn đề

```
NHẬN XÉT:
  training_accuracy = 100% (trên small val set)
  inference_wer = 110% (trên 60 mẫu real data)
  → GAP QUÁI LẠ!

CÓ THỂ LÀ:
  1. Inference code khác với training
  2. Preprocessing không match
  3. Language model không được dùng
  4. Softmax temperature sai
  5. CTC decoding sai
```

#### ✅ Giải pháp

**Bước 1: So sánh Training vs Inference Pipeline**

Tạo file `backend/debug_gap.py`:

```python
import torch
import numpy as np
from backend.app import load_model
import json

# Load training config
with open('backend/models/training_config.json') as f:
    config = json.load(f)

print("=" * 60)
print("TRAINING CONFIG:")
print(json.dumps(config, indent=2))
print("=" * 60)

# Load model
model = load_model('backend/models/sign_model_best.pt')
print(f"Model loaded: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")

# Test inference on random data
x = np.random.randn(1, 20, 4995).astype(np.float32)  # 1 video, 20 frames
with torch.no_grad():
    output = model(torch.from_numpy(x).to('cuda'))
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    pred = output.argmax(dim=-1)
    print(f"Prediction: {pred}")
```

**Bước 2: Kiểm tra Normalization**

```python
# backend/debug_normalization.py
import numpy as np

# Load norm stats
norm_mean = np.load('backend/models/norm_mean.npy')
norm_std = np.load('backend/models/norm_std.npy')

print(f"Norm mean shape: {norm_mean.shape}")
print(f"Norm std shape: {norm_std.shape}")
print(f"Norm mean[0:10]: {norm_mean[:10]}")
print(f"Norm std[0:10]: {norm_std[:10]}")

# Check for NaN, Inf
print(f"Mean has NaN: {np.isnan(norm_mean).any()}")
print(f"Std has NaN: {np.isnan(norm_std).any()}")
print(f"Mean has Inf: {np.isinf(norm_mean).any()}")
print(f"Std has Inf: {np.isinf(norm_std).any()}")
```

**Bước 3: Test trên Known Sample**

```python
# backend/test_known_sample.py
# Lấy một mẫu từ training set, chạy inference
# So sánh kết quả

import json
import numpy as np
from pathlib import Path

# Load landmark file từ backend/landmarks/
landmark_files = list(Path('backend/landmarks').glob('*.mp4.npy'))
print(f"Found {len(landmark_files)} landmark files")

# Pick first file
first_file = landmark_files[0]
landmarks = np.load(first_file)
print(f"Landmark shape: {landmarks.shape}")

# Chuẩn hóa
norm_mean = np.load('backend/models/norm_mean.npy')
norm_std = np.load('backend/models/norm_std.npy')
normalized = (landmarks - norm_mean) / (norm_std + 1e-8)
print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

# Run inference and check
# ... (call model.predict)
```

**Bước 4: Kiểm tra Continuous Pipeline**

Chạy test đơn giản:

```bash
# 1. Lấy một video từ Videos/
# 2. Chạy continuous benchmark
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py \
  --dataset benchmark/continuous/data/continuous_dataset.real.json \
  --output /tmp/test_pred.json \
  --backend http://127.0.0.1:8000 \
  --num-samples 1 \
  --verbose

# 3. Kiểm tra output
python -c "
import json
with open('/tmp/test_pred.json') as f:
    pred = json.load(f)
    print('Prediction:', pred[0]['pred_sentence_gloss'])
    print('Confidence:', pred[0].get('confidence', 'N/A'))
"
```

#### 📊 Checklist

- [ ] Chạy `debug_gap.py` → so sánh config
- [ ] Chạy `debug_normalization.py` → kiểm tra NaN/Inf
- [ ] Chạy `test_known_sample.py` → inference on training sample
- [ ] So sánh output vs expected từ training log
- [ ] Kiểm tra language model có load không
- [ ] Kiểm tra CTC/sequence decoding logic
- [ ] Viết findings vào log

**Expected Findings:**
- Có thể tìm ra lỗi preprocessing
- Có thể tìm ra language model không dùng
- Có thể tìm ra softmax temperature sai

---

### FIX #3: Tái-evaluate Benchmark Trên Dataset Lớn Hơn

#### 📋 Vấn đề

```
HIỆN TẠI:
  ├── Continuous: 60 mẫu (nhỏ)
  ├── Sentence-level: 10 mẫu (rất nhỏ)
  └── WER 1.10 trên 60 mẫu → có thể là noise

CẦN:
  ├── Continuous: ≥200 mẫu
  ├── Sentence-level: ≥100 mẫu
  └── Cross-check: Mô hình tốt hay data test xấu?
```

#### ✅ Giải pháp

**Bước 1: Tạo Continuous Dataset Lớn Hơn**

```bash
# Hiện tại có continuous_dataset.real.json với 60 mẫu
# Có thể tạo synthetic từ clip rời để mở rộng

python benchmark/continuous/scripts/build_continuous_dataset.py \
  --data-xlsx Data.xlsx \
  --videos-dir Videos \
  --output-json benchmark/continuous/data/continuous_dataset.real_v2.json \
  --num-samples 200 \
  --create-from sources \
  --min-units 2 --max-units 5 \
  --seed 42
```

**Bước 2: Evaluate Trên New Dataset**

```bash
# Start backend
python backend/app.py &

# Generate predictions
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py \
  --dataset benchmark/continuous/data/continuous_dataset.real_v2.json \
  --output benchmark/continuous/data/continuous_predictions_v2.json \
  --backend http://127.0.0.1:8000 \
  --split test

# Evaluate
python benchmark/continuous/scripts/evaluate_continuous_benchmark.py \
  --gt benchmark/continuous/data/continuous_dataset.real_v2.json \
  --pred benchmark/continuous/data/continuous_predictions_v2.json \
  --split test \
  --breakdown \
  --out benchmark/continuous/data/continuous_eval_v2.json
```

**Bước 3: Compare Kết Quả**

```python
import json

# Load both results
with open('benchmark/continuous/data/continuous_eval.real.json') as f:
    old = json.load(f)
with open('benchmark/continuous/data/continuous_eval_v2.json') as f:
    new = json.load(f)

print("BENCHMARK COMPARISON:")
print(f"OLD (60 mẫu):  WER={old['wer']:.4f}, CER={old['cer']:.4f}")
print(f"NEW (200 mẫu): WER={new['wer']:.4f}, CER={new['cer']:.4f}")
print(f"Improvement: {((old['wer']-new['wer'])/old['wer']*100):.2f}%")
```

**Bước 4: Tương tự cho Sentence-Level**

```bash
# Tạo dataset lớn hơn
python benchmark/sentence_level/scripts/generate_sentence_dataset_from_dataxlsx.py \
  --data-xlsx Data.xlsx \
  --videos-dir Videos \
  --output benchmark/sentence_level/data/sentence_dataset.real_v2.json \
  --num-samples 100+

# Split properly
python benchmark/sentence_level/scripts/split_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.real_v2.json \
  --output benchmark/sentence_level/data/sentence_dataset.real_v2.split.json \
  --group-by signer \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

# Evaluate
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
  --gt benchmark/sentence_level/data/sentence_dataset.real_v2.split.json \
  --pred benchmark/sentence_level/data/sentence_predictions_v2.json \
  --split test \
  --breakdown
```

#### 📊 Checklist

- [ ] Tạo continuous_dataset.real_v2 (200 mẫu)
- [ ] Tạo sentence_dataset.real_v2 (100 mẫu)
- [ ] Run inference trên datasets mới
- [ ] Evaluate
- [ ] So sánh kết quả new vs old
- [ ] Document findings (WER/CER trends)

**Expected Outcome:**
- Nếu WER giảm → model tốt, data test cũ xấu
- Nếu WER tăng → model thực sự không tốt
- Nếu WER ổn định → result reliable

---

## PRIORITY 2: IMPORTANT FIXES (CẦN LÀM)

**Timeline:** 1-2 tuần  
**Impact:** Tăng độ tin cậy và trình độ chuyên nghiệp

---

### FIX #4: Thêm Baseline Model Để So Sánh

#### 📋 Vấn đề

```
HIỆN TẠI:
  - Chỉ có một kiến trúc (CNN+LSTM+Attention)
  - Không biết có "tốt" hay không
  - Không có reference point
```

#### ✅ Giải pháp

**Bước 1: Chuẩn bị Baseline Models**

```python
# backend/models_baseline/vgg_baseline.py
import torch
import torch.nn as nn

class VGGBaseline(nn.Module):
    """Simple VGG-style baseline"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4995, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # x: (B, T, 4995)
        x = x.transpose(1, 2)  # (B, 4995, T)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**Bước 2: Train Baseline**

```bash
# Train VGG baseline
python backend/train_baseline.py \
  --model-name vgg_baseline \
  --model-class VGGBaseline \
  --batch-size 32 \
  --epochs 100
```

**Bước 3: Compare Results**

```python
# benchmark/compare_models.py
import json
from pathlib import Path

models = ['sign_model', 'vgg_baseline', 'simple_lstm']

results = {}
for model_name in models:
    # Load training history
    hist_file = f'backend/models/{model_name}/training_history.json'
    if Path(hist_file).exists():
        with open(hist_file) as f:
            hist = json.load(f)
            results[model_name] = {
                'val_acc_best': max(h['val_acc'] for h in hist),
                'val_loss_final': hist[-1]['val_loss'],
                'params': '?' # TODO: add param count
            }

print("MODEL COMPARISON:")
print("=" * 60)
for model, metrics in results.items():
    print(f"{model:20} | Val Acc: {metrics['val_acc_best']:.4f} | Final Loss: {metrics['val_loss_final']:.4f}")
```

#### 📊 Checklist

- [ ] Implement VGGBaseline
- [ ] Implement SimpleLSTMBaseline  
- [ ] Train cả 2 baseline
- [ ] Compare val accuracy, training time
- [ ] Document comparison table
- [ ] Nếu main model tốt hơn → tuyên bố

---

### FIX #5: Error Analysis Chi Tiết (Confusion Matrix)

#### 📋 Vấn đề

```
HIỆN TẠI:
  - Chỉ biết WER/CER
  - Không biết mô hình sai ở đâu
  - Không biết lỗi hệ thống hay random noise
```

#### ✅ Giải pháp

**Bước 1: Tạo Confusion Matrix**

```python
# benchmark/error_analysis.py
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze_errors(gt_file, pred_file, output_dir='benchmark/error_analysis'):
    """
    gt_file: benchmark/sentence_level/data/sentence_dataset.json
    pred_file: benchmark/sentence_level/data/sentence_predictions.json
    """
    
    with open(gt_file) as f:
        gt_data = json.load(f)
    with open(pred_file) as f:
        pred_data = json.load(f)
    
    # Extract class indices
    gt_labels = [sample['class_idx'] for sample in gt_data if sample.get('split') == 'test']
    pred_labels = [pred_data[i]['pred_class_idx'] for i in range(len(pred_data))]
    
    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # Plot top-20 most confused classes
    # ... (matplotlib code)
    
    # Find hardest examples
    wrong_indices = np.where(np.array(gt_labels) != np.array(pred_labels))[0]
    print(f"Error rate: {len(wrong_indices) / len(gt_labels) * 100:.2f}%")
    
    # Print sample errors
    for idx in wrong_indices[:10]:
        print(f"GT: {gt_labels[idx]}, Pred: {pred_labels[idx]}")

analyze_errors(
    'benchmark/sentence_level/data/sentence_dataset.real_v2.split.json',
    'benchmark/sentence_level/data/sentence_predictions_v2.json'
)
```

**Bước 2: Phân tích Lỗi Theo Dialect/Signer**

```python
# benchmark/error_analysis_by_group.py

def analyze_by_group(gt_file, pred_file, group_by='signer'):
    """group_by: 'signer', 'dialect', or 'class_frequency'"""
    
    errors_by_group = {}
    
    for sample, pred in zip(gt_data, pred_data):
        group_key = sample[group_by]
        is_correct = sample['class_idx'] == pred['pred_class_idx']
        
        if group_key not in errors_by_group:
            errors_by_group[group_key] = {'total': 0, 'correct': 0}
        
        errors_by_group[group_key]['total'] += 1
        if is_correct:
            errors_by_group[group_key]['correct'] += 1
    
    # Print results
    print(f"Accuracy by {group_by}:")
    print("=" * 50)
    for group, stats in sorted(errors_by_group.items()):
        acc = stats['correct'] / stats['total'] * 100
        print(f"{group:20} | Acc: {acc:5.2f}% | N: {stats['total']:4}")
```

**Bước 3: Vẫn đề Keypoint vs Model**

```python
# benchmark/debug_keypoint_vs_model.py
# Lấy 10 mẫu fail, check:
# 1. Keypoint có hợp lý không?
# 2. Model dự đoán có confidence cao không?
# 3. Top-5 predictions có chứa GT không?

def analyze_failure_case(video_path, ground_truth_class):
    """Analyze why model predicted wrong for a video"""
    
    # 1. Extract keypoint
    keypoints = extract_keypoints(video_path)
    print(f"Keypoint stats: shape={keypoints.shape}, min={keypoints.min()}, max={keypoints.max()}")
    
    # 2. Run model
    with torch.no_grad():
        logits = model(keypoints)
        probs = torch.softmax(logits, dim=-1)
    
    # 3. Get top-5
    top5 = torch.topk(probs, 5)
    print(f"Top-5 predictions: {top5}")
    print(f"Ground truth: {ground_truth_class}")
    print(f"Is GT in top-5: {ground_truth_class in top5.indices}")
    
    # 4. Visualize keypoints
    visualize_keypoints(keypoints)
```

#### 📊 Checklist

- [ ] Implement confusion matrix
- [ ] Generate heatmap
- [ ] Analyze by signer
- [ ] Analyze by dialect  
- [ ] Check top-5 accuracy
- [ ] Generate sample failure cases
- [ ] Document findings

---

### FIX #6: Giải Quyết Data Imbalance (3315 Lớp vs 4362 Samples)

#### 📋 Vấn đề

```
TOÁN HỌC:
  4362 videos / 3315 classes = 1.3 video/class average
  
ĐIỀU NÀY NGHĨA:
  - ~60% lớp có chỉ 1 video
  - ~30% lớp có 2 video
  - ~10% lớp có 3+ video
  
KẾT QUẢ:
  - Model không học class rare
  - WER/CER cao do nhiều lỗi trên minority class
```

#### ✅ Giải pháp

**Bước 1: Phân tích Class Distribution**

```python
# backend/analyze_class_dist.py
import json
from collections import Counter
from pathlib import Path

# Load data
with open('Data.xlsx', 'r') as f:
    # ... parse xlsx
    pass

class_counts = Counter(labels)
print(f"Total classes: {len(class_counts)}")
print(f"Max count: {max(class_counts.values())}")
print(f"Min count: {min(class_counts.values())}")
print(f"Mean count: {np.mean(list(class_counts.values())):.2f}")

# Distribution
one_sample_classes = sum(1 for c in class_counts.values() if c == 1)
two_sample_classes = sum(1 for c in class_counts.values() if c == 2)
many_sample_classes = sum(1 for c in class_counts.values() if c > 2)

print(f"Classes with 1 sample: {one_sample_classes} ({one_sample_classes/len(class_counts)*100:.1f}%)")
print(f"Classes with 2 samples: {two_sample_classes} ({two_sample_classes/len(class_counts)*100:.1f}%)")
print(f"Classes with 3+ samples: {many_sample_classes} ({many_sample_classes/len(class_counts)*100:.1f}%)")
```

**Bước 2: Áp dụng Weighted Sampler**

```python
# backend/train_gpu.py - sửa training

from torch.utils.data import WeightedRandomSampler

# Tính class weights (inverse frequency)
class_counts = torch.tensor([count_dict[c] for c in dataset.classes])
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[dataset.class_indices]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    sampler=sampler,  # Changed from shuffle=True
    batch_size=batch_size,
    num_workers=4,
)
```

**Bước 3: Sử dụng Focal Loss (đã có)**

```python
# backend/losses.py - Focal Loss (kiểm tra xem có không)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

**Bước 4: Data Augmentation Cho Minority Classes**

```python
# backend/spatial_augmentation.py - tăng cường thêm

# Hiện tại có:
# - Rotate, Scale, Translate
# 
# Thêm:
# - Temporal warping (slow down/speed up)
# - Keypoint dropout (simulate occlusion)
# - Noise injection

def augment_minority_class(keypoints, label, class_counts):
    """Apply stronger augmentation to minority classes"""
    
    if class_counts[label] < 5:  # Minority class
        # Apply 2-3 augmentations
        keypoints = rotate(keypoints, angle=np.random.uniform(-30, 30))
        keypoints = scale(keypoints, scale=np.random.uniform(0.8, 1.2))
        keypoints = add_noise(keypoints, std=0.01)
    
    return keypoints
```

#### 📊 Checklist

- [ ] Analyze class distribution
- [ ] Implement WeightedRandomSampler
- [ ] Kiểm tra Focal Loss có dùng
- [ ] Thêm temporal augmentation
- [ ] Re-train với balanced sampling
- [ ] Compare results (before/after)

---

### FIX #7: On-Device Testing (Mobile Latency)

#### 📋 Vấn đề

```
HIỆN TẠI:
  - Model được trained trên GPU (RTX 3060)
  - Không biết chạy trên mobile như thế nào
  - Không có latency measurements
  - Có thể accuracy drop 20-30% sau quantization
```

#### ✅ Giải pháp

**Bước 1: Quantize Model Cho Mobile**

```python
# backend/quantize_model.py
import torch

# Load trained model
model = torch.load('backend/models/sign_model_best.pt')

# Convert to ONNX (để chạy trên mobile)
dummy_input = torch.randn(1, 60, 4995)  # 1 video, 60 frames
torch.onnx.export(
    model,
    dummy_input,
    "backend/models/sign_model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

# Quantize INT8
from torch.quantization import quantize_dynamic, QInt8

quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # only quantize Linear layers
    dtype=torch.qint8
)

torch.save(quantized_model, "backend/models/sign_model_quantized.pt")

# Check size
import os
original_size = os.path.getsize("backend/models/sign_model_best.pt")
quantized_size = os.path.getsize("backend/models/sign_model_quantized.pt")
print(f"Original: {original_size / 1e6:.2f} MB")
print(f"Quantized: {quantized_size / 1e6:.2f} MB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

**Bước 2: Kiểm Tra Accuracy Drop Sau Quantization**

```bash
# Test quantized model trên validation set
python backend/test_model_load.py \
  --model backend/models/sign_model_quantized.pt \
  --data benchmark/sentence_level/data/sentence_dataset.sample.json

# Compare với original
# Expected: < 2-3% accuracy drop tùy model complexity
```

**Bước 3: Benchmark Latency**

```python
# benchmark/mobile_latency_test.py
import time
import torch
import numpy as np

def benchmark_latency(model, device, num_runs=100):
    """Benchmark inference latency"""
    
    model = model.to(device).eval()
    
    # Prepare dummy data
    x = torch.randn(1, 60, 4995).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    latency = (time.time() - start) / num_runs * 1000  # ms
    return latency

# Test on different devices
print("LATENCY BENCHMARKS:")
print("=" * 60)

# Desktop GPU
model_fp32 = torch.load("backend/models/sign_model_best.pt")
latency_gpu = benchmark_latency(model_fp32, 'cuda')
print(f"GPU (RTX 3060, FP32): {latency_gpu:.2f} ms")

# Quantized
model_int8 = torch.load("backend/models/sign_model_quantized.pt")
latency_gpu_int8 = benchmark_latency(model_int8, 'cuda')
print(f"GPU (RTX 3060, INT8):  {latency_gpu_int8:.2f} ms ({(1-latency_gpu_int8/latency_gpu)*100:.1f}% faster)")

# CPU (simulate mobile)
latency_cpu = benchmark_latency(model_int8, 'cpu')
print(f"CPU (INT8):             {latency_cpu:.2f} ms ({latency_cpu/latency_gpu:.1f}x slower than GPU)")

# Check if real-time
if latency_cpu < 100:  # < 100ms per 60 frames
    print("✅ Real-time capable (< 100ms)")
else:
    print(f"❌ Too slow for real-time ({latency_cpu:.0f}ms, need < 100ms)")
```

**Bước 4: Document Mobile Deployment**

```markdown
# Mobile Deployment Guide

## Model File
- Original: backend/models/sign_model_best.pt (size: X MB)
- Quantized: backend/models/sign_model_quantized.pt (size: Y MB)

## Latency
- GPU (RTX 3060): Z ms
- CPU (Desktop): W ms
- Expected on Mobile (Snapdragon 888+): ~150-200ms

## Accuracy  
- Original model: A%
- Quantized model: A-2.5%
- Acceptable accuracy drop: < 3%

## Deployment Steps
1. Use quantized model
2. Test on actual device
3. Monitor inference latency
4. Adjust model if needed
```

#### 📊 Checklist

- [ ] Tạo ONNX model
- [ ] Quantize INT8
- [ ] Test accuracy after quantization
- [ ] Benchmark latency (GPU/CPU/simulated mobile)
- [ ] Document findings
- [ ] Nếu quá chậm → cân nhắc model compression (pruning)

---

## PRIORITY 3: NICE-TO-HAVE (TỰA CHỌN)

**Timeline:** 1-2 tuần (nếu có thời gian)

---

### FIX #8: Cross-Validation & Statistical Significance

```bash
# backend/cross_validation.py
# Chạy 5-fold cross-validation trên full dataset
# Report: mean accuracy ± std
# Nếu std < 3% → confidence cao
```

### FIX #9: Ablation Study

```bash
# benchmark/ablation_study.py
# Test impact của từng thành phần:
# - CNN part
# - BiLSTM part
# - Attention part
# - Language model
# - TTA
```

### FIX #10: Visualization Tools

```bash
# benchmark/visualize_predictions.py
# Generate confusion matrix heatmap
# Generate t-SNE plot of embeddings
# Generate sample prediction with confidence
```

---

## TIMELINE TỔNG HỢP

```
WEEK 1:
├─ (2 ngày) FIX #1: Tái-train với val set lớn
├─ (1 ngày) FIX #2: Debug training vs inference gap
└─ (2 ngày) FIX #3: Benchmark trên dataset lớn

WEEK 2:
├─ (2 ngày) FIX #4: Thêm baseline models
├─ (1 ngày) FIX #5: Error analysis
├─ (1 ngày) FIX #6: Handle class imbalance
└─ (1 ngày) FIX #7: Mobile latency test

WEEK 3-4 (nếu cần):
├─ Priority 3 fixes
├─ Final validation
└─ Report writing

EXPECTED OUTCOME AFTER WEEK 2:
✅ WER: 1.10 → < 0.5
✅ Val accuracy: 100% (unreliable) → 85-90% (reliable)
✅ Test set: 10/60 mẫu → 100/200 mẫu
✅ Clear understanding of model performance
✅ Baseline comparisons
✅ Deploy-ready state
```

---

## SUCCESS METRICS

```
BEFORE (Hiện tại):
  ├─ WER: 1.10 ❌
  ├─ CER: 1.10 ❌
  ├─ Exact Match: 0% ❌
  ├─ Val set: 4 mẫu ❌
  ├─ Test set: 10/60 mẫu ❌
  └─ Baseline: None ❌

AFTER (Target):
  ├─ WER: < 0.5 ✅
  ├─ CER: < 0.5 ✅
  ├─ Exact Match: > 20% ✅
  ├─ Val set: > 400 mẫu ✅
  ├─ Test set: > 100/200 mẫu ✅
  ├─ Baseline: VGG, LSTM ✅
  └─ Publishable: ✅ Yes
```

---

## LIÊN HỆ & HỖ TRỢ

**Nếu gặp vấn đề:**
1. Kiểm tra logs: `backend/training.log`, `backend/inference.log`
2. Run debug scripts: `backend/debug_gap.py`, `backend/debug_normalization.py`
3. Test on small dataset trước khi scaled-up
4. Document findings để share với team

---

**Tác giả:** Architecture & ML Optimization  
**Ngày:** Tháng 4 năm 2026  
**Version:** 1.0 - Action Plan Comprehensive
