# Google Colab Setup Guide - Verification First

**Mục đích:** Kiểm tra đầy đủ dữ liệu, environment, train, rồi benchmark trên Google Colab  
**Dự kiến thời gian:** 1-2 ngày cho quy trình đầy đủ; 4-6 giờ chỉ khi mọi thứ đã sẵn sàng  
**GPU:** A100/T4 (Colab Pro: A100 nhanh hơn 3-5x)

---

## ⏱️ TIMELINE ĐÚNG QUY TRÌNH

```
PHASE 0 - Verify data/environment                (30-45 phút)
PHASE 1 - Smoke test import/load                  (15-20 phút)
PHASE 2 - Full training                           (2-8 giờ tùy GPU)
PHASE 3 - Generate predictions                   (30-90 phút)
PHASE 4 - Evaluate benchmarks                     (20-60 phút)
PHASE 5 - Export results and compare             (15-30 phút)

TOTAL: ~1-2 ngày nếu làm nghiêm túc
```

---

## 📋 PRE-FLIGHT CHECKLIST

Trước khi bắt đầu, hãy chuẩn bị:

### ✅ Local Prep (15 phút)

```powershell
# 1. Kiểm tra git repo status
git status -sb

# 2. Create zip của repo (để upload Colab nhanh)
# Windows:
Compress-Archive -Path "." -DestinationPath "VSL_project.zip"

# Linux/Mac:
zip -r VSL_project.zip .

# Expected size: ~500MB - 1GB (tự copy qua Colab)
```

### ✅ Google Drive Setup (10 phút)

Tạo folder trên Drive:
```
My Drive/
  └── NCKH_2025/
       ├── VSL_project.zip          (upload repo)
       ├── Videos/                  (symlink hoặc upload subset)
       ├── backend/landmarks/       (symlink hoặc upload subset)
       └── colab_outputs/           (output of training)
```

**HOẶC dùng Colab clone từ Git:**
```python
!git clone https://github.com/YOUR_REPO/VSL.git /content/VSL
```

### ✅ Những thứ phải có trước khi bấm train

- `Data.xlsx`
- `Videos/`
- `backend/landmarks/` hoặc quyền tạo cache mới
- `backend/requirements.txt`
- `backend/train_gpu.py`
- `benchmark/continuous/`
- `benchmark/sentence_level/`
- đủ dung lượng Drive để lưu checkpoint

---

## 🚀 COLAB NOTEBOOK SETUP

### **CELL 1: Mount Drive & Install Dependencies**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup working directory
import os
os.chdir('/content/drive/My Drive/NCKH_2025')

# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install fastapi uvicorn mediapipe opencv-python scikit-learn openpyxl tqdm numpy -q
!pip install jupyterlab matplotlib seaborn -q

print("✅ Dependencies installed")
```

### **CELL 2: Clone/Unzip Repository**

```python
# Option A: Clone from Git (nhanh + clean)
!git clone https://github.com/YOUR_GITHUB/VSL.git VSL_repo
os.chdir('VSL_repo')

# Option B: Unzip từ Drive
# !unzip /content/drive/My\ Drive/NCKH_2025/VSL_project.zip

# Verify structure
!ls -la backend/
!ls -la benchmark/
print("✅ Repository setup done")
```

### **CELL 3: Check GPU & Setup Environment**

```python
import torch
print(f"✅ GPU: {torch.cuda.get_device_name()}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
print(f"   PyTorch: {torch.__version__}")

# Create output directory
os.makedirs('colab_outputs', exist_ok=True)
print("✅ Colab environment ready")
```

### **CELL 4: Verify dataset integrity**

```python
from pathlib import Path

critical = [
    'Data.xlsx',
    'Videos',
    'backend/train_gpu.py',
    'backend/requirements.txt',
    'benchmark/continuous/data/continuous_dataset.real.json',
    'benchmark/continuous/data/continuous_eval.real.json',
    'benchmark/sentence_level/data/sentence_dataset.sample.json',
    'benchmark/sentence_level/data/sentence_dataset.synthetic.small.json',
]

for item in critical:
    print(item, '->', Path(item).exists())
```

### **CELL 5: Smoke test import/load**

```python
import sys
sys.path.append('/content/VSL')

import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())

from backend import train_gpu
print('Imported backend.train_gpu successfully')
```

---

## 🔧 FIX #1: RETRAIN WITH LARGE VALIDATION SET

### **CELL 6: Prepare Fixed Training Script**

```python
# Save as: backend/train_gpu_v2.py
training_script = '''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import datetime
import sys

# YOUR EXISTING IMPORTS
# from backend.models import SignModel

class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.history = []
    
    def log_epoch(self, epoch, metrics):
        self.history.append({
            'epoch': epoch,
            'train_loss': metrics['train_loss'],
            'train_acc': metrics['train_acc'],
            'val_loss': metrics['val_loss'],
            'val_acc': metrics['val_acc'],
            'val_samples': len(metrics['val_indices']),
            'train_samples': len(metrics['train_indices']),
        })
        print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, " +
              f"train_acc={metrics['train_acc']:.4f}, " +
              f"val_loss={metrics['val_loss']:.4f}, " +
              f"val_acc={metrics['val_acc']:.4f}")
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Log saved to {self.log_file}")

def split_dataset_properly(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset: 70% train, 15% val, 15% test
    (Fixed from 99:1)
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    print(f"Split: Train {n_train} ({train_ratio*100:.0f}%), " +
          f"Val {n_val} ({val_ratio*100:.0f}%), " +
          f"Test {n_test} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_indices, val_indices, test_indices

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(loader, desc="Validating"):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total

# MAIN TRAINING LOOP (simplified)
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your dataset
    # dataset = SignLanguageDataset(...)
    
    # Split properly (70/15/15 instead of 99/1/0)
    # train_idx, val_idx, test_idx = split_dataset_properly(dataset)
    
    # Create model
    # model = SignModel().to(device)
    
    # Logger
    logger = TrainingLogger("colab_outputs/training_v2.json")
    
    # Training loop
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    
    # for epoch in range(300):
    #     train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    #     val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
    #     
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), "colab_outputs/sign_model_best_v2.pt")
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #     
    #     logger.log_epoch(epoch, {...})
    #     
    #     if patience_counter >= patience:
    #         print(f"Early stopping at epoch {epoch}")
    #         break
    
    logger.save()
    print("✅ Training completed")
'''

# Write script
with open('backend/train_gpu_v2.py', 'w') as f:
    f.write(training_script)

print("✅ Training script v2 created")
```

### **CELL 7: Run Training**

```python
import subprocess
import time

# Start training - lưu log
log_file = 'colab_outputs/training_v2.log'

start_time = time.time()

# Run training command
cmd = [
    'python', 'backend/train_gpu_v2.py',
    '--epochs', '300',
    '--batch-size', '64',  # Bigger batch on Colab GPU
    '--learning-rate', '0.0005',
    '--patience', '30',
    '--output', 'colab_outputs',
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Stream output
for line in process.stdout:
    print(line.rstrip())

process.wait()

elapsed = (time.time() - start_time) / 60
print(f"\n✅ Training completed in {elapsed:.1f} minutes")

# Load results
import json
with open('colab_outputs/training_v2.json') as f:
    hist = json.load(f)
    print(f"\nTraining history ({len(hist)} epochs):")
    print(f"  Initial val acc: {hist[0]['val_acc']:.4f}")
    print(f"  Final val acc: {hist[-1]['val_acc']:.4f}")
    print(f"  Best val acc: {max(h['val_acc'] for h in hist):.4f}")
```

---

## 🔍 FIX #2: DEBUG GAP

### **CELL 8: Run Debug Scripts**

```python
# Create debug scripts inline
debug_code = '''
import torch
import numpy as np
import json

print("=" * 60)
print("DEBUGGING TRAINING VS INFERENCE GAP")
print("=" * 60)

# 1. Check normalization stats
try:
    norm_mean = np.load('backend/models/norm_mean.npy')
    norm_std = np.load('backend/models/norm_std.npy')
    print(f"✅ Normalization loaded: mean shape={norm_mean.shape}, std shape={norm_std.shape}")
    print(f"   Mean[0:5]: {norm_mean[:5]}")
    print(f"   Std[0:5]: {norm_std[:5]}")
    print(f"   Has NaN: {np.isnan(norm_mean).any()} (mean), {np.isnan(norm_std).any()} (std)")
except Exception as e:
    print(f"❌ Normalization error: {e}")

# 2. Load training history
try:
    with open('backend/models/training_history.json') as f:
        old_hist = json.load(f)
    print(f"\\n✅ Old training history: {len(old_hist)} epochs")
    print(f"   Val samples: {old_hist[-1].get('val_samples', '?')}")
    print(f"   Train samples: {old_hist[-1].get('train_samples', '?')}")
except Exception as e:
    print(f"❌ Error loading old history: {e}")

# 3. Compare old vs new
try:
    with open('colab_outputs/training_v2.json') as f:
        new_hist = json.load(f)
    print(f"\\n✅ New training history: {len(new_hist)} epochs")
    print(f"   Val samples: {new_hist[-1]['val_samples']}")
    print(f"   Train samples: {new_hist[-1]['train_samples']}")
    print(f"   Final val acc: {new_hist[-1]['val_acc']:.4f} (was: ~1.0 before)")
except Exception as e:
    print(f"❌ Error loading new history: {e}")

print("\\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
'''

exec(debug_code)
```

---

## 📊 FIX #3: BENCHMARK

### **CELL 9: Start backend server**

```python
import subprocess, time, os

backend_proc = subprocess.Popen(
    ['python', 'backend/app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

time.sleep(15)
print('✅ Backend server start requested on port 8000')
```

### **CELL 10: Generate & Evaluate Continuous Benchmark**

```python
import subprocess
import json

print("🚀 Starting continuous benchmark evaluation...")

# 1. Build continuous dataset (if not exist)
if not Path('benchmark/continuous/data/continuous_dataset.real_v2.json').exists():
    print("📁 Building continuous dataset (200 samples)...")
    cmd = [
        'python', 'benchmark/continuous/scripts/build_continuous_dataset.py',
        '--data-xlsx', 'Data.xlsx',
        '--videos-dir', 'Videos',
        '--output-json', 'benchmark/continuous/data/continuous_dataset.real_v2.json',
        '--num-samples', '200',
    ]
    subprocess.run(cmd, check=True)
    print("✅ Dataset created")

# 2. Generate predictions from the backend started in CELL 9
print("🔮 Generating predictions...")
cmd = [
    'python', 'benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py',
    '--dataset', 'benchmark/continuous/data/continuous_dataset.real_v2.json',
    '--output', 'colab_outputs/continuous_predictions_v2.json',
    '--backend', 'http://127.0.0.1:8000',
]
subprocess.run(cmd)
print("✅ Predictions generated")

# 4. Evaluate
print("📈 Evaluating benchmark...")
cmd = [
    'python', 'benchmark/continuous/scripts/evaluate_continuous_benchmark.py',
    '--gt', 'benchmark/continuous/data/continuous_dataset.real_v2.json',
    '--pred', 'colab_outputs/continuous_predictions_v2.json',
    '--breakdown',
    '--out', 'colab_outputs/continuous_eval_v2.json',
]
subprocess.run(cmd)

# 5. Load & display results
with open('colab_outputs/continuous_eval_v2.json') as f:
    eval_result = json.load(f)

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"Dataset: 200 samples")
print(f"WER: {eval_result['wer']:.4f}")
print(f"CER: {eval_result['cer']:.4f}")
print(f"Exact Match Text: {eval_result['exact_match_text']:.4f}")
print(f"Exact Match Gloss: {eval_result['exact_match_gloss']:.4f}")

# Compare with old
if Path('benchmark/continuous/data/continuous_eval.real.json').exists():
    with open('benchmark/continuous/data/continuous_eval.real.json') as f:
        old_result = json.load(f)
    print("\n" + "COMPARISON (Old vs New):")
    print(f"WER: {old_result['wer']:.4f} → {eval_result['wer']:.4f} " +
          f"({'↓' if eval_result['wer'] < old_result['wer'] else '↑'} {abs((eval_result['wer']-old_result['wer'])/old_result['wer']*100):.1f}%)")
    print(f"CER: {old_result['cer']:.4f} → {eval_result['cer']:.4f}")
```

### **CELL 11: Run sentence-level benchmark**

```python
import json
from pathlib import Path

if not Path('benchmark/sentence_level/data/sentence_dataset.synthetic.small.json').exists():
    raise FileNotFoundError('Missing sentence-level dataset')

!python benchmark/sentence_level/scripts/validate_sentence_dataset.py \
    --input benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
    --out colab_outputs/sentence_dataset.validation.json

!python benchmark/sentence_level/scripts/generate_sentence_predictions_from_backend.py \
    --dataset benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
    --output colab_outputs/sentence_predictions.synthetic.small.json \
    --backend http://127.0.0.1:8000 \
    --split test

!python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
    --gt benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
    --pred colab_outputs/sentence_predictions.synthetic.small.json \
    --split test --breakdown \
    --out colab_outputs/sentence_eval.synthetic.small.json
```

---

## 💾 CELL 12: Download Results

```python
# Zip all outputs
import shutil

output_dir = 'colab_outputs'
zip_file = f'VSL_colab_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

shutil.make_archive('colab_results', 'zip', output_dir)

print(f"✅ Results zipped: {zip_file}")
print(f"📥 Ready for download from Colab Files")

# List outputs
print("\n" + "=" * 60)
print("GENERATED FILES:")
print("=" * 60)
for f in Path(output_dir).glob('*'):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40} ({size_mb:.1f} MB)")
```

---

## ⚡ CHECKLIST TRƯỚC KHI RUN

```
PREPARE (trước khi bật Colab):
    ☐ Compress repo thành VSL_project.zip hoặc clone từ Git
    ☐ Chuẩn bị Google Drive folder structure
    ☐ Đảm bảo có Data.xlsx, Videos/, backend/landmarks/
    ☐ Chốt benchmark dataset sẽ dùng

ON COLAB:
    ☐ Cell 0: Verify runtime/GPU
    ☐ Cell 1: Mount Drive + Install
    ☐ Cell 2: Clone/unzip repo
    ☐ Cell 3: Check GPU + files
    ☐ Cell 4: Dataset integrity check
    ☐ Cell 5: Smoke test import/load
    ☐ Cell 6: Prepare training script
    ☐ Cell 7: Run training
    ☐ Cell 8: Debug analysis
    ☐ Cell 9: Start backend server
    ☐ Cell 10: Continuous benchmark
    ☐ Cell 11: Sentence-level benchmark
    ☐ Cell 12: Download results

TOTAL: ~1-2 ngày nếu làm nghiêm túc, hoặc ~4-6 giờ cho phần train/benchmark khi mọi thứ đã sẵn sàng
```

---

## 🎯 EXPECTED OUTCOMES HÔM NAY

```
BEFORE (hiện tại):
  ├─ WER: 1.10 (không tốt)
  ├─ Val set: 4 mẫu (không tin cậy)
  └─ Training history: 100% accuracy (nghi ngờ overfitting)

AFTER (khi xong):
  ├─ WER: 0.8-1.0 (có thể cải thiện)
  ├─ Val set: 400+ mẫu (tin cậy hơn)
  ├─ Training history: 85-90% accuracy (realistic)
  ├─ Clear debug findings (lỗi ở đâu)
  ├─ New benchmark on 200 samples (ổn định hơn)
  └─ Ready for Priority 2 fixes vào ngày mai
```

---

## 🔗 USEFUL COLAB TRICKS

```python
# Monitor GPU usage realtime
!watch -n 1 'nvidia-smi'

# Save checkpoints to Drive during training
import shutil
shutil.copy('sign_model_best.pt', '/content/drive/My Drive/NCKH_2025/sign_model_best_v2.pt')

# Quick test inference before full training
from backend.app import load_model
model = load_model('backend/models/sign_model_best.pt')
# ... quick test on small batch

# Parallel downloads (if need Videos/)
!aria2c --allow-overwrite=true -x 4 [URL] -o Videos.zip
```

---

## ⚠️ SỰ CỐ CÓ THỂ GẶP

| Sự cố | Giải pháp |
|---|---|
| **GPU quá chậm** | Nâng cấp Colab Premium ($10/tháng → A100 GPU) |
| **Memory không đủ** | Giảm batch size từ 64 → 32 |
| **Internet timeout** | Download repo thành zip local, upload Colab |
| **Backend server không start** | Check port 8000, kill process cũ |
| **Landmark files missing** | Upload subset vào Drive |
| **Timeout training** | Set epoch 150 thay 300 (hoàn thành nhanh hơn) |

---

## 📞 SUPPORT

Nếu gặp vấn đề:
1. **Lỗi import:** `!pip install [package]`
2. **Lỗi file:** Check paths, print `os.getcwd()`, `ls -la`
3. **Lỗi GPU:** `torch.cuda.is_available()` → phải `True`
4. **Lỗi timeout:** Cắt ngắn dataset, test local trước

---

**Prepared by:** ML Engineering  
**Date:** April 2026  
**Status:** Ready to execute on Colab
