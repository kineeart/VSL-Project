# Colab Pro Setup: Quick Start Guide

## 🚀 HOW TO USE THIS NOTEBOOK ON COLAB

### Step 1: Open Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook: **File → New Notebook**
3. Rename it: "VSL-Benchmark-Pipeline"

---

### Step 2: Enable Colab Pro & A100 GPU

1. Click **"Upgrade"** (top-right) to enable **Colab Pro**
2. Click **⚙ Settings** → **Runtime type**
3. Select:
   - **GPU**: A100 (best) or H100 (alternative)
   - **High RAM** (optional, but recommended)
4. Click **Save**

---

### Step 3: Copy & Paste Cells

For each cell in [COLAB_PRO_NOTEBOOK_GUIDE.md](COLAB_PRO_NOTEBOOK_GUIDE.md):

1. In Colab: Click **+ Code** to add a new cell
2. Copy the code from the guide
3. Paste into Colab cell
4. Press **Shift+Enter** to run

**Example**:
```
CELL 0 → Click "+ Code" → Paste "Clone Repository & Install Dependencies" code
CELL 1 → Click "+ Code" → Paste "Mount Google Drive & Verify GPU" code
CELL 2 → etc...
```

---

### Step 4: Monitor Execution

- **Green checkmark** = Cell succeeded ✓ Move to next!
- **Red X** = Error. See troubleshooting section below
- **Spinning wheel & elapsed time** = Still running, wait...

---

## 💾 COLAB PRO OPTIMIZATION TIPS

### A. Memory Management

**Problem**: GPU running out of VRAM during parallel training

**Solutions**:

```python
# In Cell 5-6, if you get OOM error:

# Option 1: Reduce batch size
BS = 16  # instead of 32

# Option 2: Force sequential training
RUN_PARALLEL = False

# Option 3: Clear cache between cells
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

### B. Session Timeout Prevention

Colab sessions timeout after 6 hours without activity.

**Prevention**:
```python
# At start of Cell 4-6 (long training runs), add:
!curl -s -X POST "http://[::1]:7860/api/queue" -H "Content-Type: application/json" 2>/dev/null || true
```

Or: **Enable "Notebook settings" → "Restart on disconnection"**

### C. Storage Management

Colab provides 100GB storage. Monitor with:
```python
!df -h  # Check disk space
```

If running low:
```python
!rm -rf backend/landmarks  # Landmarks can be re-extracted
# Models are lightweight (MB not GB)
```

### D. Recovery from Interruption

If Colab session crashes:

1. **Your data is auto-saved** at cells 3, 10
2. **Restart from Cell 4** (or later if further complete)
3. **Don't restart from Cell 0** (repo still exists)

```python
# Skip to Cell 4 - data from Cell 3 still available if:
X_train = np.load('/content/data_x_train.npy')  # Already saved!
```

---

## 📊 EXPECTED RUNTIME BREAKDOWN

| Phase | Cells | Duration (A100) | Can Parallel? |
|-------|-------|-----------------|---------------|
| Setup | 0-2 | 5 min | N/A (setup) |
| Data Verify | 3 | 10 min | ✗ Critical path |
| Main Model | 4 | 4-6h | Single (GPU intense) |
| Baselines | 5 | 8-12h | ✓ (if 80GB RAM) |
| Ablations | 6 | 10-15h | ✓ (if 80GB RAM) |
| Benchmarks | 7-9 | 1.5-2h | ✓ Some parallel |
| Download | 10-11 | 10-15 min | ✗ Sequential |
| **TOTAL** | **0-11** | **26-34h** | **1.5-2 days** |

---

## ✅ QUALITY CHECKLIST BEFORE STARTING

- [ ] You have **Colab Pro subscription** (not free tier)
- [ ] GPU selected is **A100 or H100** (not T4 or V100)
- [ ] **High RAM** is enabled (optional but recommended)
- [ ] You've read [COLAB_PRO_NOTEBOOK_GUIDE.md](COLAB_PRO_NOTEBOOK_GUIDE.md)
- [ ] GitHub repo is public: [https://github.com/kineeart/VSL-Project](https://github.com/kineeart/VSL-Project)
- [ ] You have Google Drive account for backup
- [ ] You have 100GB+ free space in Google Drive

---

## 🛠️ DETAILED TROUBLESHOOTING

### Issue: "CUDA out of memory" in Cell 4-6

**Root cause**: Batch size too large for GPU memory

**Solution**:
```python
# In Cell 2, before training cells:
BS = 16  # Reduce from 32
EPOCHS = 100  # Optional: reduce epochs for testing

# Then re-run Cell 4-6
```

### Issue: "Module not found" error

**Root cause**: Import failed

**Solution**:
```python
# Reinstall package
!pip install -q --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or reinstall all requirements
!cd /content/VSL-Project && pip install -r backend/requirements.txt -q
```

### Issue: "numpy.dtype size changed" when importing pandas in Cell 2

**Root cause**: Colab has an incompatible `numpy`/`pandas` binary pair after dependency installation.

**Solution**:
```python
# Re-run Cell 0 first; it now repairs the numpy/pandas pair automatically.
# If Cell 2 still fails, restart the runtime and run from Cell 0 again.

!pip install --no-cache-dir --force-reinstall numpy==1.26.4 pandas==2.2.2 -q
```

If pip still prints many dependency-conflict warnings, they are usually safe to ignore for this project as long as these imports work in Cell 2: `numpy`, `pandas`, `torch`, `cv2`, `mediapipe`, `openpyxl`, `sklearn`.

### Issue: "Notebook disconnected / Session crashed"

**Cause**: Cell ran > 12 hours without interruption

**Solution**:
```python
# Check which cell failed in runtime logs
# Then:
1. Restart runtime: Runtime → Restart runtime
2. Re-run from a checkpoint (data saved at Cell 3, 10)
3. Skip to Cell 5 if Cell 4 completed
```

### Issue: "No space left on device"

**Cause**: /tmp or Colab storage full

**Solution**:
```python
# Check space
!df -h

# Clean up
!rm -rf /tmp/*
!cd /content && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# If still issues, delete old models:
!rm -rf /content/VSL-Project/backend/models/baseline_*  # Keep main model
```

### Issue: "Training stuck / no output for 30+ min"

**Cause**: Model training is normal (each epoch takes ~60s)

**Solution**: 
- **Wait** (don't interrupt!)
- Monitor with: `!nvidia-smi` to see GPU utilization
- If GPU utilization = 0%, then something is stuck (interrupt with ⏹)

---

## 📥 WHAT TO DOWNLOAD AFTER COMPLETION

**Download these files to your local machine:**

1. ✅ **`all_models.zip`** (2.3 GB)
   - Contains all trained models
   - Keep these for future inference

2. ✅ **`all_reports.zip`** (150 MB)
   - Contains: `BENCHMARK_SUMMARY.md`, JSON results
   - Use for paper writing

3. ✅ **`BENCHMARK_SUMMARY.md`** (single file)
   - Human-readable results table
   - Copy directly into your paper!

---

## 📊 SAMPLE COLAB EXECUTION LOG

```
=== VSL BENCHMARK PIPELINE ===

CELL 0: Clone and Setup
→ Cloning repository...
✓ Repository already exists
✓ Installing PyTorch...
✓ Installing dependencies...
✅ Setup complete! [~2 min]

CELL 1: Mount Drive and Check GPU
✓ Google Drive mounted
✓ Using device: cuda
  GPU: NVIDIA A100-SXM4-80GB
  Memory: 80.0 GB
  ✓ GPU tier: EXCELLENT
✅ GPU verification complete! [~1 min]

CELL 2: Setup utilities
✓ Base directory: /content/VSL-Project
✓ Features: NF=4995, NRF=1662, SEQ=60, BS=32
✅ Utilities imported! [~2 min]

CELL 3: Verify Data Integrity
✓ Loaded 3000+ videos, 100+ classes
✓ Landmarks ready (auto-cached)
✓ Train: (50000, 60, 4995), Val: (10000, 60, 4995)
✅ Data integrity verified! [~10 min]

CELL 4: Train Main Model
=========================================================
TRAINING MAIN MODEL (CNN+BiLSTM+Attention+Cosine)
=========================================================
[TRAIN] Data loaded...
  Ep   1/200  Train: 45.20%  Val-Top1: 38.50%  ...  62s
  Ep   2/200  Train: 52.80%  Val-Top1: 42.10%  ...  61s
  .....
  Ep 185/200  Train: 98.30%  Val-Top1: 87.80%  ...  62s
  [Early stopping at epoch 185]
  ✓ Best! Top1: 88.50%  Top5: 96.20%
  THÀNH CÔNG! Thời gian: 185.5 phút
✅ Main model training complete! [~185 min = 3.1 hours]

CELL 5: Train Baselines
=========================================================
TRAINING BASELINE MODELS (PARALLEL)
=========================================================
✓ GPU has 80GB VRAM → PARALLEL mode
→ Starting baseline training...

  TRAINING: SimpleLSTM
    Ep 1/200  Train: 40.50%  Val-Top1: 35.20%  61s
    ...
    COMPLETED: SimpleLSTM
    Best Val Top-1: 71.30%  Time: 180.0 minutes

  TRAINING: GRU
    Ep 1/200  Train: 38.90%  Val-Top1: 33.80%  62s
    ...
    COMPLETED: GRU
    Best Val Top-1: 78.90%  Time: 185.0 minutes

  TRAINING: Transformer
    Ep 1/200  Train: 42.10%  Val-Top1: 36.50%  125s
    ...
    COMPLETED: Transformer
    Best Val Top-1: 82.40%  Time: 320.0 minutes

✅ Baseline models training complete! [~320 min in parallel = 5.3 hours]

CELL 6: Train Ablations
[Similar format to Cell 5]
✅ Ablation studies complete! [~540 min in parallel = 9 hours]

CELL 7: On-Device Benchmark
====================================================================
ON-DEVICE PERFORMANCE SUMMARY
====================================================================
Model               FPS (BS=1)   Latency (MS)   Size (MB)   Params (M)
SimpleLSTM          150.5        6.6            5.2         1.46
... [table continues]
✅ On-device benchmark complete! [~45 min]

CELL 8: Dataset Statistics
====================================================================
DATASET STATISTICS ANALYSIS
====================================================================
[STATS] Analyzing continuous dataset...
  Samples: 60
  Vocabulary: 52 unique glosses
  Sentence length: min=2, max=28, mean=12.5

[STATS] Analyzing sentence-level dataset...
  Samples: 30
  Splits: {'train': 21, 'val': 4, 'test': 5}
  Vocabulary: 48 unique glosses

✅ Dataset statistics complete! [~30 min]

CELL 9: Aggregate Results
✓ Main model: Top1=88.50%
✓ Baselines (3 models)
✓ Ablations (4 variants)
✓ On-device metrics loaded

BENCHMARK_SUMMARY.md generated ✨
✅ Results aggregation complete! [~30 min]

CELL 10: Download Results
→ Preparing download files...
✓ Created all_models.zip (2.3 GB)
✓ Created all_reports.zip (150 MB)
→ Downloading files...
✅ Downloads queued!
✅ Backed up to Google Drive

CELL 11: View Final Report
[Markdown summary printed]

=============================================================
TOTAL EXECUTION TIME: 26.5 hours on A100 GPU ✓
All results downloaded & backed up ✓
=============================================================
```

---

## 🎯 QUICK START (3-STEP)

1. **Open Colab Pro A100**: https://colab.research.google.com
2. **Copy-paste 12 cells** from [COLAB_PRO_NOTEBOOK_GUIDE.md](COLAB_PRO_NOTEBOOK_GUIDE.md)
3. **Run sequentially** (or parallel if 80GB+ RAM)
4. **Download results** when complete (~26-34 hours)

Note: Cell 0 may terminate/restart the runtime once after dependency repair. This is expected for fixing numpy/pandas ABI on Colab. After restart, run Cell 0 again, then continue with Cell 1.

### Drive-only Dataset Workflow

If your GitHub repo does not include `Data.xlsx` and `Videos/`, store them in Google Drive:

```text
/content/gdrive/MyDrive/VSL_Data/
├── Data.xlsx
└── Videos/
```

Cell 1 in the guide will auto-attach this dataset path into `/content/VSL-Project`.

If the dataset is in **Shared with me**, create a **Shortcut to Drive** into `MyDrive` first (recommended), then keep the same folder layout above.

### Issue: "No data mapping loaded" in Cell 3

**Root cause**: `Data.xlsx` and/or `Videos/` are missing from `/content/VSL-Project`.

**Solution**:
1. Put dataset in Drive as:
  - `/content/gdrive/MyDrive/VSL_Data/Data.xlsx`
  - `/content/gdrive/MyDrive/VSL_Data/Videos/`
2. Re-run Cell 1 (dataset attach step).
3. Re-run Cell 3.

### Issue: "BrokenProcessPool" during landmark extraction in Cell 3

**Root cause**: Too many worker processes loading MediaPipe model at once can crash the process pool on Colab.

**Solution**:
1. Use the updated Cell 3 in the notebook guide (it auto-limits workers and retries).
2. Re-run Cell 3 only (cache keeps completed items).
3. If it still crashes, set workers to 1 in Cell 3 and run again.

Notes:
- `jax_cuda12_plugin ... not compatible with jaxlib` warnings are usually harmless for this VSL pipeline.
- They do not block PyTorch/MediaPipe training unless your notebook explicitly uses JAX.

---

## 🎓 FOR ACADEMIC PUBLICATION

After Colab completes, use these outputs:

- **`BENCHMARK_SUMMARY.md`** → Copy table into your paper
- **`training_history_per_epoch.csv`** → Create convergence plots
- **`ondevice_benchmark_results.json`** → On-device deployment section
- **`dataset_statistics.json`** → Dataset description section

---

## ❓ NEED HELP?

1. Check [REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md) for detailed setup
2. Check [COLAB_PRO_NOTEBOOK_GUIDE.md](COLAB_PRO_NOTEBOOK_GUIDE.md) for cell details
3. Review "Troubleshooting" section above
4. Check GPU memory with `!nvidia-smi` during training

---

**🚀 Ready? Start copy-pasting cells into Colab Pro now!**
