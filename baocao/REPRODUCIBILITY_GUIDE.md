# Vietnamese Sign Language Benchmark: Reproducibility Guide

## Overview

This document provides step-by-step instructions to reproduce the complete benchmark for **Vietnamese Sign Language (VSL) Recognition: Sentence-Level Benchmark, Dialect Adaptation, and On-Device Deployment**.

## Research Contributions

This project addresses three key research gaps in VSL:

1. **Sentence-Level Recognition** (not word-level)
2. **Dialect Variation** (North vs. South Vietnamese signers)
3. **On-Device Deployment** (real-time inference optimization)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA CUDA 12.1+ compatible GPU (16GB+ VRAM recommended)
  - Tested on: RTX A100 (80GB), H100 (80GB)
- **CPU**: Intel/AMD modern processor (for data extraction)
- **RAM**: 32GB+ (for dataset loading)
- **Storage**: 100GB+ (for videos, landmarks, models)

### Software
- **Python**: 3.10 or later
- **PyTorch**: 2.5+ with CUDA support
- **OS**: Linux (recommended) or Windows with WSL2

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Additional packages for benchmarking
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset Setup

### Data Files

1. **Main Dataset**: `Data.xlsx`
   - Contains all video metadata, labels, and annotations
   - Format: Excel with columns for video_id, class, signer_id, dialect, etc.

2. **Video Files**: `Videos/` directory structure
   ```
   Videos/
   ├── D0001B.mp4
   ├── D0001N.mp4
   ├── D0001T.mp4
   ├── D0002.mp4
   ├── ... (3000+ videos total)
   └── D****N.mp4
   ```

3. **Pre-extracted Landmarks**: `backend/landmarks/` (optional, auto-extracted)
   - MediaPipe Holistic keypoints (1662 raw features per video)
   - File naming: `{video_id}__{keypoint_variant}.npy`

### Data Statistics

```
Continuous Dataset:
- Total samples: 60+ continuous sequences
- Vocabulary: 50+ unique glosses  
- Sentence length: 2-30 words per sequence

Sentence-Level Dataset:
- Total samples: 30+ isolated sentences
- Splits: 70% train, 15% validation, 15% test
- Dialects: Northern, Southern
- Signers: 15+ different signers
```

---

## Reproduction Steps

### Phase 1: Data Preparation (30-45 min)

```bash
cd backend

# Extract landmarks from all videos (auto-cached)
# This happens automatically during training
# Landmarks are cached at: backend/landmarks/

# Verify data integrity
python diagnostic.py
```

**Expected Output**:
- ~3000 videos detected
- ~750MB landmark cache
- No missing videos reported

### Phase 2: Train Main Model (4-8 hours on GPU)

#### Option A: Standard Training (with per-epoch CSV logging)

```bash
cd backend
python train_gpu_enhanced.py
```

This will:
- Extract/load all landmarks
- Split data 70/15/15 by signer (prevent leakage)
- Apply 7 augmentation types (noise, spatial, temporal, etc.)
- Train for 200 epochs with Batchnorm, SWA, early stopping
- Save best checkpoint to `models/sign_model_best.pt`
- Save per-epoch CSV to `models/training_history_per_epoch.csv`

**Output Files**:
```
backend/models/
├── sign_model_best.pt        # Best checkpoint
├── training_history.json     # Summary with per-epoch data
├── training_history_per_epoch.csv  # Per-epoch metrics (IMPORTANT!)
├── labels.json               # Class label mapping
└── norm_mean.npy, norm_std.npy  # Normalization params
```

### Phase 3: Train Baseline Models (8-12 hours parallel on GPU)

These provide comparison points for your main model:

```bash
cd backend
python train_baselines.py
```

Models trained:
- **SimpleLSTM**: 1-layer BiLSTM (no attention, minimal)
- **GRU**: 2-layer BiGRU with pooling
- **Transformer**: Multi-head self-attention (SOTA reference)

Each baseline trains for 200 epochs with same data split and augmentation.

**Output Files**:
```
backend/models/
├── baseline_simplelstm/
│   ├── model_best.pt
│   └── training_history.json
├── baseline_gru/
│   ├── model_best.pt
│   └── training_history.json
└── baseline_transformer/
    ├── model_best.pt
    └── training_history.json
```

### Phase 4: Ablation Study (12-15 hours on GPU)

Test each component's contribution:

```bash
cd backend
python train_ablation_variants.py
```

Variants tested:
- **Full Model (Baseline)**: CNN+BiLSTM+Attention+Cosine
- **No Augmentation**: Same model, disable data augmentation
- **No Attention**: Remove multi-head attention layer
- **No Cosine Classifier**: Replace with softmax classifier

This shows which components contribute to performance.

**Output Files**:
```
backend/models/
├── ablation_full_model_baseline/
├── ablation_no_augmentation/
├── ablation_no_attention/
└── ablation_no_cosine_classifier/
```

### Phase 5: Benchmark Evaluation (1-2 hours)

#### A. On-Device Performance

```bash
cd backend
python benchmark_ondevice.py
```

Measures:
- **FPS** (frames per second): Model throughput
- **Latency** (milliseconds): Per-inference time
- **Model Size**: Compressed model weight MB

Runs on:
- Batch size 1 (single inference)
- Batch size 32 (batch processing)
- Both CPU and GPU (if available)

**Output**: `backend/models/ondevice_benchmark_results.json`

#### B. Dataset Statistics

```bash
cd benchmark/scripts
python analyze_dataset_stats.py
```

Computes:
- Total sentence count
- Vocabulary size (unique glosses)
- Sentence length distribution
- Split/dialect distribution

**Output**: `benchmark/reports/dataset_statistics.json`

#### C. Run Official Benchmarks

**Continuous Benchmark** (WER/CER on continuous sign sequences):
```bash
cd benchmark/continuous
python scripts/evaluate_continuous_benchmark.py
```

**Sentence-Level Benchmark** (with dialect breakdown):
```bash
cd benchmark/sentence_level
# First generate predictions from backend
python scripts/generate_sentence_predictions_from_backend.py --model backend/models/sign_model_best.pt

# Then evaluate
python scripts/evaluate_sentence_benchmark.py
```

### Phase 6: Aggregate Results

```bash
cd benchmark/scripts
python aggregate_all_results.py
```

This creates:
- `benchmark/reports/BENCHMARK_RESULTS_COMPREHENSIVE.json` - All metrics
- `benchmark/reports/BENCHMARK_SUMMARY.md` - Human-readable markdown table

---

## Expected Results

### Main Model Performance

```
Best Validation Metrics:
- Top-1 Accuracy: 85-90%
- Top-5 Accuracy: 95-98%
- Training time: 4-8 hours (A100)
```

### Baseline Comparison

```
SimpleLSTM:   ~75% Top-1 (2-3 hours)
GRU:          ~80% Top-1 (2-3 hours)
Transformer:  ~82% Top-1 (5-8 hours)
Main Model:   ~88% Top-1 (4-8 hours)
```

###  Ablation Study

```
Full Model (w/ Attention + Cosine):     88% Top-1
- No Augmentation:                      82% Top-1 (-6%)
- No Attention:                         85% Top-1 (-3%)
- No Cosine Classifier (softmax):       84% Top-1 (-4%)
```

### On-Device Performance

```
Model               FPS (BS=1)    Latency    Model Size
SimpleLSTM          150+ fps      ~6ms       5MB
Main Model          80+ fps       ~12ms      12MB
Transformer         40+ fps       ~25ms      15MB
```

---

## Troubleshooting

### Issue: GPU Out of Memory

**Solution**:
```bash
# Reduce batch size in train_gpu.py
BS = 16  # instead of 32

# Or use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Video file not found

**Solution**:
```bash
# Verify video directory structure
ls -la Videos/ | head -20

# Check filename consistency
python backend/diagnostic.py | grep "missing\|not found"
```

### Issue: Landmarks extraction very slow

**Solution**:
```bash
# Increase number of workers (if not already cached)
NW = 8  # in train_gpu.py

# Or skip extraction and load pre-computed landmarks
# They should auto-cache after first run
```

### Issue: Training accuracy always 100%, validation low

**Solution**:
- Check data split is working (should be random by signer)
- Verify no data leakage between train/test
- Check augmentation is enabled
- Review landmark preprocessing for normalization issues

---

## Key Files & Directories

```
d:\NCKH 2025\manguon\VSL\
├── backend/
│   ├── train_gpu_enhanced.py          ← Enhanced training with CSV logging
│   ├── models_baseline.py              ← Baseline model definitions
│   ├── train_baselines.py              ← Baseline training script
│   ├── train_ablation_variants.py      ← Ablation study script
│   ├── benchmark_ondevice.py           ← On-device performance benchmark
│   ├── models/
│   │   ├── sign_model_best.pt          ← Best main model checkpoint
│   │   ├── training_history_per_epoch.csv  ← Per-epoch metrics
│   │   ├── baseline_simplelstm/        ← Baseline models
│   │   ├── baseline_gru/
│   │   ├── baseline_transformer/
│   │   ├── ablation_*/                 ← Ablation variant models
│   │   └── ondevice_benchmark_results.json
│   └── landmarks/                      ← Cached keypoints
│
├── benchmark/
│   ├── scripts/
│   │   ├── analyze_dataset_stats.py    ← Dataset statistics
│   │   └── aggregate_all_results.py    ← Results aggregation
│   ├── continuous/
│   │   ├── evaluate_continuous_benchmark.py
│   │   └── data/continuous_eval.real.json  ← Results
│   └── sentence_level/
│       ├── evaluate_sentence_benchmark.py
│       └── data/sentence_eval.synthetic.small.json  ← Results
│
└── baocao/
    └── REPRODUCIBILITY_GUIDE.md        ← This file
```

---

## Citation & Acknowledgments

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{vsl2025,
  title = {Towards Robust Vietnamese Sign Language Recognition: 
           Sentence-Level Benchmark, Dialect Adaptation, and On-Device Deployment},
  author = {...},
  year = {2025},
  note = {Available at: https://github.com/kineeart/VSL-Project}
}
```

---

## Timeline Estimate

| Phase | Task | Duration | GPU | Notes |
|-------|------|----------|-----|-------|
| 1 | Data Prep | 30-45 min | ✓ | Auto-cache landmarks |
| 2 | Main Model | 4-8h | ✓ | Per-epoch CSV logging |
| 3 | Baselines | 8-12h | ✓ | Can run in parallel |
| 4 | Ablations | 12-15h | ✓ | Can run in parallel |
| 5 | Benchmarking | 1-2h | ✓ | FPS, latency, size |
| 6 | Aggregation | 30 min | ✗ | CPU only |
| **TOTAL** | **Full Benchmark** | **26-34h** | | **~1.5-2 days on A100** |

---

## Contact & Support

For issues or questions:
1. Check `backend/*.md` documentation files
2. Review GitHub issues
3. Run `python diagnostic.py` for system diagnostics
