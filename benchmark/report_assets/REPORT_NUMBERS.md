# Tong hop so lieu bao cao VSL

## 1) Ket qua huan luyen mo hinh

| Model folder | So lop | Train samples | Val samples | So features | Best Val Top-1 | Best Val Top-5 | Thoi gian train (phut) |
|---|---:|---:|---:|---:|---:|---:|---:|
| backend/models | 4 | 400 | 4 | 4995 | 1.0000 | 1.0000 | 44.77 |
| backend/models_15cls_run1 | 15 | 1500 | 2 | 4995 | 1.0000 | 1.0000 | 148.37 |

## 2) Thong ke du lieu tong quan

- Tong so mau trong manifest: **4362**
- Tong so nhan (label) phan biet: **3315**
- Tong so signer: **4**
- So file landmark trong backend/landmarks: **4392**
- So file landmark holistic (__holistic.npy): **30**
- So file landmark dang .mp4.npy: **4362**

## 3) So lieu keypoint va feature

- Keypoint variant: **HOLISTIC**
- Tong raw keypoint features: **1662**
- Kich thuoc vector sau chuan hoa: **4995**

### Cau phan keypoint

| Thanh phan | So diem | So kenh | So feature |
|---|---:|---:|---:|
| Pose | 33 | 4 | 132 |
| Left Hand | 21 | 3 | 63 |
| Right Hand | 21 | 3 | 63 |
| Face | 468 | 3 | 1404 |


## 4) Ket qua benchmark de dua vao bao cao

### Continuous benchmark (real)

- So mau GT: **60**
- So mau duoc danh gia: **60**
- WER: **1.104167**
- CER: **1.103821**
- Exact match (text): **0.000000**

### Sentence-level benchmark (synthetic small)

- So mau GT: **10**
- So mau duoc danh gia: **10**
- WER: **0.933333**
- CER: **0.795604**
- Segment F1@0.5: **0.066667**

## 5) Danh sach hinh anh can chup cho bao cao

1. benchmark/continuous/data/charts/continuous_wer_cer_by_dialect.png
2. benchmark/continuous/data/charts/continuous_wer_cer_by_signer.png
3. benchmark/report_assets/charts/training_summary.png
4. benchmark/report_assets/charts/dataset_keypoint_summary.png

## 6) Nguon file so lieu

- backend/models/training_history.json
- backend/models_15cls_run1/training_history.json
- benchmark/sentence_level/data/full_dataset_manifest.json
- backend/keypoint_previews/15cls_comparison_fast/15_classes_metadata.json
- benchmark/continuous/data/continuous_eval.real.json
- benchmark/sentence_level/data/sentence_eval.synthetic.small.json
