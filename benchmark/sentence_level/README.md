# Sentence-Level Benchmark (VSL)

Khung benchmark này dùng cho chế độ nhận diện chuỗi:
- sentence-level recognition
- dialect adaptation
- robust evaluation trên dữ liệu gần với triển khai thực tế

Tài liệu định hướng đầy đủ:
- benchmark/sentence_level/roadmap/EXECUTION_PLAN_8_WEEKS.md

## 1) Cấu trúc thư mục

```
benchmark/sentence_level/
  README.md
  schemas/
    sentence_dataset.schema.json
    sentence_predictions.schema.json
  data/
    sentence_dataset.sample.json
    sentence_predictions.sample.json
  scripts/
    evaluate_sentence_benchmark.py
    split_sentence_dataset.py
    validate_sentence_dataset.py
    generate_sentence_dataset_from_dataxlsx.py
    generate_sentence_predictions_from_backend.py
```

## 2) Schema dữ liệu chuẩn

- Ground truth dataset schema: `schemas/sentence_dataset.schema.json`
- Prediction schema: `schemas/sentence_predictions.schema.json`

Dữ liệu gốc gồm:
- metadata mẫu (`sample_id`, `video_path`, `signer_id`, `dialect`)
- câu ở mức text (`sentence_text`)
- câu ở mức gloss (`sentence_gloss`)
- segment thời gian từng gloss (`segments`)
- split (`train` / `val` / `test`)

## 3) Cách chia tập (khuyến nghị)

- Không trùng sample giữa train/val/test.
- Nên chia theo người (`signer_id`) để đánh giá tổng quát tốt hơn.
- Mỗi dialect có đủ train/val/test.

Ví dụ:
- Train: signer A/B/C
- Val: signer D
- Test: signer E/F

## 4) Script đánh giá

Script: `scripts/evaluate_sentence_benchmark.py`

### Input
- `--gt`: file ground truth JSON theo schema dataset
- `--pred`: file dự đoán JSON theo schema predictions
- `--split`: split cần đánh giá (`test` mặc định)

### Metrics
- Exact Match (text)
- Exact Match (gloss)
- WER (Word Error Rate, theo gloss token)
- CER (Character Error Rate, theo text)
- Segment F1 (IoU-based, cho boundary)

### Chạy thử nhanh

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
  --gt benchmark/sentence_level/data/sentence_dataset.sample.json \
  --pred benchmark/sentence_level/data/sentence_predictions.sample.json \
  --split test
```

Danh gia chi tiet theo dialect/signer:

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
  --gt benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
  --pred benchmark/sentence_level/data/sentence_predictions.sample.json \
  --split test \
  --breakdown
```

## 5) Script split tự động theo signer/dialect

Script: `scripts/split_sentence_dataset.py`

### Mục tiêu
- Gán tự động `split` = train/val/test.
- Hỗ trợ tránh rò rỉ theo signer (mặc định `--group-by signer`).
- Có seed để tái lập kết quả.

### Chạy ví dụ

```bash
python benchmark/sentence_level/scripts/split_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.sample.json \
  --output benchmark/sentence_level/data/sentence_dataset.split.sample.json \
  --report benchmark/sentence_level/data/sentence_split_report.sample.json \
  --group-by signer \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### Tùy chọn `--group-by`
- `signer` (khuyến nghị cho generalization)
- `dialect`
- `signer_dialect`
- `sample` (split ngẫu nhiên theo sample, không chống leakage theo signer)

## 6) Script validate trước khi split/evaluate

Script: `scripts/validate_sentence_dataset.py`

### Mục tiêu
- Kiểm tra schema-like fields bắt buộc.
- Phát hiện lỗi dữ liệu: `sample_id` trùng, segment time sai, split sai.
- Kiểm tra độ phủ và leakage:
  - signer leakage giữa train/val/test
  - độ phủ dialect theo từng split
  - cảnh báo sequence quá dài/ngắn cho on-device latency

### Chạy ví dụ

```bash
python benchmark/sentence_level/scripts/validate_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.sample.json \
  --out benchmark/sentence_level/data/sentence_dataset.validation.sample.json
```

### Exit code
- `0`: không có error (có thể có warning)
- `2`: có error
- `1`: có warning khi chạy với `--strict`

### Quy trình đề xuất cho đề tài

1. `validate_sentence_dataset.py`
2. `split_sentence_dataset.py`
3. `evaluate_sentence_benchmark.py`
4. Báo cáo metrics tổng + theo dialect + theo signer

## 7) Tao sentence dataset tu Data.xlsx hien co

Neu chua the quay them du lieu, ban co the tao bo sentence-level synthetic bang cach noi cac clip tu don.

Script: `scripts/generate_sentence_dataset_from_dataxlsx.py`

Vi du (nhanh, khong render video):

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
```

Neu muon tao file video da noi clip (de phuc vu demo/e2e): them `--render-videos`.

Lưu ý:
- Synthetic sentence la giai phap tam thoi khi thieu dataset sentence that.
- Van nen bo sung mot tap test nho quay tu camera that de danh gia robust thuc te.
- `--split-policy signer` giup giam leakage theo signer.
- Script se gan `dialect` theo nhom signer (B/N/T va pseudo signer P0..Pk) de co breakdown co y nghia.

## 8) Tao prediction tu backend (khong dung sample prediction gia)

Script: `scripts/generate_sentence_predictions_from_backend.py`

Script nay goi truc tiep endpoint backend `/api/predict/video` de tao prediction JSON cho tap sentence.

```bash
python benchmark/sentence_level/scripts/generate_sentence_predictions_from_backend.py \
  --dataset benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
  --output benchmark/sentence_level/data/sentence_predictions.synthetic.small.json \
  --backend http://127.0.0.1:8000 \
  --split test
```

Sau do evaluate:

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
  --gt benchmark/sentence_level/data/sentence_dataset.synthetic.small.json \
  --pred benchmark/sentence_level/data/sentence_predictions.synthetic.small.json \
  --split test --breakdown
```

## 9) Workflow khuyến nghị

1. Thu dữ liệu câu thực tế (camera/môi trường đúng deployment).
2. Chuẩn hóa thành JSON theo schema dataset.
3. Chạy model sinh dự đoán theo schema predictions.
4. Đánh giá bằng script để lấy metrics chuẩn.
5. So sánh theo dialect (`dialect`) và theo signer (`signer_id`).

## 10) Lưu ý

- Đây là benchmark scaffold. Bạn có thể mở rộng thêm:
  - latency/FPS metrics
  - confidence calibration metrics
  - per-dialect confusion report

## 11) Chay toan bo pipeline (1 lenh)

Neu backend da chay (mac dinh `http://127.0.0.1:8000`), ban co the chay full pipeline:

```bash
benchmark\sentence_level\run_pipeline.bat
```

Tu dong thuc hien:
1. Validate dataset
2. Split signer-safe
3. Generate predictions tu backend
4. Evaluate co breakdown theo dialect/signer
5. Quality gate toi thieu (`num_evaluated > 0` va `missing_predictions = 0`)

Co the truyen tham so:

```bash
benchmark\sentence_level\run_pipeline.bat <dataset_json> <backend_url>
```

Vi du:

```bash
benchmark\sentence_level\run_pipeline.bat benchmark\sentence_level\data\sentence_dataset.synthetic.small.json http://127.0.0.1:8000
```
