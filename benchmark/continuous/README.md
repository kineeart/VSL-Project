# Continuous Recognition Benchmark

## Mục đích

Khu vực này dùng để chuẩn bị dữ liệu và pipeline cho chế độ nhận diện liên tục:
- cắt tự động phần diễn tả trong từng video rời
- ghép nhiều clip đã cắt thành chuỗi mới
- đánh giá đầu ra theo chuỗi bằng CER/WER

## Cấu trúc dữ liệu

- `schemas/continuous_dataset.schema.json`: format dataset chuỗi mới
- `scripts/detect_active_span.py`: tìm khoảng thời gian có ký hiệu trong video
- `scripts/build_continuous_dataset.py`: tạo dataset chuỗi từ dữ liệu rời
- `scripts/generate_continuous_predictions_from_backend.py`: gọi backend `/api/predict/continuous_video` để sinh dự đoán chuỗi
- `scripts/evaluate_continuous_benchmark.py`: tính Exact-Match, CER, WER cho continuous dataset
- `run_pipeline_real.bat`: chạy nhanh pipeline đánh giá trên dữ liệu chuỗi thật

## Cách dùng nhanh

### 1) Tìm phần diễn tả trong một video

```bash
python benchmark/continuous/scripts/detect_active_span.py \
  --input Videos/D0001B.mp4 \
  --output benchmark/continuous/data/D0001B.active_span.json
```

### 2) Tạo dataset chuỗi mẫu

```bash
python benchmark/continuous/scripts/build_continuous_dataset.py \
  --data-xlsx Data.xlsx \
  --videos-dir Videos \
  --output-json benchmark/continuous/data/continuous_dataset.synthetic.json \
  --num-samples 200 \
  --min-units 2 --max-units 4 \
  --seed 42
```

### 3) Nếu muốn xuất video chuỗi đã ghép

Thêm `--render-videos` vào lệnh trên.

### 4) Sinh dự đoán chuỗi từ backend

```bash
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py \
  --dataset benchmark/continuous/data/continuous_dataset.real.json \
  --output benchmark/continuous/data/continuous_predictions.real.json \
  --backend http://127.0.0.1:8000 \
  --split test --stride 3
```

### 5) Đánh giá CER/WER cho continuous dataset

```bash
python benchmark/continuous/scripts/evaluate_continuous_benchmark.py \
  --gt benchmark/continuous/data/continuous_dataset.real.json \
  --pred benchmark/continuous/data/continuous_predictions.real.json \
  --split test --breakdown \
  --out benchmark/continuous/data/continuous_eval.real.json
```

### 6) Chạy pipeline nhanh bằng batch script

```bash
benchmark\continuous\run_pipeline_real.bat benchmark\continuous\data\continuous_dataset.real.json http://127.0.0.1:8000
```

## Format dataset

Mỗi sample gồm:
- `sample_id`
- `split`
- `video_path`
- `sentence_text`
- `sentence_gloss`
- `source_segments`

`source_segments` lưu lại từng đoạn đã cắt từ video rời, gồm:
- `source_video`
- `label`
- `trim_start_ms`
- `trim_end_ms`
- `segment_index`

## Ghi chú

- Nếu chưa có dataset chuỗi thật, bộ dữ liệu sinh từ clip rời chỉ nên dùng để thử pipeline.
- Khi có video chuỗi thật, chỉ cần giữ format JSON và thay nguồn sinh dữ liệu.
- Backend đã hỗ trợ endpoint `POST /api/predict/continuous_video` để suy luận cả video chuỗi và trả về `pred_sentence_gloss`/`pred_sentence_text`.