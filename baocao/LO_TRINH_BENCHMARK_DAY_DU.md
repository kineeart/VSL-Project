# Lộ Trình Benchmark Đầy Đủ Cho Dự Án VSL

**Mục tiêu:** xây dựng và chạy benchmark một cách đầy đủ, có kiểm chứng, có so sánh baseline, có phân tích lỗi và có kết luận nghiên cứu đủ mạnh để báo cáo.  
**Tinh thần thực hiện:** không làm gấp, không chốt kết quả sớm, ưu tiên độ tin cậy hơn tốc độ.  
**Thời lượng khuyến nghị:** 1-2 tuần nếu làm nghiêm túc; 1 ngày chỉ đủ cho bản kiểm thử nhanh.

---

## 1. Nguyên tắc thực hiện

1. Không dùng validation/test quá nhỏ để kết luận thành công.
2. Không dùng riêng training accuracy để chứng minh mô hình tốt.
3. Chỉ công nhận benchmark khi có:
   - dataset rõ ràng,
   - split đúng,
   - baseline so sánh,
   - kết quả ổn định,
   - phân tích lỗi,
   - ghi nhận giới hạn.
4. Nếu có mâu thuẫn giữa training và inference, phải debug xong trước khi viết báo cáo.

---

## 2. Đầu ra cuối cùng cần có

### 2.1. Bộ dữ liệu benchmark chuẩn

- Continuous benchmark dataset
- Sentence-level benchmark dataset
- Split train/val/test không rò rỉ signer
- Có thống kê số mẫu theo signer, dialect, class, độ dài chuỗi

### 2.2. Bộ kết quả benchmark

- WER
- CER
- Exact Match
- Segment F1@0.5
- Top-k accuracy nếu áp dụng
- Kết quả theo signer
- Kết quả theo dialect
- Kết quả theo độ dài chuỗi

### 2.3. Bộ so sánh mô hình

- Mô hình chính
- Baseline đơn giản
- Bản có language model
- Bản có TTA / smoothing nếu có
- Kết quả before/after cho các cải tiến chính

### 2.4. Phân tích chất lượng nghiên cứu

- Có thành công hay không
- Thành công ở mức nào
- Điểm nào còn yếu
- Có đủ điều kiện công nhận là nghiên cứu thành công không

---

## 3. Pha 1: Kiểm tra dữ liệu đầu vào

### Mục tiêu

Đảm bảo dữ liệu benchmark là hợp lệ trước khi train hoặc evaluate.

### Việc cần làm

1. Kiểm tra manifest gốc.
2. Kiểm tra số video, số label, số signer.
3. Kiểm tra file landmark có đủ và không hỏng.
4. Kiểm tra dataset continuous và sentence-level có đúng schema.
5. Kiểm tra split train/val/test có leakage hay không.

### Lệnh gợi ý

```bash
python benchmark/sentence_level/scripts/validate_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.sample.json \
  --out benchmark/sentence_level/data/sentence_dataset.validation.json
```

```bash
python benchmark/sentence_level/scripts/split_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.sample.json \
  --output benchmark/sentence_level/data/sentence_dataset.split.json \
  --report benchmark/sentence_level/data/sentence_split_report.json \
  --group-by signer \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### Tiêu chí đạt

- Không có lỗi schema.
- Không có sample_id trùng.
- Không có signer leakage giữa train/val/test.
- Split đủ lớn để đánh giá.

---

## 4. Pha 2: Chuẩn hóa split và kích thước đánh giá

### Mục tiêu

Loại bỏ tình trạng validation/test quá nhỏ.

### Việc cần làm

1. Tạo validation đủ lớn cho training.
2. Tạo test set đủ lớn cho benchmark.
3. Chia theo signer nếu mục tiêu là tổng quát hóa thực tế.
4. Nếu cần, chia thêm theo dialect để đánh giá độ bền mô hình.

### Khuyến nghị tối thiểu

- Validation: ít nhất vài trăm mẫu.
- Test: ít nhất 100-200 mẫu cho mỗi benchmark quan trọng.
- Sentence-level: nên có đủ mẫu ở nhiều độ dài chuỗi.
- Continuous: nên có đủ mẫu theo từng độ dài và từng signer.

### Tiêu chí đạt

- Không còn split 4 mẫu hoặc 2 mẫu để kết luận.
- Có thống kê rõ ràng cho từng split.
- Có thể báo cáo độ tin cậy của metric.

---

## 5. Pha 3: Kiểm tra và tái huấn luyện mô hình

### Mục tiêu

Loại bỏ mâu thuẫn giữa training accuracy và benchmark inference.

### Việc cần làm

1. Xác nhận lại preprocessing dùng trong train và inference có giống nhau.
2. Kiểm tra normalization mean/std.
3. Kiểm tra output format của mô hình.
4. Kiểm tra decoding logic.
5. Tái huấn luyện với split chuẩn.

### Cần kiểm tra kỹ

- Có dùng đúng landmark format không.
- Có dùng đúng feature size không.
- Có padding/truncation nhất quán không.
- Có khác biệt giữa train-time augment và inference-time input không.
- Có nhầm giữa class index và class name không.

### Kết quả mong đợi

- Training accuracy và validation accuracy có khoảng cách hợp lý.
- Inference benchmark không còn lệch quá xa so với validation.
- Nếu vẫn lệch, phải ghi rõ nguyên nhân kỹ thuật.

---

## 6. Pha 4: Chạy benchmark continuous

### Mục tiêu

Đánh giá khả năng nhận diện chuỗi liên tục trên dữ liệu thực.

### Pipeline

1. Detect active span.
2. Build continuous dataset.
3. Generate predictions từ backend.
4. Evaluate CER/WER/Exact Match.
5. Breakdown theo signer hoặc dialect nếu có.

### Lệnh gợi ý

```bash
python benchmark/continuous/scripts/detect_active_span.py \
  --input Videos/D0001B.mp4 \
  --output benchmark/continuous/data/D0001B.active_span.json
```

```bash
python benchmark/continuous/scripts/build_continuous_dataset.py \
  --data-xlsx Data.xlsx \
  --videos-dir Videos \
  --output-json benchmark/continuous/data/continuous_dataset.real.json \
  --num-samples 200 \
  --min-units 2 --max-units 4 \
  --seed 42
```

```bash
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py \
  --dataset benchmark/continuous/data/continuous_dataset.real.json \
  --output benchmark/continuous/data/continuous_predictions.real.json \
  --backend http://127.0.0.1:8000 \
  --split test --stride 3
```

```bash
python benchmark/continuous/scripts/evaluate_continuous_benchmark.py \
  --gt benchmark/continuous/data/continuous_dataset.real.json \
  --pred benchmark/continuous/data/continuous_predictions.real.json \
  --split test --breakdown \
  --out benchmark/continuous/data/continuous_eval.real.json
```

### Tiêu chí đạt

- Có kết quả WER/CER ổn định.
- Có breakdown để hiểu lỗi đến từ đâu.
- Có thể so sánh với phiên bản cũ.

---

## 7. Pha 5: Chạy benchmark sentence-level

### Mục tiêu

Đánh giá mô hình trên bài toán câu hoàn chỉnh và boundary detection.

### Pipeline

1. Tạo dataset sentence-level.
2. Validate dataset.
3. Split theo signer/dialect.
4. Generate predictions từ backend.
5. Evaluate WER, CER, F1@0.5, exact match.

### Lệnh gợi ý

```bash
python benchmark/sentence_level/scripts/generate_sentence_dataset_from_dataxlsx.py \
  --data-xlsx Data.xlsx \
  --videos-dir Videos \
  --output benchmark/sentence_level/data/sentence_dataset.real.json \
  --num-samples 100
```

```bash
python benchmark/sentence_level/scripts/validate_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.real.json \
  --out benchmark/sentence_level/data/sentence_dataset.validation.json
```

```bash
python benchmark/sentence_level/scripts/split_sentence_dataset.py \
  --input benchmark/sentence_level/data/sentence_dataset.real.json \
  --output benchmark/sentence_level/data/sentence_dataset.split.json \
  --report benchmark/sentence_level/data/sentence_split_report.json \
  --group-by signer \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

```bash
python benchmark/sentence_level/scripts/generate_sentence_predictions_from_backend.py \
  --dataset benchmark/sentence_level/data/sentence_dataset.split.json \
  --output benchmark/sentence_level/data/sentence_predictions.json \
  --backend http://127.0.0.1:8000 \
  --split test
```

```bash
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py \
  --gt benchmark/sentence_level/data/sentence_dataset.split.json \
  --pred benchmark/sentence_level/data/sentence_predictions.json \
  --split test --breakdown
```

### Tiêu chí đạt

- Có WER/CER/F1@0.5 hợp lệ.
- Có thể đánh giá theo signer và dialect.
- Có thể phát hiện lỗi về segment boundary.

---

## 8. Pha 6: Thêm baseline để so sánh

### Mục tiêu

Chứng minh mô hình chính có tốt hơn phương án đơn giản hay không.

### Baseline nên có

1. Baseline VGG-style hoặc CNN đơn giản.
2. Baseline LSTM đơn giản.
3. Baseline có và không có language model.
4. Baseline không có TTA/smoothing.

### Tiêu chí đạt

- Mô hình chính phải tốt hơn baseline đơn giản.
- Nếu không tốt hơn, phải nói rõ trong báo cáo.
- Có bảng so sánh công bằng cùng dataset, cùng split.

---

## 9. Pha 7: Phân tích lỗi

### Mục tiêu

Biết mô hình sai ở đâu và vì sao sai.

### Việc cần làm

1. Tạo confusion matrix.
2. Tìm các mẫu fail điển hình.
3. Phân tích theo signer.
4. Phân tích theo dialect.
5. Phân tích theo độ dài chuỗi.
6. Xem top-5 prediction có chứa ground truth hay không.

### Tiêu chí đạt

- Không chỉ có WER/CER.
- Có mô tả nguyên nhân lỗi.
- Có khuyến nghị cải tiến cụ thể.

---

## 10. Pha 8: Kiểm thử on-device

### Mục tiêu

Xem mô hình có đủ thực tế để chạy trên thiết bị di động hay không.

### Việc cần làm

1. Quantize model nếu cần.
2. Đo latency.
3. Đo độ giảm độ chính xác sau tối ưu.
4. Kiểm tra bộ nhớ và thời gian phản hồi.
5. Ghi rõ model nào dùng được cho mobile.

### Tiêu chí đạt

- Có số liệu latency.
- Có số liệu accuracy drop.
- Có kết luận rõ: chạy được / chưa chạy được / cần tối ưu thêm.

---

## 11. Pha 9: Tổng hợp và chốt kết luận nghiên cứu

### Mục tiêu

Không chốt sớm khi chưa đủ bằng chứng.

### Cần trả lời các câu hỏi sau

1. Benchmark có đủ lớn để tin cậy chưa?
2. Kết quả inference có ổn định không?
3. Mô hình có tốt hơn baseline không?
4. Có dấu hiệu overfitting không?
5. Có phân tích lỗi đủ sâu không?
6. Có thể công nhận là nghiên cứu thành công không?

### Một nghiên cứu được xem là mạnh khi

- Dữ liệu và split hợp lệ.
- Kết quả benchmark nhất quán.
- Có baseline so sánh.
- Có phân tích lỗi.
- Có giới hạn và hướng phát triển tiếp theo.

---

## 12. Tiêu chí công nhận thành công

### Mức 1: Thành công kỹ thuật

- Pipeline chạy được.
- Dữ liệu hợp lệ.
- Có benchmark.
- Có log và kết quả lặp lại được.

### Mức 2: Thành công nghiên cứu

- Có cải thiện so với baseline.
- Kết quả có ý nghĩa hơn random hoặc baseline yếu.
- Có phân tích lỗi và đánh giá thực nghiệm.

### Mức 3: Thành công ứng dụng

- Có thể demo ổn định.
- Có thể chạy trên web hoặc mobile.
- Latency và độ chính xác đủ chấp nhận.

---

## 13. Kết luận thực dụng

Nếu làm đầy đủ các bước ở trên, dự án sẽ không còn là “benchmark cho có” mà trở thành một bộ đánh giá có giá trị nghiên cứu thật sự.

Nếu bỏ qua các bước quan trọng như split đúng, baseline, error analysis và on-device evaluation thì kết luận cuối cùng sẽ yếu, dù mô hình có thể chạy được.

**Vì vậy, hướng đúng là:** làm đầy đủ, kiểm tra kỹ, rồi mới kết luận.

---

## 14. Thứ tự thực hiện khuyến nghị

1. Chuẩn hóa dataset và split.
2. Tái huấn luyện mô hình với split đúng.
3. Chạy benchmark continuous.
4. Chạy benchmark sentence-level.
5. Thêm baseline.
6. Phân tích lỗi.
7. Đo on-device.
8. Viết kết luận cuối.

---

**Trạng thái tài liệu:** dùng làm kế hoạch triển khai benchmark đầy đủ, không phải bản rút gọn.
