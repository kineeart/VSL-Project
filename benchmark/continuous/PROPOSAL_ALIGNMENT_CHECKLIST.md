# Proposal Alignment Checklist (VSL Chuoi)

Tai lieu nay dung de doi chieu nhanh giua code hien tai va muc tieu de tai VSL chuoi.

## 1) Muc tieu va trang thai

- [x] Nhan dien ky hieu bang mo hinh hoc sau lai (CNN + BiLSTM + Attention)
- [x] Trich xuat dac trung keypoint khong gian-thoi gian bang MediaPipe
- [x] Co che nhan dien lien tuc theo stream (`/ws/continuous`) va giao dien tab Continuous
- [x] Co benchmark chuoi va metric CER/WER/F1 co ban
- [x] Tich hop Language Model (n-gram/LM) vao giai ma chuoi (baseline bigram rerank)
- [x] Ensemble nhieu mo hinh trong suy luan chuoi (baseline average prob)
- [ ] Tap du lieu VSL chuoi that (quay lien tuc) cho train/val/test
- [ ] Bao cao loi dinh tinh theo nhom loi (chen, xoa, nham ky hieu)

## 2) Bang chung trong repo

- Mo hinh va backend:
  - `backend/app.py`
- Benchmark sentence-level:
  - `benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py`
- Dataset/pipeline continuous:
  - `benchmark/continuous/scripts/detect_active_span.py`
  - `benchmark/continuous/scripts/build_continuous_dataset.py`
  - `benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py`
  - `benchmark/continuous/scripts/evaluate_continuous_benchmark.py`
  - `benchmark/continuous/schemas/continuous_dataset.schema.json`

## 3) Thu tu uu tien de dat muc tieu de tai

1. Thu thap va dong goi tap VSL chuoi that theo schema continuous.
2. Chay benchmark CER/WER tren tap test that, kem phan tich loi dinh tinh.
3. Chot bao cao so sanh: single model vs LM vs ensemble.
4. Nang cap LM (beam + higher-order n-gram/Transformer LM) neu can.
5. Nang cap ensemble (weighted blending/stacking) neu can.
