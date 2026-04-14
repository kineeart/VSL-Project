# On-device Optimization Pipeline

Thu muc nay cung cap pipeline xuat va benchmark model cho muc tieu mobile/on-device.

## Files

- `export_mobile_model.py`: xuat artifacts (TorchScript FP32, TorchScript INT8, ONNX)
- `benchmark_mobile_artifacts.py`: benchmark latency tren CPU host

## 1) Export artifacts

```bash
python backend/ondevice/export_mobile_model.py \
  --model-dir backend/models_15cls_run1 \
  --checkpoint sign_model.pt \
  --output-dir backend/ondevice/artifacts
```

## 2) Benchmark artifacts

```bash
python backend/ondevice/benchmark_mobile_artifacts.py \
  --manifest backend/ondevice/artifacts/manifest.json \
  --runs 120 --warmup 20
```

Output:
- `backend/ondevice/artifacts/benchmark_report.json`

## 3) Suggested deployment order

1. Benchmark/protocol complete first.
2. Select best checkpoint (accuracy/robustness).
3. Export + quantize for on-device.
4. Compare latency/size vs accuracy drop.
5. Integrate best artifact into mobile app.
