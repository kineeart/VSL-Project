# Towards Robust Vietnamese Sign Language Recognition

## Full Project Blueprint

### Title
Towards Robust Vietnamese Sign Language Recognition: Sentence-Level Benchmark, Dialect Adaptation, and On-Device Deployment

## 1. Problem Statement

Vietnamese Sign Language (VSL) recognition systems often report high accuracy on curated datasets but degrade sharply in real-world conditions (webcam noise, signer variation, dialect variation, device constraints).

This project addresses three gaps simultaneously:

1. Lack of sentence-level benchmark protocol.
2. Lack of robust dialect adaptation strategy.
3. Lack of practical on-device deployment pipeline for mobile use.

## 2. Research Objectives

### O1. Build a reproducible sentence-level benchmark for VSL
- Standardized dataset schema.
- Split protocol preventing signer leakage.
- Metrics for sentence quality and boundary quality.

### O2. Improve robustness across dialects and domain shifts
- Cross-dialect generalization experiments.
- Few-shot adaptation setup.
- Robust inference behavior under real-world capture.

### O3. Deploy a lightweight inference pipeline on mobile
- Quantized model.
- Latency-aware pipeline.
- Usable mini app for live testing.

## 3. Research Questions

1. How much does performance drop from video-domain to real-time camera-domain?
2. Which adaptation method best transfers between VSL dialect groups?
3. What trade-off between accuracy and latency is acceptable for mobile inference?
4. Can a compact model keep robust top-k behavior under noisy conditions?

## 4. Scope and Assumptions

### In scope
- Sentence-level VSL recognition benchmark.
- Dialect-aware evaluation and adaptation.
- On-device prototype with measurable performance.

### Out of scope (for this phase)
- Full nationwide dialect coverage.
- Large-scale language model decoding.
- Production-grade app store release.

## 5. Dataset and Annotation Design

### 5.1 Data schema
Use the JSON schema already created in:
- benchmark/sentence_level/schemas/sentence_dataset.schema.json
- benchmark/sentence_level/schemas/sentence_predictions.schema.json

### 5.2 Required fields per sample
- sample_id
- video_path
- signer_id
- dialect
- split
- sentence_text
- sentence_gloss
- segments (start_ms, end_ms, gloss)

### 5.3 Collection protocol
- Capture from real camera setup intended for deployment.
- Record multiple dialect groups (north, central, south).
- Include diverse conditions: indoor, classroom, home, varied lighting.

### 5.4 Split protocol
Primary split policy:
- Group by signer (no signer leakage across train/val/test).

Secondary split policy for dialect adaptation:
- Leave-one-dialect-out setting.
- Few-shot target-dialect adaptation setting.

## 6. Benchmark Tasks

### Task A: Sentence-level text recognition
Input: sentence video.
Output: sentence text.
Metric: Exact match text, CER.

### Task B: Sentence-level gloss recognition
Input: sentence video.
Output: gloss sequence.
Metric: Exact match gloss, WER.

### Task C: Temporal boundary quality
Input: sentence video.
Output: per-gloss temporal segments.
Metric: Segment F1 at IoU threshold(s).

### Task D: Robustness under real-world shift
Input: webcam/live data.
Output: top-k predictions + rejection behavior.
Metric: top-1/top-5, calibration, false accept/reject.

## 7. Evaluation Protocol

### 7.1 Core metrics
- Exact Match (text).
- Exact Match (gloss).
- WER (gloss sequence).
- CER (sentence text).
- Segment F1@0.5.

### 7.2 Robustness metrics
- Cross-dialect transfer gap.
- In-domain vs out-of-domain drop.
- Confidence calibration (ECE optional).
- Rejection quality (false acceptance / false rejection).

### 7.3 Reporting format
Always report:
1. Overall metrics.
2. Per-dialect metrics.
3. Per-signer metrics.
4. Inference latency metrics (desktop and mobile).

## 8. Dialect Adaptation Plan

### Baseline
- Train on mixed dialect training set.
- Evaluate on held-out signers in each dialect.

### Adaptation experiments
1. Zero-shot: train on source dialects, test on unseen target dialect.
2. Few-shot fine-tune: add small labeled target subset.
3. Domain balancing: weighted sampling by dialect/signer.

### Success criteria
- Reduced performance gap between dialects.
- Stable top-k behavior on live capture.

## 9. On-Device Deployment Plan

### 9.1 Deployment target
- Android first (mid-range devices).
- Optional iOS later.

### 9.2 Model export path
Preferred path:
1. PyTorch checkpoint.
2. Export to ONNX.
3. Quantization (int8 where feasible).
4. Mobile runtime benchmark.

Alternative path:
- TFLite conversion and delegate benchmarking.

### 9.3 On-device KPIs
- End-to-end latency per inference window.
- FPS for live camera pipeline.
- Memory footprint.
- Battery usage trend (short session).

Target guideline (initial MVP):
- <= 200ms inference per window on mid-range phone.
- >= 10 FPS UI update.

## 10. Mini Mobile App (Post-Benchmark)

Yes, after benchmark/protocol is complete, developing a simple mobile app is feasible.

### MVP features
1. Camera preview.
2. Start/stop recognition.
3. Top-5 predictions panel.
4. Confidence bar.
5. Debug toggle (raw/smoothed top-k).

### UI style
- Keep simple and close to current start.bat web app flow:
  - Home: Start camera.
  - Live: video + predictions.
  - Settings: model profile, debug toggle.

### Suggested stack
Option A (fastest prototype):
- React Native + Expo
- Backend-assisted inference first, then on-device model later

Option B (full on-device earlier):
- Native Android (Kotlin) + ONNX Runtime Mobile or TFLite

## 11. Phase-by-Phase Execution

### Phase 1: Benchmark Foundation (now)
- Finalize schema and validation.
- Build split/evaluate scripts.
- Validate dataset quality and leakage constraints.

### Phase 2: Robust Baselines
- Train sentence-level baseline.
- Report overall + per-dialect metrics.
- Reproduce with fixed seeds.

### Phase 3: Dialect Adaptation
- Run zero-shot and few-shot adaptation.
- Compare adaptation strategies.

### Phase 4: On-Device Optimization
- Export and quantize model.
- Measure latency/accuracy trade-off.

### Phase 5: Mini Mobile App
- Implement MVP UI.
- Integrate model inference.
- Conduct user-facing demo and error analysis.

## 12. Deliverables Checklist

### Scientific deliverables
- Benchmark dataset spec and protocol.
- Reproducible scripts for split/validate/evaluate.
- Baseline and adaptation experiment tables.
- Error analysis and robustness discussion.

### Engineering deliverables
- Trained model variants (baseline + adapted + quantized).
- Inference pipeline for mobile.
- Mini demo app.

### Thesis/report deliverables
- Problem + related work.
- Method + benchmark protocol.
- Results + ablation + deployment metrics.
- Limitations + future work.

## 13. Report Structure Template (for later writing)

1. Introduction
2. Related Work
3. Benchmark Design
4. Methodology
5. Dialect Adaptation Experiments
6. On-Device Deployment
7. Discussion and Limitations
8. Conclusion

## 14. Immediate Next Actions

1. Curate real sentence dataset according to schema.
2. Run validate script on full dataset.
3. Run signer-grouped split.
4. Train first sentence-level baseline.
5. Generate first benchmark table (overall + per-dialect).
