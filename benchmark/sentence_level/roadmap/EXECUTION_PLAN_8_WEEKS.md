# 8-Week Execution Plan for Robust VSL Project

## Week 1-2: Dataset and Protocol

1. Build sentence dataset manifest using schema.
2. Run validation and fix all error-level issues.
3. Define signer-safe train/val/test split.
4. Freeze benchmark version v1.0.

Output:
- validated dataset JSON
- split report
- benchmark protocol note

## Week 3-4: Sentence Baseline

1. Train baseline sentence recognizer.
2. Evaluate with exact match, WER, CER, segment F1.
3. Produce per-dialect and per-signer tables.
4. Perform failure case analysis.

Output:
- baseline checkpoint
- metrics table v1
- confusion/failure samples

## Week 5-6: Dialect Adaptation

1. Zero-shot cross-dialect evaluation.
2. Few-shot adaptation experiments.
3. Compare adaptation strategies and report gains.

Output:
- adapted checkpoints
- transfer-gap analysis
- robustness plots

## Week 7: On-Device Optimization

1. Export baseline and adapted models.
2. Quantization and latency profiling.
3. Select best model variant by accuracy-latency trade-off.

Output:
- mobile-ready model artifact
- latency/size report

## Week 8: Mobile Mini App + Final Packaging

1. Implement simple mobile demo UI.
2. Integrate inference and top-k display.
3. Demo scripts and screenshots.
4. Prepare report outline with filled figures/tables.

Output:
- mobile MVP demo
- final benchmark bundle
- report-ready assets

## Gate Criteria

Proceed to next stage only if:
1. Validation errors are zero.
2. Split leakage is controlled (signer leakage none by policy).
3. Baseline metrics are reproducible with fixed seed.
4. Mobile latency target is met for selected model.
