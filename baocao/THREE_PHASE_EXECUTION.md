# Three-Phase Execution Guide (V2)

This guide formalizes the project into three practical phases.

## Phase 1 - Reduced-Class Classification Benchmark

Goal: stable, clean benchmark for reporting.

Run from repository root:

```bash
python backend/train_phase1_reduced.py --max-classes 40 --min-samples 3 --target-per-class 20 --variant holistic_no_face --seq 50 --epochs 120 --workers 1 --output-subdir phase1_reduced
```

Notes:
- Default variant is `holistic_no_face` to reduce RAM usage.
- The script filters classes by frequency and keeps only top-K classes.

## Phase 2 - Large-Scale Few-Shot / Embedding Benchmark

Goal: retain many classes in low-data settings and evaluate with cosine nearest-prototype.

```bash
python backend/phase2_fewshot_embedding.py --variant holistic_no_face --min-samples 1 --max-classes 0 --support-per-class 1 --seq 50 --workers 1 --report-path benchmark/reports/phase2_fewshot_embedding.json
```

Notes:
- `--max-classes 0` means keep all eligible classes.
- Output report includes top-1/top-5 and query/class counts.

## Phase 3 - Continuous/Sentence Audit and Sanity Checks

Goal: keep sequence-level benchmarks and detect pipeline problems early.

```bash
python benchmark/scripts/phase3_continuous_sentence_audit.py --continuous-min-test 100 --sentence-min-test 50 --out benchmark/reports/phase3_audit.json
```

Checks include:
- dataset test split size against target minimums
- eval artifact presence
- WER sanity (`WER > 1.0` warning)
- missing prediction counts

## Recommended Workflow

1. Run Phase 1 to obtain stable classification baseline results.
2. Run Phase 2 to produce few-shot/embedding results on larger class coverage.
3. Run Phase 3 audit before writing final conclusions.

This structure supports both practical delivery and stronger research contribution.
