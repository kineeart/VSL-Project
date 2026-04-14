import argparse
import json
from collections import defaultdict
from pathlib import Path


def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def wer(ref_tokens, hyp_tokens):
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    return levenshtein_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def cer(ref_text, hyp_text):
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def segment_iou(seg_a, seg_b):
    a0, a1 = seg_a["start_ms"], seg_a["end_ms"]
    b0, b1 = seg_b["start_ms"], seg_b["end_ms"]
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def segment_f1(gt_segments, pred_segments, iou_threshold=0.5):
    matched_gt = set()
    matched_pred = set()

    for pi, p in enumerate(pred_segments):
        best_idx = None
        best_iou = 0.0
        for gi, g in enumerate(gt_segments):
            if gi in matched_gt:
                continue
            if g.get("gloss") != p.get("gloss"):
                continue
            iou = segment_iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi

        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(pred_segments) - len(matched_pred)
    fn = len(gt_segments) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_samples(gt_samples, pred_map):
    total = 0
    exact_text = 0
    exact_gloss = 0
    wer_list = []
    cer_list = []
    f1_list = []
    prec_list = []
    rec_list = []
    missing_preds = []

    for gt in gt_samples:
        sample_id = gt.get("sample_id")
        pred = pred_map.get(sample_id)
        if pred is None:
            missing_preds.append(sample_id)
            continue

        total += 1

        gt_text = gt.get("sentence_text", "")
        pred_text = pred.get("pred_sentence_text", "")
        if gt_text == pred_text:
            exact_text += 1

        gt_gloss = gt.get("sentence_gloss", [])
        pred_gloss = pred.get("pred_sentence_gloss", [])
        if gt_gloss == pred_gloss:
            exact_gloss += 1

        wer_list.append(wer(gt_gloss, pred_gloss))
        cer_list.append(cer(gt_text, pred_text))

        f1, p, r = segment_f1(gt.get("segments", []), pred.get("pred_segments", []), iou_threshold=0.5)
        f1_list.append(f1)
        prec_list.append(p)
        rec_list.append(r)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    result = {
        "num_gt_samples": len(gt_samples),
        "num_evaluated": total,
        "num_missing_predictions": len(missing_preds),
        "missing_prediction_ids": missing_preds,
        "exact_match_text": mean([1.0] * exact_text + [0.0] * (total - exact_text)) if total else 0.0,
        "exact_match_gloss": mean([1.0] * exact_gloss + [0.0] * (total - exact_gloss)) if total else 0.0,
        "wer": mean(wer_list),
        "cer": mean(cer_list),
        "segment_f1@0.5": mean(f1_list),
        "segment_precision@0.5": mean(prec_list),
        "segment_recall@0.5": mean(rec_list),
    }
    return result


def evaluate(gt_data, pred_data, split="test", breakdown=False):
    gt_samples = [s for s in gt_data.get("samples", []) if s.get("split") == split]
    pred_map = {p["sample_id"]: p for p in pred_data.get("predictions", [])}

    result = evaluate_samples(gt_samples, pred_map)
    result["split"] = split

    if not breakdown:
        return result

    by_dialect = defaultdict(list)
    by_signer = defaultdict(list)
    for s in gt_samples:
        by_dialect[s.get("dialect", "unknown")].append(s)
        by_signer[s.get("signer_id", "unknown")].append(s)

    result["by_dialect"] = {
        k: evaluate_samples(v, pred_map) for k, v in sorted(by_dialect.items(), key=lambda x: x[0])
    }
    result["by_signer"] = {
        k: evaluate_samples(v, pred_map) for k, v in sorted(by_signer.items(), key=lambda x: x[0])
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence-level VSL benchmark")
    parser.add_argument("--gt", required=True, help="Path to ground-truth dataset JSON")
    parser.add_argument("--pred", required=True, help="Path to predictions JSON")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    parser.add_argument("--breakdown", action="store_true", help="Include per-dialect and per-signer metrics")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)
    metrics = evaluate(gt_data, pred_data, split=args.split, breakdown=args.breakdown)

    print("=" * 72)
    print("Sentence-Level Benchmark Results")
    print("=" * 72)
    for k, v in metrics.items():
        if k in ("by_dialect", "by_signer"):
            continue
        if isinstance(v, float):
            print(f"{k:26s}: {v:.4f}")
        else:
            print(f"{k:26s}: {v}")

    if args.breakdown:
        print("\nBy Dialect")
        print("-" * 72)
        for k, v in metrics.get("by_dialect", {}).items():
            print(
                f"{k:12s} | n={v['num_gt_samples']:3d} eval={v['num_evaluated']:3d} "
                f"EM_gloss={v['exact_match_gloss']:.3f} WER={v['wer']:.3f} F1={v['segment_f1@0.5']:.3f}"
            )

        print("\nBy Signer")
        print("-" * 72)
        for k, v in metrics.get("by_signer", {}).items():
            print(
                f"{k:12s} | n={v['num_gt_samples']:3d} eval={v['num_evaluated']:3d} "
                f"EM_gloss={v['exact_match_gloss']:.3f} WER={v['wer']:.3f} F1={v['segment_f1@0.5']:.3f}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
