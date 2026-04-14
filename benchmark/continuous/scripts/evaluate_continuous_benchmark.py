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
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
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


def evaluate(gt_samples, pred_map):
    total = 0
    exact_text = 0
    exact_gloss = 0
    wer_list = []
    cer_list = []
    missing = []

    for gt in gt_samples:
        sid = gt.get("sample_id")
        pred = pred_map.get(sid)
        if pred is None:
            missing.append(sid)
            continue

        total += 1
        gt_text = gt.get("sentence_text", "")
        gt_gloss = gt.get("sentence_gloss", [])
        pred_text = pred.get("pred_sentence_text", "")
        pred_gloss = pred.get("pred_sentence_gloss", [])

        if gt_text == pred_text:
            exact_text += 1
        if gt_gloss == pred_gloss:
            exact_gloss += 1

        wer_list.append(wer(gt_gloss, pred_gloss))
        cer_list.append(cer(gt_text, pred_text))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "num_gt_samples": len(gt_samples),
        "num_evaluated": total,
        "num_missing_predictions": len(missing),
        "missing_prediction_ids": missing,
        "exact_match_text": (exact_text / total) if total else 0.0,
        "exact_match_gloss": (exact_gloss / total) if total else 0.0,
        "wer": mean(wer_list),
        "cer": mean(cer_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate continuous benchmark with EM/WER/CER")
    parser.add_argument("--gt", required=True, help="Ground-truth continuous dataset JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--out", default="", help="Optional output JSON")
    parser.add_argument("--breakdown", action="store_true", help="Include per-dialect and per-signer")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred not found: {pred_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_samples = [s for s in gt_data.get("samples", []) if s.get("split") == args.split]
    pred_map = {p.get("sample_id"): p for p in pred_data.get("predictions", [])}

    result = evaluate(gt_samples, pred_map)
    result["split"] = args.split

    if args.breakdown:
        by_dialect = defaultdict(list)
        by_signer = defaultdict(list)
        for s in gt_samples:
            by_dialect[s.get("dialect", "unknown")].append(s)
            by_signer[s.get("signer_id", "unknown")].append(s)
        result["by_dialect"] = {k: evaluate(v, pred_map) for k, v in sorted(by_dialect.items())}
        result["by_signer"] = {k: evaluate(v, pred_map) for k, v in sorted(by_signer.items())}

    print("=" * 72)
    print("Continuous Benchmark Results")
    print("=" * 72)
    for key, value in result.items():
        if key in ("by_dialect", "by_signer"):
            continue
        if isinstance(value, float):
            print(f"{key:26s}: {value:.4f}")
        else:
            print(f"{key:26s}: {value}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
