"""Phase 3: Continuous/Sentence benchmark health audit.

Checks:
- dataset split sizes against target minimums
- eval json presence
- WER/CER sanity (warn when WER > 1.0)
- missing prediction counts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_existing(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def count_split_samples(dataset_obj, split: str):
    samples = dataset_obj.get("samples", [])
    return sum(1 for s in samples if s.get("split") == split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit phase 3 continuous/sentence pipeline health")
    parser.add_argument("--continuous-min-test", type=int, default=100)
    parser.add_argument("--sentence-min-test", type=int, default=50)
    parser.add_argument(
        "--out",
        default="benchmark/reports/phase3_audit.json",
        help="Output report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent.parent

    continuous_dataset = find_first_existing([
        repo_root / "benchmark" / "continuous" / "data" / "continuous_dataset.real.json",
    ])
    continuous_eval = find_first_existing([
        repo_root / "benchmark" / "continuous" / "data" / "continuous_eval.real.json",
    ])

    sentence_dataset = find_first_existing([
        repo_root / "benchmark" / "sentence_level" / "data" / "sentence_dataset.synthetic.small.split.json",
        repo_root / "benchmark" / "sentence_level" / "data" / "sentence_dataset.split.sample.json",
    ])
    sentence_eval = find_first_existing([
        repo_root / "benchmark" / "sentence_level" / "data" / "sentence_eval.synthetic.small.json",
    ])

    report = {
        "phase": "phase3_continuous_sentence_audit",
        "targets": {
            "continuous_min_test": args.continuous_min_test,
            "sentence_min_test": args.sentence_min_test,
        },
        "checks": {},
        "warnings": [],
        "actions": [],
    }

    # Continuous checks
    cont_info = {
        "dataset_path": str(continuous_dataset) if continuous_dataset else None,
        "eval_path": str(continuous_eval) if continuous_eval else None,
    }
    if continuous_dataset:
        ds = load_json(continuous_dataset)
        n_test = count_split_samples(ds, "test")
        cont_info["test_samples"] = n_test
        cont_info["target_ok"] = n_test >= args.continuous_min_test
        if not cont_info["target_ok"]:
            report["warnings"].append(
                f"Continuous test set too small: {n_test} < {args.continuous_min_test}"
            )
            report["actions"].append("Increase continuous test set or rebuild split before final report")
    else:
        report["warnings"].append("Continuous dataset JSON not found")

    if continuous_eval:
        ev = load_json(continuous_eval)
        cont_info["wer"] = ev.get("wer")
        cont_info["cer"] = ev.get("cer")
        cont_info["missing_predictions"] = ev.get("num_missing_predictions")
        if isinstance(cont_info.get("wer"), (float, int)) and cont_info["wer"] > 1.0:
            report["warnings"].append(
                f"Continuous WER > 1.0 ({cont_info['wer']:.3f}) suggests decode/normalization mismatch"
            )
            report["actions"].append("Check token normalization, decode path, and prediction alignment")
    else:
        report["warnings"].append("Continuous eval JSON not found")

    report["checks"]["continuous"] = cont_info

    # Sentence checks
    sent_info = {
        "dataset_path": str(sentence_dataset) if sentence_dataset else None,
        "eval_path": str(sentence_eval) if sentence_eval else None,
    }
    if sentence_dataset:
        ds = load_json(sentence_dataset)
        n_test = count_split_samples(ds, "test")
        sent_info["test_samples"] = n_test
        sent_info["target_ok"] = n_test >= args.sentence_min_test
        if not sent_info["target_ok"]:
            report["warnings"].append(
                f"Sentence test set too small: {n_test} < {args.sentence_min_test}"
            )
            report["actions"].append("Increase sentence test set or rebuild split before final report")
    else:
        report["warnings"].append("Sentence dataset JSON not found")

    if sentence_eval:
        ev = load_json(sentence_eval)
        sent_info["wer"] = ev.get("wer")
        sent_info["cer"] = ev.get("cer")
        sent_info["segment_f1@0.5"] = ev.get("segment_f1@0.5")
        sent_info["missing_predictions"] = ev.get("num_missing_predictions")
        if isinstance(sent_info.get("wer"), (float, int)) and sent_info["wer"] > 1.0:
            report["warnings"].append(
                f"Sentence WER > 1.0 ({sent_info['wer']:.3f}) suggests decode/normalization mismatch"
            )
            report["actions"].append("Check segmentation, tokenization, and sentence decode consistency")
    else:
        report["warnings"].append("Sentence eval JSON not found")

    report["checks"]["sentence"] = sent_info

    if not report["warnings"]:
        report["actions"].append("Phase 3 metrics look sane for current artifacts")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("PHASE 3 AUDIT")
    print("=" * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
