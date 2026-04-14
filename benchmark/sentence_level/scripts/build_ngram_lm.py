import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build simple unigram/bigram LM from sentence dataset")
    parser.add_argument("--dataset", required=True, help="Path to sentence dataset JSON")
    parser.add_argument("--out", required=True, help="Output LM JSON path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], help="Use which split")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Add-k smoothing constant")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    samples = dataset.get("samples", [])
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]

    unigram = Counter()
    bigram = defaultdict(Counter)

    for sample in samples:
        gloss = [str(x) for x in sample.get("sentence_gloss", []) if str(x).strip()]
        if not gloss:
            continue
        seq = ["<s>"] + gloss + ["</s>"]
        for tok in seq[1:]:
            unigram[tok] += 1
        for i in range(len(seq) - 1):
            bigram[seq[i]][seq[i + 1]] += 1

    payload = {
        "version": "1.0",
        "dataset": args.dataset,
        "split": args.split,
        "smoothing": float(args.smoothing),
        "num_samples": len(samples),
        "vocab_size": len(unigram),
        "unigram_counts": dict(unigram),
        "bigram_counts": {k: dict(v) for k, v in bigram.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("N-gram LM built")
    print("=" * 72)
    print(f"output      : {out}")
    print(f"num_samples : {payload['num_samples']}")
    print(f"vocab_size  : {payload['vocab_size']}")


if __name__ == "__main__":
    main()
