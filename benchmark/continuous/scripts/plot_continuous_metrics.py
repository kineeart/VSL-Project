import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_grouped_bar(categories, wer_vals, cer_vals, title, out_path: Path):
    x = list(range(len(categories)))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], wer_vals, width=width, label="WER")
    plt.bar([i + width / 2 for i in x], cer_vals, width=width, label="CER")

    plt.xticks(x, categories, rotation=20)
    plt.ylabel("Error rate")
    plt.title(title)
    plt.ylim(0, max(1.4, max(wer_vals + cer_vals) * 1.1 if (wer_vals or cer_vals) else 1.0))
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def extract_group_metrics(group_obj):
    categories = []
    wer_vals = []
    cer_vals = []
    for key in sorted(group_obj.keys()):
        categories.append(key)
        wer_vals.append(float(group_obj[key].get("wer", 0.0)))
        cer_vals.append(float(group_obj[key].get("cer", 0.0)))
    return categories, wer_vals, cer_vals


def main():
    parser = argparse.ArgumentParser(description="Plot continuous benchmark WER/CER charts")
    parser.add_argument("--eval", required=True, help="Path to continuous_eval JSON")
    parser.add_argument("--out-dir", default="benchmark/continuous/data/charts", help="Output chart directory")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_path}")

    data = load_json(eval_path)
    out_dir = Path(args.out_dir)

    if "by_dialect" in data and data["by_dialect"]:
        categories, wer_vals, cer_vals = extract_group_metrics(data["by_dialect"])
        make_grouped_bar(
            categories,
            wer_vals,
            cer_vals,
            title="Continuous Benchmark: WER/CER by Dialect",
            out_path=out_dir / "continuous_wer_cer_by_dialect.png",
        )

    if "by_signer" in data and data["by_signer"]:
        categories, wer_vals, cer_vals = extract_group_metrics(data["by_signer"])
        make_grouped_bar(
            categories,
            wer_vals,
            cer_vals,
            title="Continuous Benchmark: WER/CER by Signer",
            out_path=out_dir / "continuous_wer_cer_by_signer.png",
        )

    print("=" * 72)
    print("Continuous charts generated")
    print("=" * 72)
    print(f"eval_json : {eval_path}")
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    main()
