import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def benchmark_torchscript(model_path: Path, warmup=20, runs=100, seq_len=60, num_features=4995):
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    x = torch.randn(1, seq_len, num_features)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark exported on-device artifacts")
    parser.add_argument("--manifest", default="backend/ondevice/artifacts/manifest.json")
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    seq_len = int(m.get("seq_len", 60))
    num_features = int(m.get("num_features", 4995))
    artifacts = m.get("artifacts", {})

    ts_path = Path(artifacts.get("torchscript", ""))
    qts_path = Path(artifacts.get("torchscript_int8", ""))

    report = {
        "manifest": str(manifest_path).replace("\\", "/"),
        "seq_len": seq_len,
        "num_features": num_features,
        "runs": args.runs,
        "warmup": args.warmup,
        "results": {},
    }

    if ts_path.exists():
        report["results"]["torchscript_fp32"] = benchmark_torchscript(
            ts_path,
            warmup=args.warmup,
            runs=args.runs,
            seq_len=seq_len,
            num_features=num_features,
        )

    if qts_path.exists():
        report["results"]["torchscript_int8"] = benchmark_torchscript(
            qts_path,
            warmup=args.warmup,
            runs=args.runs,
            seq_len=seq_len,
            num_features=num_features,
        )

    out_path = manifest_path.parent / "benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("On-device artifact benchmark")
    print("=" * 72)
    for name, stats in report["results"].items():
        print(
            f"{name}: mean={stats['mean_ms']:.2f}ms p50={stats['p50_ms']:.2f}ms "
            f"p90={stats['p90_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms"
        )
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
