import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of split ratios must be > 0")
    return train_ratio / total, val_ratio / total, test_ratio / total


def make_group_key(sample, group_by):
    signer = sample.get("signer_id", "unknown_signer")
    dialect = sample.get("dialect", "unknown")
    sample_id = sample.get("sample_id", "unknown_sample")

    if group_by == "signer":
        return signer
    if group_by == "dialect":
        return dialect
    if group_by == "signer_dialect":
        return f"{signer}::{dialect}"
    if group_by == "sample":
        return sample_id
    raise ValueError(f"Unsupported group_by: {group_by}")


def build_groups(samples, group_by):
    groups = defaultdict(list)
    for s in samples:
        groups[make_group_key(s, group_by)].append(s)
    return groups


def dominant_dialect(group_samples):
    c = Counter([s.get("dialect", "unknown") for s in group_samples])
    if not c:
        return "unknown"
    return c.most_common(1)[0][0]


def assign_groups_greedy(groups, train_ratio, val_ratio, test_ratio, seed=42):
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    total_samples = sum(len(v) for _, v in group_items)
    targets = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * test_ratio,
    }

    assigned = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_dialect_sets = {"train": set(), "val": set(), "test": set()}

    for group_key, group_samples in group_items:
        group_size = len(group_samples)
        group_dialect = dominant_dialect(group_samples)

        best_split = None
        best_score = None

        for split in ("train", "val", "test"):
            # Main objective: keep split sample counts close to target.
            next_count = split_counts[split] + group_size
            count_cost = abs(next_count - targets[split])

            # Secondary objective: diversify dialects in each split.
            dialect_penalty = 0.0
            if group_dialect not in split_dialect_sets[split]:
                dialect_penalty = 0.2

            score = count_cost + dialect_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assigned[best_split].append((group_key, group_samples))
        split_counts[best_split] += group_size
        split_dialect_sets[best_split].add(group_dialect)

    # Ensure no split is empty if enough groups exist.
    non_empty_splits = [s for s in ("train", "val", "test") if split_counts[s] > 0]
    if len(non_empty_splits) < 3 and len(group_items) >= 3:
        # Move one smallest group from largest split into each empty split.
        empty_splits = [s for s in ("train", "val", "test") if split_counts[s] == 0]
        for empty in empty_splits:
            largest = max(("train", "val", "test"), key=lambda s: split_counts[s])
            if not assigned[largest]:
                continue
            smallest_idx = min(range(len(assigned[largest])), key=lambda i: len(assigned[largest][i][1]))
            group = assigned[largest].pop(smallest_idx)
            moved_size = len(group[1])
            split_counts[largest] -= moved_size
            split_counts[empty] += moved_size
            assigned[empty].append(group)

    return assigned


def flatten_assignment(assigned):
    split_samples = {"train": [], "val": [], "test": []}
    split_groups = {"train": set(), "val": set(), "test": set()}

    for split, group_entries in assigned.items():
        for gk, gs in group_entries:
            split_groups[split].add(gk)
            split_samples[split].extend(gs)

    return split_samples, split_groups


def summarize(split_samples, split_groups):
    summary = {}
    for split in ("train", "val", "test"):
        samples = split_samples[split]
        summary[split] = {
            "num_samples": len(samples),
            "num_groups": len(split_groups[split]),
            "num_signers": len(set(s.get("signer_id", "unknown") for s in samples)),
            "dialect_distribution": dict(Counter(s.get("dialect", "unknown") for s in samples)),
        }
    return summary


def rebalance_non_empty_by_samples(split_samples, desired=("train", "val", "test")):
    """Rebalance by moving individual samples so each desired split is non-empty.

    This may break strict group isolation and should only be used as fallback.
    """
    changed = False
    for split in desired:
        if len(split_samples[split]) > 0:
            continue
        donor = max(desired, key=lambda s: len(split_samples[s]))
        if len(split_samples[donor]) <= 1:
            continue
        split_samples[split].append(split_samples[donor].pop())
        changed = True
    return changed


def apply_split(dataset, split_samples):
    id_to_split = {}
    for split, samples in split_samples.items():
        for s in samples:
            id_to_split[s.get("sample_id")] = split

    out = deepcopy(dataset)
    for s in out.get("samples", []):
        sid = s.get("sample_id")
        if sid in id_to_split:
            s["split"] = id_to_split[sid]
    return out


def main():
    parser = argparse.ArgumentParser(description="Auto split sentence-level dataset into train/val/test")
    parser.add_argument("--input", required=True, help="Input dataset JSON path")
    parser.add_argument("--output", required=True, help="Output dataset JSON path with split field updated")
    parser.add_argument("--report", default="", help="Optional output report JSON path")
    parser.add_argument("--group-by", default="signer", choices=["signer", "dialect", "signer_dialect", "sample"],
                        help="Grouping key for split assignment")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-non-empty-splits",
        action="store_true",
        help="Fallback to sample-level moves if grouped split leaves val/test empty",
    )
    args = parser.parse_args()

    train_ratio, val_ratio, test_ratio = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset = load_json(input_path)
    samples = dataset.get("samples", [])
    if not samples:
        raise ValueError("Dataset has no samples")

    groups = build_groups(samples, args.group_by)
    assigned = assign_groups_greedy(groups, train_ratio, val_ratio, test_ratio, seed=args.seed)
    split_samples, split_groups = flatten_assignment(assigned)

    empty_splits = [s for s in ("train", "val", "test") if len(split_samples[s]) == 0]
    if empty_splits:
        print(
            f"[WARN] Empty splits after grouped assignment: {empty_splits}. "
            f"groups={len(groups)}, samples={len(samples)}"
        )
        if args.force_non_empty_splits and len(samples) >= 3:
            changed = rebalance_non_empty_by_samples(split_samples)
            if changed:
                print("[WARN] Applied sample-level fallback to avoid empty splits (may cause signer leakage).")
                # Recompute group sets after fallback for reporting.
                split_groups = {
                    "train": set(make_group_key(s, args.group_by) for s in split_samples["train"]),
                    "val": set(make_group_key(s, args.group_by) for s in split_samples["val"]),
                    "test": set(make_group_key(s, args.group_by) for s in split_samples["test"]),
                }
        elif len(samples) < 3:
            print("[WARN] Dataset too small to populate train/val/test simultaneously.")

    out_dataset = apply_split(dataset, split_samples)
    summary = summarize(split_samples, split_groups)

    save_json(output_path, out_dataset)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "group_by": args.group_by,
        "seed": args.seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "summary": summary,
        "empty_splits": [s for s in ("train", "val", "test") if summary[s]["num_samples"] == 0],
    }

    if args.report:
        save_json(Path(args.report), report)

    print("=" * 72)
    print("Sentence Dataset Split Report")
    print("=" * 72)
    print(f"group_by: {args.group_by}")
    print(f"seed    : {args.seed}")
    print(f"ratios  : train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    for split in ("train", "val", "test"):
        s = summary[split]
        print(
            f"{split:5s} | samples={s['num_samples']:4d} | groups={s['num_groups']:3d} | "
            f"signers={s['num_signers']:3d} | dialects={s['dialect_distribution']}"
        )
    print(f"\nSaved split dataset: {output_path}")
    if args.report:
        print(f"Saved split report : {args.report}")


if __name__ == "__main__":
    main()
