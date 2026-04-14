import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def add_issue(issues, level, code, message, sample_id=None):
    item = {
        "level": level,
        "code": code,
        "message": message,
    }
    if sample_id is not None:
        item["sample_id"] = sample_id
    issues.append(item)


def validate_required_fields(sample, issues):
    sid = sample.get("sample_id", "<missing>")
    required = [
        "sample_id",
        "video_path",
        "split",
        "signer_id",
        "dialect",
        "sentence_text",
        "sentence_gloss",
        "segments",
    ]
    for field in required:
        if field not in sample:
            add_issue(issues, "error", "MISSING_FIELD", f"Missing required field: {field}", sid)


def validate_segments(sample, issues):
    sid = sample.get("sample_id", "<missing>")
    segments = sample.get("segments", [])
    if not isinstance(segments, list):
        add_issue(issues, "error", "BAD_SEGMENTS_TYPE", "segments must be a list", sid)
        return

    prev_end = None
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            add_issue(issues, "error", "BAD_SEGMENT_ITEM", f"Segment #{i} is not an object", sid)
            continue

        for k in ("gloss", "start_ms", "end_ms"):
            if k not in seg:
                add_issue(issues, "error", "BAD_SEGMENT_FIELD", f"Segment #{i} missing field: {k}", sid)

        st = seg.get("start_ms")
        ed = seg.get("end_ms")
        if isinstance(st, (int, float)) and isinstance(ed, (int, float)):
            if st < 0 or ed < 0:
                add_issue(issues, "error", "NEGATIVE_SEGMENT_TIME", f"Segment #{i} has negative time", sid)
            if ed <= st:
                add_issue(issues, "error", "NON_POSITIVE_SEGMENT_DURATION", f"Segment #{i} has end <= start", sid)
            if prev_end is not None and st < prev_end:
                add_issue(issues, "warning", "OVERLAP_OR_OUT_OF_ORDER_SEGMENT", f"Segment #{i} starts before previous segment ended", sid)
            prev_end = ed


def validate_sentence_alignment(sample, issues):
    sid = sample.get("sample_id", "<missing>")
    gloss = sample.get("sentence_gloss", [])
    segments = sample.get("segments", [])

    if not isinstance(gloss, list):
        add_issue(issues, "error", "BAD_GLOSS_TYPE", "sentence_gloss must be a list", sid)
        return

    if len(gloss) == 0:
        add_issue(issues, "warning", "EMPTY_GLOSS", "sentence_gloss is empty", sid)

    if isinstance(segments, list) and len(segments) > 0 and len(gloss) != len(segments):
        add_issue(
            issues,
            "warning",
            "GLOSS_SEGMENT_LENGTH_MISMATCH",
            f"sentence_gloss length ({len(gloss)}) != segments length ({len(segments)})",
            sid,
        )


def validate_split_values(samples, issues):
    allowed = {"train", "val", "test"}
    for s in samples:
        sid = s.get("sample_id", "<missing>")
        split = s.get("split")
        if split not in allowed:
            add_issue(issues, "error", "INVALID_SPLIT", f"Invalid split value: {split}", sid)


def check_unique_sample_ids(samples, issues):
    c = Counter([s.get("sample_id") for s in samples])
    dups = [sid for sid, n in c.items() if sid and n > 1]
    for sid in dups:
        add_issue(issues, "error", "DUPLICATE_SAMPLE_ID", f"Duplicate sample_id: {sid}", sid)


def check_split_leakage(samples, issues):
    signer_to_splits = defaultdict(set)
    for s in samples:
        signer_to_splits[s.get("signer_id", "unknown")].add(s.get("split", "unknown"))

    leaked = []
    for signer, splits in signer_to_splits.items():
        real_splits = {x for x in splits if x in {"train", "val", "test"}}
        if len(real_splits) > 1:
            leaked.append((signer, sorted(real_splits)))

    for signer, splits in leaked:
        add_issue(
            issues,
            "warning",
            "SIGNER_SPLIT_LEAKAGE",
            f"Signer appears in multiple splits: signer={signer}, splits={splits}",
        )


def check_dialect_coverage(samples, issues):
    by_split = defaultdict(list)
    for s in samples:
        by_split[s.get("split", "unknown")].append(s)

    for split in ("train", "val", "test"):
        items = by_split.get(split, [])
        if len(items) == 0:
            add_issue(issues, "warning", "EMPTY_SPLIT", f"Split '{split}' has 0 samples")
            continue

        dcount = Counter([x.get("dialect", "unknown") for x in items])
        if len(dcount) < 2:
            add_issue(
                issues,
                "warning",
                "LOW_DIALECT_DIVERSITY",
                f"Split '{split}' has low dialect diversity: {dict(dcount)}",
            )


def check_ondevice_readiness(samples, issues):
    # Practical checks useful before on-device deployment experiments.
    for s in samples:
        sid = s.get("sample_id", "<missing>")
        segs = s.get("segments", [])
        if not isinstance(segs, list) or len(segs) == 0:
            continue

        starts = [x.get("start_ms") for x in segs if isinstance(x.get("start_ms"), (int, float))]
        ends = [x.get("end_ms") for x in segs if isinstance(x.get("end_ms"), (int, float))]
        if not starts or not ends:
            continue

        duration_ms = max(ends) - min(starts)
        if duration_ms > 12000:
            add_issue(
                issues,
                "warning",
                "LONG_SEQUENCE_FOR_ONDEVICE",
                f"Sequence duration {duration_ms:.0f}ms may hurt mobile latency",
                sid,
            )
        if duration_ms < 600:
            add_issue(
                issues,
                "warning",
                "TOO_SHORT_SEQUENCE",
                f"Sequence duration {duration_ms:.0f}ms may be unstable for sentence-level recognition",
                sid,
            )


def build_summary(samples, issues):
    by_split = Counter([s.get("split", "unknown") for s in samples])
    by_dialect = Counter([s.get("dialect", "unknown") for s in samples])
    by_signer = Counter([s.get("signer_id", "unknown") for s in samples])

    level_counts = Counter([x["level"] for x in issues])

    summary = {
        "num_samples": len(samples),
        "num_signers": len(by_signer),
        "split_distribution": dict(by_split),
        "dialect_distribution": dict(by_dialect),
        "issue_counts": dict(level_counts),
    }
    return summary


def validate_schema_like(dataset, issues):
    if not isinstance(dataset, dict):
        add_issue(issues, "error", "BAD_ROOT_TYPE", "Dataset root must be an object")
        return []

    if "samples" not in dataset:
        add_issue(issues, "error", "MISSING_SAMPLES", "Missing top-level field: samples")
        return []

    samples = dataset.get("samples")
    if not isinstance(samples, list):
        add_issue(issues, "error", "BAD_SAMPLES_TYPE", "samples must be a list")
        return []

    return samples


def run_validation(dataset):
    issues = []
    samples = validate_schema_like(dataset, issues)
    if not samples:
        return {
            "is_valid": False,
            "summary": build_summary([], issues),
            "issues": issues,
        }

    check_unique_sample_ids(samples, issues)
    validate_split_values(samples, issues)

    for s in samples:
        validate_required_fields(s, issues)
        validate_sentence_alignment(s, issues)
        validate_segments(s, issues)

    # Research-focused checks for robust benchmarking.
    check_split_leakage(samples, issues)
    check_dialect_coverage(samples, issues)
    check_ondevice_readiness(samples, issues)

    has_error = any(x["level"] == "error" for x in issues)
    report = {
        "is_valid": not has_error,
        "summary": build_summary(samples, issues),
        "issues": issues,
    }
    return report


def print_report(report):
    summary = report["summary"]
    print("=" * 72)
    print("Sentence Dataset Validation Report")
    print("=" * 72)
    print(f"is_valid                : {report['is_valid']}")
    print(f"num_samples             : {summary['num_samples']}")
    print(f"num_signers             : {summary['num_signers']}")
    print(f"split_distribution      : {summary['split_distribution']}")
    print(f"dialect_distribution    : {summary['dialect_distribution']}")
    print(f"issue_counts            : {summary['issue_counts']}")

    if report["issues"]:
        print("\nTop issues:")
        for issue in report["issues"][:20]:
            sid = issue.get("sample_id", "-")
            print(f"- [{issue['level'].upper()}] {issue['code']} | sample={sid} | {issue['message']}")



def main():
    parser = argparse.ArgumentParser(
        description="Validate sentence-level VSL dataset (schema-like checks + robust benchmark checks)"
    )
    parser.add_argument("--input", required=True, help="Path to sentence dataset JSON")
    parser.add_argument("--out", default="", help="Optional output report JSON path")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code if warnings exist")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset = load_json(input_path)
    report = run_validation(dataset)
    print_report(report)

    if args.out:
        save_json(Path(args.out), report)
        print(f"\nSaved report: {args.out}")

    has_error = any(x["level"] == "error" for x in report["issues"])
    has_warning = any(x["level"] == "warning" for x in report["issues"])

    if has_error:
        raise SystemExit(2)
    if args.strict and has_warning:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
