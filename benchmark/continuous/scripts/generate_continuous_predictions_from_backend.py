import argparse
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path


def multipart_form_request(url, file_path: Path, field_name="file", extra_fields=None):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_data = f.read()

    body = bytearray()
    if extra_fields:
        for key, value in extra_fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_data)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} for {file_path}: {payload}") from e


def resolve_video_path(path_value: str, workspace_root: Path):
    # For rendered continuous dataset, video_path should be a real path (not pipe-joined).
    first = path_value.split("|")[0]
    p = Path(first)
    if not p.is_absolute():
        p = workspace_root / p
    return p


def main():
    parser = argparse.ArgumentParser(description="Generate continuous-sequence predictions from backend")
    parser.add_argument("--dataset", required=True, help="Path to continuous dataset JSON")
    parser.add_argument("--output", required=True, help="Output predictions JSON")
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to run")
    parser.add_argument("--stride", type=int, default=3, help="Frame stride for continuous endpoint")
    parser.add_argument("--limit", type=int, default=0, help="Optional max samples")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    with open(ds_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = [s for s in dataset.get("samples", []) if s.get("split") == args.split]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    api_url = args.backend.rstrip("/") + "/api/predict/continuous_video"
    root = Path.cwd()

    predictions = []
    for idx, sample in enumerate(samples, start=1):
        video = resolve_video_path(sample.get("video_path", ""), root)
        if not video.exists():
            predictions.append({
                "sample_id": sample.get("sample_id"),
                "pred_sentence_text": "",
                "pred_sentence_gloss": [],
                "pred_segments": [],
                "confidence": 0.0,
                "error": f"video_not_found:{video}",
            })
            continue

        result = multipart_form_request(api_url, video, field_name="file", extra_fields={"stride": max(1, args.stride)})
        gloss = result.get("pred_sentence_gloss", [])
        text = result.get("pred_sentence_text", "")
        confidence = 1.0 if gloss else 0.0
        predictions.append({
            "sample_id": sample.get("sample_id"),
            "pred_sentence_text": text,
            "pred_sentence_gloss": gloss,
            "pred_segments": [],
            "confidence": confidence,
        })

        if idx % 10 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] predicted")

    payload = {
        "version": "1.0",
        "model_name": "backend_api_predict_continuous_video",
        "predictions": predictions,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Continuous predictions generated")
    print("=" * 72)
    print(f"dataset      : {ds_path}")
    print(f"split        : {args.split}")
    print(f"num_samples  : {len(samples)}")
    print(f"output       : {out}")


if __name__ == "__main__":
    main()
