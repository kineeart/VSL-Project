import argparse
import json
import mimetypes
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import cv2


def slugify_label(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = txt.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "unk"


def multipart_form_request(url, file_path: Path, field_name="file"):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_data = f.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_data)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} for {file_path}: {payload}") from e


def get_video_duration_ms(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(200, int((frame_count / fps) * 1000.0))


def predict_clip(api_url, video_path: Path):
    data = multipart_form_request(api_url, video_path)
    preds = data.get("predictions", [])
    if not preds:
        return "no_sign", 0.0
    top = preds[0]
    return str(top.get("label", "no_sign")), float(top.get("confidence", 0.0))


def build_prediction_for_sample(api_url, sample, workspace_root: Path):
    sample_id = sample["sample_id"]
    video_path_value = sample.get("video_path", "")

    # Synthetic dataset may keep a pipe-separated list of source clips.
    clip_paths = [p for p in video_path_value.split("|") if p.strip()]
    if not clip_paths:
        clip_paths = [video_path_value]

    pred_text_tokens = []
    pred_gloss = []
    pred_segments = []
    t0 = 0
    conf_list = []

    for raw in clip_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = workspace_root / p
        if not p.exists():
            label = "no_sign"
            conf = 0.0
            dur = 800
        else:
            label, conf = predict_clip(api_url, p)
            dur = get_video_duration_ms(p)

        pred_text_tokens.append(label)
        pred_gloss.append(slugify_label(label))
        pred_segments.append({
            "gloss": slugify_label(label),
            "start_ms": t0,
            "end_ms": t0 + dur,
        })
        t0 += dur
        conf_list.append(conf)

    return {
        "sample_id": sample_id,
        "pred_sentence_text": " ".join(pred_text_tokens).strip(),
        "pred_sentence_gloss": pred_gloss,
        "pred_segments": pred_segments,
        "confidence": sum(conf_list) / len(conf_list) if conf_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate sentence predictions from backend /api/predict/video")
    parser.add_argument("--dataset", required=True, help="Path to sentence dataset JSON")
    parser.add_argument("--output", required=True, help="Path to output predictions JSON")
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to predict")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = [s for s in dataset.get("samples", []) if s.get("split") == args.split]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    api_url = args.backend.rstrip("/") + "/api/predict/video"
    root = Path.cwd()

    out_preds = []
    for i, s in enumerate(samples, start=1):
        pred = build_prediction_for_sample(api_url, s, root)
        out_preds.append(pred)
        if i % 10 == 0 or i == len(samples):
            print(f"[{i}/{len(samples)}] predicted")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "model_name": "backend_api_predict_video",
        "predictions": out_preds,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Sentence predictions generated from backend")
    print("=" * 72)
    print(f"dataset      : {dataset_path}")
    print(f"split        : {args.split}")
    print(f"backend      : {args.backend}")
    print(f"num_samples  : {len(samples)}")
    print(f"output       : {output_path}")


if __name__ == "__main__":
    main()
