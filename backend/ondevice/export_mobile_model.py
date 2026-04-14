import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out).mean(dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, temperature=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.temperature * (x_norm @ w_norm.t())


class SignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.attention = MultiHeadAttention(768, num_heads=4)
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = CosineClassifier(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.conv_drop(c)
        c = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm1(c)
        attn_out = self.attention(lstm_out)
        x = F.gelu(self.bn3(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)


def human_size(path: Path):
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description="Export VSL model for on-device inference")
    parser.add_argument("--model-dir", default="backend/models_15cls_run1")
    parser.add_argument("--checkpoint", default="sign_model.pt")
    parser.add_argument("--output-dir", default="backend/ondevice/artifacts")
    parser.add_argument("--seq-len", type=int, default=60)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / args.checkpoint
    labels_path = model_dir / "labels.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        num_classes = len(labels)
    else:
        labels = {}
        num_classes = 15

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    num_features = checkpoint.get("num_features", 4995)
    num_classes = checkpoint.get("num_classes", num_classes)

    model = SignModel(num_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_input = torch.randn(1, args.seq_len, num_features)

    fp32_path = out_dir / "sign_model_fp32.pt"
    torch.save({"model_state_dict": model.state_dict()}, fp32_path)

    scripted_path = out_dir / "sign_model_torchscript.pt"
    scripted = torch.jit.trace(model, sample_input)
    scripted.save(str(scripted_path))

    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    quantized.eval()

    q_scripted_path = out_dir / "sign_model_torchscript_int8.pt"
    q_scripted = torch.jit.trace(quantized, sample_input)
    q_scripted.save(str(q_scripted_path))

    onnx_path = out_dir / "sign_model.onnx"
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    manifest = {
        "created_at_unix": int(time.time()),
        "source_model_dir": str(model_dir).replace("\\", "/"),
        "checkpoint": args.checkpoint,
        "num_features": num_features,
        "num_classes": num_classes,
        "seq_len": args.seq_len,
        "labels": labels,
        "artifacts": {
            "fp32": str(fp32_path).replace("\\", "/"),
            "torchscript": str(scripted_path).replace("\\", "/"),
            "torchscript_int8": str(q_scripted_path).replace("\\", "/"),
            "onnx": str(onnx_path).replace("\\", "/"),
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("On-device export completed")
    print("=" * 72)
    for p in [fp32_path, scripted_path, q_scripted_path, onnx_path, manifest_path]:
        print(f"{p}: {human_size(p)}")


if __name__ == "__main__":
    main()
