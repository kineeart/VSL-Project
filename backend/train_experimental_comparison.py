"""Few-shot / embedding benchmark for Vietnamese Sign Language.

Context:
- Low-data regime (often 1 original video per class)
- Metric learning instead of standard classification
- Fair comparison across encoders under the same data protocol

Models compared:
- GRU encoder
- BiLSTM encoder
- Transformer encoder
- Proposed encoder (CNN + BiLSTM + Attention)

Outputs:
- comparison_results.json / comparison_results.csv
- best checkpoint for each encoder (.pt)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import train_gpu as tg
from keypoint_variants import KeypointType


# =========================
# Utilities
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_variant(variant_name: str) -> None:
    variant = KeypointType(variant_name)
    tg.KEYPOINT_VARIANT = variant
    tg.VARIANT_CONFIG = tg.VARIANTS[variant]
    tg.HAS_HANDS_POSE = (
        tg.VARIANT_CONFIG.include_pose
        and tg.VARIANT_CONFIG.include_left_hand
        and tg.VARIANT_CONFIG.include_right_hand
    )
    tg.NRF = tg.VARIANT_CONFIG.feature_size


@dataclass(frozen=True)
class FeatureConfig:
    include_velocity: bool = True
    include_acceleration: bool = True


def feature_dim(raw_dim: int, cfg: FeatureConfig) -> int:
    d = raw_dim
    if cfg.include_velocity:
        d += raw_dim
    if cfg.include_acceleration:
        d += raw_dim
    if tg.HAS_HANDS_POSE:
        d += 9
    return d


def build_features(raw_seq: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Raw keypoints -> engineered feature matrix (seq, feat_dim)."""
    s = raw_seq.astype(np.float32)

    v = np.zeros_like(s, dtype=np.float32)
    v[1:] = s[1:] - s[:-1]

    a = np.zeros_like(s, dtype=np.float32)
    a[1:] = v[1:] - v[:-1]

    parts = [s]
    if cfg.include_velocity:
        parts.append(v)
    if cfg.include_acceleration:
        parts.append(a)

    if tg.HAS_HANDS_POSE:
        pose_size = tg.VARIANT_CONFIG.pose_points * (4 if tg.VARIANT_CONFIG.include_pose_visibility else 3)
        ls = pose_size
        rs = ls + 21 * 3
        if rs + 3 <= s.shape[1]:
            nose = s[:, 0:3]
            lw = s[:, ls:ls + 3]
            rw = s[:, rs:rs + 3]
            ex = np.concatenate(
                [
                    lw - nose,
                    rw - nose,
                    np.linalg.norm(lw - rw, axis=1, keepdims=True),
                    np.linalg.norm(lw - nose, axis=1, keepdims=True),
                    np.linalg.norm(rw - nose, axis=1, keepdims=True),
                ],
                axis=1,
            ).astype(np.float32)
            parts.append(ex)

    return np.concatenate(parts, axis=1).astype(np.float32)


# =========================
# Encoders (128D output)
# =========================


class MeanPoolAttention(nn.Module):
    def __init__(self, hidden: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out.mean(dim=1)


class GRUEncoder(nn.Module):
    def __init__(self, input_size: int, emb_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        z = self.head(pooled)
        return F.normalize(z, p=2, dim=1)


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size: int, emb_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        z = self.head(pooled)
        return F.normalize(z, p=2, dim=1)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size: int, emb_dim: int = 128, max_seq: int = 80):
        super().__init__()
        hidden = 256
        self.in_proj = nn.Linear(input_size, hidden)
        self.pos_emb = nn.Embedding(max_seq, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
        )
        self.max_seq = max_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.in_proj(x)
        pos = torch.arange(min(t, self.max_seq), device=x.device).unsqueeze(0)
        x[:, : pos.shape[1], :] = x[:, : pos.shape[1], :] + self.pos_emb(pos)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        z = self.head(pooled)
        return F.normalize(z, p=2, dim=1)


class ProposedEncoder(nn.Module):
    """CNN + BiLSTM + Attention -> 128D embedding."""

    def __init__(self, input_size: int, emb_dim: int = 128):
        super().__init__()
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.drop = nn.Dropout(0.25)

        self.lstm = nn.LSTM(384, 320, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.attn = MeanPoolAttention(640, heads=4)

        self.head = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.drop(c)
        c = c.permute(0, 2, 1)

        out, _ = self.lstm(c)
        pooled = self.attn(out)
        z = self.head(pooled)
        return F.normalize(z, p=2, dim=1)


# =========================
# Data pipeline
# =========================


def build_cache_index(mapping: dict[str, str], raw_dim: int) -> dict[str, list[Path]]:
    by_label: dict[str, list[Path]] = defaultdict(list)
    skipped = 0

    for fn, lb in mapping.items():
        cp = tg.landmark_cache_path(fn, tg.KEYPOINT_VARIANT)
        if not cp.exists():
            skipped += 1
            continue
        try:
            arr = np.load(str(cp))
            if arr.ndim != 2 or arr.shape[1] != raw_dim:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        by_label[str(lb)].append(cp)

    print(f"[DATA] Classes with usable cache: {len(by_label)}")
    print(f"[DATA] Skipped cache files: {skipped}")
    return by_label


def select_classes(by_label: dict[str, list[Path]], max_classes: int, min_samples: int) -> dict[str, list[Path]]:
    ranked = [(lb, len(v)) for lb, v in by_label.items() if len(v) >= min_samples]
    ranked.sort(key=lambda x: (-x[1], str(x[0])))
    if max_classes > 0:
        ranked = ranked[:max_classes]

    if not ranked:
        raise RuntimeError("No eligible class after filtering. Reduce --min-samples or --max-classes")

    keep = {lb for lb, _ in ranked}
    out = {lb: by_label[lb] for lb in keep}
    print(f"[DATA] Classes selected: {len(out)}")
    return out


def split_support_query(
    by_label: dict[str, list[Path]],
    support_per_class: int,
    query_augs_per_class: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = np.random.default_rng(seed)

    support_samples: list[tuple[Path, int]] = []
    query_samples: list[tuple[Path, int]] = []

    labels = sorted(by_label.keys(), key=str)
    for li, lb in enumerate(labels):
        paths = by_label[lb].copy()
        rng.shuffle(paths)

        if len(paths) >= support_per_class + 1:
            support_paths = paths[:support_per_class]
            query_paths = paths[support_per_class:]
            for p in support_paths:
                support_samples.append((p, li))
            for p in query_paths:
                query_samples.append((p, li))
        else:
            # one-video-per-class case: use the same file for support and query views
            p = paths[0]
            support_samples.append((p, li))
            for _ in range(max(1, query_augs_per_class)):
                query_samples.append((p, li))

    print(f"[DATA] Support items: {len(support_samples)}")
    print(f"[DATA] Query items: {len(query_samples)}")
    return support_samples, query_samples


def compute_norm_stats(samples: list[tuple[Path, int]], seq_len: int, raw_dim: int, cfg: FeatureConfig):
    dim = feature_dim(raw_dim, cfg)
    s1 = np.zeros(dim, dtype=np.float64)
    s2 = np.zeros(dim, dtype=np.float64)
    n = 0

    for path, _ in samples:
        raw = np.load(str(path))
        if raw.shape[0] != seq_len:
            raw = tg.random_choose_seq(raw, seq_len, auto_pad=True)
        x = build_features(raw, cfg)
        s1 += x.sum(axis=0)
        s2 += (x * x).sum(axis=0)
        n += x.shape[0]

    mean = (s1 / max(n, 1)).astype(np.float32)
    var = (s2 / max(n, 1)) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def make_view(raw: np.ndarray, seq_len: int, cfg: FeatureConfig, mean: np.ndarray, std: np.ndarray, augment: bool) -> np.ndarray:
    if raw.shape[0] != seq_len:
        raw = tg.random_choose_seq(raw, seq_len, auto_pad=True)
    if augment:
        raw = tg.aug1(raw, tg.KEYPOINT_VARIANT)
    x = build_features(raw, cfg)
    x = (x - mean) / std
    return x.astype(np.float32)


class MetricPairDataset(Dataset):
    """Returns two augmented views from the same class sample for metric learning."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        seq_len: int,
        raw_dim: int,
        cfg: FeatureConfig,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.samples = samples
        self.seq_len = seq_len
        self.raw_dim = raw_dim
        self.cfg = cfg
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        raw = np.load(str(path))
        if raw.ndim != 2 or raw.shape[1] != self.raw_dim:
            raise ValueError(f"Invalid raw shape in {path}: {raw.shape}")

        v1 = make_view(raw, self.seq_len, self.cfg, self.mean, self.std, augment=True)
        v2 = make_view(raw, self.seq_len, self.cfg, self.mean, self.std, augment=True)
        return (
            torch.from_numpy(v1),
            torch.from_numpy(v2),
            torch.tensor(y, dtype=torch.long),
        )


class EvalDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        seq_len: int,
        raw_dim: int,
        cfg: FeatureConfig,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
    ):
        self.samples = samples
        self.seq_len = seq_len
        self.raw_dim = raw_dim
        self.cfg = cfg
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        raw = np.load(str(path))
        if raw.ndim != 2 or raw.shape[1] != self.raw_dim:
            raise ValueError(f"Invalid raw shape in {path}: {raw.shape}")
        x = make_view(raw, self.seq_len, self.cfg, self.mean, self.std, augment=self.augment)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =========================
# Metric losses
# =========================


def pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
    # x: (N, D)
    # returns (N, N)
    dist = torch.cdist(x, x, p=2)
    return dist


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    dist = pairwise_euclidean(emb)
    n = emb.size(0)
    labels = labels.view(-1)

    loss_vals = []
    for i in range(n):
        pos_mask = labels == labels[i]
        neg_mask = labels != labels[i]

        pos_mask[i] = False
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        hardest_pos = dist[i][pos_mask].max()
        hardest_neg = dist[i][neg_mask].min()
        loss_vals.append(F.relu(hardest_pos - hardest_neg + margin))

    if not loss_vals:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    return torch.stack(loss_vals).mean()


def contrastive_pair_loss(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Simple contrastive loss on pairs in a batch.

    Positive pairs: same label
    Negative pairs: different label
    """
    d = F.pairwise_distance(z1, z2, p=2)
    y = torch.ones_like(d, device=d.device)  # given pairs are positive
    pos_loss = y * (d ** 2)

    # Build negatives by shuffling labels until mismatch.
    idx = torch.randperm(z2.size(0), device=z2.device)
    z2n = z2[idx]
    yn = (labels != labels[idx]).float()
    dn = F.pairwise_distance(z1, z2n, p=2)
    neg_loss = yn * (F.relu(margin - dn) ** 2)

    loss = pos_loss.mean() + neg_loss.mean()
    return loss


# =========================
# Train + Evaluate
# =========================


def build_train_loader(samples: list[tuple[Path, int]], dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    class_counts = Counter([y for _, y in samples])
    weights = np.array([1.0 / class_counts[y] for _, y in samples], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(samples), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


def encode_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    embs = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            z = model(xb)
            embs.append(z.detach().cpu().numpy())
            ys.append(yb.numpy())
    if not embs:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)


def nearest_neighbor_topk(
    support_emb: np.ndarray,
    support_y: np.ndarray,
    query_emb: np.ndarray,
    query_y: np.ndarray,
    k_values=(1, 5),
) -> dict[str, float]:
    if len(query_emb) == 0 or len(support_emb) == 0:
        return {"top1": 0.0, "top5": 0.0}

    # cosine similarity because embeddings are L2 normalized
    sim = query_emb @ support_emb.T  # (Q, S)

    max_k = min(max(k_values), support_emb.shape[0])
    idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]

    order_scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-order_scores, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    pred_labels = support_y[top_idx]

    out = {}
    for k in k_values:
        kk = min(k, pred_labels.shape[1])
        hit = (pred_labels[:, :kk] == query_y[:, None]).any(axis=1)
        out[f"top{k}"] = float(hit.mean())

    return {"top1": out.get("top1", 0.0), "top5": out.get("top5", 0.0)}


def train_encoder(
    model_name: str,
    model_factory: Callable[[int, int], nn.Module],
    input_dim: int,
    emb_dim: int,
    train_loader: DataLoader,
    support_loader: DataLoader,
    query_loader: DataLoader,
    out_dir: Path,
    epochs: int,
    lr: float,
    patience: int,
    mixed_precision: bool,
    loss_name: str,
    margin: float,
    device: torch.device,
) -> dict[str, float]:
    model = model_factory(input_dim, emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == "cuda") else None

    best_top1 = -1.0
    best_top5 = -1.0
    no_improve = 0
    history = []

    ckpt = out_dir / f"{model_name}_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batch = 0

        for v1, v2, y in train_loader:
            y = y.to(device, non_blocking=True)
            x = torch.cat([v1, v2], dim=0).to(device, non_blocking=True)
            y2 = torch.cat([y, y], dim=0)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    z = model(x)
                    if loss_name == "contrastive":
                        z1, z2 = z[: y.size(0)], z[y.size(0) :]
                        loss = contrastive_pair_loss(z1, z2, y, margin=margin)
                    else:
                        loss = batch_hard_triplet_loss(z, y2, margin=margin)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                z = model(x)
                if loss_name == "contrastive":
                    z1, z2 = z[: y.size(0)], z[y.size(0) :]
                    loss = contrastive_pair_loss(z1, z2, y, margin=margin)
                else:
                    loss = batch_hard_triplet_loss(z, y2, margin=margin)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum += float(loss.item())
            n_batch += 1

        scheduler.step()
        train_loss = loss_sum / max(n_batch, 1)

        support_emb, support_y = encode_loader(model, support_loader, device)
        query_emb, query_y = encode_loader(model, query_loader, device)
        m = nearest_neighbor_topk(support_emb, support_y, query_emb, query_y, k_values=(1, 5))

        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "top1": float(m["top1"]),
                "top5": float(m["top5"]),
            }
        )

        print(
            f"[{model_name}] Ep {ep:3d}/{epochs} loss={train_loss:.4f} "
            f"Top-1={m['top1']*100:.2f}% Top-5={m['top5']*100:.2f}%"
        )

        improved = (m["top1"] > best_top1) or (m["top1"] == best_top1 and m["top5"] > best_top5)
        if improved:
            best_top1 = m["top1"]
            best_top5 = m["top5"]
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "input_dim": input_dim,
                    "embedding_dim": emb_dim,
                    "loss_name": loss_name,
                    "margin": margin,
                    "best_top1": best_top1,
                    "best_top5": best_top5,
                    "history": history,
                },
                str(ckpt),
            )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {ep}")
                break

    return {
        "top1": float(best_top1),
        "top5": float(best_top5),
        "checkpoint": str(ckpt),
        "epochs_ran": len(history),
    }


# =========================
# Reporting
# =========================


def print_table(title: str, result_dict: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Model':26s} | {'Top-1':>8s} | {'Top-5':>8s}")
    print("-" * 72)
    for name, m in result_dict.items():
        print(f"{name:26s} | {m['top1']*100:7.2f}% | {m['top5']*100:7.2f}%")


def save_results(path_json: Path, path_csv: Path, result_dict: dict[str, dict[str, float]]) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "top1", "top5", "checkpoint", "epochs_ran"])
        writer.writeheader()
        for name, m in result_dict.items():
            writer.writerow({"model": name, **m})


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot embedding benchmark for VSL")
    p.add_argument("--variant", default=KeypointType.HOLISTIC_NO_FACE.value)
    p.add_argument("--seq", type=int, default=50)
    p.add_argument("--min-samples", type=int, default=1)
    p.add_argument("--max-classes", type=int, default=0, help="0 means keep all classes")
    p.add_argument("--support-per-class", type=int, default=1)
    p.add_argument("--query-augs-per-class", type=int, default=5)

    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--loss", choices=["triplet", "contrastive"], default="triplet")
    p.add_argument("--margin", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--extract-workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", action="store_true")

    p.add_argument("--out-dir", default="backend/models/fewshot_embedding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    configure_variant(args.variant)
    tg.SEQ = args.seq
    tg.NW = max(1, args.extract_workers)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = tg.BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FEW-SHOT / EMBEDDING EXPERIMENTAL COMPARISON")
    print("=" * 72)
    print(f"Variant: {tg.KEYPOINT_VARIANT.value} ({tg.VARIANT_CONFIG.name})")
    print(f"Raw features: {tg.NRF}, seq: {tg.SEQ}, emb: {args.embedding_dim}")
    print(f"Loss: {args.loss}, margin: {args.margin}")

    mapping = tg.load_data_mapping()
    if not mapping:
        raise RuntimeError("No data mapping found")

    tg.extract_all(mapping)

    by_label = build_cache_index(mapping, tg.NRF)
    by_label = select_classes(by_label, args.max_classes, args.min_samples)

    support_samples, query_samples = split_support_query(
        by_label=by_label,
        support_per_class=args.support_per_class,
        query_augs_per_class=args.query_augs_per_class,
        seed=args.seed,
    )

    # train_samples are support files; with one video/class we rely on view augmentation for positives.
    train_samples = support_samples

    cfg = FeatureConfig(include_velocity=True, include_acceleration=True)
    mean, std = compute_norm_stats(train_samples, tg.SEQ, tg.NRF, cfg)
    in_dim = feature_dim(tg.NRF, cfg)

    train_ds = MetricPairDataset(train_samples, tg.SEQ, tg.NRF, cfg, mean, std)
    support_ds = EvalDataset(support_samples, tg.SEQ, tg.NRF, cfg, mean, std, augment=False)
    query_ds = EvalDataset(query_samples, tg.SEQ, tg.NRF, cfg, mean, std, augment=True)

    train_loader = build_train_loader(train_samples, train_ds, args.batch_size, args.num_workers)
    support_loader = DataLoader(support_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = tg.setup_gpu()

    model_factories: dict[str, Callable[[int, int], nn.Module]] = {
        "GRU encoder": lambda d, e: GRUEncoder(d, e),
        "BiLSTM encoder": lambda d, e: BiLSTMEncoder(d, e),
        "Transformer encoder": lambda d, e: TransformerEncoder(d, e, max_seq=max(80, args.seq + 4)),
        "Proposed encoder": lambda d, e: ProposedEncoder(d, e),
    }

    comparison_results: dict[str, dict[str, float]] = {}
    for name, factory in model_factories.items():
        comparison_results[name] = train_encoder(
            model_name=name.replace(" ", "_"),
            model_factory=factory,
            input_dim=in_dim,
            emb_dim=args.embedding_dim,
            train_loader=train_loader,
            support_loader=support_loader,
            query_loader=query_loader,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            mixed_precision=args.mixed_precision,
            loss_name=args.loss,
            margin=args.margin,
            device=device,
        )

    print_table("FEW-SHOT MODEL COMPARISON", comparison_results)

    save_results(
        out_dir / "comparison_results.json",
        out_dir / "comparison_results.csv",
        comparison_results,
    )

    print("\nSaved:")
    print(f"- {out_dir / 'comparison_results.json'}")
    print(f"- {out_dir / 'comparison_results.csv'}")
    print("- Best checkpoints: <model>_best.pt in same folder")


if __name__ == "__main__":
    main()
