"""Full experimental comparison framework for Vietnamese Sign Language recognition.

This script trains multiple models under the SAME data/split/preprocessing pipeline:
- GRU
- BiLSTM
- Transformer
- Proposed (CNN + BiLSTM + Attention)

Then runs ablation for proposed model:
- Without velocity features
- Without acceleration features
- Without attention
- Full model

Outputs:
- comparison_results.json / comparison_results.csv
- ablation_results.json / ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import train_gpu as tg
from keypoint_variants import KeypointType
from models_baseline import (
    AblationNoAttention,
    GRUBaseline,
    ImprovedSignModel,
    SimpleLSTMBaseline,
    TransformerBaseline,
)


@dataclass(frozen=True)
class FeatureConfig:
    include_velocity: bool
    include_acceleration: bool


FULL_FEATURE = FeatureConfig(include_velocity=True, include_acceleration=True)
NO_VELOCITY_FEATURE = FeatureConfig(include_velocity=False, include_acceleration=True)
NO_ACCELERATION_FEATURE = FeatureConfig(include_velocity=True, include_acceleration=False)


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


def build_features(raw_seq: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Build engineered features from raw landmarks with configurable v/a blocks."""
    s = raw_seq.astype(np.float32)

    v = np.zeros_like(s)
    v[1:] = s[1:] - s[:-1]

    a = np.zeros_like(s)
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
            )
            parts.append(ex.astype(np.float32))

    return np.concatenate(parts, axis=1).astype(np.float32)


def feature_dim(raw_dim: int, cfg: FeatureConfig) -> int:
    dim = raw_dim
    if cfg.include_velocity:
        dim += raw_dim
    if cfg.include_acceleration:
        dim += raw_dim
    if tg.HAS_HANDS_POSE:
        dim += 9
    return dim


class LandmarkSequenceDataset(Dataset):
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        raw = np.load(str(path))
        if raw.ndim != 2 or raw.shape[1] != self.raw_dim:
            raise ValueError(f"Invalid raw shape in {path}: {raw.shape}")
        if raw.shape[0] != self.seq_len:
            raw = tg.random_choose_seq(raw, self.seq_len, auto_pad=True)

        x = build_features(raw, self.cfg)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def build_samples(mapping: dict[str, str], seq_len: int, raw_dim: int) -> tuple[dict[str, list[Path]], list[str]]:
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

    labels = sorted(by_label.keys(), key=str)
    print(f"[DATA] Classes with usable cache: {len(labels)}")
    print(f"[DATA] Cached files skipped: {skipped}")
    return by_label, labels


def stratified_split(
    by_label: dict[str, list[Path]],
    min_samples: int,
    val_ratio: float,
    max_classes: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict[str, int]]:
    rng = np.random.default_rng(seed)

    ranked = [(lb, len(paths)) for lb, paths in by_label.items() if len(paths) >= min_samples]
    ranked.sort(key=lambda x: (-x[1], str(x[0])))
    if max_classes > 0:
        ranked = ranked[:max_classes]

    if not ranked:
        raise RuntimeError("No class has enough samples for the requested min_samples/max_classes setup")

    label_map = {lb: i for i, (lb, _) in enumerate(ranked)}

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for lb, _ in ranked:
        paths = by_label[lb].copy()
        rng.shuffle(paths)

        if len(paths) == 1:
            train_paths, val_paths = paths, []
        else:
            n_val = max(1, int(round(len(paths) * val_ratio)))
            n_val = min(n_val, len(paths) - 1)
            val_paths = paths[:n_val]
            train_paths = paths[n_val:]

        li = label_map[lb]
        train_samples.extend((p, li) for p in train_paths)
        val_samples.extend((p, li) for p in val_paths)

    print(f"[DATA] Selected classes: {len(label_map)}")
    print(f"[DATA] Train videos: {len(train_samples)}  Val videos: {len(val_samples)}")

    return train_samples, val_samples, label_map


def compute_norm_stats(train_samples: list[tuple[Path, int]], seq_len: int, raw_dim: int, cfg: FeatureConfig):
    dim = feature_dim(raw_dim, cfg)
    s1 = np.zeros(dim, dtype=np.float64)
    s2 = np.zeros(dim, dtype=np.float64)
    n = 0

    for path, _ in train_samples:
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


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        num_classes = logits.size(1)
        ks = [min(k, num_classes) for k in topk]
        maxk = max(ks)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        out = []
        for k in ks:
            c = correct[:k].reshape(-1).float().sum(0)
            out.append(c.item() / target.size(0))
        return out


def make_loaders(
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
    seq_len: int,
    raw_dim: int,
    cfg: FeatureConfig,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    num_workers: int,
):
    train_ds = LandmarkSequenceDataset(train_samples, seq_len, raw_dim, cfg, mean, std)
    val_ds = LandmarkSequenceDataset(val_samples, seq_len, raw_dim, cfg, mean, std)

    # Handle imbalance without materializing augmented arrays in RAM
    class_counts = Counter([y for _, y in train_samples])
    sample_weights = np.array([1.0 / class_counts[y] for _, y in train_samples], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_model(
    model_name: str,
    model_factory: Callable[[int, int], nn.Module],
    n_features: int,
    n_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
    epochs: int,
    lr: float,
    patience: int,
    mixed_precision: bool,
    device: torch.device,
):
    model = model_factory(n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == "cuda") else None

    best_top1 = 0.0
    best_top5 = 0.0
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    ckpt_path = out_dir / f"{model_name}_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = xb.size(0)
            tr_loss_sum += loss.item() * bs
            tr_n += bs

        train_loss = tr_loss_sum / max(tr_n, 1)

        model.eval()
        val_loss_sum = 0.0
        v_n = 0
        t1_sum = 0.0
        t5_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                t1, t5 = topk_accuracy(out, yb, topk=(1, 5))

                bs = xb.size(0)
                val_loss_sum += loss.item() * bs
                t1_sum += t1 * bs
                t5_sum += t5 * bs
                v_n += bs

        val_loss = val_loss_sum / max(v_n, 1)
        val_top1 = t1_sum / max(v_n, 1)
        val_top5 = t5_sum / max(v_n, 1)

        scheduler.step()

        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_top1": float(val_top1),
                "val_top5": float(val_top5),
            }
        )

        print(
            f"[{model_name}] Ep {ep:3d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"top1={val_top1*100:.2f}% top5={val_top5*100:.2f}%"
        )

        improved = (val_top1 > best_top1) or (math.isclose(val_top1, best_top1) and val_top5 > best_top5)
        if improved:
            best_top1 = val_top1
            best_top5 = val_top5
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": n_classes,
                    "num_features": n_features,
                    "best_val_top1": best_top1,
                    "best_val_top5": best_top5,
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                str(ckpt_path),
            )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {ep}")
                break

    return {
        "top1": float(best_top1),
        "top5": float(best_top5),
        "val_loss": float(best_val_loss),
        "checkpoint": str(ckpt_path),
        "epochs_ran": len(history),
    }


def print_table(title: str, result_dict: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Model':30s} | {'Top-1':>8s} | {'Top-5':>8s}")
    print("-" * 72)
    for name, m in result_dict.items():
        print(f"{name:30s} | {m['top1']*100:7.2f}% | {m['top5']*100:7.2f}%")


def save_json_csv(path_json: Path, path_csv: Path, result_dict: dict[str, dict[str, float]], key_name: str = "model"):
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name, "top1", "top5", "val_loss", "checkpoint", "epochs_ran"])
        writer.writeheader()
        for name, metrics in result_dict.items():
            row = {key_name: name}
            row.update(metrics)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental comparison framework (multi-model + ablation)")
    p.add_argument("--variant", default=KeypointType.HOLISTIC_NO_FACE.value)
    p.add_argument("--seq", type=int, default=50)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--max-classes", type=int, default=50)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--extract-workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--out-dir", default="backend/models/experimental_comparison")
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
    print("EXPERIMENTAL COMPARISON")
    print("=" * 72)
    print(f"Variant: {tg.KEYPOINT_VARIANT.value} ({tg.VARIANT_CONFIG.name})")
    print(f"Raw features: {tg.NRF}, Sequence length: {tg.SEQ}")

    mapping = tg.load_data_mapping()
    if not mapping:
        raise RuntimeError("No data mapping found")

    # Extract/cache landmarks once for all models.
    tg.extract_all(mapping)

    by_label, _ = build_samples(mapping, tg.SEQ, tg.NRF)
    train_samples, val_samples, label_map = stratified_split(
        by_label=by_label,
        min_samples=args.min_samples,
        val_ratio=args.val_ratio,
        max_classes=args.max_classes,
        seed=args.seed,
    )

    n_classes = len(label_map)
    device = tg.setup_gpu()

    # Build shared loaders/stats for fair model comparison.
    mean_full, std_full = compute_norm_stats(train_samples, tg.SEQ, tg.NRF, FULL_FEATURE)
    n_features_full = feature_dim(tg.NRF, FULL_FEATURE)

    train_loader_full, val_loader_full = make_loaders(
        train_samples,
        val_samples,
        tg.SEQ,
        tg.NRF,
        FULL_FEATURE,
        mean_full,
        std_full,
        args.batch_size,
        args.num_workers,
    )

    model_factories: dict[str, Callable[[int, int], nn.Module]] = {
        "GRU": lambda nf, nc: GRUBaseline(nf, num_classes=nc),
        "BiLSTM": lambda nf, nc: SimpleLSTMBaseline(nf, num_classes=nc),
        "Transformer": lambda nf, nc: TransformerBaseline(nf, num_classes=nc),
        "Proposed": lambda nf, nc: ImprovedSignModel(nf, nc),
    }

    comparison_results: dict[str, dict[str, float]] = {}
    for name, factory in model_factories.items():
        comparison_results[name] = train_one_model(
            model_name=name,
            model_factory=factory,
            n_features=n_features_full,
            n_classes=n_classes,
            train_loader=train_loader_full,
            val_loader=val_loader_full,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            mixed_precision=args.mixed_precision,
            device=device,
        )

    print_table("MODEL COMPARISON", comparison_results)
    save_json_csv(
        out_dir / "comparison_results.json",
        out_dir / "comparison_results.csv",
        comparison_results,
        key_name="model",
    )

    # Ablation: proposed variants
    ablation_setups = {
        "Without velocity features": (NO_VELOCITY_FEATURE, lambda nf, nc: ImprovedSignModel(nf, nc)),
        "Without acceleration features": (NO_ACCELERATION_FEATURE, lambda nf, nc: ImprovedSignModel(nf, nc)),
        "Without attention": (FULL_FEATURE, lambda nf, nc: AblationNoAttention(nf, nc)),
        "Full model": (FULL_FEATURE, lambda nf, nc: ImprovedSignModel(nf, nc)),
    }

    ablation_results: dict[str, dict[str, float]] = {}

    for variant_name, (feat_cfg, factory) in ablation_setups.items():
        mean_cfg, std_cfg = compute_norm_stats(train_samples, tg.SEQ, tg.NRF, feat_cfg)
        n_features_cfg = feature_dim(tg.NRF, feat_cfg)
        tr_loader, va_loader = make_loaders(
            train_samples,
            val_samples,
            tg.SEQ,
            tg.NRF,
            feat_cfg,
            mean_cfg,
            std_cfg,
            args.batch_size,
            args.num_workers,
        )

        safe_name = variant_name.replace(" ", "_").replace("/", "_")
        ablation_results[variant_name] = train_one_model(
            model_name=f"Ablation_{safe_name}",
            model_factory=factory,
            n_features=n_features_cfg,
            n_classes=n_classes,
            train_loader=tr_loader,
            val_loader=va_loader,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            mixed_precision=args.mixed_precision,
            device=device,
        )

    print_table("ABLATION RESULTS", ablation_results)
    save_json_csv(
        out_dir / "ablation_results.json",
        out_dir / "ablation_results.csv",
        ablation_results,
        key_name="variant",
    )

    print("\nSaved:")
    print(f"- {out_dir / 'comparison_results.json'}")
    print(f"- {out_dir / 'comparison_results.csv'}")
    print(f"- {out_dir / 'ablation_results.json'}")
    print(f"- {out_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
