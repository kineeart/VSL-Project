"""Train Baseline Models (LSTM, GRU, Transformer) + Ablation Variants"""
import os, sys, json, time, cv2, numpy as np, openpyxl, csv
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp_proc

# Import from train_gpu for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gpu import (
    setup_gpu, load_data_mapping, extract_all, eng_feat, mirror, aug1,
    random_choose_seq, mixup_data, mixup_criterion, topk_accuracy,
    KEYPOINT_VARIANT, VARIANT_CONFIG, SEQ, NRF, EXTRA_FEATURES, NF,
    BS, EPOCHS, PAT, LR, WARM, SWA_EP, MIXUP_ALPHA, MIN_SAMPLES,
    BASE_DIR, VIDEOS_DIR, DATA_FILE, MODEL_DIR, LANDMARKS_DIR, CUSTOM_VIDEOS_DIR,
    HAS_HANDS_POSE, MAX_DATA_ROWS, USE_FIXED_20_CLASSES, FIXED_CLASS_LIMIT,
    FIXED_20_FILES_WEBM, TARGET_SAMPLES_PER_CLASS, SPATIAL_AUG_PROB,
    SPATIAL_SCALE_MIN, SPATIAL_SCALE_MAX, SMOOTH_MOVE_PROB, RANDOM_SHIFT_PROB,
    prep_data, landmark_cache_path
)
from models_baseline import (
    SimpleLSTMBaseline, GRUBaseline, TransformerBaseline,
    AblationNoAttention, AblationNoCosineClassifier
)

NW = max(1, mp_proc.cpu_count() - 2)


# ============================================================
# Shared Training Loop
# ============================================================
def train_baseline_epoch(model, criterion, optimizer, train_loader, device, scaler=None):
    """Train one epoch"""
    model.train()
    train_correct, train_total = 0, 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        # Mixup
        mixed_x, ya, yb_mix, lam = mixup_data(xb, yb, alpha=MIXUP_ALPHA)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
            with torch.amp.autocast('cuda'):
                out = model(mixed_x)
                loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        train_correct += (out.detach().argmax(1) == ya).sum().item()
        train_total += xb.size(0)
    
    return train_correct / max(train_total, 1)


def eval_baseline_epoch(model, val_loader, device):
    """Evaluate one epoch"""
    model.eval()
    val_top1_sum, val_top5_sum, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            t1, t5 = topk_accuracy(out, yb, topk=(1, 5))
            bs = xb.size(0)
            val_top1_sum += t1 * bs
            val_top5_sum += t5 * bs
            val_total += bs
    
    val_top1 = val_top1_sum / max(val_total, 1)
    val_top5 = val_top5_sum / max(val_total, 1)
    return val_top1, val_top5


def train_baseline_model(model, model_name, X_train, y_train, X_val, y_val, 
                        label_map, device, lr_scale=1.0):
    """Train baseline model"""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*60}\n")
    
    nc = len(label_map)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * lr_scale)
    
    use_scaler = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    
    model = model.to(device)
    
    best_val_top1 = 0
    best_val_top5 = 0
    patience_counter = 0
    best_model_state = None
    
    # Training history per epoch
    history = []
    
    t0 = time.time()
    for epoch in range(EPOCHS):
        t_ep = time.time()
        
        train_acc = train_baseline_epoch(model, criterion, optimizer, train_loader, device, scaler)
        val_top1, val_top5 = eval_baseline_epoch(model, val_loader, device)
        
        scheduler.step()
        elapsed = time.time() - t_ep
        lr = optimizer.param_groups[0]['lr']
        
        print(f"  Ep {epoch+1:3d}/{EPOCHS}  Train: {train_acc*100:.2f}%  "
              f"Val-Top1: {val_top1*100:.2f}%  Val-Top5: {val_top5*100:.2f}%  "
              f"LR: {lr:.1e}  {elapsed:.0f}s")
        
        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'val_top1': float(val_top1),
            'val_top5': float(val_top5),
            'lr': float(lr),
            'time_sec': float(elapsed)
        })
        
        # Save best model
        if (val_top1 > best_val_top1) or (val_top1 == best_val_top1 and val_top5 > best_val_top5):
            best_val_top5 = val_top5
            best_val_top1 = val_top1
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"    ^ Best! Top1: {val_top1*100:.2f}%  Top5: {val_top5*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PAT and epoch >= SWA_EP:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    elapsed_total = time.time() - t0
    
    # Save best model
    baseline_dir = Path(MODEL_DIR) / f"baseline_{model_name.lower()}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_model_state, str(baseline_dir / "model_best.pt"))
    torch.save(best_model_state, str(baseline_dir / "model.pt"))
    
    with open(baseline_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    with open(baseline_dir / "training_history.json", "w") as f:
        json.dump({
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "minutes": elapsed_total / 60,
            "num_classes": nc,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs_trained": epoch + 1,
            "model_name": model_name,
            "per_epoch": history
        }, f, indent=2)
    
    # Save per-epoch history as CSV
    with open(baseline_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_acc', 'val_top1', 'val_top5', 'lr', 'time_sec'])
        writer.writeheader()
        writer.writerows(history)
    
    print(f"\n{'='*60}")
    print(f"  COMPLETED: {model_name}")
    print(f"  Best Val Top-1: {best_val_top1*100:.2f}%")
    print(f"  Best Val Top-5: {best_val_top5*100:.2f}%")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    print(f"  Model saved to: {baseline_dir}")
    print(f"{'='*60}\n")
    
    return {
        "model_name": model_name,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "minutes": elapsed_total / 60,
        "num_epochs": epoch + 1,
        "model_dir": str(baseline_dir)
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    device = setup_gpu()
    
    print("[BASELINE] Loading data...")
    mapping = load_data_mapping()
    extract_all(mapping)
    X_train, y_train, X_val, y_val, label_map, nc = prep_data(mapping)
    
    num_classes = nc
    input_size = NF
    
    print(f"[BASELINE] Data loaded: Train {X_train.shape}, Val {X_val.shape}, Classes {num_classes}")
    
    # Define baselines to train
    baselines = [
        ("SimpleLSTM", SimpleLSTMBaseline(input_size, hidden_size=256, num_classes=num_classes)),
        ("GRU", GRUBaseline(input_size, hidden_size=256, num_classes=num_classes)),
        ("Transformer", TransformerBaseline(input_size, hidden_size=256, num_classes=num_classes)),
    ]
    
    results = []
    
    for model_name, model in baselines:
        result = train_baseline_model(model, model_name, X_train, y_train, X_val, y_val,
                                     label_map, device)
        results.append(result)
    
    # Save summary
    summary_file = MODEL_DIR / "baseline_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[BASELINE] All training completed!")
    print(f"[BASELINE] Summary saved to: {summary_file}\n")
    
    # Print comparison
    print("="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    for r in results:
        print(f"{r['model_name']:20s}  Top1: {r['best_val_top1']*100:6.2f}%  "
              f"Top5: {r['best_val_top5']*100:6.2f}%  Time: {r['minutes']:6.1f}min")
    print("="*80)
