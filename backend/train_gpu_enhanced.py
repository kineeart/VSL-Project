"""Enhanced Training Script with Per-Epoch CSV Logging
This is a wrapper/extension of train_gpu.py that adds comprehensive per-epoch metric logging to CSV.
Run this instead of train_gpu.py to get detailed epoch-by-epoch records for convergence analysis.
"""
import os, sys, json, time, csv
from pathlib import Path
import numpy as np
import torch

# Import everything from the original train_gpu
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gpu import (
    setup_gpu, load_data_mapping, extract_all, prep_data, eng_feat, 
    KEYPOINT_VARIANT, MODEL_DIR, EPOCHS, PAT, SWA_EP, MIXUP_ALPHA,
    BS, NF, SEQ, NRF, EXTRA_FEATURES, HAS_HANDS_POSE, LR, WARM,
    SignModel, MultiHeadAttention, CosineClassifier,
    mixup_data, mixup_criterion, topk_accuracy
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn


def train_enhanced():
    """Enhanced training with per-epoch CSV logging"""
    device = setup_gpu()
    
    # Load data
    print("[TRAIN] Loading data...")
    mapping = load_data_mapping()
    extract_all(mapping)
    X_train, y_train, X_val, y_val, label_map, nc = prep_data(mapping)
    
    print(f"[TRAIN] Data loaded: Train {X_train.shape}, Val {X_val.shape}, Classes {nc}")
    
    # Datasets and loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = SignModel(NF, nc).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Stochastic Weight Averaging
    swa_model = AveragedModel(model)
    
    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Tracking
    best_val_top1 = 0
    best_val_top5 = 0
    patience_counter = 0
    swa_active = False
    
    # Per-epoch history
    per_epoch_history = []
    
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        t_ep = time.time()
        
        # ====== TRAINING PHASE ======
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        
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
            
            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (out.detach().argmax(1) == ya).sum().item()
            train_total += xb.size(0)
        
        # Activate SWA after SWA_EP
        if epoch >= SWA_EP and not swa_active:
            swa_active = True
            print(f"[TRAIN] SWA activated at epoch {epoch+1}")
        
        if swa_active:
            swa_model.update_parameters(model)
        
        train_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        
        # ====== VALIDATION PHASE ======
        eval_model = swa_model if swa_active else model
        eval_model.eval()
        val_top1_sum, val_top5_sum, val_loss_sum, val_total = 0, 0, 0.0, 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = eval_model(xb)
                loss = criterion(out, yb)
                t1, t5 = topk_accuracy(out, yb, topk=(1, 5))
                bs = xb.size(0)
                val_top1_sum += t1 * bs
                val_top5_sum += t5 * bs
                val_loss_sum += loss.item() * bs
                val_total += bs
        
        val_top1 = val_top1_sum / max(val_total, 1)
        val_top5 = val_top5_sum / max(val_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        
        # Learning rate scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - t_ep
        
        # Print epoch info
        print(f"  Ep {epoch+1:3d}/{EPOCHS}  Train: {train_acc*100:.2f}% (loss {train_loss:.4f})  "
              f"Val-Top1: {val_top1*100:.2f}%  Val-Top5: {val_top5*100:.2f}% (loss {val_loss:.4f})  "
              f"LR: {lr:.1e}  {elapsed:.0f}s")
        
        # ====== RECORD EPOCH ======
        per_epoch_history.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'train_loss': float(train_loss),
            'val_top1': float(val_top1),
            'val_top5': float(val_top5),
            'val_loss': float(val_loss),
            'lr': float(lr),
            'time_sec': float(elapsed),
            'swa_active': swa_active
        })
        
        # ====== SAVE BEST MODEL ======
        if (val_top1 > best_val_top1) or (val_top1 == best_val_top1 and val_top5 > best_val_top5):
            best_val_top5 = val_top5
            best_val_top1 = val_top1
            patience_counter = 0
            state = swa_model.module.state_dict() if swa_active else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'num_classes': nc,
                'num_features': NF,
                'sequence_length': SEQ,
                'num_raw_features': NRF,
                'keypoint_variant': KEYPOINT_VARIANT.value,
            }, str(MODEL_DIR / "sign_model_best.pt"))
            print(f"    ^ Best! Top1: {val_top1*100:.2f}%  Top5: {val_top5*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PAT and epoch >= SWA_EP:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    # ====== FINALIZE SWA ======
    if swa_active:
        print("\n  Updating SWA batch normalization...")
        update_bn(train_loader, swa_model, device=device)
    
    # Save final model
    best = torch.load(str(MODEL_DIR / "sign_model_best.pt"), weights_only=True)
    torch.save(best, str(MODEL_DIR / "sign_model.pt"))
    
    # Save label map
    with open(MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    elapsed_total = time.time() - t0
    
    # ====== SAVE TRAINING HISTORY ======
    # JSON format
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump({
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "minutes": elapsed_total / 60,
            "num_classes": nc,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs_trained": epoch + 1,
            "per_epoch": per_epoch_history
        }, f, indent=2)
    
    # CSV format for easy analysis
    with open(MODEL_DIR / "training_history_per_epoch.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_acc', 'train_loss', 'val_top1', 'val_top5', 'val_loss', 'lr', 'time_sec', 'swa_active'
        ])
        writer.writeheader()
        writer.writerows(per_epoch_history)
    
    print(f"\n{'='*60}")
    print(f"  HOAN TAT (Enhanced Logging)!")
    print(f"  Thoi gian: {elapsed_total/60:.1f} phut")
    print(f"  So lop: {nc}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Best Val Top-1: {best_val_top1*100:.2f}%")
    print(f"  Best Val Top-5: {best_val_top5*100:.2f}%")
    print(f"  Model: {MODEL_DIR / 'sign_model.pt'}")
    print(f"  Per-epoch history CSV: {MODEL_DIR / 'training_history_per_epoch.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_enhanced()
