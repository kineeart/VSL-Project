"""Train Ablation Variants: No Augmentation, No Attention, No Cosine Classifier"""
import os, sys, json, time, csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp_proc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gpu import (
    setup_gpu, load_data_mapping, extract_all, load_train_val,
    KEYPOINT_VARIANT, MODEL_DIR, EPOCHS, PAT, SWA_EP, MIXUP_ALPHA,
    BS, NF, SEQ, NRF, EXTRA_FEATURES, LR, WARM,
    mixup_data, mixup_criterion, topk_accuracy
)
from models_baseline import (
    AblationNoAttention, AblationNoCosineClassifier, ImprovedSignModel
)

NW = max(1, mp_proc.cpu_count() - 2)


def train_ablation_variant(model, variant_name, X_train, y_train, X_val, y_val,
                          label_map, device, enable_augmentation=True):
    """Train an ablation variant"""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {variant_name}")
    print(f"  Augmentation: {'YES' if enable_augmentation else 'NO'}")
    print(f"{'='*60}\n")
    
    nc = len(label_map)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    use_scaler = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
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
    
    history = []
    
    t0 = time.time()
    for epoch in range(EPOCHS):
        t_ep = time.time()
        
        # Training
        model.train()
        train_correct, train_total = 0, 0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            # Apply augmentation only if enabled
            if enable_augmentation:
                mixed_x, ya, yb_mix, lam = mixup_data(xb, yb, alpha=MIXUP_ALPHA)
            else:
                mixed_x, ya, yb_mix, lam = xb, yb, yb, 1.0
            
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
        
        train_acc = train_correct / max(train_total, 1)
        
        # Validation
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
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_ep
        
        print(f"  Ep {epoch+1:3d}/{EPOCHS}  Train: {train_acc*100:.2f}%  "
              f"Val-Top1: {val_top1*100:.2f}%  Val-Top5: {val_top5*100:.2f}%  "
              f"LR: {lr:.1e}  {elapsed:.0f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'val_top1': float(val_top1),
            'val_top5': float(val_top5),
            'lr': float(lr),
            'time_sec': float(elapsed)
        })
        
        # Save best
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
    
    # Save variant
    variant_dir = Path(MODEL_DIR) / f"ablation_{variant_name.lower().replace(' ', '_')}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_model_state, str(variant_dir / "model_best.pt"))
    torch.save(best_model_state, str(variant_dir / "model.pt"))
    
    with open(variant_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    with open(variant_dir / "training_history.json", "w") as f:
        json.dump({
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "minutes": elapsed_total / 60,
            "num_classes": nc,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs_trained": epoch + 1,
            "variant_name": variant_name,
            "augmentation_enabled": enable_augmentation,
            "per_epoch": history
        }, f, indent=2)
    
    with open(variant_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_acc', 'val_top1', 'val_top5', 'lr', 'time_sec'])
        writer.writeheader()
        writer.writerows(history)
    
    print(f"\n{'='*60}")
    print(f"  COMPLETED: {variant_name}")
    print(f"  Best Val Top-1: {best_val_top1*100:.2f}%")
    print(f"  Best Val Top-5: {best_val_top5*100:.2f}%")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    print(f"  Saved to: {variant_dir}")
    print(f"{'='*60}\n")
    
    return {
        "variant_name": variant_name,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "minutes": elapsed_total / 60,
        "num_epochs": epoch + 1,
        "model_dir": str(variant_dir),
        "enable_augmentation": enable_augmentation
    }


def main():
    """Train all ablation variants"""
    device = setup_gpu()
    
    print("[ABLATION] Loading data...")
    label_map, mapping = load_data_mapping()
    extract_all(mapping)
    X_train, y_train, X_val, y_val, label_map, nc = load_train_val()
    
    input_size = NF
    num_classes = nc
    
    print(f"[ABLATION] Data loaded: Train {X_train.shape}, Val {X_val.shape}, Classes {num_classes}\n")
    
    # Define ablation variants
    # Format: (model_instance, variant_name, augmentation_enabled)
    variants = [
        (ImprovedSignModel(input_size, num_classes), "Full Model (Baseline)", True),
        (ImprovedSignModel(input_size, num_classes), "No Augmentation", False),
        (AblationNoAttention(input_size, num_classes), "No Attention", True),
        (AblationNoCosineClassifier(input_size, num_classes), "No Cosine Classifier", True),
    ]
    
    results = []
    
    for model, variant_name, enable_aug in variants:
        result = train_ablation_variant(model, variant_name, X_train, y_train, X_val, y_val,
                                       label_map, device, enable_augmentation=enable_aug)
        results.append(result)
        
        # Clean up
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save summary
    summary_file = MODEL_DIR / "ablation_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[ABLATION] All ablations completed!")
    print(f"[ABLATION] Summary saved to: {summary_file}\n")
    
    # Print comparison table
    print("="*80)
    print("ABLATION STUDY COMPARISON")
    print("="*80)
    for r in results:
        print(f"{r['variant_name']:30s}  Top1: {r['best_val_top1']*100:6.2f}%  "
              f"Top5: {r['best_val_top5']*100:6.2f}%  Time: {r['minutes']:6.1f}min")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
