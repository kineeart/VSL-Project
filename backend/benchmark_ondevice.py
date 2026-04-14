"""On-Device Performance Benchmark: FPS, Latency, Model Size"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gpu import MODEL_DIR, NF, SEQ


def load_model(model_path, device):
    """Load specific model checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            # Load model structure from checkpoint
            from train_gpu import SignModel
            num_classes = checkpoint.get('num_classes', 100)
            model = SignModel(NF, num_classes).to(device)
            model.load_state_dict(state)
        else:
            # Assume it's just state dict
            from train_gpu import SignModel
            model = SignModel(NF, 100).to(device)
            model.load_state_dict(checkpoint)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def benchmark_model(model, name, device, input_shape=(1, SEQ, NF), num_iterations=100, warmup=10):
    """Benchmark model FPS and latency"""
    if model is None:
        return None
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - t0
    
    avg_latency_ms = (elapsed / num_iterations) * 1000  # milliseconds
    fps = num_iterations / elapsed  # frames per second
    
    return {
        'name': name,
        'latency_ms': round(avg_latency_ms, 3),
        'fps': round(fps, 2),
        'throughput_fps': round(fps, 2)
    }


def get_model_size(model_path):
    """Get model file size in MB and parameter count"""
    if not Path(model_path).exists():
        return None
    
    file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    
    # Load model to count parameters
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in model:
            state = model['model_state_dict']
        else:
            state = model
        
        param_count = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
        param_count_m = param_count / 1_000_000  # millions
        
        return {
            'file_size_mb': round(file_size_mb, 2),
            'parameters_millions': round(param_count_m, 2),
            'parameters': int(param_count)
        }
    except:
        return {'file_size_mb': round(file_size_mb, 2)}


def benchmark_all():
    """Benchmark main model + baselines + ablations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[BENCHMARK] Device: {device}")
    print(f"[BENCHMARK] Input shape: (1, {SEQ}, {NF})")
    print(f"[BENCHMARK] Iterations: 100 (warmup: 10)\n")
    
    results = {}
    
    # Model checkpoints to benchmark
    models_to_test = {
        'Main Model': MODEL_DIR / "sign_model_best.pt",
        'SimpleL STM': MODEL_DIR / "baseline_simplelstm" / "model_best.pt",
        'GRU': MODEL_DIR / "baseline_gru" / "model_best.pt",
        'Transformer': MODEL_DIR / "baseline_transformer" / "model_best.pt",
        'Ablation NoAttention': MODEL_DIR / "ablation_noattention" / "model_best.pt",
        'Ablation NoCosine': MODEL_DIR / "ablation_nocosine" / "model_best.pt",
    }
    
    # Benchmark each model
    for model_name, model_path in models_to_test.items():
        if not Path(model_path).exists():
            print(f"⊘ {model_name}: Model not found ({model_path})")
            continue
        
        print(f"→ Benchmarking {model_name}...")
        model = load_model(model_path, device)
        
        if model is None:
            print(f"  ✗ Failed to load model")
            continue
        
        # FPS/Latency on batch size 1
        bench_bs1 = benchmark_model(model, model_name, device, (1, SEQ, NF))
        if bench_bs1:
            fp = 's'
            print(f"  ✓ Batch 1: FPS={bench_bs1['fps']:.1f}, Latency={bench_bs1['latency_ms']:.1f}ms")
        
        # FPS/Latency on batch size 32
        try:
            bench_bs32 = benchmark_model(model, model_name, device, (32, SEQ, NF), num_iterations=10)
            if bench_bs32:
                print(f"  ✓ Batch 32: FPS={bench_bs32['fps']:.1f}, Latency={bench_bs32['latency_ms']:.1f}ms")
        except:
            bench_bs32 = None
            print(f"  ⚠ Batch 32: OOM or error")
        
        # Model size
        size_info = get_model_size(model_path)
        if size_info:
            print(f"  ✓ Size: {size_info['file_size_mb']:.2f} MB, "
                  f"Params: {size_info.get('parameters_millions', '?'):.2f}M")
        
        results[model_name] = {
            'model_path': str(model_path),
            'throughput_bs1': bench_bs1,
            'throughput_bs32': bench_bs32,
            'model_info': size_info
        }
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save results
    output_file = MODEL_DIR / "ondevice_benchmark_results.json"
    with open(output_file, "w") as f:
        # Convert to serializable format
        results_serializable = {}
        for k, v in results.items():
            results_serializable[k] = {
                'model_path': v['model_path'],
                'throughput_bs1': v['throughput_bs1'],
                'throughput_bs32': v['throughput_bs32'],
                'model_info': v['model_info']
            }
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n[BENCHMARK] Results saved to: {output_file}\n")
    
    # Print summary table
    print("="*100)
    print("ON-DEVICE PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Model':<25} {'FPS (BS=1)':<15} {'Latency (BS=1)':<18} {'Size (MB)':<15} {'Params (M)':<15}")
    print("-"*100)
    
    for model_name, data in results.items():
        bs1_data = data['throughput_bs1']
        size_data = data['model_info']
        
        if bs1_data:
            fps_str = f"{bs1_data['fps']:.1f}"
            latency_str = f"{bs1_data['latency_ms']:.1f} ms"
        else:
            fps_str = "N/A"
            latency_str = "N/A"
        
        if size_data:
            size_str = f"{size_data['file_size_mb']:.2f}"
            params_str = f"{size_data.get('parameters_millions', 0):.2f}"
        else:
            size_str = "N/A"
            params_str = "N/A"
        
        print(f"{model_name:<25} {fps_str:<15} {latency_str:<18} {size_str:<15} {params_str:<15}")
    
    print("="*100)


if __name__ == "__main__":
    benchmark_all()
