"""Test Baseline Models: Verify all models can be instantiated and forward pass works"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_baseline import (
    SimpleLSTMBaseline, GRUBaseline, TransformerBaseline,
    ImprovedSignModel, AblationNoAttention, AblationNoCosineClassifier
)


def test_models(batch_size=4, seq_len=60, input_size=4995, num_classes=100):
    """Test all models"""
    
    print("\n" + "="*70)
    print("TESTING ALL BASELINE MODELS")
    print("="*70 + "\n")
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input features: {input_size}")
    print(f"  Number of classes: {num_classes}\n")
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Models to test
    models = [
        ("SimpleLSTMBaseline", SimpleLSTMBaseline(input_size, hidden_size=256, num_classes=num_classes)),
        ("GRUBaseline", GRUBaseline(input_size, hidden_size=256, num_classes=num_classes)),
        ("TransformerBaseline", TransformerBaseline(input_size, hidden_size=256, num_classes=num_classes)),
        ("ImprovedSignModel", ImprovedSignModel(input_size, num_classes)),
        ("AblationNoAttention", AblationNoAttention(input_size, num_classes)),
        ("AblationNoCosineClassifier", AblationNoCosineClassifier(input_size, num_classes)),
    ]
    
    passed = 0
    failed = 0
    
    print("Testing models...\n")
    
    for model_name, model in models:
        try:
            # Move to device
            model = model.to(device)
            x_device = x.to(device)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(x_device)
            
            # Check output shape
            expected_shape = (batch_size, num_classes)
            actual_shape = tuple(output.shape)
            
            if actual_shape == expected_shape:
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                print(f"✓ {model_name:<30s} Output: {actual_shape}  Params: {num_params:,}")
                passed += 1
            else:
                print(f"✗ {model_name:<30s} WRONG OUTPUT SHAPE: {actual_shape} (expected {expected_shape})")
                failed += 1
        
        except Exception as e:
            print(f"✗ {model_name:<30s} ERROR: {str(e)[:60]}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return passed == len(models)


def test_model_saving_loading():
    """Test model checkpoint saving and loading"""
    print("\n" + "="*70)
    print("TESTING MODEL SAVING/LOADING")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_path = Path(__file__).resolve().parent / "test_model_checkpoint.pt"
    
    try:
        # Create and save model
        model = ImprovedSignModel(4995, 100)
        model = model.to(device)
        
        torch.save(model.state_dict(), str(temp_path))
        print(f"✓ Saved model checkpoint: {temp_path}")
        
        # Load model
        loaded_model = ImprovedSignModel(4995, 100)
        loaded_model.load_state_dict(torch.load(str(temp_path), map_location=device))
        loaded_model = loaded_model.to(device)
        print(f"✓ Loaded model checkpoint")
        
        # Verify forward pass works
        x = torch.randn(2, 60, 4995, device=device)
        with torch.no_grad():
            y = loaded_model(x)
        print(f"✓ Forward pass works: output shape {y.shape}")
        
        # Clean up
        temp_path.unlink()
        print(f"✓ Checkpoint test passed!\n")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}\n")
        if temp_path.exists():
            temp_path.unlink()
        return False


if __name__ == "__main__":
    # Test instantiation and forward pass
    test_passed = test_models()
    
    # Test saving/loading
    save_load_passed = test_model_saving_loading()
    
    if test_passed and save_load_passed:
        print("="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print("="*70)
        print("SOME TESTS FAILED ✗")
        print("="*70 + "\n")
        sys.exit(1)
