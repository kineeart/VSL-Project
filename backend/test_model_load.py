"""
Debug script: Check if model loads correctly with environment variable override.
"""
import os
import sys
from pathlib import Path

# Simulate what start.bat does
model_dir = "backend/models_15cls_run1"
vsl_model_dir = str(Path.cwd() / model_dir)
os.environ["VSL_MODEL_DIR"] = vsl_model_dir

print(f"[TEST] Current dir: {Path.cwd()}")
print(f"[TEST] VSL_MODEL_DIR set to: {vsl_model_dir}")
print(f"[TEST] Directory exists: {Path(vsl_model_dir).exists()}")

# Now import app to see if MODEL_DIR is set correctly
sys.path.insert(0, str(Path.cwd() / "backend"))

import app
print(f"[TEST] app.MODEL_DIR: {app.MODEL_DIR}")
print(f"[TEST] MODEL_DIR exists: {app.MODEL_DIR.exists()}")

# Try loading model
print("\n[TEST] Attempting to load model...")
app.load_model_if_exists()

# Check app module globals (not local scope)
print(f"[TEST] app.model is not None: {app.model is not None}")
print(f"[TEST] app.label_map size: {len(app.label_map)}")
print(f"[TEST] app.label_map: {dict(list(app.label_map.items())[:3]) if app.label_map else 'None'}")

if app.model is None:
    print("[ERROR] Model failed to load!")
    sys.exit(1)
else:
    print("[OK] Model loaded successfully!")
    print(f"[OK] Model type: {type(app.model)}")
    print(f"[OK] Num classes from label_map: {len(app.label_map)}")
