#!/usr/bin/env python3.13
"""
Test rapide disponibilite backends (ONNX vs TFLite)
"""

import sys

print("=== Test Backend Disponibilite ===\n")

# Test 1: ONNX Runtime
print("1. ONNX Runtime:")
try:
    import onnxruntime as ort
    print(f"   [OK] Disponible (version {ort.__version__})")
    providers = ort.get_available_providers()
    print(f"   Providers: {', '.join(providers)}")
except ImportError as e:
    print(f"   [FAIL] Non disponible: {e}")

print()

# Test 2: TFLite Runtime
print("2. TFLite Runtime:")
try:
    import tflite_runtime.interpreter as tflite
    print("   [OK] tflite_runtime disponible")
    TFLITE_AVAILABLE = True
except ImportError:
    print("   [WARN] tflite_runtime non trouve, essai tensorflow.lite...")
    try:
        import tensorflow.lite as tflite
        import tensorflow as tf
        print(f"   [OK] tensorflow.lite disponible (TF {tf.__version__})")
        TFLITE_AVAILABLE = True
    except ImportError as e:
        print(f"   [FAIL] Non disponible: {e}")
        TFLITE_AVAILABLE = False

print()

# Test 3: Modeles disponibles
print("3. Modeles telecharges:")
import os
from pathlib import Path

# Rechercher le dossier des modèles MediaPipe relatif au dépôt.
project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "models" / "mediapipe"

if models_dir.exists():
    files = os.listdir(models_dir)
    for f in files:
        size_kb = os.path.getsize(models_dir / f) / 1024
        print(f"   [OK] {f} ({size_kb:.1f} KB)")
else:
    print(f"   [FAIL] Dossier non trouvé: {models_dir}")

print()

# Conclusion
print("=== Recommandation ===")
if 'ort' in dir():
    print("[OK] Utiliser backend ONNX (onnxruntime disponible)")
    print("     Action: Convertir TFLite -> ONNX sur machine x86_64")
elif TFLITE_AVAILABLE:
    print("[OK] Utiliser backend TFLite (runtime disponible)")
    print("     Action: Charger directement face_detector.tflite et face_mesh.tflite")
else:
    print("[FAIL] Aucun backend disponible")
    print("       Action: Installer onnxruntime OU tflite-runtime")
    print("       pip install onnxruntime")
    print("       pip install tflite-runtime")
