#!/usr/bin/env python3.13
"""Test simple: verifier que pose_estimation supporte 468 landmarks"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.pose_estimation import calculate_head_pose

print("="*70)
print("TEST: pose_estimation avec 468 landmarks")
print("="*70)

# Créer des landmarks de test (468 points)
landmarks_468 = np.random.rand(468, 2) * 640  # Coordonnées aléatoires dans [0, 640]
image_shape = (480, 640)

print(f"\nLandmarks shape: {landmarks_468.shape}")
print(f"Image shape: {image_shape}")

# Tester le calcul de pose
try:
    result = calculate_head_pose(landmarks_468, image_shape)
    
    if result is None:
        print("\n[ERREUR] calculate_head_pose a retourne None")
        sys.exit(1)
    
    print(f"\n[OK] calculate_head_pose a fonctionne!")
    print(f"  Yaw: {result.get('yaw', 'N/A'):.1f}°")
    print(f"  Pitch: {result.get('pitch', 'N/A'):.1f}°")
    print(f"  Roll: {result.get('roll', 'N/A'):.1f}°")
    
    print("\n" + "="*70)
    print("SUCCES: pose_estimation supporte bien 468 landmarks")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERREUR] Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
