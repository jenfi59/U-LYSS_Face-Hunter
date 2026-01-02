#!/usr/bin/env python3.13
"""
Test rapide: Vérifier que 468 landmarks bruts sont utilisés par défaut
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from fr_core.config import Config
from fr_core.landmark_onnx import LandmarkDetectorONNX
import cv2

def test_default_config():
    """Vérifie que la config par défaut utilise 468"""
    config = Config()
    print(f"✓ Config par défaut:")
    print(f"  num_landmarks = {config.num_landmarks}")
    print(f"  n_landmarks = {config.n_landmarks}")
    print(f"  pca_n_components = {config.pca_n_components}")
    
    assert config.num_landmarks == 468, f"Expected 468, got {config.num_landmarks}"
    assert config.n_landmarks == 468, f"Expected 468, got {config.n_landmarks}"
    print("  ✓ Configuration correcte: 468 landmarks bruts par défaut\n")

def test_detector():
    """Vérifie que le détecteur retourne bien 468 landmarks"""
    print("✓ Test détecteur:")
    detector = LandmarkDetectorONNX()
    print(f"  num_landmarks = {detector.num_landmarks}")
    
    assert detector.num_landmarks == 468, f"Expected 468, got {detector.num_landmarks}"
    print("  ✓ Détecteur configuré pour 468 landmarks\n")
    
    # Test avec une image de test
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠ Pas de caméra disponible pour test réel")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("  ⚠ Impossible de capturer une frame")
        return
    
    print(f"  Frame: {frame.shape}")
    landmarks = detector.detect_landmarks(frame)
    
    if landmarks is None:
        print("  ⚠ Aucun visage détecté")
        return
    
    print(f"  Landmarks détectés: shape = {landmarks.shape}")
    assert landmarks.shape[0] == 468, f"Expected (468, 2), got {landmarks.shape}"
    assert landmarks.shape[1] == 2, f"Expected (468, 2), got {landmarks.shape}"
    print("  ✓ Détection OK: 468 landmarks (x, y) retournés\n")

def test_raw_mediapipe():
    """Vérifie que MediaPipeLite retourne 468 landmarks bruts"""
    from fr_core.mediapipe_lite import MediaPipeLite
    
    print("✓ Test MediaPipeLite:")
    mp = MediaPipeLite()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠ Pas de caméra disponible")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("  ⚠ Impossible de capturer une frame")
        return
    
    # Test avec extract_68=False (défaut)
    result = mp.process_frame(frame, extract_68=False)
    
    if result is None:
        print("  ⚠ Aucun visage détecté")
        return
    
    landmarks_468 = result.get('landmarks_468')
    landmarks_68 = result.get('landmarks_68')
    
    print(f"  landmarks_468: {landmarks_468.shape if landmarks_468 is not None else None}")
    print(f"  landmarks_68: {landmarks_68.shape if landmarks_68 is not None else None}")
    
    assert landmarks_468 is not None, "landmarks_468 doit être présent"
    assert landmarks_468.shape == (468, 3), f"Expected (468, 3), got {landmarks_468.shape}"
    assert landmarks_68 is None, "landmarks_68 ne devrait pas être extrait par défaut"
    
    print("  ✓ MediaPipeLite OK: 468 landmarks bruts (x, y, z) retournés\n")
    print(f"  📊 Exemple de valeurs (landmark #1 - nez):")
    print(f"     x={landmarks_468[1, 0]:.1f}, y={landmarks_468[1, 1]:.1f}, z={landmarks_468[1, 2]:.3f}")

if __name__ == '__main__':
    print("=" * 70)
    print("TEST: Vérification utilisation 468 landmarks bruts (sans mapping)")
    print("=" * 70)
    print()
    
    try:
        test_default_config()
        test_detector()
        test_raw_mediapipe()
        
        print("=" * 70)
        print("✅ TOUS LES TESTS PASSÉS")
        print("   → Configuration: 468 landmarks bruts par défaut")
        print("   → MediaPipe: Sortie directe du modèle ONNX (1404 valeurs)")
        print("   → Aucun mapping 468→68 appliqué par défaut")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ ERREUR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
