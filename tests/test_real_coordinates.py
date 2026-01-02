#!/usr/bin/env python3.13
"""Test avec vraies coordonnées de la caméra"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from fr_core.mediapipe_lite import MediaPipeLite

print("="*70)
print("TEST: Verification coordonnees 468 landmarks")
print("="*70)

detector_path = "models/mediapipe_onnx/face_detector.onnx"
mesh_path = "models/mediapipe_onnx/face_mesh.onnx"
mp = MediaPipeLite(detector_path=detector_path, mesh_path=mesh_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERREUR: Impossible d'ouvrir la camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nCapture d'une frame...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERREUR: Impossible de capturer une frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame shape: {frame.shape} (H={h}, W={w})")

# Test avec extract_68=False (défaut: 468 landmarks bruts)
print("\n--- Test 1: extract_68=False (468 landmarks bruts) ---")
result = mp.process_frame(frame, extract_68=False)

if result is None:
    print("ERREUR: Aucun visage detecte")
    sys.exit(1)

bbox = result.get('bbox')
keypoints = result.get('keypoints')
landmarks_468 = result.get('landmarks_468')
landmarks_68 = result.get('landmarks_68')

print(f"\nBBox: {bbox}")
print(f"Keypoints (6): {len(keypoints) if keypoints else None}")
print(f"landmarks_468: {landmarks_468.shape if landmarks_468 is not None else None}")
print(f"landmarks_68: {landmarks_68.shape if landmarks_68 is not None else None}")

if landmarks_468 is None:
    print("\nERREUR: landmarks_468 devrait etre present!")
    sys.exit(1)

if landmarks_68 is not None:
    print("\nATTENTION: landmarks_68 ne devrait pas etre extrait par defaut!")

# Vérifier les coordonnées
print(f"\n--- Verification coordonnees landmarks_468 ---")
print(f"Shape: {landmarks_468.shape}")
print(f"Min X: {landmarks_468[:, 0].min():.1f}")
print(f"Max X: {landmarks_468[:, 0].max():.1f}")
print(f"Min Y: {landmarks_468[:, 1].min():.1f}")
print(f"Max Y: {landmarks_468[:, 1].max():.1f}")
print(f"Min Z: {landmarks_468[:, 2].min():.3f}")
print(f"Max Z: {landmarks_468[:, 2].max():.3f}")

# Vérifier que les coordonnées sont dans les limites de l'image
in_bounds_x = np.all((landmarks_468[:, 0] >= 0) & (landmarks_468[:, 0] <= w))
in_bounds_y = np.all((landmarks_468[:, 1] >= 0) & (landmarks_468[:, 1] <= h))

print(f"\nCoordonnees X dans [0, {w}]: {in_bounds_x}")
print(f"Coordonnees Y dans [0, {h}]: {in_bounds_y}")

# Afficher quelques landmarks clés
print(f"\n--- Landmarks cles (indices MediaPipe 468) ---")
print(f"Landmark #1 (nez):     x={landmarks_468[1, 0]:.1f}, y={landmarks_468[1, 1]:.1f}, z={landmarks_468[1, 2]:.3f}")
print(f"Landmark #33 (oeil G):  x={landmarks_468[33, 0]:.1f}, y={landmarks_468[33, 1]:.1f}, z={landmarks_468[33, 2]:.3f}")
print(f"Landmark #263 (oeil D): x={landmarks_468[263, 0]:.1f}, y={landmarks_468[263, 1]:.1f}, z={landmarks_468[263, 2]:.3f}")
print(f"Landmark #61 (bouche G): x={landmarks_468[61, 0]:.1f}, y={landmarks_468[61, 1]:.1f}, z={landmarks_468[61, 2]:.3f}")
print(f"Landmark #291 (bouche D): x={landmarks_468[291, 0]:.1f}, y={landmarks_468[291, 1]:.1f}, z={landmarks_468[291, 2]:.3f}")
print(f"Landmark #152 (menton): x={landmarks_468[152, 0]:.1f}, y={landmarks_468[152, 1]:.1f}, z={landmarks_468[152, 2]:.3f}")

# Visualiser sur l'image
vis = frame.copy()

# Dessiner bbox
if bbox:
    x, y, w_box, h_box = bbox
    cv2.rectangle(vis, (int(x), int(y)), (int(x+w_box), int(y+h_box)), (0, 255, 0), 2)

# Dessiner les 6 landmarks clés en gros
key_landmarks = {
    1: ((0, 0, 255), "Nez"),
    152: ((255, 255, 0), "Menton"),
    33: ((0, 255, 0), "Oeil G"),
    263: ((0, 255, 0), "Oeil D"),
    61: ((255, 0, 255), "Bouche G"),
    291: ((255, 0, 255), "Bouche D")
}

for idx, (color, label) in key_landmarks.items():
    pt = landmarks_468[idx, :2].astype(int)
    cv2.circle(vis, tuple(pt), 8, color, -1)
    cv2.putText(vis, label, (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Dessiner tous les landmarks en petit gris
for i in range(landmarks_468.shape[0]):
    pt = landmarks_468[i, :2].astype(int)
    cv2.circle(vis, tuple(pt), 1, (128, 128, 128), -1)

cv2.imwrite('/tmp/test_coordinates_468.jpg', vis)
print(f"\n==> Image sauvegardee: /tmp/test_coordinates_468.jpg")

# Test avec extract_68=True pour comparaison
print("\n--- Test 2: extract_68=True (mapping 468->68) ---")
result_68 = mp.process_frame(frame, extract_68=True)

if result_68:
    landmarks_68_mapped = result_68.get('landmarks_68')
    if landmarks_68_mapped is not None:
        print(f"landmarks_68 shape: {landmarks_68_mapped.shape}")
        print(f"\nLandmark #30 (nez dlib):     x={landmarks_68_mapped[30, 0]:.1f}, y={landmarks_68_mapped[30, 1]:.1f}")
        print(f"Landmark #8 (menton dlib):   x={landmarks_68_mapped[8, 0]:.1f}, y={landmarks_68_mapped[8, 1]:.1f}")
        
        # Visualiser les 68
        vis_68 = frame.copy()
        for i in range(landmarks_68_mapped.shape[0]):
            pt = landmarks_68_mapped[i, :2].astype(int)
            cv2.circle(vis_68, tuple(pt), 2, (0, 255, 255), -1)
        
        # Landmarks clés en rouge
        cv2.circle(vis_68, tuple(landmarks_68_mapped[30, :2].astype(int)), 8, (0, 0, 255), -1)  # Nez
        cv2.circle(vis_68, tuple(landmarks_68_mapped[8, :2].astype(int)), 8, (255, 255, 0), -1)  # Menton
        
        cv2.imwrite('/tmp/test_coordinates_68.jpg', vis_68)
        print(f"==> Image sauvegardee: /tmp/test_coordinates_68.jpg")

print("\n" + "="*70)
if in_bounds_x and in_bounds_y:
    print("SUCCES: Coordonnees 468 dans les limites de l'image")
else:
    print("ATTENTION: Certaines coordonnees hors limites!")
print("="*70)
