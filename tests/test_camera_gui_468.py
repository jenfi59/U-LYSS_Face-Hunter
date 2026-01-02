#!/usr/bin/env python3.13
"""
Test interactif: Vérifier les coordonnées 468 landmarks en temps réel
Affiche les landmarks sur la caméra avec les indices clés
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from fr_core.mediapipe_lite import MediaPipeLite
from utils.pose_estimation import calculate_head_pose

print("="*70)
print("TEST CAMERA GUI: Verification coordonnees 468 landmarks")
print("="*70)
print("\nControles:")
print("  - ESPACE: Toggle entre affichage tous landmarks / landmarks cles")
print("  - 'p': Toggle affichage pose (Yaw/Pitch/Roll)")
print("  - 'b': Toggle affichage bbox")
print("  - 'k': Toggle affichage keypoints BlazeFace")
print("  - 'q': Quitter")
print()

detector_path = "models/mediapipe_onnx/face_detector.onnx"
mesh_path = "models/mediapipe_onnx/face_mesh.onnx"

print("Chargement MediaPipe...")
mp = MediaPipeLite(detector_path=detector_path, mesh_path=mesh_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERREUR: Impossible d'ouvrir la camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# États d'affichage
show_all_landmarks = True
show_pose = True
show_bbox = True
show_keypoints = True

# Landmarks clés MediaPipe 468
KEY_LANDMARKS = {
    1: ((0, 0, 255), "Nez"),
    152: ((0, 255, 255), "Menton"),
    33: ((0, 255, 0), "Oeil G"),
    263: ((0, 255, 0), "Oeil D"),
    61: ((255, 0, 255), "Bouche G"),
    291: ((255, 0, 255), "Bouche D")
}

print("\nCamera ouverte. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    vis = frame.copy()
    
    # Traiter avec 468 landmarks
    result = mp.process_frame(frame, extract_68=False)
    
    if result is not None:
        bbox = result.get('bbox')
        keypoints = result.get('keypoints')
        landmarks_468 = result.get('landmarks_468')
        conf_det = result.get('confidence_detection', 0)
        conf_mesh = result.get('confidence_mesh', 0)
        
        # DEBUG: Afficher les coordonnées brutes
        if bbox and keypoints:
            x_box, y_box, w_box, h_box = bbox
            print(f"DEBUG BBox: x={x_box}, y={y_box}, w={w_box}, h={h_box}")
            print(f"DEBUG Keypoint[0]: x={keypoints[0][0]}, y={keypoints[0][1]}")
            if landmarks_468 is not None:
                print(f"DEBUG Landmark #1 (nez): x={landmarks_468[1, 0]:.1f}, y={landmarks_468[1, 1]:.1f}")
                print()
        
        # Dessiner bbox
        if show_bbox and bbox:
            x, y, w_box, h_box = bbox
            cv2.rectangle(vis, (int(x), int(y)), (int(x+w_box), int(y+h_box)), (0, 255, 0), 2)
            cv2.putText(vis, f"Det: {conf_det:.2f}", (int(x), int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dessiner keypoints BlazeFace
        if show_keypoints and keypoints:
            for i, (kx, ky) in enumerate(keypoints):
                cv2.circle(vis, (int(kx), int(ky)), 4, (255, 128, 0), -1)
                cv2.putText(vis, f"K{i}", (int(kx)+5, int(ky)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
        
        if landmarks_468 is not None:
            # Dessiner tous les landmarks en petit gris
            if show_all_landmarks:
                for i in range(landmarks_468.shape[0]):
                    pt = landmarks_468[i, :2].astype(int)
                    cv2.circle(vis, tuple(pt), 1, (128, 128, 128), -1)
            
            # Dessiner les landmarks clés
            for idx, (color, label) in KEY_LANDMARKS.items():
                pt = landmarks_468[idx, :2].astype(int)
                cv2.circle(vis, tuple(pt), 6, color, -1)
                cv2.circle(vis, tuple(pt), 8, (255, 255, 255), 1)
                cv2.putText(vis, f"{idx}", (pt[0]+12, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(vis, label, (pt[0]+12, pt[1]+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Calculer et afficher pose
            if show_pose:
                pose_dict = calculate_head_pose(landmarks_468[:, :2], (h, w))
                if pose_dict:
                    yaw = pose_dict.get('yaw', 0)
                    pitch = pose_dict.get('pitch', 0)
                    roll = pose_dict.get('roll', 0)
                    
                    # Afficher angles en haut à gauche
                    y_offset = 30
                    cv2.putText(vis, f"Yaw:   {yaw:6.1f}deg", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(vis, f"Pitch: {pitch:6.1f}deg", (10, y_offset+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(vis, f"Roll:  {roll:6.1f}deg", (10, y_offset+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Dessiner axes 3D
                    nose_pt = landmarks_468[1, :2].astype(int)
                    axes = pose_dict.get('axes_2d')
                    if axes is not None:
                        # X axis (rouge)
                        cv2.line(vis, tuple(nose_pt), tuple(axes[0].astype(int)), (0, 0, 255), 3)
                        cv2.putText(vis, "X", tuple(axes[0].astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        # Y axis (vert)
                        cv2.line(vis, tuple(nose_pt), tuple(axes[1].astype(int)), (0, 255, 0), 3)
                        cv2.putText(vis, "Y", tuple(axes[1].astype(int)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # Z axis (bleu)
                        cv2.line(vis, tuple(nose_pt), tuple(axes[2].astype(int)), (255, 0, 0), 3)
                        cv2.putText(vis, "Z", tuple(axes[2].astype(int)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Afficher stats en bas
            cv2.putText(vis, f"Landmarks: 468 (bruts MediaPipe)", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis, f"Mesh conf: {conf_mesh:.3f}", (10, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher limites
            min_x, max_x = landmarks_468[:, 0].min(), landmarks_468[:, 0].max()
            min_y, max_y = landmarks_468[:, 1].min(), landmarks_468[:, 1].max()
            cv2.putText(vis, f"X:[{min_x:.0f},{max_x:.0f}] Y:[{min_y:.0f},{max_y:.0f}]",
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(vis, "Aucun visage detecte", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Afficher états
    status_y = 30
    status_x = w - 200
    cv2.putText(vis, f"[ESPACE] Tous: {'ON' if show_all_landmarks else 'OFF'}", 
               (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(vis, f"[P] Pose: {'ON' if show_pose else 'OFF'}", 
               (status_x, status_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(vis, f"[B] BBox: {'ON' if show_bbox else 'OFF'}", 
               (status_x, status_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(vis, f"[K] Keys: {'ON' if show_keypoints else 'OFF'}", 
               (status_x, status_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Test 468 Landmarks - Camera", vis)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        show_all_landmarks = not show_all_landmarks
    elif key == ord('p'):
        show_pose = not show_pose
    elif key == ord('b'):
        show_bbox = not show_bbox
    elif key == ord('k'):
        show_keypoints = not show_keypoints

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("Test termine")
print("="*70)
