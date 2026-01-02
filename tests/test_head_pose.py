#!/usr/bin/env python3
"""
Test Head Pose Estimation (Yaw, Pitch, Roll)
Affiche les angles de la tête en temps réel avec visualisation
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fr_core.mediapipe_lite import MediaPipeLite
from utils.pose_estimation import calculate_head_pose


def draw_pose_info(frame, pose_dict, x=10, y=30):
    """Dessine les informations de pose sur l'image"""
    if pose_dict is None:
        cv2.putText(frame, "Pose: N/A", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    
    yaw = pose_dict['yaw']
    pitch = pose_dict['pitch']
    roll = pose_dict['roll']
    
    # Afficher les angles
    cv2.putText(frame, f"Yaw:   {yaw:+7.2f}deg (gauche/droite)", (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {pitch:+7.2f}deg (haut/bas)", (x, y+30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Roll:  {roll:+7.2f}deg (inclinaison)", (x, y+60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Indicateurs visuels de direction
    y_indicator = y + 100
    
    # Yaw (gauche/droite)
    direction = ""
    color = (0, 255, 0)
    if yaw < -15:
        direction = "<<< GAUCHE"
        color = (255, 0, 0)
    elif yaw > 15:
        direction = "DROITE >>>"
        color = (0, 0, 255)
    else:
        direction = "FRONTAL"
        color = (0, 255, 0)
    
    cv2.putText(frame, direction, (x, y_indicator), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Pitch (haut/bas)
    direction_pitch = ""
    if pitch < -10:
        direction_pitch = "v BAS"
    elif pitch > 10:
        direction_pitch = "^ HAUT"
    else:
        direction_pitch = "CENTRE"
    
    cv2.putText(frame, direction_pitch, (x, y_indicator + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_axes(frame, pose_dict, landmarks_468, image_shape):
    """Dessine les axes 3D de la pose sur l'image"""
    if pose_dict is None or landmarks_468 is None:
        return
    
    height, width = image_shape[:2]
    
    # Origine: bout du nez (landmark 1 pour MediaPipe 468)
    nose_tip = landmarks_468[1].astype(int)
    
    # Longueur des axes
    axis_length = 100
    
    # Angles
    yaw = np.radians(pose_dict['yaw'])
    pitch = np.radians(pose_dict['pitch'])
    roll = np.radians(pose_dict['roll'])
    
    # Matrice de rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Axes dans le repère de la tête
    axes = np.array([
        [axis_length, 0, 0],  # X (rouge) - droite
        [0, axis_length, 0],  # Y (vert) - bas
        [0, 0, axis_length]   # Z (bleu) - avant
    ])
    
    # Appliquer la rotation
    axes_rotated = (R @ axes.T).T
    
    # Projeter sur l'image (approximation simple)
    focal_length = width
    
    # Dessiner les axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Rouge, Vert, Bleu
    labels = ['X', 'Y', 'Z']
    
    for i, (axis, color, label) in enumerate(zip(axes_rotated, colors, labels)):
        # Point de fin de l'axe
        end_point = nose_tip[:2] + axis[:2].astype(int)
        
        # Dessiner la ligne
        cv2.line(frame, tuple(nose_tip[:2]), tuple(end_point), color, 3)
        
        # Ajouter le label
        cv2.putText(frame, label, tuple(end_point + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    # Ouvrir caméra
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        return
    
    # Charger MediaPipe
    print("Chargement MediaPipe...")
    mp = MediaPipeLite(
        detector_path="models/mediapipe_onnx/face_detector.onnx",
        mesh_path="models/mediapipe_onnx/face_mesh.onnx"
    )
    
    print("\n" + "="*80)
    print("TEST HEAD POSE ESTIMATION (Yaw, Pitch, Roll)")
    print("="*80)
    print("\nInstructions:")
    print("- Tournez la tête GAUCHE/DROITE pour tester Yaw")
    print("- Levez/baissez la tête pour tester Pitch")
    print("- Inclinez la tête pour tester Roll")
    print("\nAppuyez sur 'q' pour quitter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Détecter les landmarks
        result = mp.process_frame(frame, extract_68=False)
        
        if result:
            bbox = result['bbox']
            keypoints = result['keypoints']
            landmarks_468 = result.get('landmarks_468')
            
            # Calculer la pose de la tête avec 468 landmarks directs
            if landmarks_468 is not None:
                pose_dict = calculate_head_pose(landmarks_468[:, :2], (h, w))
                
                # Dessiner la bbox des keypoints (cyan)
                kps = np.array(keypoints, dtype=np.float32)
                min_x = int(kps[:, 0].min())
                max_x = int(kps[:, 0].max())
                min_y = int(kps[:, 1].min())
                max_y = int(kps[:, 1].max())
                
                width_kp = max_x - min_x
                height_kp = max_y - min_y
                size = max(width_kp, height_kp)
                
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                
                margin = 0.3
                size_with_margin = int(size * (1 + margin))
                
                roi_x = center_x - size_with_margin // 2
                roi_y = center_y - size_with_margin // 2
                
                cv2.rectangle(frame, 
                            (roi_x, roi_y), 
                            (roi_x + size_with_margin, roi_y + size_with_margin), 
                            (255, 255, 0), 2)
                
                # Dessiner les keypoints
                for i, kp in enumerate(keypoints):
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
                
                # Dessiner quelques landmarks clés (en 468)
                key_landmarks_468 = [1, 152, 33, 263, 61, 291]  # nez, menton, yeux, bouche
                for idx in key_landmarks_468:
                    lm = landmarks_468[idx].astype(int)
                    cv2.circle(frame, tuple(lm[:2]), 4, (0, 255, 255), -1)
                
                # Dessiner les axes 3D
                draw_axes(frame, pose_dict, landmarks_468, (h, w))
                
                # Afficher les infos de pose
                draw_pose_info(frame, pose_dict)
            else:
                cv2.putText(frame, "Landmarks non detectes", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Aucun visage detecte", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Afficher les zones de guidance
        cv2.line(frame, (w//3, 0), (w//3, h), (100, 100, 100), 1)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (100, 100, 100), 1)
        cv2.putText(frame, "GAUCHE", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "FRONTAL", (w//3 + 10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "DROITE", (2*w//3 + 10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Afficher
        cv2.imshow("Head Pose Test - Yaw/Pitch/Roll", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test terminé")


if __name__ == "__main__":
    main()
