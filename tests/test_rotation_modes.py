#!/usr/bin/env python3.13
"""
Test différentes rotations pour trouver laquelle donne un ROI en portrait
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fr_core.landmark_onnx import LandmarkDetectorONNX


def main():
    print("=== TEST ROTATIONS ===")
    print("Test des 4 rotations possibles pour trouver la bonne\n")
    
    detector = LandmarkDetectorONNX()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERREUR camera")
        return
    
    rotation_mode = 0  # 0=none, 1=90CW, 2=180, 3=270CW(90CCW)
    
    print("Controls:")
    print("  r = changer rotation (0/1/2/3)")
    print("  q = quitter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h_orig, w_orig = frame.shape[:2]
        
        # Appliquer rotation
        if rotation_mode == 1:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == 2:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_mode == 3:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame_rot = frame.copy()
        
        h_rot, w_rot = frame_rot.shape[:2]
        
        # Detecter
        result = detector.process_frame(frame_rot, compute_pose=True)
        
        # Afficher info rotation
        rotation_names = ["0° (none)", "90° CW", "180°", "270° CW (90° CCW)"]
        cv2.putText(frame_rot, f"Rotation: {rotation_names[rotation_mode]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_rot, f"Frame: {w_rot}x{h_rot}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if result and result['bbox']:
            bbox = result['bbox']
            x, y, bw, bh = bbox
            
            # Dessiner ROI
            cv2.rectangle(frame_rot, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            
            # Info ROI
            ratio = bw / bh if bh > 0 else 0
            if bw > bh:
                mode_str = f"PAYSAGE ({bw}x{bh}, ratio={ratio:.2f})"
                color = (0, 0, 255)  # Rouge
            else:
                mode_str = f"PORTRAIT ({bw}x{bh}, ratio={ratio:.2f})"
                color = (0, 255, 0)  # Vert
            
            cv2.putText(frame_rot, mode_str, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Afficher pose si disponible
            if result['pose']:
                yaw, pitch, roll = result['pose']
                cv2.putText(frame_rot, f"Yaw:{yaw:+6.1f} Pitch:{pitch:+6.1f} Roll:{roll:+6.1f}",
                           (10, h_rot-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame_rot, "R = changer rotation  Q = quitter", (10, h_rot-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('Test Rotations', frame_rot)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            rotation_mode = (rotation_mode + 1) % 4
            print(f"Rotation: {rotation_names[rotation_mode]}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest termine")


if __name__ == "__main__":
    main()
