#!/usr/bin/env python3.13
"""
Test simple: Affiche la position du nez et vérifie le centrage
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fr_core.landmark_onnx import LandmarkDetectorONNX

def main():
    print("=== Test Position Landmarks ===")
    print("Centrez votre visage dans le frame")
    print("Le point ROUGE devrait être au centre (320, 240)")
    print()
    
    detector = LandmarkDetectorONNX(num_landmarks=68)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Appuyez sur ESPACE pour capturer et analyser")
    print("Q pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Grille de référence
        for i in range(0, w, 80):
            cv2.line(display, (i, 0), (i, h), (50, 50, 50), 1)
        for i in range(0, h, 60):
            cv2.line(display, (0, i), (w, i), (50, 50, 50), 1)
        
        # Centre théorique
        center_x, center_y = w // 2, h // 2
        cv2.circle(display, (center_x, center_y), 10, (0, 255, 255), 2)
        cv2.putText(display, f"Centre theorique: ({center_x}, {center_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Détecter landmarks
        result = detector.process_frame(frame, compute_pose=False)
        
        if result and result['landmarks'] is not None:
            landmarks = result['landmarks']
            bbox = result['bbox']
            
            # Position du nez (landmark 30)
            if len(landmarks) >= 31:
                nose_x, nose_y = landmarks[30].astype(int)
                
                # Dessiner le nez
                cv2.circle(display, (nose_x, nose_y), 12, (0, 0, 255), -1)
                cv2.circle(display, (nose_x, nose_y), 15, (255, 255, 255), 2)
                
                # Décalage par rapport au centre
                dx = nose_x - center_x
                dy = nose_y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Afficher les infos
                cv2.putText(display, f"Nez detect: ({nose_x}, {nose_y})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display, f"Decalage: dx={dx:+4d}px, dy={dy:+4d}px", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Distance: {distance:.1f}px", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Ligne du centre au nez
                cv2.line(display, (center_x, center_y), (nose_x, nose_y), (255, 0, 255), 2)
                
                # Bbox
                if bbox:
                    x, y, bw, bh = bbox
                    cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                    bbox_center_x = x + bw // 2
                    bbox_center_y = y + bh // 2
                    cv2.circle(display, (bbox_center_x, bbox_center_y), 6, (0, 255, 0), -1)
                    
                    cv2.putText(display, f"BBox: {bw}x{bh} (ratio {bw/bh:.2f})", 
                               (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Test Position Landmarks", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Analyse détaillée
            if result and result['landmarks'] is not None:
                print(f"\n=== ANALYSE ===")
                print(f"Frame size: {w}x{h}")
                print(f"Centre theorique: ({center_x}, {center_y})")
                print(f"Nez detecte: ({nose_x}, {nose_y})")
                print(f"Decalage: dx={dx:+d}px, dy={dy:+d}px")
                print(f"Distance: {distance:.1f}px")
                if bbox:
                    print(f"BBox: x={x}, y={y}, w={bw}, h={bh}")
                    print(f"BBox ratio W/H: {bw/bh:.2f} (devrait etre ~0.75 pour visage)")
                print()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
