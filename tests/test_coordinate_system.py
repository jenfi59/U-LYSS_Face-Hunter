#!/usr/bin/env python3.13
"""
Diagnostic: Test coordinate system and camera orientation.
Displays landmarks with motion tracking to identify axis inversions.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fr_core.landmark_onnx import LandmarkDetectorONNX


def main():
    print("=== Test Systeme de Coordonnees ===")
    print()
    print("Diagnostic des axes de la camera:")
    print("  - Deplacement GAUCHE physique -> ROI doit aller a GAUCHE")
    print("  - Deplacement DROITE physique -> ROI doit aller a DROITE")
    print("  - Deplacement HAUT physique -> ROI doit aller HAUT")
    print("  - Deplacement BAS physique -> ROI doit aller BAS")
    print()
    print("Instructions:")
    print("  1. Positionnez-vous face camera")
    print("  2. Deplacez votre buste lentement GAUCHE/DROITE/HAUT/BAS")
    print("  3. Observez la direction du ROI (rectangle vert)")
    print("  4. Q: Quitter")
    print()
    
    # Initialize detector
    detector = LandmarkDetectorONNX(num_landmarks=68)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la camera")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Track nose position history (for motion visualization)
    nose_history = deque(maxlen=30)
    
    # Reference position (center of first detection)
    ref_nose_pos = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect landmarks
        landmarks = detector.extract_landmarks(frame)
        bbox = detector.detect_face(frame)
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshair
        cv2.line(display_frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)
        cv2.line(display_frame, (0, h//2), (w, h//2), (100, 100, 100), 1)
        
        if landmarks is not None and len(landmarks) >= 68:
            # Get nose tip position (landmark 30)
            nose_x, nose_y = landmarks[30].astype(int)
            
            # Set reference on first detection
            if ref_nose_pos is None:
                ref_nose_pos = (nose_x, nose_y)
            
            # Add to history
            nose_history.append((nose_x, nose_y))
            
            # Draw trajectory
            if len(nose_history) > 1:
                pts = np.array(list(nose_history), dtype=np.int32)
                cv2.polylines(display_frame, [pts], False, (255, 255, 0), 2)
            
            # Draw nose position
            cv2.circle(display_frame, (nose_x, nose_y), 8, (0, 255, 0), -1)
            cv2.circle(display_frame, (nose_x, nose_y), 10, (255, 255, 255), 2)
            
            # Draw reference position
            if ref_nose_pos:
                cv2.circle(display_frame, ref_nose_pos, 6, (0, 0, 255), 2)
                cv2.line(display_frame, ref_nose_pos, (nose_x, nose_y), (255, 0, 255), 2)
            
            # Calculate displacement from reference
            if ref_nose_pos:
                dx = nose_x - ref_nose_pos[0]
                dy = nose_y - ref_nose_pos[1]
                
                # Display displacement
                cv2.putText(display_frame, f"Delta X: {dx:+4d}px (Horizontal)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Delta Y: {dy:+4d}px (Vertical)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Direction indicators
                if abs(dx) > 20:
                    direction_h = "DROITE" if dx > 0 else "GAUCHE"
                    color_h = (0, 255, 0)
                else:
                    direction_h = "CENTRE"
                    color_h = (100, 100, 100)
                    
                if abs(dy) > 20:
                    direction_v = "BAS" if dy > 0 else "HAUT"
                    color_v = (0, 255, 0)
                else:
                    direction_v = "CENTRE"
                    color_v = (100, 100, 100)
                
                cv2.putText(display_frame, f"Horizontal: {direction_h}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_h, 2)
                cv2.putText(display_frame, f"Vertical: {direction_v}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_v, 2)
            
            # Draw bounding box
            if bbox is not None:
                x, y, w_box, h_box = bbox
                cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                
                # Show bbox center
                bbox_center_x = x + w_box // 2
                bbox_center_y = y + h_box // 2
                cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 5, (0, 255, 255), -1)
            
            # Get pose
            pose = detector.estimate_pose(landmarks, frame.shape[:2])
            yaw, pitch, roll = pose
            
            # Display pose
            cv2.putText(display_frame, f"Yaw: {yaw:+.1f}deg", (10, h-90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Pitch: {pitch:+.1f}deg", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Roll: {roll:+.1f}deg", (10, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Aucun visage detecte", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "R: Reset reference | Q: Quitter", (10, h - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Test Coordonnees", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            ref_nose_pos = None
            nose_history.clear()
            print("Reference reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=== DIAGNOSTIC ===")
    print()
    print("Verifications effectuees:")
    print("  1. Deplacement physique GAUCHE -> Delta X negatif ?")
    print("  2. Deplacement physique DROITE -> Delta X positif ?")
    print("  3. Deplacement physique HAUT -> Delta Y negatif ?")
    print("  4. Deplacement physique BAS -> Delta Y positif ?")
    print()
    print("Si les signes ne correspondent PAS:")
    print("  - X inverse: probleme sur axe horizontal (deja corrige normalement)")
    print("  - Y inverse: il faut aussi flipper verticalement")
    print("  - X et Y inverses mais pas coherents: camera tournee de 90deg")


if __name__ == "__main__":
    main()
