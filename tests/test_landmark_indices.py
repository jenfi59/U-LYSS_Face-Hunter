#!/usr/bin/env python3.13
"""
Diagnostic: Verify MediaPipe landmarks correspondence with dlib 68-point indices.

This script captures a frontal face and displays the key landmarks used for pose
estimation to verify if MediaPipe indices match dlib convention.

Key landmarks to verify (dlib convention):
- 30: Nose tip (should be at the tip of the nose)
- 8:  Chin (should be at the bottom of the chin)
- 36: Left eye left corner (outer corner of left eye)
- 45: Right eye right corner (outer corner of right eye)
- 48: Left mouth corner
- 54: Right mouth corner
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fr_core.landmark_onnx import LandmarkDetectorONNX

def draw_landmark_indices(frame, landmarks, indices_to_show):
    """Draw specific landmark points with their indices."""
    colors = {
        30: (0, 255, 0),    # Nose tip - GREEN
        8:  (255, 0, 0),    # Chin - BLUE
        36: (0, 255, 255),  # Left eye - YELLOW
        45: (0, 255, 255),  # Right eye - YELLOW
        48: (255, 0, 255),  # Left mouth - MAGENTA
        54: (255, 0, 255),  # Right mouth - MAGENTA
        39: (0, 200, 200),  # Left eye inner - CYAN
        42: (0, 200, 200),  # Right eye inner - CYAN
        51: (200, 0, 200),  # Upper lip - PURPLE
        57: (200, 0, 200),  # Lower lip - PURPLE
        17: (255, 255, 0),  # Left eyebrow - CYAN
        26: (255, 255, 0),  # Right eyebrow - CYAN
        27: (0, 255, 128),  # Nose bridge - LIME
        33: (0, 128, 255),  # Nose bottom - ORANGE
    }
    
    labels = {
        30: "30:Nose_tip",
        8:  "8:Chin",
        36: "36:L_eye_outer",
        45: "45:R_eye_outer",
        48: "48:L_mouth",
        54: "54:R_mouth",
        39: "39:L_eye_inner",
        42: "42:R_eye_inner",
        51: "51:Upper_lip",
        57: "57:Lower_lip",
        17: "17:L_eyebrow",
        26: "26:R_eyebrow",
        27: "27:Nose_bridge",
        33: "33:Nose_bottom",
    }
    
    for idx in indices_to_show:
        if idx < len(landmarks):
            x, y = landmarks[idx][:2].astype(int)
            color = colors.get(idx, (255, 255, 255))
            
            # Draw circle
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 7, (0, 0, 0), 2)
            
            # Draw label
            label = labels.get(idx, f"{idx}")
            cv2.putText(frame, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    print("=== Test de correspondance des indices landmarks ===")
    print("Verification si les indices dlib correspondent aux landmarks MediaPipe")
    print()
    print("Landmarks cles attendus (convention dlib 68-point):")
    print("  30: Bout du nez (origine du repere)")
    print("  8:  Menton (bas du visage)")
    print("  36: Coin externe oeil gauche")
    print("  45: Coin externe oeil droit")
    print("  48: Coin bouche gauche")
    print("  54: Coin bouche droit")
    print()
    print("Instructions:")
    print("  1. Positionnez-vous face caméra (frontal)")
    print("  2. Vérifiez visuellement si les points correspondent")
    print("  3. ESPACE: Capturer et sauvegarder l'image")
    print("  4. Q: Quitter")
    print()
    
    # Initialize detector
    detector = LandmarkDetectorONNX(num_landmarks=68)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Indices to visualize (14 points used for pose estimation)
    key_indices = [30, 8, 36, 45, 48, 54, 39, 42, 51, 57, 17, 26, 27, 33]
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect landmarks
        landmarks = detector.extract_landmarks(frame)
        bbox = detector.detect_face(frame)
        
        display_frame = frame.copy()
        
        if landmarks is not None and len(landmarks) >= 68:
            # Draw all 68 landmarks in gray
            for i in range(68):
                x, y = landmarks[i][:2].astype(int)
                cv2.circle(display_frame, (x, y), 2, (128, 128, 128), -1)
            
            # Draw key landmarks with labels
            draw_landmark_indices(display_frame, landmarks, key_indices)
            
            # Draw bounding box
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get pose
            pose = detector.estimate_pose(landmarks, frame.shape[:2])
            yaw, pitch, roll = pose
            
            # Display pose info
            cv2.putText(display_frame, f"Yaw: {yaw:+.1f}deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Pitch: {pitch:+.1f}deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Roll: {roll:+.1f}deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            status_text = "FRONTAL - Verifiez position des landmarks"
            cv2.putText(display_frame, status_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Aucun visage detecte", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "ESPACE: Capturer | Q: Quitter", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Test Landmark Indices", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and landmarks is not None:
            # Save capture
            capture_count += 1
            output_path = project_root / f"landmark_indices_test_{capture_count}.jpg"
            cv2.imwrite(str(output_path), display_frame)
            print(f"Image sauvegardée: {output_path}")
            print(f"  Yaw={yaw:+.1f}° Pitch={pitch:+.1f}° Roll={roll:+.1f}°")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=== Resume ===")
    print(f"Images capturees: {capture_count}")
    print()
    print("Verifications a faire manuellement sur les images:")
    print("  OK Point 30 (VERT) est-il au BOUT DU NEZ ?")
    print("  OK Point 8 (BLEU) est-il au BAS DU MENTON ?")
    print("  OK Points 36/45 (JAUNE) sont-ils aux COINS EXTERNES des yeux ?")
    print("  OK Points 48/54 (MAGENTA) sont-ils aux COINS DE LA BOUCHE ?")
    print()
    print("Si les points ne correspondent PAS:")
    print("  -> MediaPipe ne suit pas la convention dlib 68-point")
    print("  -> Il faut trouver les VRAIS indices MediaPipe pour ces landmarks")
    print("  -> Consulter la doc MediaPipe FaceMesh 468-point topology")


if __name__ == "__main__":
    main()
