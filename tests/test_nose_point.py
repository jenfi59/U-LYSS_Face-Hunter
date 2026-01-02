#!/usr/bin/env python3.13
"""
Test simple : afficher UNIQUEMENT le point 30 (supposé être le nez)
pour vérifier s'il est au bon endroit après échange X/Y.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fr_core.landmark_onnx import LandmarkDetectorONNX

detector = LandmarkDetectorONNX(num_landmarks=68)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Test du point 30 (nez)")
print("Le point VERT doit être au BOUT DU NEZ")
print("Q pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = detector.extract_landmarks(frame)
    
    if landmarks is not None and len(landmarks) >= 68:
        # Point 30 uniquement
        nose_x, nose_y = landmarks[30].astype(int)
        
        cv2.circle(frame, (nose_x, nose_y), 15, (0, 255, 0), -1)
        cv2.putText(frame, "30:NEZ", (nose_x + 20, nose_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Coords: ({nose_x}, {nose_y})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Test Point 30", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
