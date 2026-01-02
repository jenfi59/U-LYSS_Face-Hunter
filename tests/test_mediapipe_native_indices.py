#!/usr/bin/env python3.13
"""
Test des vrais indices MediaPipe 468 pour les landmarks clés.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fr_core.landmark_onnx import LandmarkDetectorONNX

# Forcer l'extraction de 468 points pour avoir les indices natifs MediaPipe
detector = LandmarkDetectorONNX(num_landmarks=468)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Indices MediaPipe natifs selon la doc
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
mediapipe_indices = {
    1: "Nez (tip)",
    152: "Menton",
    33: "Oeil gauche (coin externe)",
    133: "Oeil gauche (coin interne)",  
    362: "Oeil droit (coin interne)",
    263: "Oeil droit (coin externe)",
    61: "Bouche gauche",
    291: "Bouche droite",
    0: "Centre nez",
    4: "Pont nez",
}

print("Test des indices MediaPipe natifs")
print("Verifiez que chaque point correspond a la bonne position anatomique")
print("Q pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = detector.extract_landmarks(frame)
    
    if landmarks is not None and len(landmarks) >= 468:
        # Dessiner tous les points testés
        for idx, label in mediapipe_indices.items():
            if idx < len(landmarks):
                x, y = landmarks[idx].astype(int)
                
                # Couleur selon type
                if "Nez" in label:
                    color = (0, 255, 0)  # VERT
                elif "Menton" in label:
                    color = (255, 0, 0)  # BLEU
                elif "Oeil" in label:
                    color = (0, 255, 255)  # JAUNE
                elif "Bouche" in label:
                    color = (255, 0, 255)  # MAGENTA
                else:
                    color = (255, 255, 255)  # BLANC
                
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.putText(frame, f"{idx}", (x+8, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        cv2.putText(frame, "Aucun visage", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("MediaPipe Native Indices", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nIndices MediaPipe natifs testes:")
for idx, label in mediapipe_indices.items():
    print(f"  {idx}: {label}")
