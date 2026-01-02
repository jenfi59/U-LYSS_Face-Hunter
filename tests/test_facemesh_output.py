#!/usr/bin/env python3
"""
Test pour vérifier si FaceMesh ONNX sort [x,y,z] ou [y,x,z]
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fr_core.mediapipe_lite import MediaPipeLite

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        return
    
    mp = MediaPipeLite(
        detector_path="models/mediapipe_onnx/face_detector.onnx",
        mesh_path="models/mediapipe_onnx/face_mesh.onnx"
    )
    
    print("\n" + "="*80)
    print("TEST: Ordre des coordonnées FaceMesh")
    print("="*80)
    print("\nCe test affiche 2 versions:")
    print("  VERSION 1 (GAUCHE): landmarks[:, 0]=X, landmarks[:, 1]=Y (actuel)")
    print("  VERSION 2 (DROITE): landmarks[:, 0]=Y, landmarks[:, 1]=X (inversé)")
    print("\nObservez quelle version place correctement les landmarks!")
    print("Appuyez sur 'q' pour quitter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        result = mp.process_frame(frame, extract_68=False)
        
        if result:
            bbox = result['bbox']
            landmarks_original = result['landmarks_468'].copy()
            
            # Version 1: Actuel (X, Y, Z)
            frame1 = frame.copy()
            x, y, bw, bh = bbox
            cv2.rectangle(frame1, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            
            # Dessiner landmarks clés
            key_landmarks = [1, 33, 263, 61, 291]  # nez, oeil gauche, oeil droit, bouche gauche, bouche droite
            colors = [(0, 0, 255), (255, 0, 0), (255, 0, 0), (0, 255, 255), (0, 255, 255)]
            
            for idx, color in zip(key_landmarks, colors):
                lm = landmarks_original[idx]
                cv2.circle(frame1, (int(lm[0]), int(lm[1])), 5, color, -1)
            
            # Afficher coordonnées du nez
            nose = landmarks_original[1]
            cv2.putText(frame1, f"Nez: ({int(nose[0])}, {int(nose[1])})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame1, "V1: X,Y (actuel)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Version 2: Inversé (Y, X, Z)
            frame2 = frame.copy()
            cv2.rectangle(frame2, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            
            # Créer landmarks inversés
            landmarks_swapped = landmarks_original.copy()
            landmarks_swapped[:, 0], landmarks_swapped[:, 1] = landmarks_original[:, 1].copy(), landmarks_original[:, 0].copy()
            
            for idx, color in zip(key_landmarks, colors):
                lm = landmarks_swapped[idx]
                # Vérifier que les coords sont dans l'image
                if 0 <= lm[0] < w and 0 <= lm[1] < h:
                    cv2.circle(frame2, (int(lm[0]), int(lm[1])), 5, color, -1)
            
            # Afficher coordonnées du nez inversé
            nose_swapped = landmarks_swapped[1]
            cv2.putText(frame2, f"Nez: ({int(nose_swapped[0])}, {int(nose_swapped[1])})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame2, "V2: Y,X (inverse)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Afficher côte à côte
            combined = np.hstack([frame1, frame2])
            cv2.imshow("Test FaceMesh: Gauche=Actuel(X,Y) | Droite=Inverse(Y,X) [q pour quitter]", combined)
        else:
            cv2.imshow("Test FaceMesh: Gauche=Actuel(X,Y) | Droite=Inverse(Y,X) [q pour quitter]", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
