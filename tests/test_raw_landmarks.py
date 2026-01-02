#!/usr/bin/env python3.13
"""
Test des landmarks bruts du modele ONNX
Pour identifier l'ordre exact des coordonnees
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fr_core.mediapipe_lite import MediaPipeLite

def main():
    print("=== TEST LANDMARKS BRUTS ===\n")
    
    # Initialiser
    detector = MediaPipeLite(
        detector_path="models/face_detector.onnx",
        mesh_path="models/face_mesh.onnx"
    )
    
    # Ouvrir camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR camera")
        return
    
    print("Camera ouverte")
    print("Appuyez sur ESPACE pour analyser")
    print("Appuyez sur 'q' pour quitter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Detecter
        result = detector.process_frame(frame, extract_68=True)
        
        if result:
            bbox = result['bbox']
            landmarks_468 = result['landmarks_468']
            landmarks_68 = result['landmarks_68']
            
            # Dessiner bbox
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"ROI: {bw}x{bh}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Info
            if bw > bh:
                cv2.putText(frame, "PAYSAGE (INCORRECT)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "PORTRAIT (OK)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "ESPACE = analyser", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Test Raw Landmarks', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord(' '):
            if result:
                print("\n" + "="*60)
                print("ANALYSE LANDMARKS BRUTS")
                print("="*60)
                
                bbox = result['bbox']
                landmarks_468 = result['landmarks_468']
                landmarks_68 = result['landmarks_68']
                
                x, y, bw, bh = bbox
                
                print(f"\nFrame: {w}x{h}")
                print(f"ROI bbox: x={x}, y={y}, largeur={bw}, hauteur={bh}")
                print(f"Ratio L/H: {bw/bh:.3f}")
                
                if bw > bh:
                    print("ALERTE: ROI en PAYSAGE (largeur > hauteur)")
                else:
                    print("OK: ROI en PORTRAIT")
                
                # Analyser quelques landmarks cles
                print(f"\nLandmarks 468 (premiers 5 points):")
                for i in range(5):
                    x_lm, y_lm, z_lm = landmarks_468[i]
                    print(f"  [{i}]: x={x_lm:.1f}, y={y_lm:.1f}, z={z_lm:.3f}")
                
                # Landmark 30 (nez) et 8 (menton) pour analyse geometrique
                nez = landmarks_68[30]
                menton = landmarks_68[8]
                oeil_g = landmarks_68[36]
                oeil_d = landmarks_68[45]
                
                print(f"\nLandmarks cles (68-subset):")
                print(f"  Nez [30]:     x={nez[0]:.1f}, y={nez[1]:.1f}")
                print(f"  Menton [8]:   x={menton[0]:.1f}, y={menton[1]:.1f}")
                print(f"  Oeil G [36]:  x={oeil_g[0]:.1f}, y={oeil_g[1]:.1f}")
                print(f"  Oeil D [45]:  x={oeil_d[0]:.1f}, y={oeil_d[1]:.1f}")
                
                # Distances
                dist_yeux_x = abs(oeil_d[0] - oeil_g[0])
                dist_yeux_y = abs(oeil_d[1] - oeil_g[1])
                dist_nez_menton_x = abs(menton[0] - nez[0])
                dist_nez_menton_y = abs(menton[1] - nez[1])
                
                print(f"\nAnalyse geometrique:")
                print(f"  Distance yeux X: {dist_yeux_x:.1f}")
                print(f"  Distance yeux Y: {dist_yeux_y:.1f}")
                print(f"  Distance nez-menton X: {dist_nez_menton_x:.1f}")
                print(f"  Distance nez-menton Y: {dist_nez_menton_y:.1f}")
                
                # Diagnostic
                print(f"\nDiagnostic:")
                if dist_yeux_x < dist_yeux_y:
                    print("  PROBLEME: Yeux plus espaces en Y qu'en X")
                    print("  --> X et Y inverses dans les landmarks!")
                else:
                    print("  OK: Yeux plus espaces en X")
                
                if dist_nez_menton_x > dist_nez_menton_y:
                    print("  PROBLEME: Nez-menton plus long en X qu'en Y")
                    print("  --> X et Y inverses dans les landmarks!")
                else:
                    print("  OK: Nez-menton plus long en Y")
                
                print("="*60 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
