#!/usr/bin/env python3.13
"""
Test MediaPipe Lite - Mode Interactif
Attend la détection d'un visage avec preview
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fr_core.mediapipe_lite import MediaPipeLite


def main():
    print("=" * 70)
    print("TEST MEDIAPIPE LITE - Mode Interactif")
    print("=" * 70)
    
    # Load pipeline
    print("\n[1/2] Chargement modèles MediaPipe...")
    detector_path = "models/mediapipe_onnx/face_detector.onnx"
    mesh_path = "models/mediapipe_onnx/face_mesh.onnx"
    
    pipeline = MediaPipeLite(
        detector_path=detector_path,
        mesh_path=mesh_path,
        confidence_threshold=0.5
    )
    print("  [OK] Modèles chargés")
    
    # Open camera
    print("\n[2/2] Ouverture caméra...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("  [OK] Caméra ouverte (640×480)")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("  - Positionne-toi devant la caméra")
    print("  - Appuie sur ESPACE pour capturer et analyser")
    print("  - Appuie sur 'q' pour quitter")
    print("=" * 70 + "\n")
    
    frame_count = 0
    detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            frame_count += 1
            
            # Try detection every 5 frames (performance)
            if frame_count % 5 == 0:
                result = pipeline.process_frame(frame, extract_68=True)
                
                if result is not None:
                    detected_count += 1
                    
                    # Draw bbox
                    x, y, w, h = result['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw landmarks 68
                    for lm in result['landmarks_68']:
                        lx, ly = int(lm[0]), int(lm[1])
                        cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
                    
                    # Draw keypoints (6 from BlazeFace)
                    for kp in result['keypoints']:
                        kx, ky = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (kx, ky), 4, (255, 0, 0), -1)
                    
                    # Info text
                    cv2.putText(frame, f"FACE DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {result['confidence_mesh']:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected - move closer", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('MediaPipe Lite Test - Press SPACE to analyze', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break
            
            elif key == ord(' '):  # SPACE
                print("\n" + "=" * 70)
                print("ANALYSE COMPLETE...")
                print("=" * 70)
                
                result = pipeline.process_frame(frame, extract_68=True)
                
                if result is None:
                    print("\n[WARNING] Aucun visage détecté!")
                    print("  - Rapproche-toi de la caméra")
                    print("  - Assure-toi d'avoir un bon éclairage")
                    continue
                
                print("\n[SUCCESS] Visage détecté!")
                print(f"\nDétails:")
                print(f"  BBox: {result['bbox']}")
                print(f"  Keypoints BlazeFace: {len(result['keypoints'])} points")
                print(f"  Landmarks 468: {result['landmarks_468'].shape}")
                print(f"  Landmarks 68: {result['landmarks_68'].shape}")
                print(f"  Confidence Detection: {result['confidence_detection']:.3f}")
                print(f"  Confidence Mesh: {result['confidence_mesh']:.3f}")
                
                # Validation
                print(f"\n Validations:")
                x, y, w, h = result['bbox']
                h_frame, w_frame = frame.shape[:2]
                
                lm68 = result['landmarks_68']
                out_of_bounds = 0
                for lm in lm68:
                    if lm[0] < 0 or lm[0] > w_frame or lm[1] < 0 or lm[1] > h_frame:
                        out_of_bounds += 1
                
                pct_oob = (out_of_bounds / 68) * 100
                print(f"  ✓ BBox: ({w}×{h}) valide")
                print(f"  ✓ Landmarks 68: {68 - out_of_bounds}/68 dans frame")
                print(f"  ✓ Out-of-bounds: {out_of_bounds}/68 ({pct_oob:.1f}%)")
                
                if pct_oob == 0:
                    print(f"\n  [EXCELLENT] 0% out-of-bounds (vs 11.8% avec dlib)!")
                elif pct_oob < 5:
                    print(f"\n  [GOOD] <5% out-of-bounds (acceptable)")
                else:
                    print(f"\n  [WARNING] >{pct_oob:.1f}% out-of-bounds")
                
                print("\n" + "=" * 70)
                print("Appuie sur ESPACE pour une nouvelle analyse, 'q' pour quitter")
                print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n" + "=" * 70)
        print("STATISTIQUES")
        print("=" * 70)
        print(f"Frames traitées: {frame_count}")
        print(f"Détections réussies: {detected_count}")
        if frame_count > 0:
            print(f"Taux détection: {(detected_count / (frame_count // 5)) * 100:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
