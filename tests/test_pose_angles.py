#!/usr/bin/env python3.13
"""
Test diagnostic des angles de pose
Vérifie la correspondance entre pose calculée et mouvement réel
"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from src.fr_core import LandmarkDetectorONNX
import time

def test_pose_calculation():
    """Test interactif pour vérifier les angles"""
    detector = LandmarkDetectorONNX(num_landmarks=68, confidence_threshold=0.7)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("="*60)
    print("TEST DIAGNOSTIC DES ANGLES DE POSE")
    print("="*60)
    print("\nInstructions:")
    print("1. FRONTAL: Regardez la camera de face")
    print("2. YAW+: Tournez la tete vers VOTRE DROITE")
    print("3. YAW-: Tournez la tete vers VOTRE GAUCHE")
    print("4. PITCH+: Levez la tete (regardez vers le HAUT)")
    print("5. PITCH-: Baissez la tete (regardez vers le BAS)")
    print("6. ROLL+: Penchez la tete vers EPAULE DROITE")
    print("7. ROLL-: Penchez la tete vers EPAULE GAUCHE")
    print("\nAppuyez sur ESPACE pour capturer, Q pour quitter")
    print("="*60)
    
    cv2.namedWindow('Pose Test', cv2.WINDOW_NORMAL)
    
    captures = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        result = detector.process_frame(frame, compute_pose=True)
        
        # Afficher feedback
        display = frame.copy()
        h, w = display.shape[:2]
        
        if result and result.get('pose'):
            yaw, pitch, roll = result['pose']
            
            # Afficher angles
            cv2.putText(display, f"YAW:   {yaw:+7.1f} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"PITCH: {pitch:+7.1f} deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"ROLL:  {roll:+7.1f} deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Indicateurs visuels
            # Yaw (gauche/droite)
            yaw_x = int(w/2 + yaw * 2)
            cv2.circle(display, (yaw_x, h - 80), 10, (0, 0, 255), -1)
            cv2.line(display, (w//4, h - 80), (3*w//4, h - 80), (100, 100, 100), 2)
            cv2.putText(display, "YAW: G <- -> D", (w//4, h - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Pitch (haut/bas)
            pitch_y = int(h/2 - pitch * 2)
            cv2.circle(display, (w - 40, pitch_y), 10, (255, 0, 0), -1)
            cv2.line(display, (w - 40, h//4), (w - 40, 3*h//4), (100, 100, 100), 2)
            cv2.putText(display, "P", (w - 50, h//4 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "I", (w - 50, h//4 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "T", (w - 50, h//4 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "C", (w - 50, h//4 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "H", (w - 50, h//4 + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display, "NO FACE DETECTED", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(display, "SPACE: Capturer | Q: Quitter", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Pose Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):  # Space
            if result and result.get('pose'):
                position = input("\nPosition (frontal/yaw+/yaw-/pitch+/pitch-/roll+/roll-): ")
                captures.append({
                    'position': position,
                    'pose': result['pose']
                })
                print(f"[OK] Capture: {position} -> Yaw={result['pose'][0]:.1f} Pitch={result['pose'][1]:.1f} Roll={result['pose'][2]:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyse des captures
    if captures:
        print("\n" + "="*60)
        print("ANALYSE DES CAPTURES")
        print("="*60)
        
        for cap in captures:
            pos = cap['position']
            yaw, pitch, roll = cap['pose']
            print(f"\n{pos.upper():10s}: Yaw={yaw:+7.1f}deg  Pitch={pitch:+7.1f}deg  Roll={roll:+7.1f}deg")
        
        # Vérifications
        print("\n" + "="*60)
        print("VERIFICATIONS")
        print("="*60)
        
        frontal = [c for c in captures if 'frontal' in c['position'].lower()]
        if frontal:
            yaw, pitch, roll = frontal[0]['pose']
            print(f"\nFRONTAL: Yaw={yaw:+.1f} (attendu ~0)")
            if abs(yaw) > 20:
                print(f"  [WARN] Yaw frontal devie de {abs(yaw):.1f}deg!")
        
        yaw_plus = [c for c in captures if 'yaw+' in c['position'].lower()]
        yaw_minus = [c for c in captures if 'yaw-' in c['position'].lower()]
        if yaw_plus and yaw_minus:
            yaw_p = yaw_plus[0]['pose'][0]
            yaw_m = yaw_minus[0]['pose'][0]
            print(f"\nYAW+: {yaw_p:+.1f}deg (attendu >0 pour rotation droite)")
            print(f"YAW-: {yaw_m:+.1f}deg (attendu <0 pour rotation gauche)")
            if yaw_p < 0 or yaw_m > 0:
                print("  [ERROR] YAW INVERSE! Gauche/Droite sont inverses!")
        
        pitch_plus = [c for c in captures if 'pitch+' in c['position'].lower()]
        pitch_minus = [c for c in captures if 'pitch-' in c['position'].lower()]
        if pitch_plus and pitch_minus:
            pitch_p = pitch_plus[0]['pose'][1]
            pitch_m = pitch_minus[0]['pose'][1]
            print(f"\nPITCH+: {pitch_p:+.1f}deg (attendu >0 pour tete levee)")
            print(f"PITCH-: {pitch_m:+.1f}deg (attendu <0 pour tete baissee)")
            if pitch_p < 0 or pitch_m > 0:
                print("  [ERROR] PITCH INVERSE! Haut/Bas sont inverses!")
        
        roll_plus = [c for c in captures if 'roll+' in c['position'].lower()]
        roll_minus = [c for c in captures if 'roll-' in c['position'].lower()]
        if roll_plus and roll_minus:
            roll_p = roll_plus[0]['pose'][2]
            roll_m = roll_minus[0]['pose'][2]
            print(f"\nROLL+: {roll_p:+.1f}deg (attendu >0 pour penche droite)")
            print(f"ROLL-: {roll_m:+.1f}deg (attendu <0 pour penche gauche)")
            if roll_p < 0 or roll_m > 0:
                print("  [ERROR] ROLL INVERSE! Penche gauche/droite sont inverses!")

if __name__ == '__main__':
    test_pose_calculation()
