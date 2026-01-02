#!/usr/bin/env python3.13
"""
Test de stabilité du calcul de pose AVEC GUI
Capture 30 frames en position frontale statique et analyse variance
"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from src.fr_core import LandmarkDetectorONNX
import time

def draw_pose_feedback(frame, poses_history, current_pose, frame_count, target_frames):
    """Affiche feedback visuel avec historique"""
    h, w = frame.shape[:2]
    display = frame.copy()
    
    if current_pose is None:
        cv2.putText(display, "NO FACE DETECTED", (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return display
    
    yaw, pitch, roll = current_pose
    
    # Titre
    cv2.putText(display, "TEST STABILITE - RESTEZ IMMOBILE", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Angles actuels
    cv2.putText(display, f"YAW:   {yaw:+7.1f} deg", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"PITCH: {pitch:+7.1f} deg", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"ROLL:  {roll:+7.1f} deg", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Statistiques si historique
    if len(poses_history) > 1:
        history_array = np.array(poses_history)
        yaw_std = history_array[:, 0].std()
        pitch_std = history_array[:, 1].std()
        roll_std = history_array[:, 2].std()
        
        cv2.putText(display, f"Std: {yaw_std:.1f}", (250, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display, f"Std: {pitch_std:.1f}", (250, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display, f"Std: {roll_std:.1f}", (250, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Indicateur Yaw (horizontal)
    yaw_center = w // 2
    yaw_x = int(yaw_center + yaw * 2)
    yaw_x = max(50, min(w - 50, yaw_x))  # Clamp
    cv2.line(display, (w//4, h - 150), (3*w//4, h - 150), (100, 100, 100), 3)
    cv2.circle(display, (yaw_center, h - 150), 8, (0, 255, 0), 2)  # Centre
    cv2.circle(display, (yaw_x, h - 150), 12, (0, 0, 255), -1)
    cv2.putText(display, "YAW (G <- 0 -> D)", (w//4, h - 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Indicateur Pitch (vertical)
    pitch_center = h // 2
    pitch_y = int(pitch_center - pitch * 2)
    pitch_y = max(100, min(h - 200, pitch_y))
    cv2.line(display, (w - 50, h//4), (w - 50, 3*h//4), (100, 100, 100), 3)
    cv2.circle(display, (w - 50, pitch_center), 8, (0, 255, 0), 2)
    cv2.circle(display, (w - 50, pitch_y), 12, (255, 0, 0), -1)
    cv2.putText(display, "P", (w - 70, h//4 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "I", (w - 70, h//4 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "T", (w - 70, h//4 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Graphique historique Yaw (mini)
    if len(poses_history) > 1:
        history_array = np.array(poses_history)
        yaw_history = history_array[:, 0]
        
        graph_h = 80
        graph_w = min(300, len(yaw_history) * 10)
        graph_x = 10
        graph_y = h - 100
        
        # Background
        cv2.rectangle(display, (graph_x, graph_y - graph_h), 
                     (graph_x + graph_w, graph_y), (0, 0, 0), -1)
        
        # Scale: -180 to +180
        scale = graph_h / 360.0
        zero_line = graph_y - int(180 * scale)
        cv2.line(display, (graph_x, zero_line), (graph_x + graph_w, zero_line),
                (0, 255, 0), 1)
        
        # Plot points
        for i in range(1, len(yaw_history)):
            x1 = graph_x + int((i-1) * graph_w / len(yaw_history))
            x2 = graph_x + int(i * graph_w / len(yaw_history))
            y1 = graph_y - int((yaw_history[i-1] + 180) * scale)
            y2 = graph_y - int((yaw_history[i] + 180) * scale)
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        cv2.putText(display, "YAW history", (graph_x, graph_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Progression
    progress_pct = (frame_count / target_frames) * 100
    cv2.putText(display, f"Progress: {frame_count}/{target_frames} ({progress_pct:.0f}%)",
               (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Barre de progression
    bar_w = w - 40
    bar_h = 20
    bar_x, bar_y = 20, h - 40
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    filled_w = int((frame_count / target_frames) * bar_w)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)
    
    return display

def test_stability():
    detector = LandmarkDetectorONNX(num_landmarks=468, confidence_threshold=0.7)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create named window with fixed size (3x larger)
    window_name = "Test Stabilite Poses"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1440)  # 640x3=1920, 480x3=1440
    
    print("="*60)
    print("TEST DE STABILITE DES POSES AVEC GUI")
    print("="*60)
    print("\nMettez-vous en position FRONTALE FIXE")
    print("NE BOUGEZ PAS pendant la capture (30 frames)")
    print("\nDemarrage dans 3 secondes...")
    
    cv2.namedWindow('Pose Stability Test', cv2.WINDOW_NORMAL)
    
    time.sleep(3)
    
    # Rotation de la caméra : 0=aucune, 1=90° horaire, 2=180°, 3=270° horaire (90° anti-horaire)
    # Logitech C920 filme en paysage (landscape) : pas de rotation nécessaire
    rotation_mode = 0
    
    poses = []
    target_frames = 30
    
    while len(poses) < target_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Appliquer la rotation si nécessaire
        if rotation_mode == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_mode == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        frame = cv2.flip(frame, 1)
        result = detector.process_frame(frame, compute_pose=True)
        
        current_pose = result.get('pose') if result else None
        
        # Afficher GUI
        display = draw_pose_feedback(frame, poses, current_pose, len(poses), target_frames)
        cv2.imshow(window_name, display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[WARN] Test interrompu")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        if current_pose:
            yaw, pitch, roll = current_pose
            poses.append([yaw, pitch, roll])
            print(f"Frame {len(poses):2d}: Yaw={yaw:+7.1f}  Pitch={pitch:+7.1f}  Roll={roll:+7.1f}")
        
        time.sleep(0.15)
    
    cap.release()
    
    poses = np.array(poses)
    
    print("\n" + "="*60)
    print("RESULTATS (30 frames, position fixe)")
    print("="*60)
    
    for i, name in enumerate(['YAW', 'PITCH', 'ROLL']):
        values = poses[:, i]
        mean = values.mean()
        std = values.std()
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val
        
        print(f"\n{name}:")
        print(f"  Mean:  {mean:+7.1f}deg")
        print(f"  Std:   {std:7.2f}deg")
        print(f"  Range: [{min_val:+7.1f}, {max_val:+7.1f}]deg")
        print(f"  Span:  {range_val:7.2f}deg")
        
        # Diagnostic
        if std > 5:
            print(f"  [ERROR] Trop instable! Std={std:.1f}deg (attendu <5deg)")
        elif std > 2:
            print(f"  [WARN] Instabilite moderee. Std={std:.1f}deg (attendu <2deg)")
        else:
            print(f"  [OK] Stabilite acceptable")
        
        if abs(mean) > 20:
            print(f"  [WARN] Offset frontal: {mean:+.1f}deg (attendu ~0deg)")
    
    # Test wrapping angle
    print("\n" + "="*60)
    print("DETECTION WRAPPING D'ANGLE")
    print("="*60)
    
    yaw_values = poses[:, 0]
    # Vérifier si yaw saute de -180 à +180
    yaw_diffs = np.diff(yaw_values)
    big_jumps = np.where(np.abs(yaw_diffs) > 100)[0]
    
    if len(big_jumps) > 0:
        print(f"\n[ERROR] {len(big_jumps)} sauts detectes dans YAW!")
        print("Cela indique un probleme de wrapping d'angle a +/-180deg")
        for idx in big_jumps[:3]:  # Afficher premiers 3
            print(f"  Frame {idx} -> {idx+1}: {yaw_values[idx]:+.1f}deg -> {yaw_values[idx+1]:+.1f}deg (saut: {yaw_diffs[idx]:+.1f}deg)")
    else:
        print("\n[OK] Pas de saut d'angle detecte")
    
    # Vérifier distribution
    print("\n" + "="*60)
    print("DISTRIBUTION YAW (position frontale attendue a 0deg)")
    print("="*60)
    
    bins = [(-180, -150), (-150, -120), (-120, -90), (-90, -60), (-60, -30),
            (-30, 0), (0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
    
    for start, end in bins:
        count = ((yaw_values >= start) & (yaw_values < end)).sum()
        if count > 0:
            bar = '#' * count
            print(f"  [{start:4d}, {end:4d}[deg : {count:2d} {bar}")
    
    # Conclusion
    print("\n" + "="*60)
    print("DIAGNOSTIC")
    print("="*60)
    
    yaw_std = poses[:, 0].std()
    yaw_mean = poses[:, 0].mean()
    
    if yaw_std > 10:
        print("\n[CRITICAL] Calcul de pose TRES instable!")
        print("Causes possibles:")
        print("  1. Landmarks 2D imprecis (detection MediaPipe defaillante)")
        print("  2. Modele 3D generique inadapte")
        print("  3. Points de reference mal choisis (indices landmarks)")
        print("  4. Probleme d'illumination/qualite image")
    elif abs(yaw_mean) > 90:
        print("\n[CRITICAL] Yaw offset de +/-90deg ou plus!")
        print("Le calcul est probablement INVERSE ou utilise mauvaise convention d'angles")
    else:
        print("\n[INFO] Stabilite acceptable mais peut necessiter calibration")

if __name__ == '__main__':
    test_stability()
