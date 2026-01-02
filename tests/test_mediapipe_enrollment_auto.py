#!/usr/bin/env python3
"""
Quick MediaPipe enrollment test (no user interaction)
Captures 10 frames automatically to validate pipeline.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

from fr_core import get_config, VerificationDTW
from fr_core.mediapipe_lite import MediaPipeLite


def main():
    config = get_config()
    
    print("\n" + "=" * 70)
    print("  MEDIAPIPE ENROLLMENT TEST (AUTO)")
    print("=" * 70 + "\n")
    
    # Initialize MediaPipe
    print("[1/4] Loading MediaPipe models...")
    detector = MediaPipeLite(
        detector_path=str(config.project_root / "models/mediapipe_onnx/face_detector.onnx"),
        mesh_path=str(config.project_root / "models/mediapipe_onnx/face_mesh.onnx"),
        confidence_threshold=0.3
    )
    print("  [OK] Models loaded\n")
    
    # Open camera
    print("[2/4] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return 1
    print("  [OK] Camera opened\n")
    
    # Warm-up
    print("[3/4] Camera warm-up (2s)...")
    time.sleep(2)
    print("  [OK] Ready\n")
    
    # Capture 10 frames
    print("[4/4] Capturing 10 frames...")
    landmarks_list = []
    inference_times = []
    oob_count = 0
    frame_width = 2560
    frame_height = 1472
    
    attempts = 0
    max_attempts = 400
    
    while len(landmarks_list) < 10 and attempts < max_attempts:
        attempts += 1
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_width, frame_height = w, h
        
        t0 = time.time()
        result = detector.process_frame(frame, extract_68=True)
        t1 = time.time()
        
        if result is None:
            continue
        
        landmarks = result['landmarks_68'][:, :2]  # (68, 2)
        landmarks_list.append(landmarks.astype(np.float32))
        inference_times.append((t1 - t0) * 1000)
        
        # Check out-of-bounds
        for lm in landmarks:
            x, y = lm
            if x < 0 or x >= w or y < 0 or y >= h:
                oob_count += 1
        
        print(f"  Frame {len(landmarks_list)}/10 captured "
              f"(inference: {inference_times[-1]:.0f}ms, "
              f"conf: {result['confidence_detection']:.2f})")
        
        time.sleep(0.05)
    
    cap.release()
    
    if len(landmarks_list) == 0:
        print("\n[ERROR] Failed to capture any frames")
        return 1
    
    # Statistics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70 + "\n")
    
    seq = np.stack(landmarks_list, axis=0)  # (10, 68, 2)
    
    total_landmarks = len(landmarks_list) * 68
    oob_pct = (oob_count / total_landmarks * 100)
    
    avg_inference = np.mean(inference_times)
    total_time = sum(inference_times) / 1000
    
    print(f"Frames captured: {len(landmarks_list)}/10")
    print(f"Landmarks shape: {seq.shape}")
    print(f"Frame size: {frame_width}x{frame_height}")
    print()
    print(f"Out-of-bounds: {oob_count}/{total_landmarks} ({oob_pct:.1f}%)")
    print(f"Avg inference: {avg_inference:.0f}ms/frame")
    print(f"Total time: {total_time:.1f}s")
    print()
    
    if oob_pct == 0:
        print("[SUCCESS] ZERO out-of-bounds landmarks!")
    elif oob_pct < 11.8:
        print(f"[SUCCESS] Out-of-bounds below dlib baseline (11.8%)")
    else:
        print(f"[WARNING] Out-of-bounds above dlib baseline (11.8%)")
    
    if avg_inference < 500:
        print("[SUCCESS] Inference speed acceptable (<500ms)")
    else:
        print("[WARNING] Inference speed slow (>500ms)")
    
    # Compute variance
    var_x = np.var(seq[:, :, 0], axis=0).mean()
    var_y = np.var(seq[:, :, 1], axis=0).mean()
    var_total = var_x + var_y
    
    print(f"\nIntra-user variance: {var_total:.2f} px^2")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
