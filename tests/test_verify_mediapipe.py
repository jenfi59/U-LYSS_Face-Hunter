#!/usr/bin/env python3
"""
Test verification with MediaPipe landmarks
==========================================

Loads a MediaPipe-enrolled user and verifies using MediaPipe detector.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

from fr_core import get_config
from fr_core.verification_multimodal import MultimodalVerifier
from fr_core.mediapipe_lite import MediaPipeLite


def main():
    config = get_config()
    
    print("\n" + "=" * 70)
    print("  MEDIAPIPE VERIFICATION TEST")
    print("=" * 70 + "\n")
    
    # Check available users
    users_dir = config.users_models_dir
    user_files = list(users_dir.glob("*.npz"))
    
    if not user_files:
        print("[ERROR] No enrolled users found")
        print(f"Directory: {users_dir}")
        return 1
    
    print(f"Available users: {len(user_files)}")
    for f in user_files:
        print(f"  - {f.stem}")
    print()
    
    # Load verifier with enrolled users
    print("[1/4] Loading enrolled users...")
    verifier = MultimodalVerifier()
    gallery = verifier.load_gallery_from_dir(users_dir)
    print(f"  [OK] {len(gallery)} users loaded\n")
    
    # Initialize MediaPipe
    print("[2/4] Loading MediaPipe detector...")
    detector = MediaPipeLite(
        detector_path=str(config.project_root / "models/mediapipe_onnx/face_detector.onnx"),
        mesh_path=str(config.project_root / "models/mediapipe_onnx/face_mesh.onnx"),
        confidence_threshold=0.3
    )
    print("  [OK] MediaPipe loaded\n")
    
    # Open camera
    print("[3/4] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return 1
    
    if config.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.camera_width))
    if config.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.camera_height))
    
    print("  [OK] Camera opened\n")
    
    # Warm-up
    print("Warming up camera (2s)...")
    time.sleep(2)
    
    # Capture and verify
    print("\n[4/4] Capturing and verifying...")
    print("-" * 70)
    
    # Accumulate frames for DTW verification
    verify_frames = 10  # Same as test_user enrollment
    landmarks_buffer = []
    
    attempts = 0
    max_attempts = 200
    verified = False
    
    try:
        while attempts < max_attempts and not verified:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Detect landmarks with MediaPipe
            t0 = time.time()
            result = detector.process_frame(frame, extract_68=True)
            t1 = time.time()
            
            if result is None:
                if attempts % 10 == 0:
                    print(f"  [{attempts:3d}] No face detected...")
                continue
            
            landmarks = result['landmarks_68'][:, :2].astype(np.float32)  # (68, 2)
            landmarks_buffer.append(landmarks)
            
            # Wait until we have enough frames
            if len(landmarks_buffer) < verify_frames:
                if len(landmarks_buffer) % 3 == 0:
                    print(f"  Collecting frames: {len(landmarks_buffer)}/{verify_frames}")
                continue
            
            # Keep only last N frames (sliding window)
            landmarks_buffer = landmarks_buffer[-verify_frames:]
            landmarks_seq = np.stack(landmarks_buffer, axis=0)  # (N, 68, 2)
            
            # Verify with 1:N matching
            t2 = time.time()
            user_id, distance, confidence_level, details = verifier.verify_1_to_n_with_margin(landmarks_seq, gallery)
            t3 = time.time()
            
            inference_time = (t1 - t0) * 1000
            verify_time = (t3 - t2) * 1000
            total_time = (t3 - t0) * 1000
            
            if user_id is not None and confidence_level == "VERIFIED":
                verified = True
                print("\n" + "=" * 70)
                print("  VERIFICATION SUCCESS")
                print("=" * 70 + "\n")
                print(f"User ID: {user_id}")
                print(f"Distance: {distance:.2f}")
                print(f"Confidence Level: {confidence_level}")
                print(f"Confidence: {result['confidence_detection']:.2f}")
                print()
                print(f"MediaPipe detection: {inference_time:.0f}ms")
                print(f"DTW verification:    {verify_time:.0f}ms")
                print(f"Total pipeline:      {total_time:.0f}ms")
                print()
            else:
                if attempts % 10 == 0:
                    print(f"  [{attempts:3d}] {confidence_level} (dist: {distance:.1f})")
            
            time.sleep(0.05)
    
    finally:
        cap.release()
    
    if not verified:
        print("\n[WARNING] No successful verification after", attempts, "attempts")
        print("Possible reasons:")
        print("  - Different person than enrolled users")
        print("  - Poor lighting or angle")
        print("  - Distance threshold too strict")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
