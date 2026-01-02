#!/usr/bin/env python3
"""
Compare dlib vs MediaPipe ONNX enrollment quality
==================================================

Captures 45 frames with both backends and compares:
1. Out-of-bounds percentage
2. Intra-user variance (σ²)
3. Inference speed
4. Landmarks stability

Usage:
    python3 scripts/test_enrollment_comparison.py [--frames 45]
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

from fr_core import get_config, LandmarkDetectorONNX
from fr_core.mediapipe_lite import MediaPipeLite


def capture_landmarks_dlib(
    detector: LandmarkDetectorONNX,
    num_frames: int,
    camera_id: int = 0
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Capture landmarks using dlib.
    
    Returns:
        (landmarks_list, inference_times_ms)
    """
    config = get_config()
    cap = cv2.VideoCapture(camera_id)
    
    if config.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.camera_width))
    if config.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.camera_height))
    
    landmarks_list: List[np.ndarray] = []
    inference_times: List[float] = []
    attempts = 0
    max_attempts = num_frames * 40
    
    print(f"[dlib] Capturing {num_frames} frames...")
    
    try:
        while len(landmarks_list) < num_frames and attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            
            t0 = time.time()
            result = detector.process_frame(frame)
            t1 = time.time()
            
            if result is None:
                continue
            
            landmarks = result["landmarks"][:, :2].astype(np.float32)  # (68, 2)
            landmarks_list.append(landmarks)
            inference_times.append((t1 - t0) * 1000)  # ms
            
            print(f"  Frame {len(landmarks_list)}/{num_frames} ({inference_times[-1]:.1f}ms)")
            time.sleep(0.05)
    finally:
        cap.release()
    
    return landmarks_list, inference_times


def capture_landmarks_mediapipe(
    detector: MediaPipeLite,
    num_frames: int,
    camera_id: int = 0
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Capture landmarks using MediaPipe ONNX.
    
    Returns:
        (landmarks_list, inference_times_ms)
    """
    config = get_config()
    cap = cv2.VideoCapture(camera_id)
    
    if config.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.camera_width))
    if config.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.camera_height))
    
    landmarks_list: List[np.ndarray] = []
    inference_times: List[float] = []
    attempts = 0
    max_attempts = num_frames * 40
    
    print(f"[MediaPipe] Capturing {num_frames} frames...")
    
    try:
        while len(landmarks_list) < num_frames and attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            
            t0 = time.time()
            result = detector.process_frame(frame, extract_68=True)
            t1 = time.time()
            
            if result is None:
                continue
            
            landmarks = result['landmarks_68'][:, :2].astype(np.float32)  # (68, 2)
            landmarks_list.append(landmarks)
            inference_times.append((t1 - t0) * 1000)  # ms
            
            print(f"  Frame {len(landmarks_list)}/{num_frames} ({inference_times[-1]:.1f}ms)")
            time.sleep(0.05)
    finally:
        cap.release()
    
    return landmarks_list, inference_times


def compute_out_of_bounds(
    landmarks_list: List[np.ndarray],
    frame_width: int,
    frame_height: int
) -> Dict[str, float]:
    """
    Compute out-of-bounds statistics.
    
    Returns:
        {
            'total_landmarks': int,
            'out_of_bounds': int,
            'percentage': float
        }
    """
    total = 0
    oob = 0
    
    for landmarks in landmarks_list:
        for lm in landmarks:
            x, y = lm
            total += 1
            if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
                oob += 1
    
    return {
        'total_landmarks': total,
        'out_of_bounds': oob,
        'percentage': (oob / total * 100) if total > 0 else 0.0
    }


def compute_variance(landmarks_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute intra-user variance.
    
    Returns:
        {
            'variance_x': float,
            'variance_y': float,
            'variance_total': float
        }
    """
    if len(landmarks_list) == 0:
        return {'variance_x': 0.0, 'variance_y': 0.0, 'variance_total': 0.0}
    
    # Stack all landmarks: (N_frames, 68, 2)
    landmarks_array = np.stack(landmarks_list, axis=0)
    
    # Compute variance per landmark point across frames
    var_x = np.var(landmarks_array[:, :, 0], axis=0).mean()  # Average variance of x coords
    var_y = np.var(landmarks_array[:, :, 1], axis=0).mean()  # Average variance of y coords
    var_total = var_x + var_y
    
    return {
        'variance_x': float(var_x),
        'variance_y': float(var_y),
        'variance_total': float(var_total)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare dlib vs MediaPipe enrollment quality")
    parser.add_argument("--frames", type=int, default=45, help="Number of frames to capture (default 45)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    
    config = get_config()
    
    print("\n" + "=" * 80)
    print("  ENROLLMENT COMPARISON: dlib vs MediaPipe ONNX")
    print("=" * 80 + "\n")
    print(f"Frames to capture: {args.frames}")
    print(f"Frame size: {config.camera_width}x{config.camera_height}")
    print("\n")
    
    input("Press ENTER to start dlib capture...")
    
    # 1. Capture with dlib
    detector_dlib = LandmarkDetectorONNX(conf_threshold=0.7)
    landmarks_dlib, times_dlib = capture_landmarks_dlib(detector_dlib, args.frames, args.camera)
    
    print("\n[PAUSE] Position yourself again for MediaPipe capture...")
    time.sleep(3)
    
    input("Press ENTER to start MediaPipe capture...")
    
    # 2. Capture with MediaPipe
    detector_mediapipe = MediaPipeLite(
        detector_path=str(config.project_root / "models/mediapipe_onnx/face_detector.onnx"),
        mesh_path=str(config.project_root / "models/mediapipe_onnx/face_mesh.onnx"),
        confidence_threshold=0.3
    )
    landmarks_mediapipe, times_mediapipe = capture_landmarks_mediapipe(
        detector_mediapipe, args.frames, args.camera
    )
    
    # 3. Compute statistics
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80 + "\n")
    
    # Out-of-bounds
    oob_dlib = compute_out_of_bounds(landmarks_dlib, config.camera_width, config.camera_height)
    oob_mediapipe = compute_out_of_bounds(landmarks_mediapipe, config.camera_width, config.camera_height)
    
    print("1. OUT-OF-BOUNDS ANALYSIS")
    print("-" * 40)
    print(f"  dlib:      {oob_dlib['out_of_bounds']}/{oob_dlib['total_landmarks']} "
          f"({oob_dlib['percentage']:.1f}%)")
    print(f"  MediaPipe: {oob_mediapipe['out_of_bounds']}/{oob_mediapipe['total_landmarks']} "
          f"({oob_mediapipe['percentage']:.1f}%)")
    
    improvement_oob = oob_dlib['percentage'] - oob_mediapipe['percentage']
    if improvement_oob > 0:
        print(f"  => IMPROVEMENT: -{improvement_oob:.1f}%")
    elif improvement_oob < 0:
        print(f"  => REGRESSION: +{abs(improvement_oob):.1f}%")
    else:
        print(f"  => SAME: {oob_mediapipe['percentage']:.1f}%")
    
    # Variance
    var_dlib = compute_variance(landmarks_dlib)
    var_mediapipe = compute_variance(landmarks_mediapipe)
    
    print("\n2. INTRA-USER VARIANCE")
    print("-" * 40)
    print(f"  dlib:      σ² = {var_dlib['variance_total']:.2f} px²")
    print(f"  MediaPipe: σ² = {var_mediapipe['variance_total']:.2f} px²")
    
    if var_dlib['variance_total'] > 0:
        var_change = ((var_mediapipe['variance_total'] - var_dlib['variance_total']) 
                      / var_dlib['variance_total'] * 100)
        if var_change < 0:
            print(f"  => IMPROVEMENT: {var_change:.1f}% (lower variance)")
        else:
            print(f"  => REGRESSION: +{var_change:.1f}% (higher variance)")
    
    # Speed
    avg_time_dlib = np.mean(times_dlib)
    avg_time_mediapipe = np.mean(times_mediapipe)
    
    print("\n3. INFERENCE SPEED")
    print("-" * 40)
    print(f"  dlib:      {avg_time_dlib:.1f} ms/frame")
    print(f"  MediaPipe: {avg_time_mediapipe:.1f} ms/frame")
    
    speed_ratio = avg_time_mediapipe / avg_time_dlib
    print(f"  => MediaPipe is {speed_ratio:.1f}x slower")
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    if oob_mediapipe['percentage'] == 0 and oob_dlib['percentage'] > 0:
        print("  [SUCCESS] MediaPipe eliminates out-of-bounds landmarks!")
    
    if var_mediapipe['variance_total'] < var_dlib['variance_total']:
        print("  [SUCCESS] MediaPipe reduces variance!")
    
    if avg_time_mediapipe < 500:
        print("  [OK] MediaPipe speed acceptable (<500ms/frame)")
    else:
        print("  [WARNING] MediaPipe slower than target (>500ms/frame)")
    
    print()


if __name__ == "__main__":
    main()
