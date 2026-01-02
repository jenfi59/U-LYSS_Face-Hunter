#!/usr/bin/env python3.13
"""
Test MediaPipe Lite integration
Validates BlazeFace + FaceMesh pipeline with real camera
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fr_core.mediapipe_lite import MediaPipeLite


def test_mediapipe_lite():
    """Test MediaPipe Lite on camera frame."""
    
    print("=" * 70)
    print("TEST MEDIAPIPE LITE - Integration")
    print("=" * 70)
    
    # Paths
    detector_path = "models/mediapipe_onnx/face_detector.onnx"
    mesh_path = "models/mediapipe_onnx/face_mesh.onnx"
    
    print(f"\n[1/4] Loading models...")
    print(f"  BlazeFace: {detector_path}")
    print(f"  FaceMesh:  {mesh_path}")
    
    try:
        pipeline = MediaPipeLite(
            detector_path=detector_path,
            mesh_path=mesh_path,
            confidence_threshold=0.5
        )
        print("  [OK] Models loaded")
    except Exception as e:
        print(f"  [ERROR] Failed to load models: {e}")
        return False
    
    # Capture frame
    print(f"\n[2/4] Capturing frame from camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("  [ERROR] Cannot capture frame")
        return False
    
    print(f"  [OK] Frame captured: {frame.shape}")
    
    # Test full pipeline
    print(f"\n[3/4] Processing frame (full pipeline)...")
    try:
        result = pipeline.process_frame(frame, extract_68=True)
        
        if result is None:
            print("  [WARNING] No face detected")
            return False
        
        print(f"  [OK] Face detected!")
        print(f"    BBox: {result['bbox']}")
        print(f"    Keypoints (6): {len(result['keypoints'])} points")
        print(f"    Landmarks 468: {result['landmarks_468'].shape}")
        print(f"    Landmarks 68: {result['landmarks_68'].shape}")
        print(f"    Conf Detection: {result['confidence_detection']:.3f}")
        print(f"    Conf Mesh: {result['confidence_mesh']:.3f}")
        
    except Exception as e:
        print(f"  [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate outputs
    print(f"\n[4/4] Validating outputs...")
    
    checks = []
    
    # Check bbox
    x, y, w, h = result['bbox']
    bbox_valid = w > 0 and h > 0 and x >= 0 and y >= 0
    checks.append(("BBox valid", bbox_valid))
    
    # Check keypoints
    kps_valid = len(result['keypoints']) == 6
    checks.append(("Keypoints count", kps_valid))
    
    # Check landmarks 468
    lm468_valid = result['landmarks_468'].shape == (468, 3)
    checks.append(("Landmarks 468 shape", lm468_valid))
    
    # Check landmarks 68
    lm68_valid = result['landmarks_68'].shape == (68, 3)
    checks.append(("Landmarks 68 shape", lm68_valid))
    
    # Check coordinates in frame
    h_frame, w_frame = frame.shape[:2]
    lm68_coords_valid = (
        np.all(result['landmarks_68'][:, 0] >= 0) and
        np.all(result['landmarks_68'][:, 0] <= w_frame) and
        np.all(result['landmarks_68'][:, 1] >= 0) and
        np.all(result['landmarks_68'][:, 1] <= h_frame)
    )
    checks.append(("Landmarks 68 coords in frame", lm68_coords_valid))
    
    # Print checks
    all_passed = True
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] MediaPipe Lite integration working!")
        print("\nNext steps:")
        print("  1. Compare landmarks variance vs dlib")
        print("  2. Test on multiple frames")
        print("  3. Measure inference speed")
        print("  4. Check out-of-bounds percentage (target: 0%)")
    else:
        print("[PARTIAL] Some checks failed, review above")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_mediapipe_lite()
    sys.exit(0 if success else 1)
