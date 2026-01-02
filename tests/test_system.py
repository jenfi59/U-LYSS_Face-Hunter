#!/usr/bin/env python3.11
"""
D-Face Hunter ARM64 v1.0 - Quick Validation Test

This script performs basic validation tests to ensure the system is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("[1/5] Testing imports...")
    try:
        import numpy
        import cv2
        import mediapipe
        import scipy
        import sklearn
        from src.fr_core import VerificationDTW, get_config
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_mediapipe_model():
    """Test that MediaPipe model is present."""
    print("[2/5] Testing MediaPipe model...")
    model_path = Path(__file__).parent.parent / "models" / "mediapipe" / "face_landmarker_v2_with_blendshapes.task"
    if model_path.exists():
        print(f"  ✅ Model found: {model_path}")
        return True
    else:
        print(f"  ❌ Model not found: {model_path}")
        print("     Run: ./install.sh to download the model")
        return False

def test_verification_dtw():
    """Test VerificationDTW class."""
    print("[3/5] Testing VerificationDTW...")
    try:
        from src.fr_core import VerificationDTW
        verifier = VerificationDTW()
        
        # Check spatial mode methods
        has_verify_auto = hasattr(verifier, 'verify_auto')
        has_verify_pose_based = hasattr(verifier, 'verify_pose_based')
        
        if has_verify_auto and has_verify_pose_based:
            print(f"  ✅ VerificationDTW ready (mode: {verifier.config.matching_mode})")
            return True
        else:
            print("  ❌ Missing spatial mode methods")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_config():
    """Test configuration."""
    print("[4/5] Testing configuration...")
    try:
        from src.fr_core import get_config
        config = get_config()
        
        # Check key config values
        assert config.matching_mode == "spatial", "Mode should be 'spatial'"
        assert config.pose_threshold == 3.0, "Pose threshold should be 3.0"
        assert config.num_landmarks == 468, "Should use 468 landmarks"
        
        print(f"  ✅ Configuration OK")
        print(f"     - Mode: {config.matching_mode}")
        print(f"     - Threshold: {config.pose_threshold}")
        print(f"     - Landmarks: {config.num_landmarks}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_spatial_mode_simulation():
    """Test spatial mode with synthetic data."""
    print("[5/5] Testing spatial mode (simulation)...")
    try:
        import numpy as np
        from src.fr_core import VerificationDTW
        
        verifier = VerificationDTW()
        
        # Create synthetic data
        enrolled_landmarks = np.random.rand(50, 468, 3).astype(np.float32)
        enrolled_poses = np.random.rand(50, 3).astype(np.float32) * 30 - 15  # [-15, 15] degrees
        
        probe_landmarks = enrolled_landmarks[:10]  # Use same frames
        probe_poses = enrolled_poses[:10]
        
        # Test verify_auto
        is_match, distance, details = verifier.verify_auto(
            probe_landmarks, probe_poses, enrolled_landmarks, enrolled_poses
        )
        
        if distance < 3.0:  # Should match (same data)
            print(f"  ✅ Spatial mode OK (distance: {distance:.6f})")
            print(f"     - Coverage: {details.get('coverage', 0):.1f}%")
            return True
        else:
            print(f"  ❌ Unexpected distance: {distance}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("  D-Face Hunter ARM64 v1.0 - Validation Tests")
    print("=" * 70 + "\n")
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("MediaPipe Model", test_mediapipe_model()))
    results.append(("VerificationDTW", test_verification_dtw()))
    results.append(("Configuration", test_config()))
    results.append(("Spatial Mode", test_spatial_mode_simulation()))
    
    print("\n" + "=" * 70)
    print("  Test Results")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20s} {status}")
    
    print("=" * 70 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\nSystem ready! Try:")
        print("  python3.11 enroll_interactive.py")
        return 0
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        print("\nPlease check the errors above and run:")
        print("  ./install.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
