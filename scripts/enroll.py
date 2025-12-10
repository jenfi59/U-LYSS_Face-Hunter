#!/usr/bin/env python3
"""
FR_VERS_JP 2.1 - User Enrollment
=================================

Enroll a new user by capturing facial landmarks.

Usage:
    python scripts/enroll.py <username>

Version: 2.1.0
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fr_core.landmark_utils import extract_landmarks_from_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def enroll_user(username: str, num_frames: int = 10, video_source: int = 0):
    """Enroll a new user.
    
    Parameters
    ----------
    username : str
        User identifier
    num_frames : int
        Number of frames to capture (default: 10)
    video_source : int
        Video source (default: 0 for webcam)
    """
    print(f"\n{'='*70}")
    print(f"  FR_VERS_JP 2.1 - ENROLLMENT")
    print(f"{'='*70}\n")
    print(f"User: {username}")
    print(f"Frames: {num_frames}")
    print(f"\nInstructions:")
    print("  • Look at the camera frontally")
    print("  • Normal lighting")
    print("  • Stay still\n")
    
    input("Press ENTER to start...")
    
    # Extract landmarks
    print(f"\n🎥 Capturing {num_frames} frames...")
    features, n_valid = extract_landmarks_from_video(
        video_source=video_source,
        num_frames=num_frames
    )
    
    if features is None or len(features) == 0:
        print("\n❌ ERROR: Failed to extract landmarks")
        return False
    
    print(f"✓ Captured {n_valid}/{num_frames} valid frames")
    
    # Apply PCA transformation
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    from fr_core import config
    pca = PCA(n_components=config.PCA_N_COMPONENTS)
    pca_sequence = pca.fit_transform(features_scaled)
    
    # Save model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{username}.npz"
    
    import numpy as np
    np.savez(
        model_path,
        dtw_template=pca_sequence,  # Use same key as v2.0
        pca=pca,
        scaler=scaler,
        use_dtw=True,
        metadata={
            'username': username,
            'num_frames': num_frames,
            'n_valid': n_valid,
            'features': 'landmarks_68',
            'version': '2.1.0'
        }
    )
    
    print(f"\n✅ SUCCESS")
    print(f"Model saved: {model_path}")
    print(f"Shape: {pca_sequence.shape}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/enroll.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    success = enroll_user(username)
    
    sys.exit(0 if success else 1)
