"""
FR_VERS_JP 2.1 - Core Modules
==============================

Facial recognition system using:
- 68 landmarks (MediaPipe)
- DTW matching
- DDTW (velocity features)
- Liveness detection

Version: 2.1.0
"""

__version__ = "2.1.0"

from fr_core.verification_dtw import verify_dtw, load_model
from fr_core.landmark_utils import extract_landmarks_from_video, is_landmark_model

__all__ = [
    '__version__',
    'verify_dtw',
    'load_model',
    'extract_landmarks_from_video',
    'is_landmark_model',
]
