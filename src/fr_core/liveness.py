"""
U-LYSS Face Recognition - Liveness Detection
Anti-spoofing using Eye Aspect Ratio, motion, and texture analysis
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging
from collections import deque

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from .config import get_config

logger = logging.getLogger(__name__)


class LivenessDetector:
    """
    Multi-cue liveness detection for anti-spoofing.
    - Eye Aspect Ratio (EAR) for blink detection
    - Nose tip motion for 3D movement
    - Texture LBP variance for print/screen detection
    """

    def __init__(self):
        """Initialize liveness detector."""
        self.config = get_config()

        # Blink detection state
        self.ear_history = deque(maxlen=10)
        self.blink_count = 0
        self.last_ear = None

        # Motion tracking state
        self.nose_history = deque(maxlen=10)
        self.motion_detected = False

        # Texture analysis state
        self.lbp_variance_history = deque(maxlen=5)

    def reset(self):
        """Reset detector state."""
        self.ear_history.clear()
        self.blink_count = 0
        self.last_ear = None
        self.nose_history.clear()
        self.motion_detected = False
        self.lbp_variance_history.clear()

    def compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio (EAR).

        Args:
            eye_landmarks: Eye landmarks (6, 3) with 3D coords - uses only x,y for 2D distances
                          [outer_corner, top1, top2, inner_corner, bottom2, bottom1]

        Returns:
            EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
        """
        # Use only x,y coordinates (ignore z) for 2D eye aspect ratio
        eye_2d = eye_landmarks[:, :2]
        
        # Vertical distances
        v1 = np.linalg.norm(eye_2d[1] - eye_2d[5])
        v2 = np.linalg.norm(eye_2d[2] - eye_2d[4])

        # Horizontal distance
        h = np.linalg.norm(eye_2d[0] - eye_2d[3])

        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)

        return ear

    def detect_blink(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Detect blink from 68-point landmarks.

        Args:
            landmarks: Full face landmarks (68, 3) with 3D coordinates

        Returns:
            Tuple (blink_detected, current_ear)
        """
        # Extract eye landmarks (dlib 68-point format)
        left_eye = landmarks[36:42]  # Points 36-41
        right_eye = landmarks[42:48]  # Points 42-47

        # Compute EAR for both eyes
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0

        # Update history
        self.ear_history.append(avg_ear)

        # Detect blink (EAR drops below threshold)
        decision_threshold = self.config.liveness_ear_threshold
        blink_detected = False

        if len(self.ear_history) >= 3:
            # Check for blink pattern: normal -> low -> normal
            if (
                self.ear_history[-2] < decision_threshold
                and self.ear_history[-1] > decision_threshold
                and (self.last_ear is None or self.last_ear > decision_threshold)
            ):
                blink_detected = True
                self.blink_count += 1
                logger.debug(f"Blink detected! Total: {self.blink_count}")

        self.last_ear = avg_ear

        return blink_detected, avg_ear

    def detect_motion(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Detect 3D head motion using nose tip tracking.

        Args:
            landmarks: Full face landmarks (68, 3) with 3D coordinates - uses x,y for 2D tracking

        Returns:
            Tuple (motion_detected, displacement_pixels)
        """
        # Nose tip is point 30 in dlib 68-point format (use x,y only)
        nose_tip = landmarks[30, :2]

        # Update history
        self.nose_history.append(nose_tip)

        motion_detected = False
        displacement = 0.0

        if len(self.nose_history) >= 2:
            # Compute displacement from previous frame
            displacement = np.linalg.norm(self.nose_history[-1] - self.nose_history[-2])

            # Check if displacement exceeds threshold
            decision_threshold = self.config.liveness_motion_threshold
            motion_detected = displacement > decision_threshold

        return motion_detected, displacement

    def analyze_texture(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Analyze texture using Local Binary Pattern (LBP) variance.
        Low variance suggests flat surface (print/screen).

        Args:
            face_image: Cropped face region (BGR or grayscale)

        Returns:
            Tuple (lbp_variance, is_live)
        """
        if not HAS_SKIMAGE:
            logger.warning("scikit-image not available, skipping texture analysis")
            return 0.0, True

        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # Resize to fixed size for consistency
        gray_resized = cv2.resize(gray, (128, 128))

        # Compute LBP
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_resized, n_points, radius, method="uniform")

        # Compute variance of LBP
        variance = np.var(lbp)

        # Update history
        self.lbp_variance_history.append(variance)

        # Check if variance is above threshold (indicates texture)
        decision_threshold = self.config.liveness_texture_threshold
        is_live = variance > decision_threshold

        return variance, is_live

    def check_liveness(self, landmarks: np.ndarray, face_image: Optional[np.ndarray] = None) -> dict:
        """
        Comprehensive liveness check combining multiple cues.

        Args:
            landmarks: Full face landmarks (68, 2)
            face_image: Optional face crop for texture analysis

        Returns:
            Dict with liveness results and scores
        """
        results = {
            "is_live": False,
            "confidence": 0.0,
            "blink_detected": False,
            "ear": 0.0,
            "motion_detected": False,
            "motion_displacement": 0.0,
            "texture_variance": 0.0,
            "texture_live": True,
            "blink_count": self.blink_count,
        }

        # 1. Blink detection
        blink_detected, ear = self.detect_blink(landmarks)
        results["blink_detected"] = blink_detected
        results["ear"] = ear

        # 2. Motion detection
        motion_detected, displacement = self.detect_motion(landmarks)
        results["motion_detected"] = motion_detected
        results["motion_displacement"] = displacement

        # 3. Texture analysis (if face image provided)
        texture_live = True
        if face_image is not None:
            texture_var, texture_live = self.analyze_texture(face_image)
            results["texture_variance"] = texture_var
            results["texture_live"] = texture_live

        # Combine cues for final decision
        # Require at least one blink AND texture check pass
        min_blinks = self.config.liveness_min_blinks
        has_enough_blinks = self.blink_count >= min_blinks

        # Calculate confidence score
        confidence = 0.0
        if has_enough_blinks:
            confidence += 0.5
        if motion_detected:
            confidence += 0.2
        if texture_live:
            confidence += 0.3

        results["is_live"] = has_enough_blinks and texture_live
        results["confidence"] = confidence

        return results

    def get_blink_count(self) -> int:
        """Get current blink count."""
        return self.blink_count


def create_liveness_detector() -> LivenessDetector:
    """Create liveness detector."""
    return LivenessDetector()


if __name__ == "__main__":
    # Test liveness detection
    print("Testing liveness detector...")

    detector = create_liveness_detector()

    # Simulate frames with blinking
    n_frames = 50
    for i in range(n_frames):
        # Create dummy landmarks
        landmarks = np.random.randn(68, 2) * 10 + 200

        # Simulate blink at frames 10, 25, 40
        if i in [10, 25, 40]:
            # Reduce eye opening (lower EAR)
            landmarks[36:48] *= 0.5

        # Check liveness
        result = detector.check_liveness(landmarks)

        if result["blink_detected"]:
            print(
                f"Frame {i}: BLINK! EAR={result['ear']:.3f}, "
                f"Total blinks={result['blink_count']}"
            )

    print(f"\nFinal liveness check:")
    print(f"  Total blinks: {detector.get_blink_count()}")
    print(f"  Is live: {result['is_live']}")
    print(f"  Confidence: {result['confidence']:.2f}")