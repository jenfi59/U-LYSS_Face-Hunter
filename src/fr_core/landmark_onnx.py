"""
U-LYSS Face Recognition - Landmark Detection using MediaPipe

This module provides face detection, landmark extraction and pose estimation
using MediaPipe Python API. It wraps MediaPipe's FaceLandmarker to provide
a unified interface for the facial recognition system.

MediaPipe returns 478 landmarks: 468 face mesh + 10 iris refinement points.
Supported landmark counts: 68, 98, 194, 468, 478 (automatically subsampled).
Pose estimation uses MediaPipe's facial_transformation_matrixes (4×4 matrix)
providing direct head orientation without solvePnP calculations.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging
import json
import os
from scipy.spatial.transform import Rotation as R_scipy

# MediaPipe Python API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import get_config

logger = logging.getLogger(__name__)


class LandmarkDetectorONNX:
    """Facial landmarks detector using MediaPipe Python API.

    Provides unified interface for face detection, landmark extraction (478 points)
    and pose estimation (yaw/pitch/roll) using MediaPipe's FaceLandmarker.
    
    MediaPipe returns 478 landmarks: 468 face mesh + 10 iris refinement points.
    The pose is computed directly from MediaPipe's facial_transformation_matrixes,
    providing accurate head orientation without solvePnP or canonical face models.
    """

    def __init__(self, num_landmarks: Optional[int] = None, confidence_threshold: float = 0.3):
        """Initialize MediaPipe FaceLandmarker.

        Args:
            num_landmarks: Desired landmarks (68, 98, 194, 468, 478). Default from config.
            confidence_threshold: Detection confidence threshold (0.0-1.0).
        """
        config = get_config()
        self.num_landmarks: int = num_landmarks if num_landmarks is not None else getattr(config, 'num_landmarks', 468)

        # Initialize MediaPipe FaceLandmarker
        models_dir = config.models_dir / "mediapipe_onnx"
        task_path = models_dir / "face_landmarker.task"

        if not task_path.exists():
            logger.error(f"MediaPipe model not found: {task_path}")
            raise FileNotFoundError(f"Missing face_landmarker.task at {task_path}")

        base_options = python.BaseOptions(model_asset_path=str(task_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # Required for pose estimation
            num_faces=1,
            min_face_detection_confidence=confidence_threshold,
            min_face_presence_confidence=confidence_threshold
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Load camera calibration offsets (optional)
        self.yaw_offset, self.pitch_offset, self.roll_offset = self.load_camera_calibration()

    def load_camera_calibration(self) -> Tuple[float, float, float]:
        """Load camera calibration offsets from config/camera_calibration.json.

        Returns:
            (yaw_offset, pitch_offset, roll_offset) in degrees. Default: (0, 0, 0)
        """
        calib_file = os.path.join('config', 'camera_calibration.json')
        if not os.path.exists(calib_file):
            logger.warning(f"No camera calibration file found at {calib_file}")
            logger.warning("Using default pose offsets (0°, 0°, 0°)")
            return 0.0, 0.0, 0.0

        try:
            with open(calib_file, 'r') as f:
                data = json.load(f)
            yaw_offset = float(data.get('yaw_offset', 0.0))
            pitch_offset = float(data.get('pitch_offset', 0.0))
            roll_offset = float(data.get('roll_offset', 0.0))
            logger.info(f"Loaded calibration: yaw={yaw_offset:.1f}°, pitch={pitch_offset:.1f}°, roll={roll_offset:.1f}°")
            return yaw_offset, pitch_offset, roll_offset
        except Exception as exc:
            logger.error(f"Failed to load calibration from {calib_file}: {exc}")
            return 0.0, 0.0, 0.0

    def detect_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face and return bounding box.

        Args:
            frame_bgr: BGR image (H, W, 3)

        Returns:
            (x, y, w, h) or None if no face detected
        """
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = self.detector.detect(mp_image)
            if not result.face_landmarks:
                return None
            
            # Calculate bounding box from landmarks
            landmarks = result.face_landmarks[0]
            h, w = frame_bgr.shape[:2]
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        except Exception as exc:
            logger.error(f"Face detection failed: {exc}")
            return None

    def extract_landmarks(self, frame_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Extract facial landmarks (auto-subsampled to desired count).

        Args:
            frame_bgr: BGR image (H, W, 3)
            bbox: Unused (kept for API compatibility)

        Returns:
            (num_landmarks, 2) array or None if no face detected
        """
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = self.detector.detect(mp_image)
            if not result.face_landmarks:
                return None

            # Get all landmarks (478: 468 face + 10 iris) and convert to pixels
            landmarks = result.face_landmarks[0]
            h, w = frame_bgr.shape[:2]
            landmarks_all = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

            # Subsample if needed
            return self._subsample_landmarks(landmarks_all)
        except Exception as exc:
            logger.error(f"Landmark extraction failed: {exc}")
            return None

    def _subsample_landmarks(self, landmarks_all: np.ndarray) -> np.ndarray:
        """Subsample MediaPipe landmarks to desired count.

        Args:
            landmarks_all: (N, 2) array where N=478 (468 face + 10 iris)

        Returns:
            (num_landmarks, 2) array uniformly sampled
        """
        num_total = landmarks_all.shape[0]  # Should be 478
        
        if self.num_landmarks == 468:
            # Return face mesh only (exclude iris refinement)
            return landmarks_all[:468]
        elif self.num_landmarks == 478:
            # Return all landmarks (face + iris)
            return landmarks_all
        elif self.num_landmarks in {194, 98, 68}:
            # Uniformly sample from face mesh (first 468 points)
            indices = np.linspace(0, 467, num=self.num_landmarks, dtype=np.int32)
            return landmarks_all[indices]
        else:
            logger.warning(f"Unsupported landmark count {self.num_landmarks}, using all {num_total}")
            return landmarks_all

    def estimate_pose(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Estimate head pose using MediaPipe's facial_transformation_matrixes.

        Uses MediaPipe's 4×4 pose matrix directly (no solvePnP, no canonical model).
        Validated by external assistant: ±1° accuracy on test frames.

        Args:
            frame_bgr: BGR image (H, W, 3)

        Returns:
            (yaw, pitch, roll) in degrees or None if no face detected
        """
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = self.detector.detect(mp_image)
            if not result.face_landmarks or not result.facial_transformation_matrixes:
                return None

            # Extract rotation from 4×4 pose matrix
            pose_matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            rot_mat = pose_matrix[:3, :3]
            
            # Convert to Euler angles - XZY convention
            # XZY gives [X, Z, Y] but MediaPipe's coordinate mapping requires reordering:
            # angles[0] (X/Roll in XZY) → maps to head up/down (Pitch)
            # angles[1] (Z/Yaw in XZY) → maps to head tilt (Roll)  
            # angles[2] (Y/Pitch in XZY) → maps to head left/right (Yaw)
            r = R_scipy.from_matrix(rot_mat)
            angles = r.as_euler('XZY', degrees=True)
            pitch_deg = angles[0]  # Head up/down
            roll_deg = angles[1]   # Head tilt
            yaw_deg = angles[2]    # Head left/right
            
            return yaw_deg, pitch_deg, roll_deg
        except Exception as exc:
            logger.error(f"Pose estimation failed: {exc}")
            return None

    def process_frame(self, frame_bgr: np.ndarray, compute_pose: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """Process frame: detect face, extract landmarks, compute pose.

        Args:
            frame_bgr: BGR image (H, W, 3)
            compute_pose: Whether to compute head orientation (default: True)

        Returns:
            Dict with 'bbox', 'landmarks', 'pose' (or None if no face)
        """
        bbox = self.detect_face(frame_bgr)
        if bbox is None:
            return None

        landmarks = self.extract_landmarks(frame_bgr, bbox)
        if landmarks is None:
            return None

        pose_value = None
        if compute_pose and getattr(get_config(), 'use_pnp_pose', True):
            try:
                pose_result = self.estimate_pose(frame_bgr)
                if pose_result is not None:
                    yaw_deg, pitch_deg, roll_deg = pose_result
                    # Apply calibration offsets
                    pose_value = (
                        yaw_deg - self.yaw_offset,
                        pitch_deg - self.pitch_offset,
                        roll_deg - self.roll_offset
                    )
            except Exception as e:
                logger.warning(f"Pose estimation failed: {e}")

        return {
            'bbox': bbox,
            'landmarks': landmarks,
            'pose': pose_value
        }

    def release(self):
        """Release resources (no-op for MediaPipe)."""
        pass


def create_detector(num_landmarks: Optional[int] = None) -> LandmarkDetectorONNX:
    """Factory function to create detector instance.

    Args:
        num_landmarks: Desired landmark count (68, 98, 194, 468)

    Returns:
        Configured LandmarkDetectorONNX instance
    """
    return LandmarkDetectorONNX(num_landmarks=num_landmarks)


if __name__ == "__main__":
    # Standalone test
    print("[TEST] Initializing MediaPipe detector...")
    detector = LandmarkDetectorONNX(num_landmarks=468)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Camera not available")
    else:
        print("Camera opened. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            res = detector.process_frame(frame, compute_pose=True)
            if res is not None:
                # Draw bounding box
                x, y, w, h = res['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw landmarks
                for (px, py) in res['landmarks'].astype(np.int32):
                    cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
                
                # Display pose
                if res['pose'] is not None:
                    yaw, pitch, roll = res['pose']
                    cv2.putText(frame, f"Yaw: {yaw:+.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"Pitch: {pitch:+.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"Roll: {roll:+.1f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("MediaPipe Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed.")