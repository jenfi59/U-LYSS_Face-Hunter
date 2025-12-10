"""
Landmark Utilities - FR_VERS_JP 2.0
====================================

Centralized utilities for 68-landmark facial geometry features.
Avoids code duplication and provides consistent landmark extraction.

Features:
- 68 landmark indices (MediaPipe subset compatible with dlib)
- Extraction function with error handling
- Model type detection (landmarks vs Gabor+LBP)

Version: 1.0.0
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import logging

# ==============================================================================
# 68 LANDMARK INDICES (MediaPipe subset)
# ==============================================================================
# Based on dlib 68-point model, extracted from MediaPipe's 468 landmarks
# Structure: Contour (17) + Eyebrows (10) + Nose (9) + Eyes (12) + Mouth (20)

LANDMARK_INDICES = [
    # Contour du visage (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    
    # Sourcil gauche (5 points)
    70, 63, 105, 66, 107,
    
    # Sourcil droit (5 points)
    336, 296, 334, 293, 300,
    
    # Nez (9 points)
    168, 6, 197, 195, 5, 4, 19, 94, 2,
    
    # Œil gauche (6 points)
    33, 160, 158, 133, 153, 144,
    
    # Œil droit (6 points)
    362, 385, 387, 263, 373, 380,
    
    # Bouche extérieure (12 points)
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 181,
    
    # Bouche intérieure (8 points)
    78, 191, 80, 81, 82, 13, 312, 311
]  # Total: 68 landmarks

# Feature dimensions
N_LANDMARKS = 68
N_COORDS = 2  # x, y (ignoring z from MediaPipe)
N_LANDMARK_FEATURES = N_LANDMARKS * N_COORDS  # 136

# Legacy Gabor+LBP feature count
N_GABOR_LBP_FEATURES = 275  # 16 (Gabor) + 256 (LBP) + 3 (pose)


# ==============================================================================
# MEDIAPIPE DETECTOR
# ==============================================================================

class FaceLandmarkDetector:
    """Simple wrapper for MediaPipe Face Mesh detector."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except ImportError:
            logging.error("MediaPipe not installed. Install with: pip install mediapipe")
            raise
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks in frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Input image (BGR format from OpenCV)
        
        Returns
        -------
        landmarks : np.ndarray or None
            Array of shape (468, 3) with normalized coordinates [x, y, z]
            Returns None if no face detected
        """
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array (468 landmarks, 3 coords each)
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in face_landmarks.landmark
        ])
        
        return landmarks
    
    def close(self):
        """Close the detector."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# ==============================================================================
# EXTRACTION FUNCTIONS
# ==============================================================================

def extract_landmarks_from_frame(
    frame: np.ndarray,
    detector,  # FaceLandmarkDetector instance
) -> Optional[np.ndarray]:
    """Extract 68 landmark features from a single frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input image (full resolution, color or grayscale)
    detector : FaceLandmarkDetector
        Initialized landmark detector
    
    Returns
    -------
    features : np.ndarray or None
        Flattened (136,) array of landmark coordinates [x1,y1,x2,y2,...]
        Returns None if no landmarks detected
    """
    landmarks = detector.detect_landmarks(frame)
    
    if landmarks is None:
        logging.debug("No landmarks detected in frame")
        return None
    
    # Extract 68 landmark subset (only x, y - ignore z)
    landmarks_68 = landmarks[LANDMARK_INDICES][:, :N_COORDS]
    
    # Flatten to 1D array (136 features)
    features = landmarks_68.flatten()
    
    return features


def extract_landmarks_from_video(
    video_source: int | str,
    num_frames: int = 10,
    show_preview: bool = True,
) -> Tuple[np.ndarray, int]:
    """Extract landmarks from multiple video frames.
    
    Parameters
    ----------
    video_source : int or str
        Camera index or video file path
    num_frames : int
        Number of frames to capture
    show_preview : bool
        If True, display video preview with landmarks overlay
    
    Returns
    -------
    features : np.ndarray
        Array of shape (n_valid_frames, 136)
    n_valid : int
        Number of frames with valid landmarks
    """
    import cv2
    
    detector = FaceLandmarkDetector()
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logging.error(f"Failed to open video source: {video_source}")
        return np.array([]), 0
    
    features_list = []
    frame_count = 0
    
    if show_preview:
        cv2.namedWindow('Enrollment - Press Q to quit', cv2.WINDOW_NORMAL)
    
    print(f"Capturing {num_frames} frames... (Press Q to cancel)")
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect landmarks
        landmarks = detector.detect_landmarks(frame)
        
        # Extract features if face detected
        features = None
        if landmarks is not None:
            landmarks_68 = landmarks[LANDMARK_INDICES][:, :N_COORDS]
            features = landmarks_68.flatten()
            features_list.append(features)
            frame_count += 1
        
        # Show preview
        if show_preview:
            display_frame = frame.copy()
            
            # Draw landmarks if detected
            if landmarks is not None:
                h, w = frame.shape[:2]
                landmarks_68 = landmarks[LANDMARK_INDICES]
                
                for i, (x, y, _) in enumerate(landmarks_68):
                    # Green for detected landmarks
                    color = (0, 255, 0) if features is not None else (0, 0, 255)
                    cv2.circle(display_frame, (int(x), int(y)), 2, color, -1)
                
                # Show progress
                cv2.putText(display_frame, 
                           f"Frame: {frame_count}/{num_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, 
                           "Face detected - Hold still", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(display_frame, 
                           "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)
                cv2.putText(display_frame, 
                           "Look at the camera", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            cv2.imshow('Enrollment - Press Q to quit', display_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nEnrollment cancelled by user")
                break
    
    cap.release()
    detector.close()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    if not features_list:
        logging.warning(f"No valid landmarks found in {num_frames} frames")
        return np.array([]), 0
    
    features_array = np.array(features_list)
    n_valid = len(features_list)
    
    if n_valid < num_frames:
        logging.warning(f"Only {n_valid}/{num_frames} frames with valid landmarks")
    
    return features_array, n_valid


# ==============================================================================
# MODEL TYPE DETECTION
# ==============================================================================

def detect_model_type(scaler) -> str:
    """Detect if model uses landmarks or Gabor+LBP based on feature count.
    
    Parameters
    ----------
    scaler : sklearn.preprocessing.RobustScaler
        Fitted scaler from loaded model
    
    Returns
    -------
    model_type : str
        'landmarks' (136 features) or 'gabor_lbp' (275 features)
    
    Raises
    ------
    ValueError
        If feature count doesn't match known types
    """
    n_features = scaler.n_features_in_
    
    if n_features == N_LANDMARK_FEATURES:
        return 'landmarks'
    elif n_features == N_GABOR_LBP_FEATURES:
        return 'gabor_lbp'
    else:
        raise ValueError(
            f"Unknown feature count: {n_features}. "
            f"Expected {N_LANDMARK_FEATURES} (landmarks) or {N_GABOR_LBP_FEATURES} (Gabor+LBP)"
        )


def is_landmark_model(scaler) -> bool:
    """Check if model uses landmark features.
    
    Parameters
    ----------
    scaler : sklearn.preprocessing.RobustScaler
        Fitted scaler from loaded model
    
    Returns
    -------
    bool
        True if landmarks (136 features), False otherwise
    """
    return scaler.n_features_in_ == N_LANDMARK_FEATURES
