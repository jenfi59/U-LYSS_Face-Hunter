#!/usr/bin/env python3
"""
Core preprocessing functions for robust landmark detection.

Implements:
- ROI extraction with margin
- CLAHE histogram equalization
- Skin segmentation (YCrCb)
- Bilateral filtering
- Temporal Kalman filtering
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""
    roi_clean: np.ndarray  # Preprocessed ROI (H, W, 3)
    skin_mask: np.ndarray  # Skin mask (H, W)
    transform_matrix: np.ndarray  # 3x3 transform for landmark reprojection
    bbox_roi: Tuple[int, int, int, int]  # (x, y, w, h) of ROI in original frame
    processing_time_ms: float  # Processing time


def extract_facial_roi(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_percent: float = 0.15  # 15% validated 2025-12-21 (optimal tradeoff)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract facial ROI with margin.
    
    Args:
        frame_bgr: Original BGR image (H, W, 3)
        bbox: Face bounding box (x, y, w, h)
        margin_percent: Margin to add around face (default 15%)
    
    Returns:
        roi_face: Extracted ROI (H', W', 3)
        transform_matrix: 3x3 matrix for landmark reprojection
        bbox_roi: (x1, y1, w_roi, h_roi) in original coordinates
    """
    x, y, w, h = bbox
    
    # Calculate margin
    margin_x = int(w * margin_percent)
    margin_y = int(h * margin_percent)
    
    # Expand bbox with clipping
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_bgr.shape[1], x + w + margin_x)
    y2 = min(frame_bgr.shape[0], y + h + margin_y)
    
    # Extract ROI
    roi_face = frame_bgr[y1:y2, x1:x2].copy()
    
    # Transform matrix for landmark reprojection
    # landmarks_original = landmarks_roi + [x1, y1, 0]
    transform_matrix = np.array([
        [1, 0, x1],
        [0, 1, y1],
        [0, 0, 1]
    ], dtype=np.float32)
    
    bbox_roi = (x1, y1, x2 - x1, y2 - y1)
    
    return roi_face, transform_matrix, bbox_roi


def apply_clahe(roi_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Works on L channel of LAB color space for better results.
    
    Args:
        roi_bgr: Input ROI in BGR (H, W, 3)
        clip_limit: Contrast limit (higher = more contrast)
        tile_size: Grid size for local equalization
    
    Returns:
        roi_equalized: ROI with enhanced contrast (H, W, 3)
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)
    
    # Merge and convert back to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    roi_equalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return roi_equalized


def segment_skin(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Segment skin pixels using YCrCb color space.
    
    Args:
        roi_bgr: Input ROI in BGR (H, W, 3)
    
    Returns:
        skin_mask: Binary mask (H, W) where 255 = skin
    """
    # Convert to YCrCb (better for skin detection)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Empirical thresholds for human skin
    # Y: 0-255, Cr: 133-173, Cb: 77-127
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create mask
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    return skin_mask


def apply_bilateral_filter(
    roi_bgr: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving smoothing).
    
    Args:
        roi_bgr: Input ROI (H, W, 3)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        roi_filtered: Filtered ROI (H, W, 3)
    """
    roi_filtered = cv2.bilateralFilter(roi_bgr, d, sigma_color, sigma_space)
    return roi_filtered


def preprocess_roi_robust(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_percent: float = 0.15,
    apply_skin_mask: bool = True
) -> PreprocessingResult:
    """
    Full preprocessing pipeline for robust landmark detection.
    
    Pipeline:
    1. Extract ROI with margin
    2. CLAHE histogram equalization
    3. Skin segmentation
    4. Bilateral filtering
    5. Apply skin mask
    
    Args:
        frame_bgr: Original BGR frame (H, W, 3)
        bbox: Face bounding box (x, y, w, h)
        margin_percent: ROI margin (default 15%)
        apply_skin_mask: Whether to mask non-skin regions
    
    Returns:
        PreprocessingResult with processed ROI and metadata
    """
    import time
    start_time = time.perf_counter()
    
    # Step 1: Extract ROI
    roi_face, transform_matrix, bbox_roi = extract_facial_roi(frame_bgr, bbox, margin_percent)
    
    # Step 2: CLAHE
    roi_equalized = apply_clahe(roi_face)
    
    # Step 3: Skin segmentation
    skin_mask = segment_skin(roi_equalized)
    
    # Step 4: Bilateral filter
    roi_filtered = apply_bilateral_filter(roi_equalized)
    
    # Step 5: Apply skin mask (optional)
    if apply_skin_mask:
        # Create 3-channel mask
        mask_3ch = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        roi_clean = cv2.bitwise_and(roi_filtered, mask_3ch)
    else:
        roi_clean = roi_filtered
    
    processing_time = (time.perf_counter() - start_time) * 1000  # ms
    
    return PreprocessingResult(
        roi_clean=roi_clean,
        skin_mask=skin_mask,
        transform_matrix=transform_matrix,
        bbox_roi=bbox_roi,
        processing_time_ms=processing_time
    )


class TemporalLandmarkFilter:
    """
    Kalman filter for temporal smoothing of landmarks.
    
    Reduces jitter and noise across frames.
    """
    
    def __init__(self, n_landmarks: int = 68, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman filters for all landmark coordinates.
        
        Args:
            n_landmarks: Number of landmarks (68 for dlib)
            process_noise: Process noise covariance (smaller = more stable)
            measurement_noise: Measurement noise covariance (larger = smoother)
        """
        self.n_landmarks = n_landmarks
        self.n_dims = 3  # x, y, z
        self.filters = []
        
        # Create Kalman filter for each coordinate (68 * 3 = 204 filters)
        for _ in range(n_landmarks * self.n_dims):
            kf = cv2.KalmanFilter(2, 1, 0)  # 2 states (pos, vel), 1 measurement
            
            # State transition matrix: [1 dt; 0 1]
            kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
            
            # Measurement matrix: [1 0]
            kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
            
            # Process noise covariance
            kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise
            
            # Measurement noise covariance
            kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)
            
            # Initial state
            kf.statePost = np.zeros((2, 1), dtype=np.float32)
            kf.errorCovPost = np.eye(2, dtype=np.float32)
            
            self.filters.append(kf)
    
    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Update filter with new landmark measurements.
        
        Args:
            landmarks: (n_landmarks, 3) array of landmarks
        
        Returns:
            landmarks_smooth: (n_landmarks, 3) smoothed landmarks
        """
        landmarks_flat = landmarks.flatten()  # (204,)
        landmarks_smooth = np.zeros_like(landmarks_flat)
        
        for i, (kf, measurement) in enumerate(zip(self.filters, landmarks_flat)):
            # Predict
            kf.predict()
            
            # Correct with measurement
            kf.correct(np.array([[measurement]], dtype=np.float32))
            
            # Extract smoothed position
            landmarks_smooth[i] = kf.statePost[0, 0]
        
        return landmarks_smooth.reshape(self.n_landmarks, self.n_dims)
    
    def reset(self):
        """Reset all filters (e.g., when changing user)."""
        for kf in self.filters:
            kf.statePost = np.zeros((2, 1), dtype=np.float32)
            kf.errorCovPost = np.eye(2, dtype=np.float32)


def reproject_landmarks(
    landmarks_roi: np.ndarray,
    transform_matrix: np.ndarray
) -> np.ndarray:
    """
    Reproject landmarks from ROI coordinates to original frame coordinates.
    
    Args:
        landmarks_roi: Landmarks in ROI space (N, 3) with (x, y, z)
        transform_matrix: 3x3 transform matrix from extract_facial_roi
    
    Returns:
        landmarks_original: Landmarks in original frame space (N, 3)
    """
    landmarks_original = landmarks_roi.copy()
    
    # Apply 2D translation to x and y (z unchanged)
    # transform_matrix[0, 2] = x offset
    # transform_matrix[1, 2] = y offset
    landmarks_original[:, 0] += transform_matrix[0, 2]  # x + x_offset
    landmarks_original[:, 1] += transform_matrix[1, 2]  # y + y_offset
    # landmarks_original[:, 2] unchanged (z coordinate)
    
    return landmarks_original


if __name__ == "__main__":
    print("✅ Preprocessing module loaded")
    print(f"   - ROI extraction with margin")
    print(f"   - CLAHE histogram equalization")
    print(f"   - Skin segmentation (YCrCb)")
    print(f"   - Bilateral filtering")
    print(f"   - Temporal Kalman filtering")
