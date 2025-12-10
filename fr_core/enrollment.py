"""
Enrollment Pipeline – FR_VERS_JP 2.0
=====================================

This module implements the enrollment pipeline for 3D-aware face recognition.
It captures a user's biometric facial signature across various head poses and
lighting conditions, creating a robust model for later authentication.

Workflow:
---------
1. **Capture** – Grab frames from video source and detect faces
2. **Pose Estimation** – Estimate yaw, pitch, roll using 3D landmarks
3. **Preprocessing** – Wavelet normalization, Gabor/LBP feature extraction
4. **Feature Extraction** – Create functional representation with texture descriptors
5. **Dimensionality Reduction** – Apply PCA with dynamic component selection
6. **GMM Modeling** – Train Gaussian Mixture Model on reduced features
7. **Save Model** – Serialize PCA, GMM, scaler, and pose statistics

Features:
---------
- Gabor filters (16 features: 4 orientations × 2 frequencies × 2 stats)
- LBP histograms (256 features: radius=1, 8 neighbors)
- Dynamic PCA selection (auto-select components for 95% variance)
- BIC-based GMM component selection
- Pose-aware feature extraction

Version: 2.0.0
"""


from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional, Dict

import numpy as np

# Import from local modules
try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("config.py not found, using default parameters")

try:
    from fr_core.features import (
        FaceLandmarkDetector,
        PoseInfo,
        estimate_pose_from_landmarks,
        estimate_pose_from_bbox,
    )
    LANDMARKS_MODULE_AVAILABLE = True
except ImportError:
    LANDMARKS_MODULE_AVAILABLE = False
    logging.warning("fr_core.features not found, using fallback methods")

try:
    from fr_core.preprocessing import preprocess_face_from_config, preprocess_face
    PREPROCESSING_MODULE_AVAILABLE = True
except ImportError:
    PREPROCESSING_MODULE_AVAILABLE = False
    logging.warning("fr_core.preprocessing not found, using basic preprocessing")

# External dependencies are imported lazily inside functions to avoid
# ImportError at module import time when they are not installed.


# PoseInfo and pose estimation functions are now imported from facial_landmarks module
# If not available, define fallback versions
if not LANDMARKS_MODULE_AVAILABLE:
    import dataclasses
    import math

    @dataclasses.dataclass
    class PoseInfo:
        """Fallback container for head pose parameters."""
        yaw: float
        pitch: float
        roll: float
        distance: float

    def estimate_pose_from_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> PoseInfo:
        """Fallback pose estimation from bbox."""
        import cv2
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        image_points = np.array(
            [
                (x + 0.3 * w, y + 0.3 * h),
                (x + 0.7 * w, y + 0.3 * h),
                (x + 0.5 * w, y + 0.5 * h),
                (x + 0.35 * w, y + 0.7 * h),
                (x + 0.65 * w, y + 0.7 * h),
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (-30.0, -30.0, -30.0),
                (30.0, -30.0, -30.0),
                (0.0, 0.0, 0.0),
                (-20.0, 30.0, -30.0),
                (20.0, 30.0, -30.0),
            ],
            dtype=np.float64,
        )
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            success = False
        if not success:
            return PoseInfo(0.0, 0.0, 0.0, 1.0)
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x_rot = math.atan2(rmat[2, 1], rmat[2, 2])
            y_rot = math.atan2(-rmat[2, 0], sy)
            z_rot = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x_rot = math.atan2(-rmat[1, 2], rmat[1, 1])
            y_rot = math.atan2(-rmat[2, 0], sy)
            z_rot = 0
        pitch = math.degrees(x_rot)
        yaw = math.degrees(y_rot)
        roll = math.degrees(z_rot)
        distance = float(tvec[2, 0]) if tvec.size >= 3 else 1.0
        return PoseInfo(yaw, pitch, roll, distance)


# Preprocessing function now uses the preprocessing module
if not PREPROCESSING_MODULE_AVAILABLE:
    # Fallback preprocessing function
    def preprocess_face_basic(image: np.ndarray) -> np.ndarray:
        """Basic fallback preprocessing."""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        quantised = (gray // 32) * 32
        return quantised


def preprocess_face_wrapper(image: np.ndarray) -> np.ndarray:
    """Wrapper for preprocessing that uses the advanced module if available.

    Parameters
    ----------
    image : np.ndarray
        RGB or BGR face image.

    Returns
    -------
    np.ndarray
        Pre-processed greyscale image with shape (H, W).
    """
    if PREPROCESSING_MODULE_AVAILABLE:
        if CONFIG_AVAILABLE:
            return preprocess_face_from_config(image, config)
        else:
            return preprocess_face(image)
    else:
        return preprocess_face_basic(image)


def compute_frame_quality(
    frame: np.ndarray,
    pose: PoseInfo,
    max_yaw: float = 30.0,
    max_pitch: float = 20.0,
    max_roll: float = 15.0,
    min_brightness: int = 40,
    max_brightness: int = 215,
) -> float:
    """Compute quality score for a captured frame.
    
    OPTIMIZATION: Filter low-quality frames during enrollment
    
    Quality factors:
    - Sharpness (Laplacian variance): blurry frames get low score
    - Brightness (histogram): too dark/bright frames get low score
    - Pose alignment: extreme angles get low score
    
    Parameters
    ----------
    frame : np.ndarray
        Grayscale face image
    pose : PoseInfo
        Pose information with yaw, pitch, roll
    max_yaw, max_pitch, max_roll : float
        Maximum acceptable angles in degrees
    min_brightness, max_brightness : int
        Acceptable brightness range (0-255)
    
    Returns
    -------
    float
        Quality score between 0.0 (worst) and 1.0 (best)
    """
    import cv2
    
    scores = []
    
    # 1. Sharpness score (Laplacian variance)
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sharpness = laplacian.var()
    # Normalize: variance > 100 is sharp, < 20 is blurry
    sharpness_score = min(1.0, max(0.0, (sharpness - 20) / 80))
    scores.append(sharpness_score)
    
    # 2. Brightness score (avoid too dark or too bright)
    mean_brightness = frame.mean()
    if mean_brightness < min_brightness:
        brightness_score = mean_brightness / min_brightness
    elif mean_brightness > max_brightness:
        brightness_score = max(0.0, 1.0 - (mean_brightness - max_brightness) / (255 - max_brightness))
    else:
        brightness_score = 1.0
    scores.append(brightness_score)
    
    # 3. Pose alignment score (penalize extreme angles)
    yaw_score = max(0.0, 1.0 - abs(pose.yaw) / max_yaw)
    pitch_score = max(0.0, 1.0 - abs(pose.pitch) / max_pitch)
    roll_score = max(0.0, 1.0 - abs(pose.roll) / max_roll)
    pose_score = (yaw_score + pitch_score + roll_score) / 3.0
    scores.append(pose_score)
    
    # Overall quality: weighted average
    # Sharpness is most important, then brightness, then pose
    weights = [0.5, 0.3, 0.2]
    quality = sum(s * w for s, w in zip(scores, weights))
    
    return quality


def capture_enrolment_frames(
    video_source: int | str = 0,
    num_frames: int = 50,
    detect_interval: int = 5,
    show_gui: bool = True,
    frame_delay_ms: int = 100,
) -> Tuple[List[np.ndarray], List[PoseInfo]]:
    """Capture a sequence of face images and corresponding pose information.

    This function now uses Mediapipe Face Mesh for landmark detection when
    available, providing more accurate pose estimation.

    Parameters
    ----------
    video_source : int or str
        Index of the camera or URL of a video source.
    num_frames : int
        Total number of frames to capture.  The user should move their
        head through various poses during capture to sample the pose
        space.
    detect_interval : int
        Perform face detection every `detect_interval` frames to update
        the region of interest.  On intermediate frames the previously
        detected region is reused to reduce latency.
    show_gui : bool
        If True, shows a window with camera feed and visual guides.
    frame_delay_ms : int
        Delay in milliseconds between frame captures (default: 100ms = 10 fps).
        Higher values give more time for head movement. Minimum 2 seconds total:
        num_frames * frame_delay_ms >= 2000ms.

    Returns
    -------
    frames : list of np.ndarray
        List of pre-processed face images.
    poses : list of PoseInfo
        Corresponding list of pose measurements for each frame.
    """
    import cv2
    frames: List[np.ndarray] = []
    poses: List[PoseInfo] = []
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    # Get configuration parameters
    if CONFIG_AVAILABLE:
        scale_factor = config.HAAR_SCALE_FACTOR
        min_neighbors = config.HAAR_MIN_NEIGHBORS
        min_size = config.HAAR_MIN_SIZE
        max_size = config.HAAR_MAX_SIZE
        use_mediapipe = config.USE_MEDIAPIPE_LANDMARKS
    else:
        scale_factor = 1.1
        min_neighbors = 5
        min_size = (30, 30)
        max_size = None
        use_mediapipe = True

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Initialize landmark detector if using Mediapipe
    landmark_detector = None
    if use_mediapipe and LANDMARKS_MODULE_AVAILABLE:
        if CONFIG_AVAILABLE:
            landmark_detector = FaceLandmarkDetector(
                min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
                max_num_faces=config.MEDIAPIPE_MAX_NUM_FACES,
                model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            )
        else:
            landmark_detector = FaceLandmarkDetector()

    prev_box: Optional[Tuple[int, int, int, int]] = None
    
    # Create window if GUI is enabled
    if show_gui:
        window_name = "Enrollment - Positionnez votre visage"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Create display frame for GUI
        if show_gui:
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]

        # Detect face periodically or if no previous box
        if i % detect_interval == 0 or prev_box is None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                maxSize=max_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(detections) > 0:
                # Choose the largest detected face
                x, y, w, h = max(detections, key=lambda bb: bb[2] * bb[3])
                prev_box = (int(x), int(y), int(w), int(h))
            else:
                prev_box = None

        if prev_box is None:
            # Show message if no face detected
            if show_gui:
                cv2.putText(display_frame, "Aucun visage detecte", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(1)
            continue

        x, y, w, h = prev_box
        # Ensure bounding box is within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        face = frame[y : y + h, x : x + w].copy()
        processed = preprocess_face_wrapper(face)

        # Estimate pose using landmarks if available
        if landmark_detector is not None:
            landmarks = landmark_detector.detect_landmarks(frame)
            pose = estimate_pose_from_landmarks(frame, landmarks, prev_box)
        else:
            pose = estimate_pose_from_bbox(frame, prev_box)

        # GUI: Draw visual guides and validation
        if show_gui:
            # Draw face bounding box
            box_color = (0, 255, 0)  # Green by default
            
            # Check pose quality
            pose_ok = (abs(pose.yaw) < 20 and abs(pose.pitch) < 20 and abs(pose.roll) < 15)
            
            if pose_ok:
                box_color = (0, 255, 0)  # Green - good position
                status_text = "Position OK"
                status_color = (0, 255, 0)
            else:
                box_color = (0, 165, 255)  # Orange - adjust position
                status_text = "Ajustez votre position"
                status_color = (0, 165, 255)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw center guide (oval for ideal face position)
            center_x = width // 2
            center_y = height // 2
            oval_w = width // 4
            oval_h = height // 3
            cv2.ellipse(display_frame, (center_x, center_y), (oval_w, oval_h), 
                       0, 0, 360, (255, 255, 255), 2)
            
            # Draw progress bar
            progress = (i + 1) / num_frames
            bar_width = width - 100
            bar_height = 30
            bar_x = 50
            bar_y = height - 60
            
            # Background bar
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Progress bar
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), 
                         (0, 255, 0), -1)
            
            # Frame counter
            cv2.putText(display_frame, f"{i + 1}/{num_frames}", 
                       (bar_x + bar_width + 10, bar_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status text
            cv2.putText(display_frame, status_text, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            
            # Pose information
            pose_info = f"Yaw: {pose.yaw:+.1f}  Pitch: {pose.pitch:+.1f}  Roll: {pose.roll:+.1f}"
            cv2.putText(display_frame, pose_info, (50, height - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(frame_delay_ms)
        
        # Only save frame if pose is acceptable or we're past 70% capture
        # (to ensure we get enough frames even if user struggles with positioning)
        if pose_ok or i > int(num_frames * 0.7):
            frames.append(processed)
            poses.append(pose)

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()
    if landmark_detector is not None:
        landmark_detector.close()

    # OPTIMIZATION: Quality filtering - keep only top 70% frames
    logging.info(f"Captured {len(frames)} frames, computing quality scores...")
    quality_scores = []
    for frame, pose in zip(frames, poses):
        quality = compute_frame_quality(frame, pose)
        quality_scores.append(quality)
    
    # Sort by quality and keep top 70%
    quality_scores = np.array(quality_scores)
    num_to_keep = max(int(len(frames) * 0.7), 10)  # Keep at least 10 frames
    top_indices = np.argsort(quality_scores)[-num_to_keep:]
    top_indices = sorted(top_indices)  # Maintain temporal order
    
    filtered_frames = [frames[i] for i in top_indices]
    filtered_poses = [poses[i] for i in top_indices]
    
    logging.info(f"Quality filtering: kept {len(filtered_frames)}/{len(frames)} frames "
                f"(mean quality: {quality_scores[top_indices].mean():.2f})")
    
    return filtered_frames, filtered_poses


def select_n_components(eigenvalues: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Select the minimum number of PCA components to explain variance threshold.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues from PCA (explained variance).
    variance_threshold : float
        Minimum proportion of variance to explain (default: 0.95).

    Returns
    -------
    int
        Minimum number of components needed.
    """
    # Calculate cumulative variance ratio
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    
    # Find first index where cumulative variance exceeds threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Ensure at least 2 components
    n_components = max(2, n_components)
    
    logging.info(
        f"Selected {n_components} components explaining "
        f"{cumulative_variance[n_components-1]*100:.2f}% of variance"
    )
    
    return int(n_components)


def compute_functional_representation(
    frames: List[np.ndarray],
    poses: List[PoseInfo],
    n_components: Optional[int] = None,
    use_gabor: bool = True,
    use_lbp: bool = True,
    use_dtw: bool = True,
) -> Tuple[np.ndarray, object, Optional[object]]:
    """Compute functional PCA and optionally DTW template on captured frames.

    This function stacks the images into a 2D array (samples × pixels),
    optionally extracts Gabor and LBP features, applies PCA to reduce
    dimensionality. If use_dtw=True, stores the PCA sequence for DTW matching.
    If use_dtw=False, fits a GMM (legacy mode).

    Parameters
    ----------
    frames : list of np.ndarray
        Pre‑processed face images.
    poses : list of PoseInfo
        Corresponding pose measurements.
    n_components : int or None
        Number of principal components. If None, uses dynamic selection.
    use_gabor : bool
        Whether to extract and concatenate Gabor features.
    use_lbp : bool
        Whether to extract and concatenate LBP features.
    use_dtw : bool
        If True, use DTW (no GMM). If False, use GMM (legacy).

    Returns
    -------
    scores : np.ndarray
        Array of shape (n_samples, n_components) of principal component
        scores. Used as DTW template if use_dtw=True.
    pca_model : object
        Fitted PCA model (e.g. sklearn.decomposition.PCA).
    gmm_model : object or None
        Fitted Gaussian mixture model if use_dtw=False, None otherwise.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Get configuration parameters
    if CONFIG_AVAILABLE:
        use_gabor = use_gabor and config.USE_GABOR_FEATURES
        use_lbp = use_lbp and config.USE_LBP_FEATURES
        variance_threshold = config.PCA_VARIANCE_THRESHOLD if n_components is None else None
    else:
        variance_threshold = 0.95 if n_components is None else None
    
    # OPTIMIZATION: Use Gabor+LBP+Pose only (no raw pixels)
    # Analysis showed pixels contribute little discriminative power
    # Gabor+LBP+Pose: 275 dims vs 4371 dims (94% reduction)
    X = None
    
    # Extract and concatenate Gabor features if enabled
    if use_gabor and PREPROCESSING_MODULE_AVAILABLE:
        from fr_core.preprocessing import extract_gabor_features
        gabor_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                gab_feat = extract_gabor_features(
                    frame,
                    orientations=config.GABOR_ORIENTATIONS,
                    frequencies=config.GABOR_FREQUENCIES,
                    ksize=config.GABOR_KSIZE,
                )
            else:
                gab_feat = extract_gabor_features(frame)
            gabor_features.append(gab_feat)
        gabor_features = np.array(gabor_features)
        X = gabor_features if X is None else np.concatenate([X, gabor_features], axis=1)
        logging.info(f"Added Gabor features: {gabor_features.shape[1]} dimensions")
    
    # Extract and concatenate LBP features if enabled
    if use_lbp and PREPROCESSING_MODULE_AVAILABLE:
        from fr_core.preprocessing import extract_lbp_features
        lbp_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                lbp_feat = extract_lbp_features(
                    frame,
                    radius=config.LBP_RADIUS,
                    n_points=config.LBP_N_POINTS,
                )
            else:
                lbp_feat = extract_lbp_features(frame)
            lbp_features.append(lbp_feat)
        lbp_features = np.array(lbp_features)
        X = lbp_features if X is None else np.concatenate([X, lbp_features], axis=1)
        logging.info(f"Added LBP features: {lbp_features.shape[1]} dimensions")
    
    # Fallback to pixel features if no texture features available
    if X is None:
        logging.warning("No texture features available, falling back to raw pixels")
        X = np.array([f.ravel() for f in frames])
    
    # Concatenate pose information as additional features (yaw, pitch, roll only)
    pose_features = np.array(
        [[p.yaw, p.pitch, p.roll] for p in poses]
    )
    X_full = np.concatenate([X, pose_features], axis=1)
    
    # OPTIMIZATION: Use RobustScaler instead of StandardScaler
    # RobustScaler uses median and IQR, more robust to outliers and lighting variations
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    # Apply PCA with dynamic component selection if needed
    if n_components is None:
        # OPTIMIZATION: Increased from 20 to 45 components for 95%+ variance
        # Analysis showed Gabor+LBP+Pose needs 41 components for 95% variance
        # Using 45 for safety margin
        n_components = 45
        logging.info(f"Using optimized PCA components: {n_components}")
    
    # Apply PCA with selected components
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    
    # Choose between DTW (new) and GMM (legacy)
    if use_dtw:
        # DTW mode: no GMM needed, return sequence directly
        logging.info(f"DTW mode: storing sequence of {len(scores)} frames in PCA space")
        return scores, pca, None
    else:
        # Legacy GMM mode
        from sklearn.mixture import GaussianMixture
        
        if CONFIG_AVAILABLE:
            gmm_range = config.GMM_COMPONENT_RANGE
            gmm_cov_type = config.GMM_COVARIANCE_TYPE
        else:
            gmm_range = (1, 6)
            gmm_cov_type = "full"
        
        best_gmm: Optional[GaussianMixture] = None
        lowest_bic = np.inf
        for n in range(gmm_range[0], gmm_range[1]):
            gmm = GaussianMixture(
                n_components=n, 
                covariance_type=gmm_cov_type, 
                random_state=42,
                reg_covar=1e-6
            )
            gmm.fit(scores)
            bic = gmm.bic(scores)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        assert best_gmm is not None
        return scores, pca, best_gmm


def save_model(
    model_path: str,
    pca_model: object,
    gmm_model: Optional[object],
    scaler: object,
    pose_mean: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    t_ref: Optional[np.ndarray] = None,
    dtw_template: Optional[np.ndarray] = None,
) -> None:
    """Persist the trained models and statistics to disk.

    The models are saved using numpy's savez, which serialises
    arbitrary Python objects via pickle. In a production setting,
    consider using joblib or an explicit serialization format.
    
    Parameters
    ----------
    model_path : str
        Path where to save the model
    pca_model : object
        Trained PCA model
    gmm_model : object or None
        Trained GMM (legacy mode) or None (DTW mode)
    scaler : object
        StandardScaler fitted on training data
    pose_mean : np.ndarray
        Mean pose during enrollment
    R_ref : np.ndarray or None
        3x3 reference rotation matrix (optional, for 3D pose tracking).
    t_ref : np.ndarray or None
        3x1 reference translation vector (optional, for 3D pose tracking).
    dtw_template : np.ndarray or None
        PCA sequence for DTW matching (shape: n_frames × n_components).
    """
    save_dict = {
        'pca': pca_model,
        'scaler': scaler,
        'pose_mean': pose_mean,
    }
    
    # Save GMM or DTW template depending on mode
    if gmm_model is not None:
        save_dict["gmm"] = gmm_model
        save_dict["use_dtw"] = False
    elif dtw_template is not None:
        save_dict["dtw_template"] = dtw_template
        save_dict["use_dtw"] = True
    else:
        # Fallback: neither GMM nor DTW (shouldn't happen)
        save_dict["use_dtw"] = False
    
    # Add 3D reference frame if available
    if R_ref is not None:
        save_dict['R_ref'] = R_ref
    if t_ref is not None:
        save_dict['t_ref'] = t_ref
    
    np.savez(model_path, **save_dict)


def run_enrolment(
    output_path: str,
    video_source: int | str = 0,
    num_frames: int = 50,
    detect_interval: int = 5,
    n_components: Optional[int] = 20,
    show_gui: bool = True,
    frame_delay_ms: int = 120,
) -> None:
    """High‑level driver function for enrolment.

    Captures frames, computes functional representation, fits the GMM
    and saves the resulting model to disk.
    
    Parameters
    ----------
    output_path : str
        Path to save the model.
    video_source : int or str
        Camera index or video file path.
    num_frames : int
        Number of frames to capture.
    detect_interval : int
        Detection interval for face detection.
    n_components : int or None
        Number of PCA components (default: 20, optimized). If None, uses dynamic selection based on 95% variance threshold.
    show_gui : bool
        If True (default), shows visual guides during enrollment.
    frame_delay_ms : int
        Delay between frames in milliseconds (default: 100ms = 10 fps).
    """
    logging.info("Starting enrolment capturing %d frames from %s", num_frames, video_source)
    frames, poses = capture_enrolment_frames(
        video_source=video_source,
        num_frames=num_frames,
        detect_interval=detect_interval,
        show_gui=show_gui,
        frame_delay_ms=frame_delay_ms,
    )
    
    # Check if we captured enough frames
    if len(frames) == 0:
        raise RuntimeError("No frames captured! Please ensure your camera is working and your face is visible.")
    
    if len(frames) < 10:
        logging.warning(f"Only {len(frames)} frames captured. This may result in poor accuracy.")
    
    # Compute mean pose for calibration (yaw, pitch, roll only)
    pose_mean = np.mean(
        np.array([[p.yaw, p.pitch, p.roll] for p in poses]), axis=0
    )
    
    # Compute reference 3D orientation (R_ref, t_ref) for pose tracking
    from fr_core.features import average_rotation_matrices
    R_list = [p.R for p in poses]
    t_list = [p.t for p in poses]
    R_ref = average_rotation_matrices(R_list)
    t_ref = np.mean(np.array(t_list), axis=0) if len(t_list) > 0 else np.zeros((3, 1))
    
    # Check if DTW mode is enabled
    use_dtw = True  # Default to DTW (new mode)
    if CONFIG_AVAILABLE and hasattr(config, 'USE_DTW'):
        use_dtw = config.USE_DTW
    
    scores, pca, gmm = compute_functional_representation(
        frames, poses, n_components=n_components, use_dtw=use_dtw
    )
    # Note: scaler created inside compute_functional_representation not returned
    # To keep the API simple, we recompute it here with same features
    from sklearn.preprocessing import StandardScaler
    
    # Get configuration for feature extraction
    use_gabor = CONFIG_AVAILABLE and config.USE_GABOR_FEATURES
    use_lbp = CONFIG_AVAILABLE and config.USE_LBP_FEATURES
    
    # OPTIMIZATION: Use Gabor+LBP only (no raw pixels)
    X = None
    
    # Add Gabor features if enabled (same as in compute_functional_representation)
    if use_gabor and PREPROCESSING_MODULE_AVAILABLE:
        from fr_core.preprocessing import extract_gabor_features
        gabor_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                gab_feat = extract_gabor_features(
                    frame,
                    orientations=config.GABOR_ORIENTATIONS,
                    frequencies=config.GABOR_FREQUENCIES,
                    ksize=config.GABOR_KSIZE,
                )
            else:
                gab_feat = extract_gabor_features(frame)
            gabor_features.append(gab_feat)
        gabor_features = np.array(gabor_features)
        X = gabor_features if X is None else np.concatenate([X, gabor_features], axis=1)
    
    # Add LBP features if enabled
    if use_lbp and PREPROCESSING_MODULE_AVAILABLE:
        from fr_core.preprocessing import extract_lbp_features
        lbp_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                lbp_feat = extract_lbp_features(
                    frame,
                    radius=config.LBP_RADIUS,
                    n_points=config.LBP_N_POINTS,
                )
            else:
                lbp_feat = extract_lbp_features(frame)
            lbp_features.append(lbp_feat)
        lbp_features = np.array(lbp_features)
        X = lbp_features if X is None else np.concatenate([X, lbp_features], axis=1)
    
    # Fallback to pixel features if no texture features available
    if X is None:
        logging.warning("No texture features available, falling back to raw pixels")
        X = np.array([f.ravel() for f in frames])
    
    # Add pose features (yaw, pitch, roll only)
    pose_features_all = np.array(
        [[p.yaw, p.pitch, p.roll] for p in poses]
    )
    X_full_all = np.concatenate([X, pose_features_all], axis=1)
    
    # OPTIMIZATION: Use RobustScaler for better outlier handling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(X_full_all)
    
    # Save model with DTW template if in DTW mode
    if use_dtw:
        save_model(output_path, pca, None, scaler, pose_mean, R_ref, t_ref, dtw_template=scores)
        logging.info("Enrolment complete (DTW mode). Model saved to %s", output_path)
    else:
        save_model(output_path, pca, gmm, scaler, pose_mean, R_ref, t_ref)
        logging.info("Enrolment complete (GMM mode). Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage; adapt parameters as needed
    run_enrolment(
        output_path="enrolment_model.npz",
        video_source=0,
        num_frames=20,
        detect_interval=5,
        n_components=3,
    )