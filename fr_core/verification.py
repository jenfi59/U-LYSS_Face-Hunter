"""
Verification Pipeline – FR_VERS_JP 2.0
=======================================

This module implements the verification/authentication pipeline for face recognition.
It loads a previously enrolled user model and performs real-time verification with
confidence scoring and optional Log-Likelihood Ratio (LLR) analysis.

Features:
---------
- Real-time video verification with pose tracking
- Confidence scoring based on statistical robustness
- LLR verification with world/impostor model
- Dual threshold support (base_threshold + llr_threshold)
- Gabor and LBP feature integration
- Adaptive frame capture with quality filtering

Returns:
--------
- is_verified: Boolean authentication result
- score: Robust log-likelihood score
- confidence: Statistical confidence level (0-1)
- llr: Log-Likelihood Ratio (if world model available)

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

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


def compute_orientation_penalty(
    R_current: np.ndarray,
    t_current: np.ndarray,
    R_ref: Optional[np.ndarray],
    t_ref: Optional[np.ndarray],
    orient_weights: Tuple[float, float, float, float] = (1.0, 0.8, 0.6, 0.5)
) -> Tuple[float, float]:
    """Compute orientation penalty and quality from relative pose.
    
    Calculates the deviation from reference 3D orientation using rotation
    matrix and translation vector. Returns penalty for threshold adaptation
    and quality metric for confidence scoring.
    
    Parameters
    ----------
    R_current : np.ndarray
        Current 3x3 rotation matrix.
    t_current : np.ndarray
        Current 3x1 translation vector.
    R_ref : np.ndarray or None
        Reference 3x3 rotation matrix. If None, returns zero penalty.
    t_ref : np.ndarray or None
        Reference 3x1 translation vector. If None, returns zero penalty.
    orient_weights : tuple of 4 floats
        Weights for (yaw, pitch, roll, distance) penalties.
        
    Returns
    -------
    penalty : float
        Weighted orientation penalty (0 if R_ref or t_ref is None).
    quality : float
        Orientation quality in [0, 1] (1 if R_ref or t_ref is None).
    """
    # Backward compatibility: if no reference, return neutral values
    if R_ref is None or t_ref is None:
        return 0.0, 1.0
    
    # Compute relative rotation: R_rel = R_current @ R_ref.T
    R_rel = R_current @ R_ref.T
    
    # Extract Euler angles from relative rotation (in degrees)
    # Using ZYX convention (yaw-pitch-roll)
    sy = np.sqrt(R_rel[0, 0]**2 + R_rel[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        pitch = np.arctan2(-R_rel[2, 0], sy)
        roll = np.arctan2(R_rel[2, 1], R_rel[2, 2])
    else:
        yaw = np.arctan2(-R_rel[1, 2], R_rel[1, 1])
        pitch = np.arctan2(-R_rel[2, 0], sy)
        roll = 0.0
    
    # Convert to degrees and take absolute values
    yaw_deg = abs(np.degrees(yaw))
    pitch_deg = abs(np.degrees(pitch))
    roll_deg = abs(np.degrees(roll))
    
    # Compute translation difference (depth change)
    t_diff = t_current - t_ref
    # Use z-component (depth) as distance metric
    distance_change = abs(float(t_diff[2, 0])) if t_diff.shape == (3, 1) else abs(float(t_diff[2]))
    
    # Weighted penalty (normalize to reasonable ranges)
    # Typical ranges: yaw ±20°, pitch ±15°, roll ±10°, distance ±50 units
    penalty = (
        orient_weights[0] * (yaw_deg / 20.0) +
        orient_weights[1] * (pitch_deg / 15.0) +
        orient_weights[2] * (roll_deg / 10.0) +
        orient_weights[3] * (distance_change / 50.0)
    )
    
    # Quality metric: inverse of penalty, clipped to [0, 1]
    quality = float(np.clip(1.0 / (1.0 + penalty), 0.0, 1.0))
    
    return float(penalty), quality


# Fallback functions if modules not available
if not LANDMARKS_MODULE_AVAILABLE:
    import math

    def estimate_pose_from_bbox_fallback(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
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
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            success = False
        if not success:
            return 0.0, 0.0, 0.0, 1.0
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
        return yaw, pitch, roll, distance


if not PREPROCESSING_MODULE_AVAILABLE:
    def preprocess_face_basic(image: np.ndarray) -> np.ndarray:
        """Basic fallback preprocessing."""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        quantised = (gray // 32) * 32
        return quantised


def preprocess_face_wrapper(image: np.ndarray) -> np.ndarray:
    """Wrapper for preprocessing."""
    if PREPROCESSING_MODULE_AVAILABLE:
        if CONFIG_AVAILABLE:
            return preprocess_face_from_config(image, config)
        else:
            return preprocess_face(image)
    else:
        return preprocess_face_basic(image)


def load_model(model_path: str) -> Tuple[object, object, object, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load the PCA, GMM, scaler and reference pose from disk.

    Parameters
    ----------
    model_path : str
        Path to the ``.npz`` file saved by Engine 1.

    Returns
    -------
    pca : object
        The PCA model.
    gmm : object
        The Gaussian mixture model.
    scaler : object
        The standardisation transformer.
    pose_mean : np.ndarray
        Array of shape (4,) containing [yaw, pitch, roll, distance].
    R_ref : np.ndarray or None
        3x3 reference rotation matrix (None for backward compatibility).
    t_ref : np.ndarray or None
        3x1 reference translation vector (None for backward compatibility).
    """
    data = np.load(model_path, allow_pickle=True)
    pca = data["pca"].item()
    scaler = data["scaler"].item()
    pose_mean = data["pose_mean"]
    
    # Load 3D reference frame with backward compatibility
    R_ref = data.get("R_ref", None)
    t_ref = data.get("t_ref", None)
    
    # Check if DTW mode or GMM mode
    use_dtw = data.get("use_dtw", False)
    dtw_template = None
    gmm = None
    
    if use_dtw:
        dtw_template = data.get("dtw_template", None)
    else:
        gmm = data["gmm"].item()
    
    return pca, gmm, scaler, pose_mean, R_ref, t_ref, use_dtw, dtw_template


def load_world_model(world_model_path: str) -> Optional[object]:
    """Load the world/impostor GMM model for LLR calculation.

    Parameters
    ----------
    world_model_path : str
        Path to the world GMM model file (.npz format).

    Returns
    -------
    gmm_world : object or None
        The world GMM model, or None if loading failed.
    """
    try:
        data = np.load(world_model_path, allow_pickle=True)
        gmm_world = data["gmm"].item()
        logging.info(f"Loaded world model from {world_model_path}")
        return gmm_world
    except Exception as e:
        logging.warning(f"Failed to load world model: {e}")
        return None


def extract_additional_features(
    frames: List[np.ndarray],
    use_gabor: bool = True,
    use_lbp: bool = True
) -> np.ndarray:
    """Extract Gabor and LBP features from frames to match enrollment pipeline.
    
    Parameters
    ----------
    frames : list of np.ndarray
        Pre-processed face images.
    use_gabor : bool
        Whether to extract Gabor features.
    use_lbp : bool
        Whether to extract LBP features.
    
    Returns
    -------
    X : np.ndarray
        Flattened images with optional Gabor and LBP features concatenated.
    """
    # Get configuration parameters
    if CONFIG_AVAILABLE:
        use_gabor = use_gabor and config.USE_GABOR_FEATURES
        use_lbp = use_lbp and config.USE_LBP_FEATURES
    
    # OPTIMIZATION: Use Gabor+LBP only (no raw pixels)
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
    
    # Fallback to pixel features if no texture features available
    if X is None:
        logging.warning("No texture features available, falling back to raw pixels")
        X = np.array([f.ravel() for f in frames])
    
    return X



def capture_verification_frames(
    video_source: int | str = 0,
    num_frames: int = 10,
    detect_interval: int = 5,
) -> Tuple[List[np.ndarray], List[PoseInfo]]:
    """Capture a small number of frames and associated pose measurements.

    Now uses Mediapipe Face Mesh for landmark detection when available.

    Returns
    -------
    frames : list of np.ndarray
        Pre-processed face images.
    poses : list of PoseInfo
        Pose information including yaw, pitch, roll, distance, R, and t.
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

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i % detect_interval == 0 or prev_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                maxSize=max_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(detections) > 0:
                x, y, w, h = max(detections, key=lambda bb: bb[2] * bb[3])
                prev_box = (int(x), int(y), int(w), int(h))
            else:
                prev_box = None

        if prev_box is None:
            continue

        x, y, w, h = prev_box
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        face = frame[y : y + h, x : x + w].copy()
        processed = preprocess_face_wrapper(face)
        frames.append(processed)

        # Estimate pose using landmarks if available
        if landmark_detector is not None:
            landmarks = landmark_detector.detect_landmarks(frame)
            pose = estimate_pose_from_landmarks(frame, landmarks, prev_box)
            poses.append(pose)
        else:
            if LANDMARKS_MODULE_AVAILABLE:
                pose = estimate_pose_from_bbox(frame, prev_box)
                poses.append(pose)
            else:
                yaw, pitch, roll, dist = estimate_pose_from_bbox_fallback(frame, prev_box)
                # Create PoseInfo with identity R and zero t for fallback
                pose = PoseInfo(yaw=yaw, pitch=pitch, roll=roll, distance=dist,
                               R=np.eye(3), t=np.zeros((3, 1)))
                poses.append(pose)

    cap.release()
    if landmark_detector is not None:
        landmark_detector.close()

    return frames, poses


def decision_threshold(base_threshold: float, pose_diff: np.ndarray) -> float:
    """Adapt the log‑likelihood threshold based on pose difference.

    A simple linear scaling is used here: the more the current pose
    deviates from the reference pose, the lower the required log
    likelihood (i.e. the threshold is loosened).  This can be tuned
    experimentally.

    Parameters
    ----------
    base_threshold : float
        Baseline log‑likelihood threshold determined empirically.
    pose_diff : np.ndarray
        Absolute difference between current pose and reference pose
        (shape (4,)).

    Returns
    -------
    float
        Adjusted threshold.
    """
    weight = 0.5  # scaling factor; tune based on experiments
    return base_threshold - weight * np.linalg.norm(pose_diff)


def verify(
    model_path: str,
    video_source: int | str = 0,
    num_frames: int = 10,
    detect_interval: int = 5,
    base_threshold: float = -50.0,
) -> Tuple[bool, float]:
    """Run the authentication pipeline and return a decision and score.

    Parameters
    ----------
    model_path : str
        Path to the saved model produced by Engine 1.
    video_source : int or str
        Camera index or URL.
    num_frames : int
        Number of frames to capture during verification.
    detect_interval : int
        Face detection frequency.
    base_threshold : float
        Baseline log‑likelihood threshold.  Empirically determine this
        based on false accept/reject rates.

    Returns
    -------
    is_verified : bool
        True if the user is authenticated, False otherwise.
    score : float
        Mean log‑likelihood score of the captured samples.
    """
    pca, gmm, scaler, pose_mean, R_ref, t_ref = load_model(model_path)
    frames, poses = capture_verification_frames(
        video_source=video_source,
        num_frames=num_frames,
        detect_interval=detect_interval,
    )
    if not frames:
        logging.warning("No frames captured during verification")
        return False, float("-inf")
    
    # Extract features (with Gabor and LBP to match enrollment)
    X = extract_additional_features(frames, use_gabor=True, use_lbp=True)
    
    # Convert PoseInfo to arrays (yaw, pitch, roll only)
    pose_arr = np.array([[p.yaw, p.pitch, p.roll] for p in poses])
    X_full = np.concatenate([X, pose_arr], axis=1)
    X_scaled = scaler.transform(X_full)
    scores = pca.transform(X_scaled)
    log_likelihood = gmm.score_samples(scores)
    mean_ll = float(np.mean(log_likelihood))
    current_pose_mean = np.mean(pose_arr, axis=0)
    pose_diff = np.abs(current_pose_mean - pose_mean)
    threshold = decision_threshold(base_threshold, pose_diff)
    is_verified = mean_ll >= threshold
    return is_verified, mean_ll


def verify_with_confidence(
    model_path: str,
    video_source: int | str = 0,
    num_frames: int = 10,
    detect_interval: int = 5,
    base_threshold: float | None = None,
    user_id: str | None = None,
    calibration_dir: str = "calibration_results",
    world_model_path: str | None = None,
) -> Tuple[bool, float, float, Optional[float]]:
    """Run authentication with robust scoring and confidence estimation.

    This improved verification function uses:
    - Robust statistics (median + Q1) instead of simple mean
    - Multi-factor confidence score
    - Anomaly detection
    - Adaptive threshold based on confidence
    - Differential pose weighting
    - Optional LLR (log-likelihood ratio) with world model

    Parameters
    ----------
    model_path : str
        Path to the saved model produced by Engine 1.
    video_source : int or str
        Camera index or URL.
    num_frames : int
        Number of frames to capture during verification.
    detect_interval : int
        Face detection frequency.
    base_threshold : float or None
        Baseline log‑likelihood threshold. If None and user_id provided,
        will attempt to load from calibration file.
    user_id : str or None
        User identifier for loading calibrated threshold.
    calibration_dir : str
        Directory containing calibration results.
    world_model_path : str or None
        Path to world/impostor GMM model for LLR calculation.
        If None, uses config.WORLD_GMM_MODEL_PATH.

    Returns
    -------
    is_verified : bool
        True if the user is authenticated, False otherwise.
    score : float
        Robust log‑likelihood score (weighted median + Q1).
    confidence : float
        Confidence score in range [0.0, 1.0].
    llr : float or None
        Log-likelihood ratio if world model used, else None.
    """
    from pathlib import Path
    import json
    
    # Load configuration parameters
    if CONFIG_AVAILABLE:
        pose_weights = np.array(config.POSE_WEIGHTS)
        min_confidence = config.MIN_CONFIDENCE_THRESHOLD
        max_score_std = config.MAX_SCORE_STD
        strict_offset = config.STRICT_THRESHOLD_OFFSET
        max_bad_ratio = config.MAX_BAD_FRAME_RATIO
        bad_threshold_offset = config.BAD_FRAME_THRESHOLD_OFFSET
        confidence_weights = np.array(config.CONFIDENCE_WEIGHTS)
        robust_weights = np.array(config.ROBUST_SCORE_WEIGHTS)
    else:
        pose_weights = np.array([1.0, 0.8, 0.6, 0.5])
        min_confidence = 0.3
        max_score_std = 15.0
        strict_offset = 10.0
        max_bad_ratio = 0.3
        bad_threshold_offset = 30.0
        confidence_weights = np.array([0.3, 0.3, 0.4])
        robust_weights = np.array([0.6, 0.4])
    
    # Load world model for LLR if available
    gmm_world = None
    use_llr = False
    llr_threshold = None
    
    if world_model_path is None and CONFIG_AVAILABLE:
        world_model_path = config.WORLD_GMM_MODEL_PATH
        use_llr = config.USE_LLR_VERIFICATION
    
    if world_model_path is not None:
        gmm_world = load_world_model(world_model_path)
        if gmm_world is not None:
            use_llr = True
    
    # Load calibrated threshold if available
    if base_threshold is None:
        if user_id is not None:
            threshold_file = Path(calibration_dir) / f"{user_id}_threshold.json"
            if threshold_file.exists():
                try:
                    with open(threshold_file, 'r') as f:
                        calibration_data = json.load(f)
                    base_threshold = calibration_data['base_threshold']
                    # Load LLR threshold if using LLR
                    if use_llr:
                        llr_threshold = calibration_data.get('llr_threshold')
                        if llr_threshold is not None:
                            logging.info(f"Loaded LLR threshold for {user_id}: {llr_threshold:.4f}")
                    logging.info(f"Loaded calibrated threshold for {user_id}: {base_threshold:.4f}")
                except Exception as e:
                    logging.warning(f"Failed to load calibrated threshold: {e}")
                    base_threshold = -50.0
            else:
                logging.warning(f"No calibration file found for {user_id}, using default threshold")
                base_threshold = -50.0
        else:
            base_threshold = -50.0
            logging.info("No user_id provided, using default threshold")
    
    # Load model and capture frames
    pca, gmm, scaler, pose_mean, R_ref, t_ref, use_dtw, dtw_template = load_model(model_path)
    frames, poses = capture_verification_frames(
        video_source=video_source,
        num_frames=num_frames,
        detect_interval=detect_interval,
    )
    
    if not frames:
        logging.warning("No frames captured during verification")
        return False, float("-inf"), 0.0, None
    
    # Extract features (with Gabor and LBP to match enrollment)
    X = extract_additional_features(frames, use_gabor=True, use_lbp=True)
    
    # Convert PoseInfo to arrays for PCA (yaw, pitch, roll only)
    pose_arr = np.array([[p.yaw, p.pitch, p.roll] for p in poses])
    X_full = np.concatenate([X, pose_arr], axis=1)
    X_scaled = scaler.transform(X_full)
    scores = pca.transform(X_scaled)
    
    # Calculate log-likelihood for user model
    log_likelihood = gmm.score_samples(scores)
    
    # Calculate LLR if world model available
    llr_scores = None
    decision_scores = log_likelihood
    decision_threshold_val = base_threshold
    score_type = "LL"
    
    if use_llr and gmm_world is not None:
        log_likelihood_world = gmm_world.score_samples(scores)
        llr_scores = log_likelihood - log_likelihood_world
        decision_scores = llr_scores
        decision_threshold_val = llr_threshold if llr_threshold is not None else 0.0
        score_type = "LLR"
        logging.info(f"LLR range: [{llr_scores.min():.2f}, {llr_scores.max():.2f}]")
    
    # 1. Robust score calculation (median + Q1)
    median_score = float(np.median(decision_scores))
    q1_score = float(np.percentile(decision_scores, 25))
    robust_score = robust_weights[0] * median_score + robust_weights[1] * q1_score
    
    # 2. Consistency (inverse of variance)
    score_std = float(np.std(decision_scores))
    consistency = 1.0 / (1.0 + score_std)
    
    # 3. Pose quality (differential weighting)
    current_pose_mean = np.mean(pose_arr, axis=0)
    pose_diff = np.abs(current_pose_mean - pose_mean)
    weighted_pose_diff = np.dot(pose_diff, pose_weights)
    pose_quality = 1.0 / (1.0 + weighted_pose_diff)
    
    # 3b. Orientation quality (3D reference frame)
    # Load orientation weights from config
    if CONFIG_AVAILABLE and hasattr(config, 'USE_3D_REPERE') and config.USE_3D_REPERE:
        use_3d_repere = True
        orient_weights = config.ORIENT_WEIGHTS
    else:
        use_3d_repere = False
        orient_weights = (1.0, 0.8, 0.6, 0.5)
    
    # Compute mean orientation penalty and quality
    if use_3d_repere and R_ref is not None and t_ref is not None:
        # Calculate orientation penalty for each frame
        orient_penalties = []
        orient_qualities = []
        for pose_info in poses:
            penalty, quality = compute_orientation_penalty(
                pose_info.R, pose_info.t, R_ref, t_ref, orient_weights
            )
            orient_penalties.append(penalty)
            orient_qualities.append(quality)
        
        # Use median orientation quality for robustness
        orientation_quality = float(np.median(orient_qualities))
        mean_orient_penalty = float(np.mean(orient_penalties))
        logging.debug(f"  3D orientation: quality={orientation_quality:.3f}, "
                     f"penalty={mean_orient_penalty:.3f}")
    else:
        orientation_quality = 1.0  # Neutral if not using 3D repere
        mean_orient_penalty = 0.0
    
    # 4. Good score ratio (using appropriate threshold)
    if use_llr:
        strict_threshold = decision_threshold_val + strict_offset
    else:
        strict_threshold = base_threshold + strict_offset
    good_ratio = float(np.mean(decision_scores >= strict_threshold))
    
    # 5. Composite confidence score
    # Integrate orientation quality: adjust confidence weights dynamically
    if use_3d_repere and orientation_quality < 1.0:
        # When using 3D orientation, blend it with pose quality
        combined_pose_quality = 0.6 * pose_quality + 0.4 * orientation_quality
        confidence = (
            confidence_weights[0] * consistency +
            confidence_weights[1] * combined_pose_quality +
            confidence_weights[2] * good_ratio
        )
    else:
        confidence = (
            confidence_weights[0] * consistency +
            confidence_weights[1] * pose_quality +
            confidence_weights[2] * good_ratio
        )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    # 6. Anomaly detection
    # Reject if variance too high (possible spoofing)
    if max_score_std is not None and score_std > max_score_std:
        logging.warning(f"High score variance detected: {score_std:.2f} > {max_score_std}")
        return False, robust_score, 0.0, (robust_score if use_llr else None)
    
    # Reject if too many very bad frames
    if use_llr:
        very_bad_threshold = decision_threshold_val - bad_threshold_offset
    else:
        very_bad_threshold = base_threshold - bad_threshold_offset
    very_bad_frames = int(np.sum(decision_scores < very_bad_threshold))
    bad_frame_ratio = very_bad_frames / len(decision_scores)
    if bad_frame_ratio > max_bad_ratio:
        logging.warning(
            f"Too many bad frames: {very_bad_frames}/{len(decision_scores)} "
            f"({bad_frame_ratio:.1%} > {max_bad_ratio:.1%})"
        )
        return False, robust_score, 0.0, (robust_score if use_llr else None)
    
    # 7. Adaptive threshold based on confidence
    # Higher confidence → less pose adaptation
    adaptation_factor = 0.5 * (1.0 - confidence)
    pose_penalty = adaptation_factor * weighted_pose_diff
    
    # Add orientation penalty if using 3D repere
    if use_3d_repere and mean_orient_penalty > 0:
        total_penalty = pose_penalty + adaptation_factor * mean_orient_penalty
    else:
        total_penalty = pose_penalty
    
    threshold = decision_threshold_val - total_penalty
    
    # 8. Final decision
    is_verified = (robust_score >= threshold) and (confidence >= min_confidence)
    
    # Logging
    logging.debug(f"Verification details ({score_type}):")
    logging.debug(f"  Robust score: {robust_score:.2f} (median={median_score:.2f}, Q1={q1_score:.2f})")
    if use_3d_repere and orientation_quality < 1.0:
        logging.debug(f"  Confidence: {confidence:.3f} (consistency={consistency:.3f}, "
                     f"pose_quality={pose_quality:.3f}, orient_quality={orientation_quality:.3f}, "
                     f"good_ratio={good_ratio:.3f})")
    else:
        logging.debug(f"  Confidence: {confidence:.3f} (consistency={consistency:.3f}, "
                     f"pose_quality={pose_quality:.3f}, good_ratio={good_ratio:.3f})")
    if use_3d_repere and mean_orient_penalty > 0:
        logging.debug(f"  Threshold: {threshold:.2f} (base={decision_threshold_val:.2f}, "
                     f"pose_penalty={pose_penalty:.2f}, orient_penalty={mean_orient_penalty:.2f})")
    else:
        logging.debug(f"  Threshold: {threshold:.2f} (base={decision_threshold_val:.2f}, "
                     f"penalty={pose_penalty:.2f})")
    logging.debug(f"  Score std: {score_std:.2f}, Bad frames: {very_bad_frames}/{len(decision_scores)}")
    if use_llr:
        logging.debug(f"  LLR enabled: user_ll range=[{log_likelihood.min():.2f}, {log_likelihood.max():.2f}]")
    logging.debug(f"  Decision: {'ACCEPT' if is_verified else 'REJECT'}")
    
    # Return LLR score if used
    llr_value = robust_score if use_llr else None
    return is_verified, robust_score, confidence, llr_value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, score = verify(
        model_path="enrolment_model.npz",
        video_source=0,
        num_frames=5,
        detect_interval=5,
        base_threshold=-30.0,
    )
    logging.info("Verification result: %s (score=%.2f)", ok, score)