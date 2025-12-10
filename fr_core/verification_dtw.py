"""
Verification with DTW - FR_VERS_JP 2.0
========================================

DTW-based verification pipeline that matches PCA sequences instead of using GMM.
This module provides a more stable and interpretable alternative to GMM-based verification.

Features:
- Dynamic Time Warping for sequence alignment
- Distance-based decision (more intuitive than log-likelihood)
- Compatible with existing 3D pose tracking
- Optimizable architecture for future enhancements

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np

# Import DTW library
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logging.warning("dtaidistance not available. Install with: pip install dtaidistance")

# Import from verification module
from fr_core.verification import (
    load_model,
    capture_verification_frames,
    extract_additional_features,
    compute_orientation_penalty,
)

# Import landmark utilities
from fr_core.landmark_utils import (
    is_landmark_model,
    extract_landmarks_from_video,
    N_LANDMARK_FEATURES,
)

# Import configuration
try:
    from fr_core.config import (
        DTW_THRESHOLD as DEFAULT_DTW_THRESHOLD,
        USE_DDTW,
        DDTW_METHOD,
        DDTW_NORMALIZE
    )
    CONFIG_AVAILABLE = True
except ImportError:
    DEFAULT_DTW_THRESHOLD = 6.71  # Fallback value (calibrated for landmarks)
    USE_DDTW = False
    DDTW_METHOD = 'none'
    DDTW_NORMALIZE = True
    CONFIG_AVAILABLE = False

# Import DDTW module
try:
    from fr_core.ddtw import compute_ddtw_distance
    DDTW_AVAILABLE = True
except ImportError:
    DDTW_AVAILABLE = False
    logging.warning("DDTW module not available")


def compute_dtw_distance(
    sequence1: np.ndarray,
    sequence2: np.ndarray,
    window: Optional[int] = None,
    use_c: bool = True,
) -> float:
    """Compute DTW distance between two PCA sequences.
    
    This is the core DTW matching function. It can be optimized later
    without changing the overall architecture.
    
    Parameters
    ----------
    sequence1 : np.ndarray
        First sequence (n_frames1 × n_components)
    sequence2 : np.ndarray
        Second sequence (n_frames2 × n_components)
    window : int, optional
        Sakoe-Chiba window constraint for faster computation
    use_c : bool
        Use C-optimized DTW implementation (faster)
    
    Returns
    -------
    float
        DTW distance (unnormalized)
    """
    if not DTW_AVAILABLE:
        # Fallback to simple euclidean distance between means
        logging.warning("DTW not available, using simple distance")
        mean1 = np.mean(sequence1, axis=0)
        mean2 = np.mean(sequence2, axis=0)
        return float(np.linalg.norm(mean1 - mean2))
    
    # For multivariate sequences, compute DTW on each dimension and sum
    # sequence1: (n_frames1, n_features), sequence2: (n_frames2, n_features)
    if sequence1.ndim == 2 and sequence2.ndim == 2:
        total_distance = 0.0
        n_features = sequence1.shape[1]
        
        for i in range(n_features):
            seq1_1d = sequence1[:, i].astype(np.float64)
            seq2_1d = sequence2[:, i].astype(np.float64)
            
            if window is not None:
                dist = dtw.distance(seq1_1d, seq2_1d, window=window, use_c=use_c)
            else:
                dist = dtw.distance(seq1_1d, seq2_1d, use_c=use_c)
            
            total_distance += dist
        
        return float(total_distance)
    
    # For 1D sequences (legacy)
    else:
        if window is not None:
            distance = dtw.distance(sequence1, sequence2, window=window, use_c=use_c)
        else:
            distance = dtw.distance(sequence1, sequence2, use_c=use_c)
        
        return float(distance)


def normalize_dtw_distance(
    dtw_distance: float,
    len1: int,
    len2: int,
    normalization_method: str = "average",
) -> float:
    """Normalize DTW distance by sequence lengths.
    
    Parameters
    ----------
    dtw_distance : float
        Raw DTW distance
    len1, len2 : int
        Lengths of the two sequences
    normalization_method : str
        'average', 'min', 'max', or 'sum'
    
    Returns
    -------
    float
        Normalized DTW distance
    """
    if normalization_method == "average":
        norm_factor = (len1 + len2) / 2.0
    elif normalization_method == "min":
        norm_factor = min(len1, len2)
    elif normalization_method == "max":
        norm_factor = max(len1, len2)
    elif normalization_method == "sum":
        norm_factor = len1 + len2
    else:
        norm_factor = 1.0
    
    return dtw_distance / max(norm_factor, 1.0)


def verify_dtw(
    model_path: str | Path,
    video_source: int | str = 0,
    num_frames: int = 10,
    detect_interval: int = 5,
    dtw_threshold: float = None,  # Will use DEFAULT_DTW_THRESHOLD if None
    window: Optional[int] = 10,
    normalize: bool = True,
    check_liveness: bool = True,  # Enable liveness detection
) -> Tuple[bool, float]:
    """Verify user identity using DTW distance.
    
    Parameters
    ----------
    model_path : str
        Path to enrolled model (.npz file)
    video_source : int or str
        Camera index or video file
    num_frames : int
        Number of frames to capture for verification
    detect_interval : int
        Face detection interval
    dtw_threshold : float, optional
        DTW distance threshold for verification (default: None, uses config value 6.71)
        Calibrated for landmarks: 6.71 (separation +1.02, FAR 0%, FRR 0%)
        Lower = stricter, Higher = more permissive
    window : int, optional
        Sakoe-Chiba window for DTW (default: 10, optimized for consistency)
    normalize : bool
        Whether to normalize DTW distance by sequence length
    
    Returns
    -------
    is_verified : bool
        True if verified
    dtw_distance : float
        DTW distance score
    """
    # Use default threshold from config if not provided
    if dtw_threshold is None:
        dtw_threshold = DEFAULT_DTW_THRESHOLD
        logging.info(f"Using calibrated DTW threshold: {dtw_threshold:.2f}")
    
    # STEP 1: Liveness Detection (Anti-Spoofing)
    if check_liveness:
        try:
            from fr_core.config import (
                USE_LIVENESS,
                LIVENESS_METHODS,
                LIVENESS_CONFIDENCE_THRESHOLD
            )
            from fr_core.liveness import check_liveness_fusion
            
            if USE_LIVENESS:
                logging.info("🛡️  Starting liveness detection...")
                
                liveness_result = check_liveness_fusion(
                    video_source=video_source,
                    use_blink='blink' in LIVENESS_METHODS,
                    use_motion='motion' in LIVENESS_METHODS,
                    use_texture='texture' in LIVENESS_METHODS,
                    show_debug=False
                )
                
                logging.info(f"Liveness: {liveness_result.is_live}, confidence={liveness_result.confidence:.2%}")
                
                if not liveness_result.is_live or liveness_result.confidence < LIVENESS_CONFIDENCE_THRESHOLD:
                    logging.warning(f"⚠️  Liveness check FAILED (confidence={liveness_result.confidence:.2%})")
                    return False, float('inf')  # Reject: suspected spoof
                
                logging.info("✓ Liveness check PASSED")
        except ImportError:
            logging.warning("Liveness module not available, skipping")
        except Exception as e:
            logging.warning(f"Liveness check error: {e}, proceeding without liveness")
    
    # STEP 2: Load model and verify identity
    # Load model
    pca, gmm, scaler, pose_mean, R_ref, t_ref, use_dtw, dtw_template = load_model(model_path)
    
    if not use_dtw or dtw_template is None:
        raise ValueError("Model was not trained in DTW mode. Re-enroll with DTW enabled.")
    
    # Detect feature type and extract accordingly
    use_landmarks = is_landmark_model(scaler)
    
    if use_landmarks:
        # Extract landmarks directly (136 features per frame)
        X_full, n_valid = extract_landmarks_from_video(
            video_source=video_source,
            num_frames=num_frames,
        )
        
        if n_valid == 0:
            logging.warning("No frames with valid landmarks during verification")
            return False, float("inf")
    else:
        # Legacy Gabor+LBP extraction
        frames, poses = capture_verification_frames(
            video_source=video_source,
            num_frames=num_frames,
            detect_interval=detect_interval,
        )
        
        if not frames:
            logging.warning("No frames captured during verification")
            return False, float("inf")
        
        # Extract Gabor + LBP features
        X = extract_additional_features(frames, use_gabor=True, use_lbp=True)
        
        # Add pose features
        pose_arr = np.array([[p.yaw, p.pitch, p.roll] for p in poses])
        X_full = np.concatenate([X, pose_arr], axis=1)
    
    # Apply scaling and PCA
    X_scaled = scaler.transform(X_full)
    verification_sequence = pca.transform(X_scaled)
    
    # Compute DTW distance (with optional DDTW augmentation)
    if USE_DDTW and DDTW_AVAILABLE and DDTW_METHOD != 'none':
        # Use Derivative DTW for enhanced discrimination
        dtw_dist, static_dist = compute_ddtw_distance(
            dtw_template,
            verification_sequence,
            method=DDTW_METHOD,
            normalize=DDTW_NORMALIZE,
            window=window,
        )
        logging.info(f"DDTW enabled: method={DDTW_METHOD}, static={static_dist:.2f}, ddtw={dtw_dist:.2f}")
    else:
        # Classic DTW
        dtw_dist = compute_dtw_distance(
            dtw_template,
            verification_sequence,
            window=window,
        )
    
    # Normalize if requested
    if normalize:
        dtw_dist = normalize_dtw_distance(
            dtw_dist,
            len(dtw_template),
            len(verification_sequence),
            normalization_method="average"
        )
    
    # Decision
    is_verified = (dtw_dist < dtw_threshold)
    
    logging.info(f"DTW verification: distance={dtw_dist:.2f}, threshold={dtw_threshold:.2f}, verified={is_verified}")
    
    return is_verified, dtw_dist


# Backward compatibility: if GMM model, use original verify
def verify(
    model_path: str,
    video_source: int | str = 0,
    num_frames: int = 10,
    detect_interval: int = 5,
    base_threshold: float = -50.0,
    pose_weights: Tuple[float, float, float] = (1.0, 0.8, 0.6),
) -> Tuple[bool, float]:
    """Unified verify function that auto-detects DTW vs GMM mode.
    
    This function checks the model type and calls the appropriate verification method.
    """
    # Try to load model and detect mode
    try:
        _, _, _, _, _, _, use_dtw, dtw_template = load_model(model_path)
        
        if use_dtw and dtw_template is not None:
            # DTW mode
            return verify_dtw(
                model_path=model_path,
                video_source=video_source,
                num_frames=num_frames,
                detect_interval=detect_interval,
            )
        else:
            # GMM mode - use original verify
            from fr_core.verification import verify as verify_gmm
            return verify_gmm(
                model_path=model_path,
                video_source=video_source,
                num_frames=num_frames,
                detect_interval=detect_interval,
                base_threshold=base_threshold,
                pose_weights=pose_weights,
            )
    except Exception as e:
        logging.error(f"Error in verify: {e}")
        # Fallback to GMM
        from fr_core.verification import verify as verify_gmm
        return verify_gmm(
            model_path=model_path,
            video_source=video_source,
            num_frames=num_frames,
            detect_interval=detect_interval,
            base_threshold=base_threshold,
            pose_weights=pose_weights,
        )
