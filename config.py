"""
Configuration module for face recognition pipeline
--------------------------------------------------

This module centralizes all configurable parameters for the face
detection and recognition pipeline, allowing easy tuning without
modifying the core engine code.

Adjust these parameters based on your hardware, lighting conditions
and accuracy requirements.
"""

from __future__ import annotations

# ========================================================================
# Haar Cascade face detection parameters
# ========================================================================

# Scale factor: specifies how much the image size is reduced at each
# image scale. Smaller values (e.g., 1.05) increase detection accuracy
# but reduce speed. Larger values (e.g., 1.3) speed up detection but
# may miss faces.
HAAR_SCALE_FACTOR: float = 1.05

# Minimum neighbors: specifies how many neighbors each candidate
# rectangle should have to retain it. Higher values result in fewer
# detections but with higher quality. Lower values increase sensitivity
# but may produce false positives.
HAAR_MIN_NEIGHBORS: int = 3

# Minimum face size: minimum possible object size. Objects smaller than
# this are ignored (width, height).
HAAR_MIN_SIZE: tuple[int, int] = (30, 30)

# Maximum face size: maximum possible object size. Objects larger than
# this are ignored. None means no limit.
HAAR_MAX_SIZE: tuple[int, int] | None = None


# ========================================================================
# Preprocessing parameters
# ========================================================================

# Target face image size after preprocessing (width, height)
FACE_SIZE: tuple[int, int] = (64, 64)

# Quantization levels: reduce grayscale to this many levels to
# reduce noise. 8 levels means divide by 32 (256 / 32 = 8).
QUANTIZATION_LEVELS: int = 8

# Wavelet decomposition level for illumination normalization
WAVELET_LEVEL: int = 2

# Wavelet levels (alternative name for consistency, same as WAVELET_LEVEL)
WAVELET_LEVELS: int = 2

# Wavelet family to use (e.g., 'db1', 'haar', 'sym5', 'coif1')
WAVELET_FAMILY: str = 'db1'

# Noise threshold factor for wavelet detail coefficient filtering
WAVELET_NOISE_THRESHOLD: float = 0.5

# Enable wavelet-based illumination normalization (requires pywavelets)
USE_WAVELET_NORMALIZATION: bool = True

# Enable bilateral filtering for noise reduction
USE_BILATERAL_FILTER: bool = True

# Bilateral filter parameters
BILATERAL_D: int = 9  # diameter of pixel neighborhood
BILATERAL_SIGMA_COLOR: float = 75.0  # filter sigma in color space
BILATERAL_SIGMA_SPACE: float = 75.0  # filter sigma in coordinate space


# ========================================================================
# PCA and GMM parameters
# ========================================================================

# Number of principal components to retain in PCA
N_PCA_COMPONENTS: int = 5

# PCA variance threshold: retain components explaining this % of variance
PCA_VARIANCE_THRESHOLD: float = 0.95

# Range of GMM components to test when selecting optimal model
GMM_COMPONENT_RANGE: tuple[int, int] = (1, 6)

# GMM covariance type ('full', 'tied', 'diag', 'spherical')
GMM_COVARIANCE_TYPE: str = 'full'

# Path to world/impostor GMM model (optional, for LLR calculation)
WORLD_GMM_MODEL_PATH: str | None = None

# Use LLR (log-likelihood ratio) for verification (requires world model)
USE_LLR_VERIFICATION: bool = False


# ========================================================================
# Gabor and LBP feature extraction parameters
# ========================================================================

# Enable Gabor feature extraction
USE_GABOR_FEATURES: bool = True

# Number of Gabor filter orientations
GABOR_ORIENTATIONS: int = 4

# Gabor filter frequencies
GABOR_FREQUENCIES: tuple[float, ...] = (0.1, 0.3)

# Gabor kernel size
GABOR_KSIZE: int = 31

# Enable LBP feature extraction
USE_LBP_FEATURES: bool = True

# LBP radius
LBP_RADIUS: int = 1

# LBP number of points
LBP_N_POINTS: int = 8

# Random seed for reproducibility
RANDOM_SEED: int = 42


# ========================================================================
# Capture parameters
# ========================================================================

# Number of frames to capture during enrollment
ENROLMENT_NUM_FRAMES: int = 50

# Face detection interval: detect face every N frames
ENROLMENT_DETECT_INTERVAL: int = 5

# Number of frames to capture during verification
VERIFICATION_NUM_FRAMES: int = 10

# Face detection interval during verification
VERIFICATION_DETECT_INTERVAL: int = 5


# ========================================================================
# World model and LLR parameters
# ========================================================================

# Path to world/impostor GMM model (optional, for LLR calculation)
# Set to None to disable LLR verification
WORLD_GMM_MODEL_PATH: str | None = None

# Use LLR (log-likelihood ratio) for verification (requires world model)
USE_LLR_VERIFICATION: bool = False


# ========================================================================
# Authentication parameters
# ========================================================================

# Baseline log-likelihood threshold for authentication
# Lower (more negative) values are more permissive
BASE_THRESHOLD: float = -50.0

# Weight for pose-based threshold adaptation
POSE_ADAPTATION_WEIGHT: float = 0.5

# Pose component weights for differential adaptation
# Order: [yaw, pitch, roll, distance]
POSE_WEIGHTS: tuple[float, float, float, float] = (1.0, 0.8, 0.6, 0.5)

# Minimum confidence score for acceptance (0.0 - 1.0)
MIN_CONFIDENCE_THRESHOLD: float = 0.3

# Maximum standard deviation of log-likelihood scores (anomaly detection)
# Set to None to disable this check, or use a high value like 500.0
MAX_SCORE_STD: float | None = None  # Disabled - use other metrics for anomaly detection

# Strict threshold offset for "good score" ratio calculation
STRICT_THRESHOLD_OFFSET: float = 10.0

# Maximum proportion of very bad frames allowed
MAX_BAD_FRAME_RATIO: float = 0.5  # Allow up to 50% bad frames

# Very bad frame threshold offset (below base_threshold)
BAD_FRAME_THRESHOLD_OFFSET: float = 30.0

# Confidence calculation weights
# Order: [consistency, pose_quality, good_ratio]
CONFIDENCE_WEIGHTS: tuple[float, float, float] = (0.3, 0.3, 0.4)

# Robust score calculation weights
# Order: [median, Q1]
ROBUST_SCORE_WEIGHTS: tuple[float, float] = (0.6, 0.4)


# ========================================================================
# 3D Reference Frame (Repère Tridimensionnel) parameters
# ========================================================================

# Enable 3D head reference frame tracking
# When enabled, uses rotation matrix R and translation vector t from solvePnP
# to compute orientation penalties based on the head's 3D pose relative to
# enrollment reference pose
USE_3D_REPERE: bool = True

# Orientation component weights for 3D reference frame penalty calculation
# Order: [yaw_penalty, pitch_penalty, roll_penalty, distance_penalty]
# These weights determine how much each orientation component affects the
# adaptive threshold during verification
# - yaw_penalty: rotation around vertical axis (left/right head turn)
# - pitch_penalty: rotation around horizontal axis (up/down head tilt)
# - roll_penalty: rotation around depth axis (head tilt left/right)
# - distance_penalty: translation along depth axis (closer/farther from camera)
ORIENT_WEIGHTS: tuple[float, float, float, float] = (1.0, 0.8, 0.6, 0.5)


# ========================================================================
# Mediapipe Face Mesh parameters
# ========================================================================

# Enable Mediapipe face landmark detection (requires mediapipe)
USE_MEDIAPIPE_LANDMARKS: bool = True

# Mediapipe detection confidence threshold
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5

# Mediapipe tracking confidence threshold
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

# Maximum number of faces to detect with Mediapipe
MEDIAPIPE_MAX_NUM_FACES: int = 1

# Mediapipe model complexity (0, 1, or 2)
# Higher values are more accurate but slower
MEDIAPIPE_MODEL_COMPLEXITY: int = 1
