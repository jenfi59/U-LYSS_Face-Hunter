"""
Feature Extraction Module – FR_VERS_JP 2.0
===========================================

3D facial landmark detection and pose estimation using MediaPipe Face Mesh.

Features:
---------
- 468-point facial landmark detection
- 3D head pose estimation (yaw, pitch, roll)
- Distance-to-camera estimation
- Robust pose tracking across frames

Pose Features (4 dimensions):
-----------------------------
- yaw: Horizontal rotation (-180° to 180°)
- pitch: Vertical rotation (-90° to 90°)
- roll: Head tilt (-180° to 180°)  
- distance: Relative distance to camera (normalized)

Version: 2.0.0
"""

import logging
from typing import Tuple, Optional
import dataclasses

import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning(
        "Mediapipe not available. Facial landmark detection will fall back "
        "to synthetic landmarks based on bounding box. Install mediapipe for "
        "improved accuracy: pip install mediapipe"
    )


@dataclasses.dataclass
class PoseInfo:
    """Container for head pose parameters.

    Attributes
    ----------
    yaw : float
        Rotation around the vertical axis (in degrees).  Positive
        values denote a rotation to the left (looking right from the
        subject's perspective).
    pitch : float
        Rotation around the horizontal axis (in degrees).  Positive
        values denote an upward tilt (looking down from the subject's
        perspective).
    roll : float
        Rotation around the axis pointing away from the camera (in
        degrees).  Positive values denote a clockwise tilt.
    distance : float
        Estimated distance from the camera, often derived from the
        translation vector returned by `solvePnP`.
    R : np.ndarray
        3x3 rotation matrix from solvePnP representing the head orientation
        in camera space. Columns represent the head's local axes:
        - R[:,0]: horizontal axis (ear-to-ear, left to right)
        - R[:,1]: vertical axis (chin-to-forehead, bottom to top)
        - R[:,2]: depth axis (back-to-front, away from camera)
    t : np.ndarray
        3x1 translation vector from solvePnP representing the head position
        in camera space (in arbitrary units, typically normalized).
    """

    yaw: float
    pitch: float
    roll: float
    distance: float
    R: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((3, 1)))


def compute_head_axes(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized orthogonal axes from a rotation matrix.

    This function extracts the three orthonormal axes that define the head's
    3D reference frame from the rotation matrix returned by cv2.solvePnP.

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix from solvePnP.

    Returns
    -------
    axis_x : np.ndarray
        Horizontal axis (ear-to-ear, left to right), shape (3,).
        Corresponds to the first column of R, normalized.
    axis_y : np.ndarray
        Vertical axis (chin-to-forehead, bottom to top), shape (3,).
        Corresponds to the second column of R, normalized.
    axis_z : np.ndarray
        Depth axis (back-to-front, away from camera), shape (3,).
        Corresponds to the third column of R, normalized.

    Notes
    -----
    - axis_x represents lateral head rotation (left/right ear movement)
    - axis_y represents vertical head movement (chin/forehead)
    - axis_z represents depth/distance from camera
    - All axes are orthonormal by construction if R is a valid rotation matrix
    """
    # Extract columns from rotation matrix
    axis_x = R[:, 0].flatten()
    axis_y = R[:, 1].flatten()
    axis_z = R[:, 2].flatten()

    # Normalize to ensure unit vectors (should already be normalized for valid R)
    norm_x = np.linalg.norm(axis_x)
    norm_y = np.linalg.norm(axis_y)
    norm_z = np.linalg.norm(axis_z)

    if norm_x > 1e-6:
        axis_x = axis_x / norm_x
    if norm_y > 1e-6:
        axis_y = axis_y / norm_y
    if norm_z > 1e-6:
        axis_z = axis_z / norm_z

    return axis_x, axis_y, axis_z


def average_rotation_matrices(R_list: list) -> np.ndarray:
    """Compute the average of multiple rotation matrices.

    This function computes a mean rotation matrix from a list of rotation
    matrices using quaternion averaging for better accuracy.

    Parameters
    ----------
    R_list : list of np.ndarray
        List of 3x3 rotation matrices.

    Returns
    -------
    R_mean : np.ndarray
        Average 3x3 rotation matrix.

    Notes
    -----
    Uses quaternion representation for averaging to avoid gimbal lock issues.
    If scipy is not available, falls back to simple matrix averaging with
    SVD orthogonalization.
    """
    if len(R_list) == 0:
        return np.eye(3)
    
    if len(R_list) == 1:
        return R_list[0]
    
    try:
        from scipy.spatial.transform import Rotation as R_scipy
        # Convert rotation matrices to quaternions
        rotations = R_scipy.from_matrix(np.array(R_list))
        # Average quaternions
        mean_rotation = rotations.mean()
        # Convert back to matrix
        return mean_rotation.as_matrix()
    except ImportError:
        # Fallback: average matrices and orthogonalize using SVD
        R_mean = np.mean(np.array(R_list), axis=0)
        # Ensure orthogonality using SVD
        U, _, Vt = np.linalg.svd(R_mean)
        R_mean_ortho = U @ Vt
        # Ensure det(R) = 1 (proper rotation, not reflection)
        if np.linalg.det(R_mean_ortho) < 0:
            Vt[-1, :] *= -1
            R_mean_ortho = U @ Vt
        return R_mean_ortho


class FaceLandmarkDetector:
    """Facial landmark detector using Mediapipe Face Mesh.

    This class wraps Mediapipe's Face Mesh solution to detect 468 3D
    facial landmarks. If Mediapipe is not available, it falls back to
    synthetic landmark estimation.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        model_complexity: int = 1,
    ):
        """Initialize the landmark detector.

        Parameters
        ----------
        min_detection_confidence : float
            Minimum confidence for face detection.
        min_tracking_confidence : float
            Minimum confidence for face tracking.
        max_num_faces : int
            Maximum number of faces to detect.
        model_complexity : int
            Mediapipe model complexity (0, 1, or 2).
            Higher values are more accurate but slower.
        """
        self.use_mediapipe = MEDIAPIPE_AVAILABLE

        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                refine_landmarks=True,
            )
            logging.info("Mediapipe Face Mesh initialized successfully")
        else:
            self.face_mesh = None
            logging.warning("Using synthetic landmarks (Mediapipe not available)")

    def detect_landmarks(
        self, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect facial landmarks in an image.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR format).

        Returns
        -------
        landmarks : np.ndarray or None
            Array of shape (N, 3) containing 3D landmark coordinates,
            or None if no face detected. Coordinates are normalized
            to image dimensions.
        """
        if not self.use_mediapipe:
            return None

        # Convert BGR to RGB for Mediapipe
        import cv2
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # Extract landmarks from first detected face
        face_landmarks = results.multi_face_landmarks[0]
        height, width = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        landmarks = np.array([
            [lm.x * width, lm.y * height, lm.z * width]
            for lm in face_landmarks.landmark
        ])

        return landmarks

    def close(self):
        """Release resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()


def get_pose_estimation_points(landmarks: Optional[np.ndarray]) -> np.ndarray:
    """Extract key points for pose estimation from landmarks.

    Uses the following Mediapipe Face Mesh indices:
    - 33: Left eye left corner
    - 133: Left eye right corner
    - 362: Right eye left corner
    - 263: Right eye right corner
    - 1: Nose tip
    - 61: Left mouth corner
    - 291: Right mouth corner

    Parameters
    ----------
    landmarks : np.ndarray or None
        Full set of facial landmarks (468 points) or None.

    Returns
    -------
    points_2d : np.ndarray
        Array of shape (6, 2) containing 2D coordinates of key points.
        If landmarks is None, returns empty array.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.array([])

    # Mediapipe Face Mesh landmark indices for pose estimation
    POSE_INDICES = [
        33,   # Left eye left corner
        263,  # Right eye right corner
        1,    # Nose tip
        61,   # Left mouth corner
        291,  # Right mouth corner
        199,  # Chin
    ]

    points_2d = landmarks[POSE_INDICES, :2]  # Extract x, y coordinates
    return points_2d


def estimate_pose_from_landmarks(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> PoseInfo:
    """Estimate head pose using facial landmarks or bounding box fallback.

    Parameters
    ----------
    frame : np.ndarray
        Original video frame from which the face was detected.
    landmarks : np.ndarray or None
        Facial landmarks array of shape (468, 3), or None to use bbox.
    bbox : tuple of int or None
        Bounding box (x, y, w, h) of the detected face.
        Used as fallback if landmarks is None.

    Returns
    -------
    PoseInfo
        Estimated pose information.
    """
    import cv2
    import math

    height, width = frame.shape[:2]

    # Try to use landmarks first
    if landmarks is not None and len(landmarks) > 0:
        image_points = get_pose_estimation_points(landmarks)

        if len(image_points) == 0:
            # Fall back to bbox
            return estimate_pose_from_bbox(frame, bbox)

        # 3D model points for the key facial features
        # These are approximate 3D coordinates in millimeters
        model_points = np.array([
            (-30.0, -30.0, -30.0),  # Left eye left corner
            (30.0, -30.0, -30.0),   # Right eye right corner
            (0.0, 0.0, 0.0),        # Nose tip
            (-20.0, 30.0, -30.0),   # Left mouth corner
            (20.0, 30.0, -30.0),    # Right mouth corner
            (0.0, 50.0, -30.0),     # Chin
        ], dtype=np.float64)

    else:
        # Fall back to bbox-based synthetic landmarks
        return estimate_pose_from_bbox(frame, bbox)

    # Camera matrix approximation
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
    except Exception as e:
        logging.warning(f"solvePnP failed: {e}")
        success = False

    if not success:
        return PoseInfo(0.0, 0.0, 0.0, 1.0)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Compute Euler angles from rotation matrix
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

    # Distance: use normalized euclidean norm of translation vector
    # Dividing by 1000 to keep values in reasonable range (typically 0-10)
    if tvec.size >= 3:
        distance = float(np.linalg.norm(tvec) / 1000.0)
    else:
        distance = 1.0

    return PoseInfo(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        distance=distance,
        R=rmat,
        t=tvec
    )


def estimate_pose_from_bbox(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
) -> PoseInfo:
    """Approximate head pose from a face bounding box using solvePnP.

    This is a fallback method when Mediapipe landmarks are not available.
    It places synthetic 2D landmarks within the bounding box.

    Parameters
    ----------
    frame : np.ndarray
        Original video frame from which the face was detected.
    bbox : tuple of int or None
        Bounding box (x, y, w, h) of the detected face.

    Returns
    -------
    PoseInfo
        Estimated pose information.
    """
    import cv2
    import math

    if bbox is None:
        return PoseInfo(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            distance=1.0,
            R=np.eye(3),
            t=np.zeros((3, 1))
        )

    x, y, w, h = bbox
    height, width = frame.shape[:2]

    # Define 2D image points relative to the bounding box
    image_points = np.array(
        [
            (x + 0.3 * w, y + 0.3 * h),  # left eye
            (x + 0.7 * w, y + 0.3 * h),  # right eye
            (x + 0.5 * w, y + 0.5 * h),  # nose tip
            (x + 0.35 * w, y + 0.7 * h),  # left mouth corner
            (x + 0.65 * w, y + 0.7 * h),  # right mouth corner
        ],
        dtype=np.float64,
    )

    # Corresponding 3D model points (arbitrary scale)
    model_points = np.array(
        [
            (-30.0, -30.0, -30.0),  # left eye
            (30.0, -30.0, -30.0),   # right eye
            (0.0, 0.0, 0.0),        # nose tip
            (-20.0, 30.0, -30.0),   # left mouth
            (20.0, 30.0, -30.0),    # right mouth
        ],
        dtype=np.float64,
    )

    # Camera matrix approximation
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
        return PoseInfo(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            distance=1.0,
            R=np.eye(3),
            t=np.zeros((3, 1))
        )

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Compute Euler angles from rotation matrix
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

    # Distance: use normalized euclidean norm of translation vector
    # Dividing by 1000 to keep values in reasonable range (typically 0-10)
    if tvec.size >= 3:
        distance = float(np.linalg.norm(tvec) / 1000.0)
    else:
        distance = 1.0

    return PoseInfo(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        distance=distance,
        R=rmat,
        t=tvec
    )
