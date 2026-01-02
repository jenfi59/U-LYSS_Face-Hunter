"""
Helper functions for recalculating head poses for legacy models.

This module provides a utility to compute yaw, pitch and roll values for
pre‑recorded landmark sequences that do not include pose information.  It
leverages the existing ``utils.pose_estimation.calculate_head_pose`` function
to derive the head pose from 2‑D landmarks and a given image shape.  If the
pose cannot be computed for a frame, a default pose of (0, 0, 0) is used.

Usage:

    from utils.recalculate import recalculate_poses_from_landmarks

    poses = recalculate_poses_from_landmarks(landmarks, image_shape=(480, 640))

The function returns an array of shape (N, 3) with columns [yaw, pitch, roll].
"""

from typing import Tuple
import numpy as np

from .pose_estimation import calculate_head_pose


def recalculate_poses_from_landmarks(
    landmarks_seq: np.ndarray, image_shape: Tuple[int, int] = (480, 640)
) -> np.ndarray:
    """Recalculate head poses for a sequence of landmarks.

    This helper is intended for legacy enrollment models that were saved
    without pose information (version 2.2 and earlier).  It iterates over
    each frame's landmarks, computes the corresponding head pose using
    ``calculate_head_pose`` and returns the yaw, pitch and roll values for
    all frames.  If a pose cannot be determined (e.g., solver fails), a
    default pose of (0.0, 0.0, 0.0) is recorded.

    Args:
        landmarks_seq: Array of shape ``(N, num_landmarks, dims)`` containing
            2‑D landmarks for each frame.
        image_shape: A tuple (height, width) representing the original image
            dimensions used when capturing the landmarks.  Defaults to (480, 640).

    Returns:
        An array of shape ``(N, 3)`` with pose angles in degrees for each
        frame, in the order [yaw, pitch, roll].
    """
    if landmarks_seq.ndim != 3:
        raise ValueError(
            f"landmarks_seq must be 3‑D (N, num_landmarks, dims), got shape {landmarks_seq.shape}"
        )
    poses = []
    for lm in landmarks_seq:
        try:
            pose = calculate_head_pose(lm, image_shape)
            if pose is not None:
                yaw = float(pose['yaw'])
                pitch = float(pose['pitch'])
                roll = float(pose['roll'])
                poses.append([yaw, pitch, roll])
            else:
                poses.append([0.0, 0.0, 0.0])
        except Exception:
            # On exception, fallback to zero pose
            poses.append([0.0, 0.0, 0.0])
    return np.array(poses, dtype=np.float32)
