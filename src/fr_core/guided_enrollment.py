"""
U-LYSS Face Recognition - Guided Enrollment
3-zone enrollment with pose guidance (FRONTAL, LEFT, RIGHT)
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from enum import Enum
import logging
from collections import deque

from .config import get_config

logger = logging.getLogger(__name__)


class EnrollmentZone(Enum):
    """Enrollment zones based on head pose."""
    FRONTAL = "frontal"
    LEFT = "left"
    RIGHT = "right"


class GuidedEnrollment:
    """
    Guided enrollment with 3-zone pose coverage.
    Collects frames from FRONTAL, LEFT, and RIGHT poses.
    """

    def __init__(self):
        """Initialize guided enrollment."""
        self.config = get_config()

        # Zone definitions (yaw angles in degrees)
        self.zone_ranges = {
            EnrollmentZone.FRONTAL: (-15, 15),
            EnrollmentZone.LEFT: (-40, -10),
            EnrollmentZone.RIGHT: (10, 40)
        }

        # Frames per zone
        self.frames_per_zone = self.config.enrollment_n_frames // 3

        # Storage for collected frames
        self.collected_frames = {
            EnrollmentZone.FRONTAL: [],
            EnrollmentZone.LEFT: [],
            EnrollmentZone.RIGHT: []
        }

        # Track all captured poses per zone (for diversity check)
        self.all_captured_poses = {
            EnrollmentZone.FRONTAL: [],
            EnrollmentZone.LEFT: [],
            EnrollmentZone.RIGHT: []
        }

        # Current state
        self.current_zone = EnrollmentZone.FRONTAL
        self.is_complete = False

        # Minimum angle change between frames (from ALL previous captures in zone)
        self.min_angle_change = self.config.enrollment_min_angle_change

    def reset(self):
        """Reset enrollment state."""
        for zone in EnrollmentZone:
            self.collected_frames[zone].clear()
            self.all_captured_poses[zone].clear()
        self.current_zone = EnrollmentZone.FRONTAL
        self.is_complete = False

    def _pose_changed_enough_from_all(self, current_pose: Tuple[float, float, float],
                                      zone: EnrollmentZone) -> bool:
        """
        Check if current pose is sufficiently different from ALL previously captured poses
        in the specified zone. This ensures every frame has a distinct head position.

        Based on original code logic: checks against ALL previous captures, not just last one.

        Args:
            current_pose: (yaw, pitch, roll) tuple
            zone: EnrollmentZone to check against

        Returns:
            True if pose is different enough from ALL previous captures in this zone
        """
        previous_poses = self.all_captured_poses[zone]

        if len(previous_poses) == 0:
            return True  # First capture for this zone - always accept

        yaw, pitch, roll = current_pose

        # Check against ALL previously captured poses (not just the last one)
        for prev_yaw, prev_pitch, prev_roll in previous_poses:
            yaw_diff = abs(yaw - prev_yaw)
            pitch_diff = abs(pitch - prev_pitch)
            roll_diff = abs(roll - prev_roll)

            # If current pose is too similar to ANY previous pose, reject it
            # Must differ by at least min_angle_change in AT LEAST ONE axis
            if (yaw_diff < self.min_angle_change and
                pitch_diff < self.min_angle_change and
                roll_diff < self.min_angle_change):
                return False  # Too similar to this previous pose

        # Current pose is different enough from ALL previous poses
        return True

    def _get_current_zone_from_pose(self, yaw: float) -> Optional[EnrollmentZone]:
        """
        Determine which zone the current pose belongs to.

        Args:
            yaw: Head yaw angle in degrees

        Returns:
            EnrollmentZone or None if outside all zones
        """
        for zone, (min_yaw, max_yaw) in self.zone_ranges.items():
            if min_yaw <= yaw <= max_yaw:
                return zone
        return None

    def add_frame(self, landmarks: np.ndarray,
                  pose: Tuple[float, float, float]) -> dict:
        """
        Add a frame to enrollment if it meets quality criteria.

        Args:
            landmarks: Face landmarks (68, 3) with 3D coordinates
            pose: Head pose (yaw, pitch, roll) in degrees

        Returns:
            Dict with status information
        """
        yaw, pitch, roll = pose

        result = {
            'accepted': False,
            'zone': None,
            'frames_in_zone': 0,
            'total_frames': 0,
            'progress': 0.0,
            'next_zone': None,
            'is_complete': False,
            'message': ''
        }

        # Check if already complete
        if self.is_complete:
            result['message'] = "Enrollment already complete"
            result['is_complete'] = True
            return result

        # Determine current zone from pose (yaw only)
        detected_zone = self._get_current_zone_from_pose(yaw)

        if detected_zone is None:
            result['message'] = f"Pose outside target zones (yaw={yaw:.1f}°)"
            return result
        # Check if this is the zone we're currently collecting
        if detected_zone != self.current_zone:
            # Check if current zone is complete
            if len(self.collected_frames[self.current_zone]) < self.frames_per_zone:
                min_yaw, max_yaw = self.zone_ranges[self.current_zone]
                result['message'] = (f"Please stay in {self.current_zone.value} zone "
                                   f"({min_yaw}° to {max_yaw}°)")
                return result

        # Check if pose is different enough from ALL previous captures in this zone
        # This allows frames in any order, but ensures ≥1° spacing between all frames
        if not self._pose_changed_enough_from_all(pose, self.current_zone):
            result['message'] = f"Move head more - too close to existing frame (need ≥1° spacing)"
            return result

        # Add frame to current zone
        self.collected_frames[self.current_zone].append({
            'landmarks': landmarks.copy(),
            'pose': pose
        })

        # Track this pose
        self.all_captured_poses[self.current_zone].append(pose)

        frames_in_zone = len(self.collected_frames[self.current_zone])
        total_frames = sum(len(frames) for frames in self.collected_frames.values())

        result['accepted'] = True
        result['zone'] = self.current_zone
        result['frames_in_zone'] = frames_in_zone
        result['total_frames'] = total_frames
        result['progress'] = total_frames / self.config.enrollment_n_frames

        # Check if current zone is complete
        if frames_in_zone >= self.frames_per_zone:
            # Move to next zone
            if self.current_zone == EnrollmentZone.FRONTAL:
                self.current_zone = EnrollmentZone.LEFT
                result['next_zone'] = EnrollmentZone.LEFT
                result['message'] = "Frontal complete! Turn left (-40° to -10°)"
            elif self.current_zone == EnrollmentZone.LEFT:
                self.current_zone = EnrollmentZone.RIGHT
                result['next_zone'] = EnrollmentZone.RIGHT
                result['message'] = "Left complete! Turn right (10° to 40°)"
            elif self.current_zone == EnrollmentZone.RIGHT:
                # Enrollment complete!
                self.is_complete = True
                result['is_complete'] = True
                result['message'] = "Enrollment complete!"
        else:
            remaining = self.frames_per_zone - frames_in_zone
            result['message'] = f"{self.current_zone.value}: {remaining} frames remaining"

        return result

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------
    def get_all_landmarks_and_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all collected landmarks and poses across all zones.

        This helper concatenates the landmarks and poses accumulated during
        guided enrollment, preserving the order FRONTAL → LEFT → RIGHT.  It
        returns two arrays: the first of shape (N, 68, 3) containing the
        landmark coordinates and the second of shape (N, 3) containing the
        corresponding head poses (yaw, pitch, roll).

        Returns:
            Tuple ``(landmarks_array, poses_array)`` where
                ``landmarks_array`` has shape (N, 68, 3) and
                ``poses_array`` has shape (N, 3).
        """
        all_landmarks: List[np.ndarray] = []
        all_poses: List[Tuple[float, float, float]] = []
        # Order zones consistently
        for zone in [EnrollmentZone.FRONTAL, EnrollmentZone.LEFT, EnrollmentZone.RIGHT]:
            for frame_data in self.collected_frames[zone]:
                all_landmarks.append(frame_data['landmarks'])
                all_poses.append(frame_data['pose'])
        landmarks_array = np.array(all_landmarks, dtype=np.float32)
        poses_array = np.array(all_poses, dtype=np.float32)
        return landmarks_array, poses_array

    def get_all_landmarks(self) -> np.ndarray:
        """
        Get all collected landmarks as a single array.

        Returns:
            Array (n_frames, 68, 2) with all landmarks
        """
        all_landmarks = []

        for zone in [EnrollmentZone.FRONTAL, EnrollmentZone.LEFT, EnrollmentZone.RIGHT]:
            for frame_data in self.collected_frames[zone]:
                all_landmarks.append(frame_data['landmarks'])

        if not all_landmarks:
            raise ValueError("No frames collected yet")

        return np.array(all_landmarks)

    def get_enrollment_summary(self) -> dict:
        """Get summary of enrollment status."""
        return {
            'frontal_frames': len(self.collected_frames[EnrollmentZone.FRONTAL]),
            'left_frames': len(self.collected_frames[EnrollmentZone.LEFT]),
            'right_frames': len(self.collected_frames[EnrollmentZone.RIGHT]),
            'total_frames': sum(len(frames) for frames in self.collected_frames.values()),
            'target_frames': self.config.enrollment_n_frames,
            'is_complete': self.is_complete,
            'current_zone': self.current_zone.value
        }

    def draw_guidance(self, frame_bgr: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
        """
        Draw enrollment guidance overlay on frame.

        Args:
            frame: BGR image
            pose: Current head pose (yaw, pitch, roll)

        Returns:
            Frame with guidance overlay
        """
        yaw, pitch, roll = pose
        h, w = frame_bgr.shape[:2]

        # Create overlay
        overlay = frame_bgr.copy()

        # Draw zone indicator
        zone_y = 30
        for zone in [EnrollmentZone.FRONTAL, EnrollmentZone.LEFT, EnrollmentZone.RIGHT]:
            frames_collected = len(self.collected_frames[zone])
            target = self.frames_per_zone
            progress = min(frames_collected / target, 1.0)

            # Color: green if complete, yellow if current, gray if pending
            if frames_collected >= target:
                color = (0, 255, 0)
            elif zone == self.current_zone:
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)

            # Draw progress bar
            bar_x = 10
            bar_width = 200
            bar_height = 20

            cv2.rectangle(overlay, (bar_x, zone_y),
                         (bar_x + bar_width, zone_y + bar_height),
                         (200, 200, 200), -1)
            cv2.rectangle(overlay, (bar_x, zone_y),
                         (bar_x + int(bar_width * progress), zone_y + bar_height),
                         color, -1)

            # Text
            text = f"{zone.value.upper()}: {frames_collected}/{target}"
            cv2.putText(overlay, text, (bar_x + 5, zone_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            zone_y += 30

        # Draw pose indicator
        cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Pitch: {pitch:.1f}°", (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Roll: {roll:.1f}°", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw target zone
        if not self.is_complete:
            min_yaw, max_yaw = self.zone_ranges[self.current_zone]
            zone_text = f"TARGET: {self.current_zone.value.upper()} ({min_yaw}° to {max_yaw}°)"
            cv2.putText(overlay, zone_text, (w // 2 - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(overlay, "ENROLLMENT COMPLETE!", (w // 2 - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return overlay


# Convenience function
def create_enrollment() -> GuidedEnrollment:
    """Create guided enrollment session."""
    return GuidedEnrollment()


if __name__ == "__main__":
    # Test guided enrollment
    print("Testing guided enrollment...")

    enrollment = create_enrollment()

    # Simulate enrollment with different poses
    poses = [
        # Frontal frames
        *[(0, 0, 0) for _ in range(5)],
        *[(i * 3, 0, 0) for i in range(5)],  # Small yaw variations

        # Left frames
        *[(-20 - i * 2, 0, 0) for i in range(15)],

        # Right frames
        *[(20 + i * 2, 0, 0) for i in range(15)]
    ]

    for i, (yaw, pitch, roll) in enumerate(poses):
        # Create dummy landmarks
        landmarks = np.random.randn(68, 2) * 10 + 200

        # Add frame
        result = enrollment.add_frame(landmarks, (yaw, pitch, roll))

        if result['accepted']:
            print(f"Frame {i}: {result['message']}")
            print(f"  Progress: {result['progress']:.1%} "
                  f"({result['total_frames']}/{enrollment.config.enrollment_n_frames})")

        if result['is_complete']:
            break

    # Get summary
    summary = enrollment.get_enrollment_summary()
    print(f"\nEnrollment Summary:")
    print(f"  Frontal: {summary['frontal_frames']} frames")
    print(f"  Left: {summary['left_frames']} frames")
    print(f"  Right: {summary['right_frames']} frames")
    print(f"  Total: {summary['total_frames']}/{summary['target_frames']}")
    print(f"  Complete: {summary['is_complete']}")