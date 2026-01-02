"""
U-LYSS Face Recognition - Pose-based Landmark Matching

This module provides utilities to compare facial landmarks based on head pose,
allowing matching between faces in similar orientations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class PoseMatcher:
    """Match and compare facial landmarks based on head pose similarity."""
    
    def __init__(self, 
                 yaw_threshold: float = 15.0,
                 pitch_threshold: float = 15.0,
                 roll_threshold: float = 15.0):
        """Initialize pose matcher with angle thresholds.
        
        Args:
            yaw_threshold: Maximum yaw difference in degrees (default: 15°)
            pitch_threshold: Maximum pitch difference in degrees (default: 15°)
            roll_threshold: Maximum roll difference in degrees (default: 15°)
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
    
    def pose_distance(self, 
                      pose1: Tuple[float, float, float],
                      pose2: Tuple[float, float, float]) -> float:
        """Calculate angular distance between two head poses.
        
        Args:
            pose1: (yaw, pitch, roll) in degrees
            pose2: (yaw, pitch, roll) in degrees
        
        Returns:
            Euclidean distance in angle space (degrees)
        """
        yaw1, pitch1, roll1 = pose1
        yaw2, pitch2, roll2 = pose2
        
        # Euclidean distance in 3D angle space
        distance = np.sqrt(
            (yaw1 - yaw2)**2 + 
            (pitch1 - pitch2)**2 + 
            (roll1 - roll2)**2
        )
        
        return distance
    
    def is_pose_similar(self,
                        pose1: Tuple[float, float, float],
                        pose2: Tuple[float, float, float]) -> bool:
        """Check if two poses are similar within thresholds.
        
        Args:
            pose1: (yaw, pitch, roll) in degrees
            pose2: (yaw, pitch, roll) in degrees
        
        Returns:
            True if poses are similar, False otherwise
        """
        yaw1, pitch1, roll1 = pose1
        yaw2, pitch2, roll2 = pose2
        
        yaw_diff = abs(yaw1 - yaw2)
        pitch_diff = abs(pitch1 - pitch2)
        roll_diff = abs(roll1 - roll2)
        
        return (yaw_diff <= self.yaw_threshold and
                pitch_diff <= self.pitch_threshold and
                roll_diff <= self.roll_threshold)
    
    def find_closest_pose(self,
                          target_pose: Tuple[float, float, float],
                          pose_library: List[Tuple[float, float, float]]) -> Tuple[int, float]:
        """Find the closest matching pose in a library.
        
        Args:
            target_pose: Target (yaw, pitch, roll) to match
            pose_library: List of reference poses
        
        Returns:
            Tuple of (index, distance) of closest pose
        """
        if not pose_library:
            return -1, float('inf')
        
        distances = [self.pose_distance(target_pose, ref_pose) 
                    for ref_pose in pose_library]
        
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        return min_idx, min_distance
    
    def filter_similar_poses(self,
                            target_pose: Tuple[float, float, float],
                            pose_library: List[Tuple[float, float, float]]) -> List[int]:
        """Find all poses similar to target within thresholds.
        
        Args:
            target_pose: Target (yaw, pitch, roll) to match
            pose_library: List of reference poses
        
        Returns:
            List of indices of similar poses
        """
        similar_indices = []
        
        for idx, ref_pose in enumerate(pose_library):
            if self.is_pose_similar(target_pose, ref_pose):
                similar_indices.append(idx)
        
        return similar_indices


class Landmarks3DExtractor:
    """Extract and manipulate 3D landmark coordinates from MediaPipe."""
    
    @staticmethod
    def extract_3d_landmarks(result: Dict) -> Optional[np.ndarray]:
        """Extract 3D landmark coordinates from MediaPipe result.
        
        MediaPipe returns landmarks with x, y (normalized) and z (depth in same scale).
        
        Args:
            result: MediaPipe detection result with 'landmarks_3d' or 'landmarks'
        
        Returns:
            Array of shape (N, 3) with x, y, z coordinates, or None
        """
        # Try to get 3D landmarks if available
        if result is None:
            return None
        
        landmarks = result.get('landmarks')
        if landmarks is None:
            return None
        
        # MediaPipe landmarks are already (x, y) in 2D
        # We need the z coordinate from the 3D version
        # For now, return 2D landmarks (will need MediaPipe 3D landmarks API)
        
        logger.warning("3D landmarks extraction not yet implemented. Using 2D landmarks.")
        return landmarks
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to zero mean and unit variance.
        
        Args:
            landmarks: (N, 2) or (N, 3) array of landmark coordinates
        
        Returns:
            Normalized landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return landmarks
        
        # Center landmarks
        centered = landmarks - landmarks.mean(axis=0)
        
        # Scale to unit variance
        std = landmarks.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        normalized = centered / std
        
        return normalized
    
    @staticmethod
    def compute_landmark_distance(landmarks1: np.ndarray,
                                  landmarks2: np.ndarray,
                                  metric: str = 'euclidean') -> float:
        """Compute distance between two sets of landmarks.
        
        Args:
            landmarks1: First set of landmarks (N, 2/3)
            landmarks2: Second set of landmarks (N, 2/3)
            metric: Distance metric ('euclidean', 'cosine')
        
        Returns:
            Distance value
        """
        if landmarks1 is None or landmarks2 is None:
            return float('inf')
        
        if landmarks1.shape != landmarks2.shape:
            logger.error(f"Landmark shape mismatch: {landmarks1.shape} vs {landmarks2.shape}")
            return float('inf')
        
        if metric == 'euclidean':
            # Mean Euclidean distance per landmark
            distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
            return distances.mean()
        
        elif metric == 'cosine':
            # Flatten and compute cosine distance
            flat1 = landmarks1.flatten()
            flat2 = landmarks2.flatten()
            
            norm1 = np.linalg.norm(flat1)
            norm2 = np.linalg.norm(flat2)
            
            if norm1 == 0 or norm2 == 0:
                return float('inf')
            
            cosine_sim = np.dot(flat1, flat2) / (norm1 * norm2)
            return 1.0 - cosine_sim
        
        else:
            logger.error(f"Unknown metric: {metric}")
            return float('inf')


class PoseAwareMatcher:
    """Combine pose matching with landmark comparison for face recognition."""
    
    def __init__(self,
                 pose_matcher: Optional[PoseMatcher] = None,
                 pose_weight: float = 0.3):
        """Initialize pose-aware matcher.
        
        Args:
            pose_matcher: PoseMatcher instance (default: create new)
            pose_weight: Weight of pose similarity in final score (0-1)
        """
        self.pose_matcher = pose_matcher or PoseMatcher()
        self.pose_weight = pose_weight
        self.landmarks_extractor = Landmarks3DExtractor()
    
    def compare_with_pose(self,
                          query_result: Dict,
                          query_pose: Tuple[float, float, float],
                          reference_results: List[Dict],
                          reference_poses: List[Tuple[float, float, float]]) -> List[Tuple[int, float]]:
        """Compare query face with references, considering pose similarity.
        
        Args:
            query_result: MediaPipe result for query face
            query_pose: (yaw, pitch, roll) for query
            reference_results: List of MediaPipe results for references
            reference_poses: List of (yaw, pitch, roll) for references
        
        Returns:
            List of (index, score) sorted by similarity (lower is better)
        """
        if len(reference_results) != len(reference_poses):
            logger.error("Mismatch between results and poses length")
            return []
        
        query_landmarks = query_result.get('landmarks')
        if query_landmarks is None:
            return []
        
        scores = []
        
        for idx, (ref_result, ref_pose) in enumerate(zip(reference_results, reference_poses)):
            ref_landmarks = ref_result.get('landmarks')
            if ref_landmarks is None:
                continue
            
            # Compute pose distance
            pose_dist = self.pose_matcher.pose_distance(query_pose, ref_pose)
            
            # Compute landmark distance
            landmark_dist = self.landmarks_extractor.compute_landmark_distance(
                query_landmarks, ref_landmarks, metric='euclidean'
            )
            
            # Combined score (weighted average)
            combined_score = (self.pose_weight * pose_dist + 
                            (1 - self.pose_weight) * landmark_dist)
            
            scores.append((idx, combined_score))
        
        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[1])
        
        return scores
    
    def get_best_pose_match(self,
                           query_result: Dict,
                           query_pose: Tuple[float, float, float],
                           reference_results: List[Dict],
                           reference_poses: List[Tuple[float, float, float]],
                           require_similar_pose: bool = True) -> Optional[Tuple[int, float]]:
        """Find best matching reference with optional pose constraint.
        
        Args:
            query_result: MediaPipe result for query face
            query_pose: (yaw, pitch, roll) for query
            reference_results: List of MediaPipe results for references
            reference_poses: List of (yaw, pitch, roll) for references
            require_similar_pose: If True, only consider similar poses
        
        Returns:
            Tuple of (index, score) for best match, or None
        """
        if require_similar_pose:
            # Filter to similar poses first
            similar_indices = self.pose_matcher.filter_similar_poses(
                query_pose, reference_poses
            )
            
            if not similar_indices:
                logger.warning("No reference with similar pose found")
                return None
            
            # Compare only with similar poses
            filtered_results = [reference_results[i] for i in similar_indices]
            filtered_poses = [reference_poses[i] for i in similar_indices]
            
            scores = self.compare_with_pose(
                query_result, query_pose,
                filtered_results, filtered_poses
            )
            
            if not scores:
                return None
            
            # Map back to original indices
            best_filtered_idx, best_score = scores[0]
            best_original_idx = similar_indices[best_filtered_idx]
            
            return best_original_idx, best_score
        
        else:
            # Compare with all references
            scores = self.compare_with_pose(
                query_result, query_pose,
                reference_results, reference_poses
            )
            
            if not scores:
                return None
            
            return scores[0]
