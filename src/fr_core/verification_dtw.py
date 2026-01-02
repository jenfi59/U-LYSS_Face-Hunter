"""
U-LYSS Face Recognition - DTW-based Verification
Implements PCA projection and DTW distance comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# -------------------------------------------------------------------------
# Optional import of sequential validation components.  These modules are
# placed in the project root (``src``) and provide a more robust matching
# strategy based on groups of landmarks, pose validation and facial ratios.
# Sequential validation is only used if the configuration specifies
# ``matching_mode`` equal to ``'sequential'`` and if the modules are
# available.  Fallback to temporal/spatial modes is automatic.
try:
    from ..sequential_validator import SequentialValidator  # type: ignore
    from ..config_sequential import ConfigSequential  # type: ignore
    _SEQUENTIAL_AVAILABLE = True
except Exception:
    # If sequential modules are missing, disable sequential mode silently.
    SequentialValidator = None  # type: ignore
    ConfigSequential = None  # type: ignore
    _SEQUENTIAL_AVAILABLE = False

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict
import logging
import pickle

from utils.pose_estimation import find_similar_poses

from .dtw_backend import dtw_distance, dtw_distance_normalized
from .config import get_config

logger = logging.getLogger(__name__)


class VerificationDTW:
    """
    DTW-based face verification with 3D landmarks.

    This class supports sequences of arbitrary landmark counts.  Landmark
    sequences are flattened to a vector of length ``num_landmarks * dims``
    (typically dims=2 or 3) before being normalized and projected into PCA
    space.  Distances between sequences are computed using Dynamic Time
    Warping (DTW).  The number of PCA components is chosen dynamically
    based on the number of available samples and features.
    """

    def __init__(self, pca_model_path: Optional[Path] = None):
        """
        Initialize verification engine.

        Args:
            pca_model_path: Path to saved PCA model. If None, creates new PCA.
        """
        self.config = get_config()
        self.pca = None
        self.scaler = None
        self.pca_model_path = pca_model_path

        if pca_model_path and pca_model_path.exists():
            self._load_pca_model(pca_model_path)

        # -----------------------------------------------------------------
        # Optional sequential validator.  When ``matching_mode`` is set to
        # ``'sequential'`` in the configuration and the sequential modules
        # are available, a SequentialValidator instance will be created to
        # perform multi–criteria validation.  If the modules are not
        # available or the mode is different, this attribute remains None.
        self.sequential_validator = None
        if _SEQUENTIAL_AVAILABLE and getattr(self.config, 'matching_mode', '') == 'sequential':
            try:
                # Initialise with the configuration.  If the current config
                # is not a ConfigSequential, we pass it directly; the
                # SequentialValidator can still read thresholds via getattr.
                self.sequential_validator = SequentialValidator(self.config)
                logger.info("SequentialValidator initialised for sequential matching mode")
            except Exception as e:
                logger.error(f"Failed to initialise SequentialValidator: {e}")

    def _load_pca_model(self, path: Path):
        """Load pre-trained PCA model."""
        try:
            with open(path, "rb") as f:
                self.pca = pickle.load(f)
            logger.info(f"Loaded PCA model: {self.pca.n_components} components")
        except Exception as e:
            logger.error(f"Failed to load PCA model: {e}")
            raise

    def _save_pca_model(self, path: Path):
        """Save PCA model to file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.pca, f)
            logger.info(f"Saved PCA model: {path}")
        except Exception as e:
            logger.error(f"Failed to save PCA model: {e}")

    def fit_pca(self, landmarks_sequences: List[np.ndarray]) -> PCA:
        """Fit a PCA model from a list of landmark sequences.

        Each sequence should have shape ``(n_frames, num_landmarks, dims)``.  All
        sequences must have the same number of landmarks and dimensions.  The
        sequences are flattened across landmarks and dimensions (i.e., a frame
        becomes a vector of length ``num_landmarks * dims``) and concatenated
        across all sequences.  A `RobustScaler` is used to normalize the
        flattened data before fitting PCA.  The number of PCA components is
        limited to the minimum of the configured maximum, the number of
        samples and the feature dimension.

        Args:
            landmarks_sequences: List of landmark sequences, each of shape
                ``(n_frames, num_landmarks, dims)``.

        Returns:
            The fitted PCA model.
        """
        # Flatten all frames across sequences
        all_frames = []
        for seq in landmarks_sequences:
            n_frames, n_landmarks, dims = seq.shape
            flattened = seq.reshape(n_frames, n_landmarks * dims)
            all_frames.append(flattened)
        X = np.vstack(all_frames)

        # Normalize using RobustScaler for robustness to outliers
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Determine number of components: cannot exceed number of samples or features
        n_components = self.config.pca_n_components
        n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_components, whiten=False)
        self.pca.fit(X_scaled)

        explained_var = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info(
            f"PCA fitted: {n_components} components, explained variance: {explained_var:.3f}"
        )
        return self.pca

    def _normalize_landmarks(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Normalize landmarks to [0, 1] range per frame.
        
        Args:
            landmarks_sequence: Array of shape (n_frames, num_landmarks, dims)
            
        Returns:
            Normalized landmarks with same shape
        """
        normalized = np.zeros_like(landmarks_sequence)
        for i in range(landmarks_sequence.shape[0]):
            frame = landmarks_sequence[i]  # (num_landmarks, dims)
            # Normalize x and y independently to [0, 1]
            min_vals = frame.min(axis=0)
            max_vals = frame.max(axis=0)
            range_vals = max_vals - min_vals
            # Avoid division by zero
            range_vals = np.where(range_vals > 1e-6, range_vals, 1.0)
            normalized[i] = (frame - min_vals) / range_vals
        return normalized

    def project_landmarks(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Project a landmarks sequence to PCA space.

        Args:
            landmarks_sequence: Array of shape ``(n_frames, num_landmarks, dims)``.

        Returns:
            Array of shape ``(n_frames, n_components)`` representing the PCA
            projection of each frame in the sequence.
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")

        n_frames, n_landmarks, dims = landmarks_sequence.shape
        flattened = landmarks_sequence.reshape(n_frames, n_landmarks * dims)
        scaled = self.scaler.transform(flattened)
        projected = self.pca.transform(scaled)
        return projected

    def compute_distance(self, seq1: np.ndarray, seq2: np.ndarray, normalized: bool = False) -> float:
        """
        Compute DTW distance between two PCA-projected sequences.

        Args:
            seq1: First sequence (n_frames1, n_components)
            seq2: Second sequence (n_frames2, n_components)
            normalized: Whether to normalize by sequence length

        Returns:
            DTW distance
        """
        window = self.config.dtw_window

        if normalized:
            dist = dtw_distance_normalized(seq1, seq2, window=window)
        else:
            dist = dtw_distance(seq1, seq2, window=window)

        return dist

    def verify(
        self, probe_landmarks: np.ndarray, gallery_landmarks: np.ndarray, normalized: bool = True
    ) -> Tuple[bool, float]:
        """Verify if a probe sequence matches a gallery sequence.

        Args:
            probe_landmarks: Probe sequence with shape ``(n_frames, num_landmarks, dims)``.
            gallery_landmarks: Gallery sequence with shape ``(n_frames, num_landmarks, dims)``.
            normalized: Whether to normalize DTW distance by sequence length.

        Returns:
            Tuple ``(is_match, distance)`` where ``is_match`` is a boolean
            indicating whether the DTW distance is below the configured
            threshold, and ``distance`` is the computed DTW distance.
        """
        probe_pca = self.project_landmarks(probe_landmarks)
        gallery_pca = self.project_landmarks(gallery_landmarks)
        match_score = self.compute_distance(probe_pca, gallery_pca, normalized)
        decision_threshold = self.config.dtw_threshold
        verified = float(match_score) <= float(decision_threshold)
        logger.debug(
            f"Verification: distance={match_score:.3f}, threshold={decision_threshold:.3f}, match={verified}"
        )
        return verified, float(match_score)

    def verify_multi_gallery(
        self,
        probe_landmarks: np.ndarray,
        gallery_list: List[Tuple[str, np.ndarray, Optional[np.ndarray]]],
        probe_poses: Optional[np.ndarray] = None,
        normalized: bool = True,
    ) -> Tuple[Optional[str], float]:
        """Verify a probe sequence against multiple gallery entries.

        This method performs 1:N verification using the matching mode defined
        in the configuration.  It iterates over each gallery entry,
        computing the distance using :meth:`verify_auto`.  The best match
        (smallest distance) below the corresponding threshold is returned.

        Args:
            probe_landmarks: Probe sequence of shape ``(M, n, d)``.
            gallery_list: A list of tuples ``(user_id, landmarks, poses)`` where
                ``landmarks`` is the enrolled landmark sequence and ``poses`` is
                an optional array of head poses for the user.  If ``poses`` is
                ``None``, temporal matching is used for that user.
            probe_poses: Optional array of probe poses (M, 3).  If ``None``,
                pose‑based matching falls back to temporal.
            normalized: Ignored (for API compatibility).

        Returns:
            A tuple ``(best_user_id or None, best_distance)``.  If no
            candidate passes the configured threshold, ``best_user_id`` is
            ``None`` and ``best_distance`` is the minimum distance.
        """
        #
        # NEW: Enhanced 1:N identification with composite scoring and coverage
        #
        # To improve robustness in a crowd, we compute a composite score for
        # sequential matching that combines normalised distances from
        # invariant and stable landmark groups, pose similarity and
        # anthropometric ratios.  A majority‑vote–style coverage ratio is
        # also computed based on pose matching.  Candidates are ranked by
        # composite score; the best candidate must satisfy both absolute
        # thresholds and relative margin criteria on the score and
        # coverage.  For other matching modes, the original DTW/pose
        # distance and triplet margin logic is retained.

        # Store candidate results for later ranking
        candidate_scores: List[Tuple[str, float, float, float]] = []  # (user_id, score, coverage, distance)

        # Helper to compute composite score from sequential validation details
        def _compute_composite(details: Dict) -> float:
            """Compute a weighted, normalised composite score from sequential details."""
            # Extract stage distances and thresholds
            groups = details.get('stages', {}).get('groups', {})
            group_dist = groups.get('distances', {}) or {}
            group_thresh = groups.get('thresholds', {}) or {}
            # Normalised group distances
            inv_dist = group_dist.get('invariant', 0.0)
            inv_thresh = group_thresh.get('invariant', 1.0)
            norm_inv = inv_dist / inv_thresh if inv_thresh > 0 else inv_dist
            st_dist = group_dist.get('stable', 0.0)
            st_thresh = group_thresh.get('stable', 1.0)
            norm_st = st_dist / st_thresh if st_thresh > 0 else st_dist
            # Pose stage distances
            pose_stage = details.get('stages', {}).get('poses', {})
            pose_distances = pose_stage.get('distances', {}) or {}
            pose_threshs = pose_stage.get('thresholds', {}) or {}
            norm_pose = 0.0
            if pose_distances:
                # use the maximum normalised pose distance across frontal/left/right
                maxima = []
                for pname, dist in pose_distances.items():
                    th = pose_threshs.get(pname, 1.0)
                    maxima.append(dist / th if th > 0 else dist)
                if maxima:
                    norm_pose = float(max(maxima))
            # Ratio errors
            ratio_stage = details.get('stages', {}).get('ratios', {})
            ratio_err = ratio_stage.get('errors', {}) or {}
            ratio_thresh = ratio_stage.get('threshold', 1.0)
            norm_ratio = 0.0
            if ratio_err:
                max_err = max(ratio_err.values())
                norm_ratio = max_err / ratio_thresh if ratio_thresh > 0 else max_err
            # Weighted sum
            w_inv = getattr(self.config, 'weight_invariant', 0.4)
            w_st = getattr(self.config, 'weight_stable', 0.3)
            w_pose = getattr(self.config, 'weight_pose', 0.2)
            w_ratio = getattr(self.config, 'weight_ratio', 0.1)
            score = (
                w_inv * norm_inv +
                w_st * norm_st +
                w_pose * norm_pose +
                w_ratio * norm_ratio
            )
            return float(score)

        # Helper to compute pose coverage ratio
        def _compute_coverage(probe_p: Optional[np.ndarray], gallery_p: Optional[np.ndarray]) -> float:
            """Return fraction of probe frames that have at least one similar pose in gallery."""
            if probe_p is None or gallery_p is None:
                return 1.0  # Consider full coverage if poses unavailable
            try:
                # Convert arrays to list of dicts for find_similar_poses
                gallery_dicts = [
                    {'yaw': float(p[0]), 'pitch': float(p[1]), 'roll': float(p[2])}
                    for p in gallery_p
                ]
                match_count = 0
                eps = (
                    getattr(self.config, 'pose_epsilon_yaw', 10.0),
                    getattr(self.config, 'pose_epsilon_pitch', 10.0),
                    getattr(self.config, 'pose_epsilon_roll', 10.0)
                )
                from utils.pose_estimation import find_similar_poses as _find
                for p in probe_p:
                    target = {'yaw': float(p[0]), 'pitch': float(p[1]), 'roll': float(p[2])}
                    # If at least one similar gallery pose exists, count this frame
                    if _find(target, gallery_dicts, epsilon=eps):
                        match_count += 1
                return float(match_count) / float(len(probe_p)) if len(probe_p) > 0 else 0.0
            except Exception:
                return 0.0

        # Iterate through gallery entries and compute scores/distances
        for entry in gallery_list:
            try:
                if len(entry) == 3:
                    user_id, gallery_lm, gallery_poses = entry
                else:
                    user_id, gallery_lm = entry[:2]
                    gallery_poses = None
                # Compute distance and details via router
                is_match, distance, details = self.verify_auto(
                    probe_landmarks, probe_poses, gallery_lm, gallery_poses
                )
                # Determine composite score if sequential mode
                if getattr(self.config, 'matching_mode', '') == 'sequential' and details.get('mode') == 'sequential':
                    comp_score = _compute_composite(details)
                    coverage = _compute_coverage(probe_poses, gallery_poses)
                    candidate_scores.append((user_id, comp_score, coverage, float(distance)))
                else:
                    # Use plain distance and full coverage for other modes
                    candidate_scores.append((user_id, float(distance), 1.0, float(distance)))
                # Logging for each user
                logger.debug(
                    f"User {user_id}: score={candidate_scores[-1][1]:.3f}, coverage={candidate_scores[-1][2]:.3f}, distance={distance:.3f}, mode={details.get('mode')}"
                )
            except Exception as e:
                logger.warning(f"Failed to verify against {entry[0]}: {e}")
                continue

        # If no candidates computed, return none
        if not candidate_scores:
            logger.info("No candidates to compare")
            return None, float('inf')

        # Separate handling for sequential/composite mode
        if getattr(self.config, 'matching_mode', '') == 'sequential':
            # Sort by composite score (ascending)
            sorted_candidates = sorted(candidate_scores, key=lambda x: x[1])
            best_user_id, best_score, best_coverage, best_distance = sorted_candidates[0]
            # Determine second best if exists
            second_best_score = float('inf')
            second_coverage = 0.0
            if len(sorted_candidates) > 1:
                _, second_best_score, second_coverage, _ = sorted_candidates[1]
            # Check absolute composite threshold
            score_threshold = getattr(self.config, 'composite_threshold', 1.0)
            margin_threshold = getattr(self.config, 'composite_margin', 0.2)
            coverage_threshold = getattr(self.config, 'coverage_threshold', 0.3)
            coverage_margin = getattr(self.config, 'coverage_margin', 0.2)
            # Pass if score below threshold
            if best_score <= score_threshold:
                # Margin on score
                score_margin = float('inf')
                if np.isfinite(second_best_score) and best_score > 0:
                    score_margin = (second_best_score - best_score) / best_score
                # Margin on coverage
                cov_margin = float('inf')
                if second_coverage > 0 and best_coverage > 0:
                    cov_margin = (best_coverage - second_coverage) / best_coverage
                # Decide acceptance
                if score_margin >= margin_threshold and best_coverage >= coverage_threshold and cov_margin >= coverage_margin:
                    logger.info(
                        f"Composite match: {best_user_id} (score={best_score:.3f}, coverage={best_coverage:.3f}, score_margin={score_margin:.3f}, coverage_margin={cov_margin:.3f})"
                    )
                    return best_user_id, float(best_score)
                else:
                    logger.info(
                        f"Ambiguous composite match: best={best_score:.3f}, second={second_best_score:.3f}, score_margin={score_margin:.3f}, coverage={best_coverage:.3f}/{second_coverage:.3f}" )
            # If thresholds or margins fail, return None (no match)
            logger.info(f"No composite match (best score={best_score:.3f}, coverage={best_coverage:.3f})")
            return None, float(best_score)
        else:
            # Fallback: use legacy distance and triplet margin logic
            # Determine best and second best distances
            best_distance = float('inf')
            best_user_id = None
            second_best_distance = float('inf')
            for user_id, score, coverage, dist in candidate_scores:
                # Use distance for ranking
                if dist < best_distance:
                    second_best_distance = best_distance
                    best_distance = dist
                    best_user_id = user_id
                elif dist < second_best_distance:
                    second_best_distance = dist
            if best_user_id is not None:
                mode = self.config.matching_mode
                # Determine base threshold according to mode
                if mode == 'temporal':
                    overall_threshold = float(self.config.dtw_threshold)
                elif mode == 'spatial':
                    overall_threshold = float(self.config.pose_threshold)
                elif mode == 'spatiotemporal':
                    overall_threshold = float(self.config.spatiotemporal_threshold)
                else:
                    overall_threshold = float('inf')
                if best_distance <= overall_threshold:
                    margin_ratio = float('inf')
                    if np.isfinite(second_best_distance) and best_distance > 0:
                        margin_ratio = (second_best_distance - best_distance) / best_distance
                    try:
                        margin_threshold = float(getattr(self.config, 'threshold_triplet_margin', 0.15))
                    except Exception:
                        margin_threshold = 0.15
                    if margin_ratio >= margin_threshold:
                        logger.info(
                            f"Match found: {best_user_id} (distance={best_distance:.3f}, margin={margin_ratio:.3f})"
                        )
                        return best_user_id, best_distance
                    else:
                        logger.info(
                            f"Ambiguous match: best={best_distance:.3f}, second={second_best_distance:.3f}, margin={margin_ratio:.3f}" )
            logger.info(f"No match (best distance={best_distance:.3f})")
            return None, best_distance

    def save_enrollment(
        self,
        user_id: str,
        landmarks_sequence: np.ndarray,
        output_dir: Path,
        poses_sequence: Optional[np.ndarray] = None,
    ) -> None:
        """Save an enrollment sequence to disk with optional head poses.

        This method persists the enrollment data for a user into a compressed
        ``.npz`` file.  The saved contents include the raw landmarks sequence,
        its PCA projection, the fitted PCA model, the scaler used for
        normalisation and a metadata dictionary.  When ``poses_sequence`` is
        provided, the 3‑D head pose (yaw, pitch, roll) for each frame is
        stored and metadata fields are updated accordingly.  The metadata
        version is bumped to ``2.3`` when poses are present and includes a
        boolean ``has_poses`` flag.

        Args:
            user_id: Identifier for the enrolled user.
            landmarks_sequence: Enrollment sequence of shape
                ``(n_frames, num_landmarks, dims)``.
            output_dir: Directory in which to save the enrollment file.
            poses_sequence: Optional array of shape ``(n_frames, 3)`` with
                [yaw, pitch, roll] in degrees for each frame.  If ``None``,
                the enrollment is saved without pose information and remains
                compatible with legacy v2.2 models.
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Project landmarks into PCA space
        pca_sequence = self.project_landmarks(landmarks_sequence)
        # Compose metadata
        meta = {
            'user_id': user_id,
            'num_landmarks': int(landmarks_sequence.shape[1]),
            'detector': 'mediapipe',
            'use_dtw': True,
            'version': '2.3' if poses_sequence is not None else '2.2',
            'has_poses': poses_sequence is not None,
        }
        # Prepare dictionary to save
        save_dict = {
            'landmarks': landmarks_sequence,
            'pca_sequence': pca_sequence,
            'pca': self.pca,
            'scaler': self.scaler,
            'metadata': meta,
        }
        # Include poses if provided
        if poses_sequence is not None:
            save_dict['poses'] = poses_sequence
        enrollment_path = output_dir / f"{user_id}.npz"
        np.savez_compressed(enrollment_path, **save_dict)
        logger.info(
            f"Saved enrollment for {user_id}: {enrollment_path} (has_poses={meta['has_poses']})"
        )

    def load_enrollment(
        self, user_id: str, enrollment_dir: Path
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Load an enrolled user's data from a ``.npz`` file.

        This method restores the saved landmarks sequence along with any
        associated head pose sequence.  The PCA model and scaler are also
        reloaded into the verifier if present.  Legacy v2.2 models lacking
        pose information are handled gracefully: ``poses`` is returned as
        ``None`` and callers are expected to fall back to temporal matching or
        recalculate poses on demand.

        Args:
            user_id: Identifier of the user to load.
            enrollment_dir: Directory where enrollment files are stored.

        Returns:
            A tuple ``(landmarks, poses)`` where ``landmarks`` is the raw
            landmark sequence of shape ``(n_frames, num_landmarks, dims)`` and
            ``poses`` is an array of shape ``(n_frames, 3)`` with [yaw, pitch,
            roll] in degrees if available, or ``None`` if the model does not
            contain pose data.  Returns ``None`` if the file is not found or
            cannot be loaded.
        """
        enrollment_path = enrollment_dir / f"{user_id}.npz"
        if not enrollment_path.exists():
            logger.warning(f"Enrollment not found: {enrollment_path}")
            return None
        try:
            data = np.load(enrollment_path, allow_pickle=True)
            # Restore PCA and scaler if present (pickle stored as array of objects)
            if 'pca' in data:
                self.pca = data['pca'].item()
            if 'scaler' in data:
                self.scaler = data['scaler'].item()
            if 'metadata' in data:
                meta = data['metadata'].item()
                logger.debug(f"Loaded metadata for {user_id}: {meta}")
            landmarks = data['landmarks']
            poses = None
            if 'poses' in data:
                poses = data['poses']
                logger.debug(f"Loaded poses for {user_id}: shape={poses.shape}")
            else:
                # Legacy model without poses: attempt to recalculate
                logger.warning(f"No poses found for {user_id} (legacy model). Recalculating...")
                try:
                    from utils.recalculate import recalculate_poses_from_landmarks  # local import to avoid circular
                    poses = recalculate_poses_from_landmarks(landmarks)
                    logger.debug(f"Recalculated poses for {user_id}: shape={poses.shape}")
                except Exception as exc_inner:
                    logger.error(f"Failed to recalculate poses for {user_id}: {exc_inner}")
                    poses = None
            logger.debug(f"Loaded enrollment for {user_id}: {landmarks.shape}")
            return landmarks, poses
        except Exception as exc:
            logger.error(f"Failed to load enrollment: {exc}")
            return None

    # ------------------------------------------------------------------
    # Pose‑based verification
    # ------------------------------------------------------------------
    def verify_pose_based(
        self,
        probe_landmarks: np.ndarray,
        probe_poses: Optional[np.ndarray],
        gallery_landmarks: np.ndarray,
        gallery_poses: Optional[np.ndarray],
    ) -> Tuple[bool, float, Dict]:
        """Verify a probe sequence against a gallery using pose‑based matching.

        The spatial approach disregards temporal order and compares each probe
        frame only with gallery frames that exhibit a similar head pose.  A
        frame pair is considered similar if the normalized pose distance is
        less than 1.0, where the normalization uses the pose epsilon values
        from the configuration.  The distance between two frames is the
        Euclidean norm of the difference between their landmark vectors.  The
        final score is the mean of all per‑frame distances for which at least
        one similar gallery frame exists; frames with no corresponding pose
        matches are ignored.  If no frames match, the score is set to
        ``float('inf')``.

        Args:
            probe_landmarks: Probe landmarks array of shape
                ``(M, num_landmarks, dims)``.
            probe_poses: Probe poses array of shape ``(M, 3)`` or ``None``.
            gallery_landmarks: Gallery landmarks array of shape
                ``(N, num_landmarks, dims)``.
            gallery_poses: Gallery poses array of shape ``(N, 3)`` or ``None``.

        Returns:
            Tuple ``(is_match, distance, details)`` where ``is_match`` is
            ``True`` if the spatial distance is below the configured
            ``pose_threshold``.  ``distance`` is the average landmark distance
            across matching frames, or ``float('inf')`` when no frames
            match.  ``details`` contains diagnostic information including
            coverage (ratio of matched frames) and per‑frame distances.
        """
        details: Dict[str, any] = {}
        # Fallback to temporal if poses are missing
        if probe_poses is None or gallery_poses is None:
            logger.warning("Pose data missing; falling back to temporal verification")
            is_match, dist = self.verify(probe_landmarks, gallery_landmarks, normalized=True)
            return is_match, dist, {
                'mode': 'temporal',
                'distance_temporal': dist,
                'fallback': True,
            }

        # Convert epsilon from config
        eps_yaw, eps_pitch, eps_roll = self.config.get_pose_epsilon()
        epsilon_tuple = (eps_yaw, eps_pitch, eps_roll)

        # Build list of gallery pose dicts
        gallery_pose_dicts = [
            {'yaw': float(y), 'pitch': float(p), 'roll': float(r)}
            for y, p, r in gallery_poses
        ]

        per_frame_distances: List[float] = []
        matched_frames = 0
        total_probe_frames = probe_landmarks.shape[0]

        # Normalize landmarks to [0, 1] range for consistent distance calculation
        probe_normalized = self._normalize_landmarks(probe_landmarks)
        gallery_normalized = self._normalize_landmarks(gallery_landmarks)

        for idx, (lm_probe, pose_arr) in enumerate(zip(probe_normalized, probe_poses)):
            # Build target pose dict
            target_pose = {'yaw': float(pose_arr[0]), 'pitch': float(pose_arr[1]), 'roll': float(pose_arr[2])}
            # Find similar gallery frame indices
            similar_idx = find_similar_poses(target_pose, gallery_pose_dicts, epsilon=epsilon_tuple)
            if len(similar_idx) == 0:
                # No matching poses; record inf and continue
                per_frame_distances.append(float('inf'))
                continue
            matched_frames += 1
            distances_for_frame = []
            # Compute Euclidean distances for each similar frame
            for i in similar_idx:
                diff = lm_probe.reshape(-1) - gallery_normalized[i].reshape(-1)
                dist_val = float(np.linalg.norm(diff))
                distances_for_frame.append(dist_val)
            avg_dist = float(np.mean(distances_for_frame))
            per_frame_distances.append(avg_dist)

        # Compute final distance ignoring inf values
        finite_distances = [d for d in per_frame_distances if np.isfinite(d)]
        if len(finite_distances) == 0:
            final_distance = float('inf')
        else:
            final_distance = float(np.mean(finite_distances))

        coverage = matched_frames / total_probe_frames if total_probe_frames > 0 else 0.0
        threshold = float(self.config.pose_threshold)
        is_match = final_distance <= threshold
        details.update({
            'mode': 'spatial',
            'matched_frames': matched_frames,
            'total_probe_frames': total_probe_frames,
            'coverage': coverage,
            'per_frame_distances': per_frame_distances,
            'distance_spatial': final_distance,
            'threshold': threshold,
        })
        return is_match, final_distance, details

    # ------------------------------------------------------------------
    # Spatio‑temporal fusion verification
    # ------------------------------------------------------------------
    def verify_spatiotemporal(
        self,
        probe_landmarks: np.ndarray,
        probe_poses: Optional[np.ndarray],
        gallery_landmarks: np.ndarray,
        gallery_poses: Optional[np.ndarray],
    ) -> Tuple[bool, float, Dict]:
        """Verify a probe sequence using a weighted fusion of DTW and pose‑based distances.

        This method combines the temporal DTW distance and the spatial
        pose‑based distance into a single score via a convex combination.  It
        first computes the DTW distance using the existing temporal verifier,
        then computes the pose‑based distance via :meth:`verify_pose_based`.
        The fused distance is computed as:

        ``fused = alpha * dist_temporal + (1 - alpha) * dist_spatial``

        where ``alpha`` is the fusion weight from configuration.  The probe
        is accepted if ``fused`` is below ``config.spatiotemporal_threshold``.

        Args:
            probe_landmarks: Probe landmarks of shape ``(M, n, d)``.
            probe_poses: Probe poses array (M, 3) or ``None``.
            gallery_landmarks: Gallery landmarks of shape ``(N, n, d)``.
            gallery_poses: Gallery poses array (N, 3) or ``None``.

        Returns:
            Tuple ``(is_match, fused_distance, details)`` containing the final
            decision, the fused distance and a dictionary with intermediate
            values and coverage.
        """
        # Temporal distance
        is_match_temporal, dist_temporal = self.verify(probe_landmarks, gallery_landmarks, normalized=True)
        # Spatial distance
        is_match_spatial, dist_spatial, spatial_details = self.verify_pose_based(
            probe_landmarks, probe_poses, gallery_landmarks, gallery_poses
        )
        # If pose data missing triggers fallback, treat as temporal only
        if spatial_details.get('mode') == 'temporal' and spatial_details.get('fallback', False):
            fused_distance = dist_temporal
            is_match_final = fused_distance <= float(self.config.dtw_threshold)
            details = {
                'mode': 'temporal',
                'distance_temporal': dist_temporal,
                'distance_spatial': dist_spatial,
                'fused_distance': fused_distance,
                'alpha': 1.0,
                'threshold': float(self.config.dtw_threshold),
                'coverage': spatial_details.get('coverage'),
                'fallback': True,
            }
            return is_match_final, fused_distance, details

        alpha = float(self.config.fusion_alpha)
        if not np.isfinite(dist_spatial):
            fused_distance = float('inf')
        else:
            fused_distance = float(alpha * dist_temporal + (1.0 - alpha) * dist_spatial)
        threshold = float(self.config.spatiotemporal_threshold)
        is_match_final = fused_distance <= threshold
        details = {
            'mode': 'spatiotemporal',
            'distance_temporal': dist_temporal,
            'distance_spatial': dist_spatial,
            'fused_distance': fused_distance,
            'alpha': alpha,
            'threshold': threshold,
            'coverage': spatial_details.get('coverage'),
            'matched_frames': spatial_details.get('matched_frames'),
            'total_probe_frames': spatial_details.get('total_probe_frames'),
            'per_frame_distances': spatial_details.get('per_frame_distances'),
        }
        return is_match_final, fused_distance, details

    # ------------------------------------------------------------------
    # Sequential multi‑criteria verification
    # ------------------------------------------------------------------
    def verify_sequential(
        self,
        probe_landmarks: np.ndarray,
        probe_poses: Optional[np.ndarray],
        gallery_landmarks: np.ndarray,
        gallery_poses: Optional[np.ndarray],
        user_id: Optional[str] = None,
    ) -> Tuple[bool, float, Dict]:
        """Verify a probe sequence against a gallery using sequential
        multi‑criteria validation.

        This method delegates to the external ``SequentialValidator`` if it
        has been initialised.  If the validator is not available or not
        configured, the method falls back to pose‑based matching.  The
        returned details dictionary mirrors that of the sequential validator.

        Args:
            probe_landmarks: Landmarks for the probe sequence, shape
                ``(M, n, d)``.
            probe_poses: Poses for the probe sequence (M, 3) or ``None``.
            gallery_landmarks: Landmarks for the gallery sequence, shape
                ``(N, n, d)``.
            gallery_poses: Poses for the gallery sequence (N, 3) or ``None``.
            user_id: Optional identifier for logging purposes.

        Returns:
            Tuple ``(is_match, distance, details)``.  The meaning of
            ``distance`` depends on the underlying validator.  In case of
            fallback, this is the spatial distance.
        """
        # Use sequential validator if available
        if self.sequential_validator is not None:
            try:
                is_match, dist, details = self.sequential_validator.verify_sequential(
                    probe_landmarks,
                    probe_poses,
                    gallery_landmarks,
                    gallery_poses,
                    gallery_user_id=user_id or 'unknown',
                )
                return is_match, dist, details
            except Exception as e:
                logger.error(f"Sequential validation failed: {e}; falling back to pose-based verification")
        # Fallback: pose-based matching
        is_match, dist, details = self.verify_pose_based(
            probe_landmarks, probe_poses, gallery_landmarks, gallery_poses
        )
        return is_match, dist, details

    # ------------------------------------------------------------------
    # Automatic router
    # ------------------------------------------------------------------
    def verify_auto(
        self,
        probe_landmarks: np.ndarray,
        probe_poses: Optional[np.ndarray],
        gallery_landmarks: np.ndarray,
        gallery_poses: Optional[np.ndarray],
        user_id: Optional[str] = None,
    ) -> Tuple[bool, float, Dict]:
        """Route verification according to the configured matching mode.

        Depending on :attr:`config.matching_mode`, this method delegates to
        the temporal, spatial or spatiotemporal verification routines.  For
        legacy models lacking pose information, the router automatically
        falls back to temporal mode.

        Args:
            probe_landmarks: Probe landmarks array.
            probe_poses: Probe poses array or ``None``.
            gallery_landmarks: Gallery landmarks array.
            gallery_poses: Gallery poses array or ``None``.

        Returns:
            Tuple ``(is_match, distance, details)`` where ``distance`` is
            the relevant score (DTW, spatial or fused) and ``details``
            includes diagnostic information and the selected mode.
        """
        # Determine the matching mode from configuration.  Supported modes
        # include 'temporal', 'spatial', 'spatiotemporal' and 'sequential'.
        mode = getattr(self.config, 'matching_mode', 'temporal')
        if mode == 'temporal':
            is_match, dist = self.verify(probe_landmarks, gallery_landmarks, normalized=True)
            return is_match, dist, {'mode': 'temporal', 'distance_temporal': dist}
        if mode == 'spatial':
            return self.verify_pose_based(probe_landmarks, probe_poses, gallery_landmarks, gallery_poses)
        if mode == 'spatiotemporal':
            return self.verify_spatiotemporal(probe_landmarks, probe_poses, gallery_landmarks, gallery_poses)
        if mode == 'sequential':
            # Delegate to sequential validator.  If not available, falls back
            # internally to pose‑based verification.
            return self.verify_sequential(
                probe_landmarks, probe_poses, gallery_landmarks, gallery_poses, user_id=user_id
            )
        # Unknown mode: error
        raise ValueError(f"Invalid matching_mode: {mode}")


def create_verifier(pca_model_path: Optional[Path] = None) -> VerificationDTW:
    """Create verification engine."""
    return VerificationDTW(pca_model_path=pca_model_path)


if __name__ == "__main__":
    print("Testing DTW verification...")

    n_frames = 45
    seq1 = np.random.randn(n_frames, 68, 2)
    seq2 = seq1 + np.random.randn(n_frames, 68, 2) * 0.1
    seq3 = np.random.randn(n_frames, 68, 2)

    verifier = create_verifier()
    verifier.fit_pca([seq1, seq2, seq3])

    verified_1, dist_1 = verifier.verify(seq1, seq2)
    verified_2, dist_2 = verifier.verify(seq1, seq3)

    print(f"Seq1 vs Seq2 (similar): distance={dist_1:.3f}, match={verified_1}")
    print(f"Seq1 vs Seq3 (different): distance={dist_2:.3f}, match={verified_2}")

    gallery = [("user_1", seq2), ("user_2", seq3)]

    matched_id, best_dist = verifier.verify_multi_gallery(seq1, gallery)
    print(f"Multi-gallery: matched={matched_id}, distance={best_dist:.3f}")