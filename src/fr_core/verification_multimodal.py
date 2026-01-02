"""
U-LYSS Face Recognition - Multimodal Verification
==================================================

Implements secure 1:N verification with margin, adaptive thresholds,
and multimodal score fusion.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

from .verification_dtw import VerificationDTW
from .config import get_config

logger = logging.getLogger(__name__)


class MultimodalVerifier:
    """
    Système de vérification multimodal avec support 1:N sécurisé.
    
    Features:
    - 1-to-N verification (compare with ALL enrolled users)
    - Margin-based rejection (ambiguity detection)
    - Adaptive thresholds per user
    - Top-K prefiltering for performance
    - Multimodal score fusion (DTW + geometric ratios)
    """

    def __init__(self):
        """Initialize multimodal verifier."""
        self.config = get_config()
        self.dtw_verifier = VerificationDTW()
        logger.info("MultimodalVerifier initialized")

    def verify_1_to_n_with_margin(
        self,
        probe_landmarks: np.ndarray,
        gallery: List[Tuple[str, np.ndarray, Optional[np.ndarray], Dict]],
        probe_poses: Optional[np.ndarray] = None,
        use_adaptive_threshold: bool = True,
    ) -> Tuple[Optional[str], float, str, Dict]:
        """Perform secure 1:N verification with margin and adaptive thresholds.

        This method compares a probe sequence against all enrolled users.  It
        supports temporal, spatial and spatiotemporal matching via the
        underlying :class:`VerificationDTW` router.  For each gallery entry,
        the appropriate distance is computed using
        :meth:`VerificationDTW.verify_auto`, which returns a distance and
        details depending on the configured matching mode.  The best match
        must be below its corresponding threshold and have a sufficient
        margin relative to the second best match.  Entries with mismatched
        landmark counts are skipped.

        Args:
            probe_landmarks: Probe sequence of shape ``(n_frames, num_landmarks, dims)``.
            gallery: List of tuples ``(user_id, enrolled_landmarks, enrolled_poses, metadata)``.
                If ``enrolled_poses`` is ``None``, temporal matching is used.
            probe_poses: Optional array of probe poses (``(n_frames, 3)``) used for spatial
                and spatiotemporal modes.  If ``None`` and the mode is not temporal,
                temporal matching is used as a fallback.
            use_adaptive_threshold: Whether to use per‑user adaptive thresholds from
                metadata if available.  For spatial or spatiotemporal modes, the
                adaptive threshold is ignored and the global pose/spatiotemporal
                thresholds from config are used.

        Returns:
            A tuple ``(user_id or None, distance, confidence_level, details)``.  The
            ``confidence_level`` takes values ``VERIFIED``, ``NO_GALLERY``,
            ``THRESHOLD_REJECT`` or ``AMBIGUOUS``.
        """
        # Vérifier configuration
        group_security = self.config.biometric_groups["GROUP_1_SECURITY"]
        if not group_security.enabled:
            logger.warning("GROUP_1_SECURITY disabled, using legacy 1-to-1 verification")
            # Fallback: vérification simple avec premier utilisateur
            if len(gallery) > 0:
                user_id, enrolled_landmarks, metadata = gallery[0]
                verified, distance = self.dtw_verifier.verify(probe_landmarks, enrolled_landmarks)
                if verified:
                    return user_id, distance, "VERIFIED", {"legacy_mode": True}
                else:
                    return None, distance, "THRESHOLD_REJECT", {"legacy_mode": True}
            return None, float('inf'), "NO_GALLERY", {}

        min_margin = group_security.parameters["min_margin"]
        
        if len(gallery) == 0:
            logger.warning("Empty gallery")
            return None, float('inf'), "NO_GALLERY", {}

        # Compute distances to all users using auto router
        distances: List[Tuple[str, float, float]] = []
        for entry in gallery:
            try:
                if len(entry) == 4:
                    user_id, enrolled_landmarks, enrolled_poses, metadata = entry
                else:
                    # Legacy tuple (user_id, landmarks, metadata)
                    user_id, enrolled_landmarks, metadata = entry
                    enrolled_poses = None

                # Skip if landmark dimensionality differs
                if enrolled_landmarks.shape[1] != probe_landmarks.shape[1]:
                    logger.warning(
                        f"Skipping {user_id}: mismatched landmark count (gallery={enrolled_landmarks.shape[1]}, probe={probe_landmarks.shape[1]})"
                    )
                    continue

                # Compute distance using auto router (temporal/spatial/spatiotemporal)
                is_match, distance, verify_details = self.dtw_verifier.verify_auto(
                    probe_landmarks,
                    probe_poses,
                    enrolled_landmarks,
                    enrolled_poses,
                )

                # Determine threshold based on mode
                mode = verify_details.get('mode', 'temporal')
                if mode == 'temporal':
                    if use_adaptive_threshold:
                        # Use per-user threshold from metadata if available
                        threshold = float(metadata.get('threshold', self.config.dtw_threshold))
                    else:
                        threshold = float(self.config.dtw_threshold)
                elif mode == 'spatial':
                    threshold = float(self.config.pose_threshold)
                elif mode == 'spatiotemporal':
                    threshold = float(self.config.spatiotemporal_threshold)
                else:
                    threshold = float('inf')

                distances.append((user_id, float(distance), threshold))
                logger.debug(
                    f"User {user_id}: distance={distance:.3f}, threshold={threshold:.3f}, mode={mode}"
                )

            except Exception as e:
                logger.warning(f"Failed to verify against {entry[0]}: {e}")
                continue

        if len(distances) == 0:
            return None, float('inf'), "NO_GALLERY", {}

        # Sort by increasing distance
        distances.sort(key=lambda x: x[1])

        best_user, best_dist, best_threshold = distances[0]

        # Threshold rejection
        if best_dist > best_threshold:
            logger.info(
                f"Threshold rejection: distance={best_dist:.3f} > threshold={best_threshold:.3f}"
            )
            return None, best_dist, "THRESHOLD_REJECT", {
                "best_candidate": best_user,
                "distance": best_dist,
                "threshold": best_threshold,
            }

        # Margin check (compare against second best)
        if len(distances) > 1:
            second_user, second_dist, _ = distances[1]
            margin = (second_dist - best_dist) / max(best_dist, 1e-6)
            if margin < min_margin:
                logger.warning(
                    f"Ambiguous match: margin={margin:.3f} < min_margin={min_margin:.3f}"
                )
                return None, best_dist, "AMBIGUOUS", {
                    "candidate_1": best_user,
                    "distance_1": best_dist,
                    "candidate_2": second_user,
                    "distance_2": second_dist,
                    "margin": margin,
                    "min_margin": min_margin,
                }

        # Verified
        details = {
            "user_id": best_user,
            "distance": best_dist,
            "threshold": best_threshold,
            "margin": (distances[1][1] - best_dist) / max(best_dist, 1e-6) if len(distances) > 1 else None,
            "top_3_candidates": [
                {"user": u, "distance": d, "threshold": t}
                for u, d, t in distances[: min(3, len(distances))]
            ],
        }
        margin_str = (
            f"{details['margin']:.3f}" if details['margin'] is not None else "N/A"
        )
        logger.info(
            f"Verified: {best_user} (distance={best_dist:.3f}, margin={margin_str})"
        )
        return best_user, best_dist, "VERIFIED", details

    def prefilter_top_k(
        self,
        probe_landmarks: np.ndarray,
        gallery: List[Tuple[str, np.ndarray, Dict]],
        k: int = 10,
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Pré-filtre rapide par signature moyenne (distance euclidienne).
        
        Réduit la complexité de O(N) DTW complets à O(k) DTW sur top-K candidats,
        avec un pré-filtrage O(N) très rapide (distance L2 dans espace PCA).

        Args:
            probe_landmarks: Probe sequence of shape ``(n_frames, num_landmarks, dims)``.
            gallery: Complete gallery list ``[(user_id, landmarks, metadata), ...]``.  The
                ``metadata`` dictionary should contain a precomputed signature (mean
                PCA vector).  Entries lacking a signature are included in the
                top‑K by default (conservative approach).
            k: Number of candidates to retain.

        Returns:
            A sublist of ``gallery`` containing the top‑K candidates with the
            smallest Euclidean distance between probe and gallery signatures.
        """
        group_perf = self.config.biometric_groups["GROUP_4_PERFORMANCE"]
        if not group_perf.enabled or not group_perf.parameters["use_topk_prefilter"]:
            logger.debug("Top-K prefilter disabled, using full gallery")
            return gallery

        # Compute probe signature: mean of probe PCA sequence
        probe_pca = self.dtw_verifier.project_landmarks(probe_landmarks)
        sig_probe = np.mean(probe_pca, axis=0)
        # Compute fast distances
        fast_distances = []
        for user_id, enrolled_landmarks, metadata in gallery:
            sig_enrolled = metadata.get('signature')
            if sig_enrolled is not None:
                # Both signatures must be same length; skip if mismatch
                try:
                    d_fast = float(np.linalg.norm(sig_probe - sig_enrolled))
                except Exception as e:
                    logger.warning(f"Failed to compute signature distance for {user_id}: {e}")
                    d_fast = float('inf')
            else:
                logger.warning(f"User {user_id} has no signature, including in top-K by default")
                d_fast = 0.0
            fast_distances.append((user_id, enrolled_landmarks, metadata, d_fast))

        # Trier et prendre top-K
        fast_distances.sort(key=lambda x: x[3])
        top_k = [(uid, landmarks, meta) for uid, landmarks, meta, _ in fast_distances[:k]]

        logger.info(f"Prefilter: {len(gallery)} users -> top-{k} candidates")
        return top_k

    def load_gallery_from_dir(self, models_dir: Path) -> List[Tuple[str, np.ndarray, Optional[np.ndarray], Dict]]:
        """
        Charge tous les modèles utilisateurs d'un répertoire.

        Chaque fichier ``.npz`` peut contenir des landmarks, des poses (facultatif)
        et des métadonnées.  Cette méthode renvoie une liste de tuples
        ``(user_id, landmarks, poses, metadata)``.  Si le modèle ne contient
        pas de poses, ``poses`` est ``None``.

        Args:
            models_dir: Répertoire contenant les fichiers ``.npz``.

        Returns:
            Liste de tuples ``(user_id, landmarks, poses, metadata)``.  Le
            champ ``poses`` est ``None`` si absent.
        """
        gallery: List[Tuple[str, np.ndarray, Optional[np.ndarray], Dict]] = []
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return gallery
        for model_path in models_dir.glob("*.npz"):
            try:
                user_id = model_path.stem
                data = np.load(model_path, allow_pickle=True)
                if 'landmarks' not in data:
                    logger.warning(f"Skipping {user_id}: no landmarks in file")
                    continue
                landmarks = data['landmarks']
                poses: Optional[np.ndarray] = None
                if 'poses' in data:
                    poses = data['poses']
                # Restore PCA and scaler if present
                if 'pca' in data:
                    self.dtw_verifier.pca = data['pca'].item()
                if 'scaler' in data:
                    self.dtw_verifier.scaler = data['scaler'].item()
                # Load metadata
                metadata: Dict = {}
                if 'metadata' in data:
                    try:
                        metadata.update(data['metadata'].item())
                    except Exception:
                        pass
                # Legacy fields for backward compatibility
                for key in ['threshold', 'signature', 'mu_intra', 'sigma_intra']:
                    if key in data and key not in metadata:
                        metadata[key] = data[key].item() if isinstance(data[key], np.ndarray) else data[key]
                # Compute signature if missing and PCA available
                if 'signature' not in metadata and self.dtw_verifier.pca is not None:
                    try:
                        seq_pca = self.dtw_verifier.project_landmarks(landmarks)
                        metadata['signature'] = np.mean(seq_pca, axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to compute signature for {user_id}: {e}")
                gallery.append((user_id, landmarks, poses, metadata))
                logger.debug(
                    f"Loaded user {user_id}: landmarks={landmarks.shape}, poses={'None' if poses is None else poses.shape}, metadata keys: {list(metadata.keys())}"
                )
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
                continue
        logger.info(f"Loaded gallery: {len(gallery)} users from {models_dir}")
        return gallery


def create_multimodal_verifier() -> MultimodalVerifier:
    """Factory function to create multimodal verifier."""
    return MultimodalVerifier()


if __name__ == "__main__":
    # Test basique
    print("Testing MultimodalVerifier...")
    
    verifier = create_multimodal_verifier()
    
    # Simuler données
    n_frames = 45
    probe = np.random.randn(n_frames, 68, 3).astype(np.float32)
    
    gallery = [
        ("user1", probe + np.random.randn(n_frames, 68, 3) * 0.1, {"threshold": 6.71}),
        ("user2", np.random.randn(n_frames, 68, 3).astype(np.float32), {"threshold": 6.71}),
    ]
    
    # Fit PCA
    verifier.dtw_verifier.fit_pca([probe, gallery[0][1], gallery[1][1]])
    
    # Test 1:N
    user_id, distance, confidence, details = verifier.verify_1_to_n_with_margin(probe, gallery)
    
    print(f"\nResult: {confidence}")
    print(f"User: {user_id}, Distance: {distance:.3f}")
    print(f"Details: {details}")
