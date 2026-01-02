"""Test multi-galerie (1:N) avec données synthétiques.

Ce test illustre l'utilisation du mode séquentiel pour sélectionner
le meilleur candidat parmi plusieurs utilisateurs enregistrés.  Il
génère des séquences de repères aléatoires pour plusieurs identités
et vérifie que l'algorithme identifie correctement la séquence qui
correspond au probe.
"""

import numpy as np

from src.fr_core.config import Config
from src.fr_core.verification_dtw import VerificationDTW


def _create_synthetic_landmarks(n_frames: int = 20, offset: float = 0.0, scale: float = 0.01) -> np.ndarray:
    """Génère une séquence de repères 3D synthétiques.

    Args:
        n_frames: Nombre de frames dans la séquence.
        offset: Décalage appliqué aux coordonnées pour simuler des individus différents.
        scale: Écart type pour la génération aléatoire.

    Returns:
        np.ndarray de forme (n_frames, 478, 3)
    """
    base = np.zeros((478, 3), dtype=np.float32) + offset
    noise = np.random.normal(loc=0.0, scale=scale, size=(n_frames, 478, 3)).astype(np.float32)
    return base[np.newaxis, :, :] + noise


def test_sequential_multi_gallery():
    """Vérifie que le mode séquentiel sélectionne la bonne identité parmi plusieurs candidats."""
    cfg = Config()
    cfg.matching_mode = "sequential"
    # Ajuster les marges pour ne pas exiger de différence de score ni de couverture minimale
    cfg.composite_margin = 0.0
    cfg.coverage_threshold = 0.0
    cfg.coverage_margin = 0.0
    # Générer des identités synthétiques
    n_frames = 20
    landmarks_A = _create_synthetic_landmarks(n_frames=n_frames, offset=0.0)
    poses_A = np.zeros((n_frames, 3), dtype=np.float32)
    landmarks_B = _create_synthetic_landmarks(n_frames=n_frames, offset=5.0)
    poses_B = np.zeros((n_frames, 3), dtype=np.float32)
    landmarks_C = _create_synthetic_landmarks(n_frames=n_frames, offset=-5.0)
    poses_C = np.zeros((n_frames, 3), dtype=np.float32)
    # Probe identique à A
    probe = landmarks_A.copy()
    probe_poses = poses_A.copy()
    gallery = [
        ("A", landmarks_A, poses_A),
        ("B", landmarks_B, poses_B),
        ("C", landmarks_C, poses_C),
    ]
    verifier = VerificationDTW(config=cfg)
    best_user_id, score = verifier.verify_multi_gallery(probe, gallery, probe_poses)
    assert best_user_id == "A", f"Le meilleur candidat doit être A (score={score})"