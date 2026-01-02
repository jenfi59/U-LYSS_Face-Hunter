"""Tests basiques d'enrôlement et de vérification avec des données synthétiques.

Les fonctions de cette suite utilisent des repères 3D aléatoires pour valider
le comportement du mode spatial et du mode séquentiel.  Ces tests ne
reflètent pas des performances réelles mais permettent de détecter des
erreurs logiques ou des régressions dans l'API.
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
    # Base shape (ici un visage neutre) : zéro
    base = np.zeros((478, 3), dtype=np.float32)
    # Appliquer un décalage constant pour simuler un autre individu
    base += offset
    # Répliquer base sur plusieurs frames et ajouter du bruit
    noise = np.random.normal(loc=0.0, scale=scale, size=(n_frames, 478, 3)).astype(np.float32)
    return base[np.newaxis, :, :] + noise


def test_spatial_same_person():
    """Vérifie qu'une même séquence est reconnue en mode spatial."""
    cfg = Config()
    cfg.matching_mode = "spatial"
    # augmenter la tolérance pour augmenter la couverture lors de tests synthétiques
    cfg.pose_epsilon_yaw = 30.0
    cfg.pose_epsilon_pitch = 30.0
    cfg.pose_epsilon_roll = 30.0
    verifier = VerificationDTW(config=cfg)
    landmarks = _create_synthetic_landmarks()
    poses = np.zeros((landmarks.shape[0], 3), dtype=np.float32)
    is_match, dist, details = verifier.verify_auto(landmarks, poses, landmarks.copy(), poses.copy())
    assert is_match, f"La même séquence devrait être reconnue (distance={dist})"


def test_spatial_different_person():
    """Vérifie qu'une séquence différente est rejetée en mode spatial."""
    cfg = Config()
    cfg.matching_mode = "spatial"
    # Tolérances larges pour augmenter la couverture
    cfg.pose_epsilon_yaw = 30.0
    cfg.pose_epsilon_pitch = 30.0
    cfg.pose_epsilon_roll = 30.0
    verifier = VerificationDTW(config=cfg)
    landmarks_probe = _create_synthetic_landmarks()
    poses_probe = np.zeros((landmarks_probe.shape[0], 3), dtype=np.float32)
    # Individu différent : applique un gros décalage
    landmarks_other = _create_synthetic_landmarks(offset=10.0)
    poses_other = np.zeros((landmarks_other.shape[0], 3), dtype=np.float32)
    is_match, dist, details = verifier.verify_auto(landmarks_probe, poses_probe, landmarks_other, poses_other)
    # La distance devrait être significativement plus grande, donc pas de match
    assert not is_match, f"Les séquences différentes ne devraient pas correspondre (distance={dist})"


def test_sequential_multi_gallery():
    """Teste la logique 1:N en mode séquentiel avec des données synthétiques."""
    cfg = Config()
    cfg.matching_mode = "sequential"
    # Désactiver les marges et la couverture pour ce test
    cfg.composite_margin = 0.0
    cfg.coverage_threshold = 0.0
    cfg.coverage_margin = 0.0
    # Générer trois identités : A (référence), B (différent), C (différent)
    n_frames = 20
    landmarks_A = _create_synthetic_landmarks(n_frames=n_frames)
    poses_A = np.zeros((n_frames, 3), dtype=np.float32)
    landmarks_B = _create_synthetic_landmarks(n_frames=n_frames, offset=5.0)
    poses_B = np.zeros((n_frames, 3), dtype=np.float32)
    landmarks_C = _create_synthetic_landmarks(n_frames=n_frames, offset=-5.0)
    poses_C = np.zeros((n_frames, 3), dtype=np.float32)
    # Probe est identique à A (même personne)
    probe = landmarks_A.copy()
    probe_poses = poses_A.copy()
    # Préparer la galerie : liste de tuples (user_id, landmarks, poses)
    gallery = [
        ("user_A", landmarks_A, poses_A),
        ("user_B", landmarks_B, poses_B),
        ("user_C", landmarks_C, poses_C),
    ]
    verifier = VerificationDTW(config=cfg)
    best_user_id, score = verifier.verify_multi_gallery(probe, gallery, probe_poses)
    assert best_user_id == "user_A", f"Le meilleur candidat devrait être user_A (score={score})"