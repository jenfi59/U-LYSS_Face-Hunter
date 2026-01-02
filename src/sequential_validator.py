"""
Sequential Validator - Validation Multi-Criteres pour Reconnaissance Faciale
=============================================================================

Module de validation sequentielle robuste utilisant :
- Groupement de landmarks par importance biometrique
- Validation par pose (frontal, gauche, droite)
- Ratios anthropometriques invariants
- Distance angulaire (ArcFace-inspired)

Auteur: Assistant IA - Base sur recherche scientifique (FaceNet, ArcFace, SphereFace)
Date: 29 decembre 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DEFINITION DES GROUPES DE LANDMARKS (Base sur Recherche Anthropometrique)
# =============================================================================

LANDMARK_GROUPS = {
    # GROUPE 1 : ZONES INVARIANTES (Structure Osseuse - Haute Priorite)
    'invariant': {
        'description': 'Orbites oculaires, nez, menton (structure osseuse stable)',
        'weight': 0.4,
        'threshold': 1.0,
        'landmarks': {
            # Contours yeux (orbites osseuses)
            'eyes': [
                # il gauche (15 landmarks)
                33, 133, 160, 144, 145, 153, 154, 155, 157, 158, 159, 161, 163, 173, 246,
                # il droit (15 landmarks)
                362, 263, 387, 373, 374, 380, 381, 382, 384, 385, 386, 388, 390, 398, 466
            ],
            # Structure nasale (24 landmarks)
            'nose': [
                1, 2, 98, 327, 4, 5, 6, 19, 94, 141, 125, 241, 242, 238, 239, 129,
                358, 279, 420, 355, 305, 197, 195, 168
            ],
            # Menton et mandibule (40 landmarks)
            'chin_jaw': [
                152, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 389, 356,
                454, 323, 361, 288, 397, 365, 379, 378, 400, 377
            ]
        }
    },
    
    # GROUPE 2 : ZONES STABLES (Priorite Moyenne)
    'stable': {
        'description': 'Sourcils, pommettes (relativement fixes)',
        'weight': 0.2,
        'threshold': 1.3,
        'landmarks': {
            # Sourcils (20 landmarks)
            'eyebrows': [
                # Sourcil gauche
                107, 66, 105, 63, 70, 46, 53, 52, 65, 55,
                # Sourcil droit
                336, 296, 334, 293, 300, 276, 283, 282, 295, 285
            ],
            # Pommettes/zygomatiques (20 landmarks)
            'cheekbones': [
                116, 123, 50, 187, 207, 216, 206, 203, 205, 36,
                345, 352, 280, 330, 426, 436, 432, 434, 426, 266
            ]
        }
    },
    
    # GROUPE 3 : ZONES VARIABLES (Priorite Faible - Usage Limite)
    'variable': {
        'description': 'Bouche, levres (haute mobilite, non discriminant)',
        'weight': 0.1,
        'threshold': 2.0,
        'landmarks': {
            # Bouche externe (20 landmarks)
            'mouth_outer': [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409
            ],
            # Bouche interne (20 landmarks)
            'mouth_inner': [
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            ]
        }
    }
}


# =============================================================================
# RATIOS ANTHROPOMETRIQUES (Inspiration: Kanade, Farkas)
# =============================================================================

FACIAL_RATIOS = {
    # RATIOS HORIZONTAUX
    'R1_nose_width_to_interpupillary': {
        'description': 'Largeur nez / distance inter-pupillaire',
        'landmarks': {'nose_left': 98, 'nose_right': 327, 'eye_left_center': 33, 'eye_right_center': 263}
    },
    'R2_mouth_width_to_face_width': {
        'description': 'Largeur bouche / largeur visage',
        'landmarks': {'mouth_left': 61, 'mouth_right': 291, 'face_left': 234, 'face_right': 454}
    },
    'R3_jaw_width_to_cheekbone_width': {
        'description': 'Largeur machoire / largeur pommettes',
        'landmarks': {'jaw_left': 172, 'jaw_right': 397, 'cheekbone_left': 123, 'cheekbone_right': 352}
    },
    
    # RATIOS VERTICAUX
    'R11_nose_height_to_face_height': {
        'description': 'Hauteur nez / hauteur visage',
        'landmarks': {'nasion': 6, 'subnasale': 2, 'forehead_top': 10, 'chin_bottom': 152}
    },
    'R12_eye_to_eyebrow_distance': {
        'description': 'Distance il-sourcil / hauteur il',
        'landmarks': {'eye_top': 159, 'eyebrow_bottom': 70, 'eye_bottom': 145, 'eye_top_ref': 159}
    }
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def extract_group_landmarks(landmarks_full: np.ndarray, group_name: str) -> np.ndarray:
    """
    Extrait les landmarks d'un groupe specifique.
    
    Args:
        landmarks_full: Array (n_frames, 468, 3) - tous les landmarks
        group_name: 'invariant', 'stable', ou 'variable'
    
    Returns:
        Array (n_frames, n_landmarks_group, 3)
    """
    if group_name not in LANDMARK_GROUPS:
        raise ValueError(f"Groupe inconnu : {group_name}")
    
    group_info = LANDMARK_GROUPS[group_name]
    indices = []
    
    # Collecter tous les indices du groupe
    for subgroup_landmarks in group_info['landmarks'].values():
        indices.extend(subgroup_landmarks)
    
    # Eliminer doublons et trier
    indices = sorted(set(indices))
    
    # Verifier validite des indices
    max_idx = landmarks_full.shape[1]
    indices = [idx for idx in indices if idx < max_idx]
    
    return landmarks_full[:, indices, :]


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Distance angulaire (cosine similarity normalisee).
    Inspire de ArcFace : mesure l'angle entre vecteurs sur hypersphere.
    
    Args:
        v1, v2: Vecteurs de features (flattened landmarks ou embeddings)
    
    Returns:
        Distance  [0, 1] (0 = identique, 1 = oppose)
    """
    # Normaliser les vecteurs
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Cosine similarity
    cos_sim = np.dot(v1_norm, v2_norm)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Angle (radians)  normaliser a [0, 1]
    angle = np.arccos(cos_sim)
    return float(angle / np.pi)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance euclidienne normalisee."""
    return float(np.linalg.norm(v1 - v2))


def compute_distance_group(
    probe_group: np.ndarray,
    gallery_group: np.ndarray,
    metric: str = 'angular'
) -> float:
    """
    Calcule la distance entre deux sequences de landmarks d'un meme groupe.
    
    Args:
        probe_group: (n_frames_probe, n_landmarks, 3)
        gallery_group: (n_frames_gallery, n_landmarks, 3)
        metric: 'angular' ou 'euclidean'
    
    Returns:
        Distance scalaire
    """
    # Moyenne temporelle : representation stable
    probe_mean = probe_group.mean(axis=0).flatten()     # (n_landmarks * 3,)
    gallery_mean = gallery_group.mean(axis=0).flatten()
    
    if metric == 'angular':
        return angular_distance(probe_mean, gallery_mean)
    else:  # euclidean
        return euclidean_distance(probe_mean, gallery_mean)


def identify_pose_frames(poses_sequence: np.ndarray, config) -> Dict[str, List[int]]:
    """
    Groupe les frames par pose (frontal, gauche, droite).
    
    Args:
        poses_sequence: Array (n_frames, 3) - [yaw, pitch, roll]
        config: Configuration avec ranges de poses
    
    Returns:
        Dict {'frontal': [indices], 'left': [indices], 'right': [indices]}
    """
    pose_groups = {
        'frontal': [],
        'left': [],
        'right': []
    }
    
    # Ranges depuis config
    frontal_range = getattr(config, 'zone_frontal_range', (-15.0, 15.0))
    left_range = getattr(config, 'zone_left_range', (-40.0, -10.0))
    right_range = getattr(config, 'zone_right_range', (10.0, 40.0))
    
    for i, pose in enumerate(poses_sequence):
        yaw = pose[0]  # Angle yaw (gauche/droite)
        
        if frontal_range[0] <= yaw <= frontal_range[1]:
            pose_groups['frontal'].append(i)
        elif left_range[0] <= yaw <= left_range[1]:
            pose_groups['left'].append(i)
        elif right_range[0] <= yaw <= right_range[1]:
            pose_groups['right'].append(i)
    
    return pose_groups


def compute_facial_ratio(landmarks_mean: np.ndarray, ratio_name: str) -> float:
    """
    Calcule un ratio anthropometrique specifique.
    
    Args:
        landmarks_mean: Array (468, 3) - landmarks moyens
        ratio_name: Nom du ratio (ex: 'R1_nose_width_to_interpupillary')
    
    Returns:
        Valeur du ratio
    """
    if ratio_name not in FACIAL_RATIOS:
        return 0.0
    
    ratio_def = FACIAL_RATIOS[ratio_name]
    lm_indices = ratio_def['landmarks']
    
    # Extraire points
    points = {key: landmarks_mean[idx] for key, idx in lm_indices.items()}
    
    # Calcul selon le type de ratio
    if 'nose_left' in lm_indices and 'nose_right' in lm_indices:
        # R1: largeur nez / distance interpupillaire
        nose_width = np.linalg.norm(points['nose_right'] - points['nose_left'])
        eye_distance = np.linalg.norm(points['eye_right_center'] - points['eye_left_center'])
        return nose_width / (eye_distance + 1e-6)
    
    elif 'mouth_left' in lm_indices and 'face_left' in lm_indices:
        # R2: largeur bouche / largeur visage
        mouth_width = np.linalg.norm(points['mouth_right'] - points['mouth_left'])
        face_width = np.linalg.norm(points['face_right'] - points['face_left'])
        return mouth_width / (face_width + 1e-6)
    
    elif 'jaw_left' in lm_indices:
        # R3: largeur machoire / largeur pommettes
        jaw_width = np.linalg.norm(points['jaw_right'] - points['jaw_left'])
        cheekbone_width = np.linalg.norm(points['cheekbone_right'] - points['cheekbone_left'])
        return jaw_width / (cheekbone_width + 1e-6)
    
    elif 'nasion' in lm_indices and 'subnasale' in lm_indices:
        # R11: hauteur nez / hauteur visage
        nose_height = np.linalg.norm(points['subnasale'] - points['nasion'])
        face_height = np.linalg.norm(points['chin_bottom'] - points['forehead_top'])
        return nose_height / (face_height + 1e-6)
    
    elif 'eye_top' in lm_indices and 'eyebrow_bottom' in lm_indices:
        # R12: distance il-sourcil / hauteur il
        eye_eyebrow_dist = np.linalg.norm(points['eyebrow_bottom'] - points['eye_top'])
        eye_height = np.linalg.norm(points['eye_bottom'] - points['eye_top_ref'])
        return eye_eyebrow_dist / (eye_height + 1e-6)
    
    return 0.0


# =============================================================================
# CLASSE PRINCIPALE : SEQUENTIAL VALIDATOR
# =============================================================================

class SequentialValidator:
    """
    Validateur sequentiel multi-criteres pour reconnaissance faciale robuste.
    
    Implemente 3 strategies :
        - STRICT_SEQUENTIAL : Tous criteres doivent passer (AND)
        - WEIGHTED : Moyenne ponderee des distances
        - ADAPTIVE : Seuils adaptatifs selon qualite enrollment
    """
    
    def __init__(self, config, mode: str = 'strict_sequential'):
        """
        Args:
            config: Configuration du systeme (Config dataclass)
            mode: 'strict_sequential', 'weighted', 'adaptive'
        """
        self.config = config
        self.mode = mode
        self.landmark_groups = LANDMARK_GROUPS
        self.facial_ratios = FACIAL_RATIOS
        
        # Seuils depuis config (avec fallback)
        self.thresholds = {
            'invariant_group': getattr(config, 'threshold_invariant_group', 1.0),
            'stable_group': getattr(config, 'threshold_stable_group', 1.3),
            'variable_group': getattr(config, 'threshold_variable_group', 2.0),
            'frontal_pose': getattr(config, 'threshold_frontal_pose', 1.2),
            'left_pose': getattr(config, 'threshold_left_pose', 1.4),
            'right_pose': getattr(config, 'threshold_right_pose', 1.4),
            'ratio_threshold': getattr(config, 'threshold_ratios', 0.15),
            'global_weighted': getattr(config, 'threshold_global_weighted', 1.3),
            'triplet_margin': getattr(config, 'threshold_triplet_margin', 0.3)
        }
        
        # Poids pour mode weighted
        self.weights = {
            'invariant': getattr(config, 'weight_invariant', 0.4),
            'stable': getattr(config, 'weight_stable', 0.2),
            'poses': getattr(config, 'weight_poses', 0.3),
            'ratios': getattr(config, 'weight_ratios', 0.1)
        }
        
        # Metrique de distance
        self.distance_metric = getattr(config, 'distance_metric', 'angular')
        
        logger.info(f"SequentialValidator initialise - Mode: {self.mode}, Metrique: {self.distance_metric}")
    
    def verify_sequential(
        self,
        probe_landmarks: np.ndarray,
        probe_poses: Optional[np.ndarray],
        gallery_landmarks: np.ndarray,
        gallery_poses: Optional[np.ndarray],
        gallery_user_id: str = 'unknown'
    ) -> Tuple[bool, float, Dict]:
        """
        Validation sequentielle multi-stages.
        
        Args:
            probe_landmarks: (n_frames_probe, 468, 3)
            probe_poses: (n_frames_probe, 3) ou None
            gallery_landmarks: (n_frames_gallery, 468, 3)
            gallery_poses: (n_frames_gallery, 3) ou None
            gallery_user_id: Identifiant utilisateur (pour logs)
        
        Returns:
            (is_verified, final_distance, details)
        """
        details = {
            'mode': self.mode,
            'metric': self.distance_metric,
            'user_id': gallery_user_id,
            'stages': {}
        }
        
        logger.debug(f"Validation sequentielle pour {gallery_user_id} - Mode: {self.mode}")
        
        # STAGE 1 : Validation par groupes de landmarks
        stage1_valid, stage1_distances = self._validate_by_groups(
            probe_landmarks, gallery_landmarks
        )
        details['stages']['groups'] = {
            'valid': stage1_valid,
            'distances': stage1_distances,
            'thresholds': {
                'invariant': self.thresholds['invariant_group'],
                'stable': self.thresholds['stable_group']
            }
        }
        
        logger.debug(f"Stage 1 (Groupes): {' PASS' if stage1_valid else ' FAIL'} - {stage1_distances}")
        
        if not stage1_valid and self.mode == 'strict_sequential':
            final_distance = max(stage1_distances.values())
            logger.info(f" Rejet au Stage 1 (Groupes) - Distance max: {final_distance:.3f}")
            return False, final_distance, details
        
        # STAGE 2 : Validation par pose (si poses disponibles)
        stage2_valid = True
        stage2_distances = {}
        
        if probe_poses is not None and gallery_poses is not None:
            stage2_valid, stage2_distances = self._validate_by_pose(
                probe_landmarks, probe_poses,
                gallery_landmarks, gallery_poses
            )
            details['stages']['poses'] = {
                'valid': stage2_valid,
                'distances': stage2_distances,
                'thresholds': {
                    'frontal': self.thresholds['frontal_pose'],
                    'left': self.thresholds['left_pose'],
                    'right': self.thresholds['right_pose']
                }
            }
            
            logger.debug(f"Stage 2 (Poses): {' PASS' if stage2_valid else ' FAIL'} - {stage2_distances}")
            
            if not stage2_valid and self.mode == 'strict_sequential':
                final_distance = max(stage2_distances.values()) if stage2_distances else 999.0
                logger.info(f" Rejet au Stage 2 (Poses) - Distance max: {final_distance:.3f}")
                return False, final_distance, details
        else:
            details['stages']['poses'] = {'valid': True, 'note': 'Poses non disponibles, skipped'}
        
        # STAGE 3 : Validation par ratios anthropometriques
        stage3_valid, stage3_errors = self._validate_by_ratios(
            probe_landmarks, gallery_landmarks
        )
        details['stages']['ratios'] = {
            'valid': stage3_valid,
            'errors': stage3_errors,
            'threshold': self.thresholds['ratio_threshold']
        }
        
        logger.debug(f"Stage 3 (Ratios): {' PASS' if stage3_valid else ' FAIL'} - Max error: {max(stage3_errors.values()) if stage3_errors else 0:.3f}")
        
        if not stage3_valid and self.mode == 'strict_sequential':
            final_distance = max(stage3_errors.values()) if stage3_errors else 999.0
            logger.info(f" Rejet au Stage 3 (Ratios) - Max error: {final_distance:.3f}")
            return False, final_distance, details
        
        # Decision finale selon mode
        if self.mode == 'strict_sequential':
            is_verified = stage1_valid and stage2_valid and stage3_valid
            final_distance = self._compute_global_distance(stage1_distances, stage2_distances)
            details['decision'] = 'all_stages_passed' if is_verified else 'at_least_one_stage_failed'
        
        elif self.mode == 'weighted':
            is_verified, final_distance = self._weighted_decision(
                stage1_distances, stage2_distances, stage3_errors
            )
            details['decision'] = 'weighted_score'
        
        else:  # adaptive
            is_verified, final_distance = self._adaptive_decision(
                stage1_distances, stage2_distances, stage3_errors,
                probe_landmarks, gallery_landmarks
            )
            details['decision'] = 'adaptive_threshold'
        
        result_icon = "" if is_verified else ""
        logger.info(f"{result_icon} Decision finale: {'VERIFIED' if is_verified else 'REJECTED'} - Distance: {final_distance:.3f}")
        
        return is_verified, final_distance, details
    
    def _validate_by_groups(
        self,
        probe: np.ndarray,
        gallery: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        STAGE 1: Validation par groupes de landmarks.
        """
        distances = {}
        all_valid = True
        
        for group_name in ['invariant', 'stable']:
            try:
                probe_group = extract_group_landmarks(probe, group_name)
                gallery_group = extract_group_landmarks(gallery, group_name)
                
                distance = compute_distance_group(
                    probe_group, gallery_group, metric=self.distance_metric
                )
                distances[group_name] = distance
                
                threshold = self.thresholds[f'{group_name}_group']
                if distance > threshold:
                    all_valid = False
                    logger.debug(f"  Groupe '{group_name}': {distance:.3f} > {threshold:.3f} ")
                else:
                    logger.debug(f"  Groupe '{group_name}': {distance:.3f}  {threshold:.3f} ")
            
            except Exception as e:
                logger.error(f"Erreur validation groupe '{group_name}': {e}")
                distances[group_name] = 999.0
                all_valid = False
        
        return all_valid, distances
    
    def _validate_by_pose(
        self,
        probe_lm: np.ndarray,
        probe_poses: np.ndarray,
        gallery_lm: np.ndarray,
        gallery_poses: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        STAGE 2: Validation par pose (frontal, gauche, droite).
        """
        probe_pose_groups = identify_pose_frames(probe_poses, self.config)
        gallery_pose_groups = identify_pose_frames(gallery_poses, self.config)
        
        distances_by_pose = {}
        all_valid = True
        
        for pose_name in ['frontal', 'left', 'right']:
            probe_indices = probe_pose_groups[pose_name]
            gallery_indices = gallery_pose_groups[pose_name]
            
            if not probe_indices or not gallery_indices:
                logger.debug(f"  Pose '{pose_name}': frames manquantes, skipped")
                continue
            
            # Sous-ensembles pour cette pose
            probe_subset = probe_lm[probe_indices]
            gallery_subset = gallery_lm[gallery_indices]
            
            # Distance (moyenne temporelle puis metrique)
            distance = compute_distance_group(
                probe_subset, gallery_subset, metric=self.distance_metric
            )
            distances_by_pose[pose_name] = distance
            
            # Verifier seuil
            threshold = self.thresholds[f'{pose_name}_pose']
            if distance > threshold:
                all_valid = False
                logger.debug(f"  Pose '{pose_name}': {distance:.3f} > {threshold:.3f} ")
            else:
                logger.debug(f"  Pose '{pose_name}': {distance:.3f}  {threshold:.3f} ")
        
        return all_valid, distances_by_pose
    
    def _validate_by_ratios(
        self,
        probe: np.ndarray,
        gallery: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        STAGE 3: Validation par ratios anthropometriques.
        """
        # Moyenne temporelle
        probe_mean = probe.mean(axis=0)      # (468, 3)
        gallery_mean = gallery.mean(axis=0)
        
        # Calcul ratios
        probe_ratios = {name: compute_facial_ratio(probe_mean, name) for name in FACIAL_RATIOS.keys()}
        gallery_ratios = {name: compute_facial_ratio(gallery_mean, name) for name in FACIAL_RATIOS.keys()}
        
        # Validation
        ratio_errors = {}
        all_valid = True
        threshold = self.thresholds['ratio_threshold']
        
        for ratio_name in probe_ratios.keys():
            probe_val = probe_ratios[ratio_name]
            gallery_val = gallery_ratios[ratio_name]
            
            # Ecart relatif
            if gallery_val != 0:
                error = abs(probe_val - gallery_val) / gallery_val
            else:
                error = abs(probe_val - gallery_val)
            
            ratio_errors[ratio_name] = error
            
            if error > threshold:
                all_valid = False
                logger.debug(f"  Ratio '{ratio_name}': error {error:.3f} > {threshold:.3f} ")
        
        return all_valid, ratio_errors
    
    def _compute_global_distance(
        self,
        stage1_dist: Dict[str, float],
        stage2_dist: Dict[str, float]
    ) -> float:
        """Calcule distance globale (moyenne simple)."""
        all_distances = list(stage1_dist.values()) + list(stage2_dist.values())
        return float(np.mean(all_distances)) if all_distances else 999.0
    
    def _weighted_decision(
        self,
        stage1_dist: Dict[str, float],
        stage2_dist: Dict[str, float],
        stage3_err: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Mode WEIGHTED: Somme ponderee des distances.
        """
        # Distances moyennes par stage
        d_invariant = stage1_dist.get('invariant', 0)
        d_stable = stage1_dist.get('stable', 0)
        d_poses = np.mean(list(stage2_dist.values())) if stage2_dist else 0
        d_ratios = np.mean(list(stage3_err.values())) if stage3_err else 0
        
        # Score pondere
        score = (
            self.weights['invariant'] * d_invariant +
            self.weights['stable'] * d_stable +
            self.weights['poses'] * d_poses +
            self.weights['ratios'] * d_ratios
        )
        
        is_verified = score <= self.thresholds['global_weighted']
        return is_verified, float(score)
    
    def _adaptive_decision(
        self,
        stage1_dist: Dict[str, float],
        stage2_dist: Dict[str, float],
        stage3_err: Dict[str, float],
        probe: np.ndarray,
        gallery: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Mode ADAPTIVE: Seuils adaptatifs selon qualite enrollment.
        """
        # Qualite = inverse de la variance intra-gallery
        gallery_variance = np.var(gallery, axis=0).mean()
        quality_score = 1.0 / (1.0 + gallery_variance)  #  [0, 1]
        
        # Ajuster seuil (meilleure qualite  seuil plus strict)
        base_threshold = self.thresholds['global_weighted']
        adaptive_threshold = base_threshold * (1 - 0.3 * quality_score)
        
        # Distance globale
        score = self._compute_global_distance(stage1_dist, stage2_dist)
        
        is_verified = score <= adaptive_threshold
        logger.debug(f"  Mode adaptatif: quality={quality_score:.3f}, threshold={adaptive_threshold:.3f}")
        
        return is_verified, float(score)
