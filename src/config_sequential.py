"""
Configuration Etendue pour Validation Sequentielle
===================================================

Extension de la configuration systeme pour integrer les parametres
du SequentialValidator multi-criteres.

Auteur: Assistant IA
Date: 29 decembre 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class ConfigSequential:
    """
    Configuration complete pour le systeme de validation sequentielle.
    
    Cette classe etend la configuration originale avec tous les parametres
    necessaires pour la validation multi-criteres (groupes, poses, ratios).
    """
    
    # ======================================================================
    # PARAMETRES ORIGINAUX (Herites)
    # ======================================================================
    
    # Chemins
    data_dir: str = "./datasets"
    models_dir: str = "./models"
    
    # MediaPipe
    num_landmarks: int = 468
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Camera
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    
    # Enrollment/Verification
    num_enrollment_frames: int = 50
    num_verification_frames: int = 30
    
    # Mode de matching
    matching_mode: str = "spatial"  # 'dtw_only', 'spatial', 'sequential'
    
    # DTW Legacy (pour compatibilite ascendante)
    dtw_threshold: float = 6.71
    pose_threshold: float = 3.0
    
    
    # ======================================================================
    # NOUVEAUX PARAMETRES - VALIDATION SEQUENTIELLE
    # ======================================================================
    
    # Activation du mode sequentiel
    sequential_validation_enabled: bool = True
    
    # Mode de validation : 'strict_sequential', 'weighted', 'adaptive'
    sequential_mode: str = 'strict_sequential'
    
    # Metrique de distance : 'angular' (recommande), 'euclidean'
    distance_metric: str = 'angular'
    
    
    # ----------------------------------------------------------------------
    # SEUILS PAR GROUPE DE LANDMARKS
    # ----------------------------------------------------------------------
    
    # Groupe INVARIANT (Structure osseuse - Haute priorite)
    # Landmarks: orbites oculaires (30), nez (24), menton/mandibule (40)
    threshold_invariant_group: float = 1.0
    
    # Groupe STABLE (Priorite moyenne)
    # Landmarks: sourcils (20), pommettes (20)
    threshold_stable_group: float = 1.3
    
    # Groupe VARIABLE (Priorite faible - Usage limite)
    # Landmarks: bouche/levres (40)
    threshold_variable_group: float = 2.0
    
    
    # ----------------------------------------------------------------------
    # SEUILS PAR POSE (Frontal, Gauche, Droite)
    # ----------------------------------------------------------------------
    
    # Pose FRONTAL (meilleure qualite)
    threshold_frontal_pose: float = 1.2
    
    # Pose GAUCHE (legere degradation)
    threshold_left_pose: float = 1.4
    
    # Pose DROITE (legere degradation)
    threshold_right_pose: float = 1.4
    
    
    # ----------------------------------------------------------------------
    # SEUILS RATIOS ANTHROPOMETRIQUES
    # ----------------------------------------------------------------------
    
    # Seuil pour ratios (deviation relative maximale)
    # Ex: 0.15 = 15% de deviation max entre probe et gallery
    threshold_ratios: float = 0.15
    
    
    # ----------------------------------------------------------------------
    # SEUIL GLOBAL (Mode WEIGHTED)
    # ----------------------------------------------------------------------
    
    # Seuil pour score pondere global (mode 'weighted')
    threshold_global_weighted: float = 1.3
    
    
    # ----------------------------------------------------------------------
    # SEUIL TRIPLET MARGIN (Verification 1:N)
    # ----------------------------------------------------------------------
    
    # Marge minimale entre meilleur match et 2eme match (triplet loss)
    threshold_triplet_margin: float = 0.3
    
    
    # ----------------------------------------------------------------------
    # POIDS POUR MODE WEIGHTED
    # ----------------------------------------------------------------------
    
    # Poids pour fusion ponderee (somme = 1.0)
    weight_invariant: float = 0.4    # Groupe invariant (structure osseuse)
    weight_stable: float = 0.2       # Groupe stable (sourcils, pommettes)
    weight_poses: float = 0.3        # Validation par poses
    weight_ratios: float = 0.1       # Ratios anthropometriques
    
    
    # ----------------------------------------------------------------------
    # RANGES POUR IDENTIFICATION DES POSES
    # ----------------------------------------------------------------------
    
    # Range yaw pour pose FRONTAL (degres)
    zone_frontal_range: Tuple[float, float] = (-15.0, 15.0)
    
    # Range yaw pour pose GAUCHE (degres)
    zone_left_range: Tuple[float, float] = (-40.0, -10.0)
    
    # Range yaw pour pose DROITE (degres)
    zone_right_range: Tuple[float, float] = (10.0, 40.0)
    
    
    # ----------------------------------------------------------------------
    # PARAMETRES ADAPTATIFS (Mode ADAPTIVE)
    # ----------------------------------------------------------------------
    
    # Facteur d'ajustement adaptatif base sur qualite
    # Seuil effectif = base_threshold * (1 - adaptive_factor * quality_score)
    adaptive_quality_factor: float = 0.3
    
    # Variance minimale pour considerer enrollment de qualite
    adaptive_min_variance: float = 0.001
    
    
    # ----------------------------------------------------------------------
    # LOGGING & DEBUGGING
    # ----------------------------------------------------------------------
    
    # Niveau de log pour sequential_validator
    log_level_sequential: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Sauvegarder les details de validation dans un fichier JSON
    save_validation_details: bool = False
    validation_details_path: str = "./validation_results.json"
    
    
    # ======================================================================
    # METHODES UTILITAIRES
    # ======================================================================
    
    def get_threshold_for_group(self, group_name: str) -> float:
        """Retourne le seuil configure pour un groupe de landmarks."""
        if group_name == 'invariant':
            return self.threshold_invariant_group
        elif group_name == 'stable':
            return self.threshold_stable_group
        elif group_name == 'variable':
            return self.threshold_variable_group
        else:
            raise ValueError(f"Groupe inconnu: {group_name}")
    
    def get_threshold_for_pose(self, pose_name: str) -> float:
        """Retourne le seuil configure pour une pose."""
        if pose_name == 'frontal':
            return self.threshold_frontal_pose
        elif pose_name == 'left':
            return self.threshold_left_pose
        elif pose_name == 'right':
            return self.threshold_right_pose
        else:
            raise ValueError(f"Pose inconnue: {pose_name}")
    
    def get_weights(self) -> Dict[str, float]:
        """Retourne les poids pour le mode weighted."""
        return {
            'invariant': self.weight_invariant,
            'stable': self.weight_stable,
            'poses': self.weight_poses,
            'ratios': self.weight_ratios
        }
    
    def validate_weights(self) -> bool:
        """Verifie que la somme des poids = 1.0 (0.01)."""
        total = self.weight_invariant + self.weight_stable + self.weight_poses + self.weight_ratios
        return abs(total - 1.0) < 0.01
    
    def summary(self) -> str:
        """Genere un resume lisible de la configuration."""
        return f"""

                CONFIGURATION VALIDATION SEQUENTIELLE                 


Mode                : {self.sequential_mode}
Metrique            : {self.distance_metric}
Enabled             : {self.sequential_validation_enabled}


 SEUILS PAR GROUPE                                                    

  Invariant (structure osseuse)  : {self.threshold_invariant_group:.2f}                              
  Stable (sourcils, pommettes)   : {self.threshold_stable_group:.2f}                              
  Variable (bouche)              : {self.threshold_variable_group:.2f}                              



 SEUILS PAR POSE                                                      

  Frontal  : {self.threshold_frontal_pose:.2f}  (Range: {self.zone_frontal_range[0]:+.1f}  {self.zone_frontal_range[1]:+.1f})        
  Gauche   : {self.threshold_left_pose:.2f}  (Range: {self.zone_left_range[0]:+.1f}  {self.zone_left_range[1]:+.1f})       
  Droite   : {self.threshold_right_pose:.2f}  (Range: {self.zone_right_range[0]:+.1f}  {self.zone_right_range[1]:+.1f})        



 RATIOS & GLOBAL                                                      

  Ratios (deviation max)         : {self.threshold_ratios:.2%}                           
  Global weighted                : {self.threshold_global_weighted:.2f}                              
  Triplet margin                 : {self.threshold_triplet_margin:.2f}                              



 POIDS (Mode WEIGHTED)                                                

  1 (Invariant)  : {self.weight_invariant:.1%}                                      
  2 (Stable)     : {self.weight_stable:.1%}                                      
  3 (Poses)      : {self.weight_poses:.1%}                                      
  4 (Ratios)     : {self.weight_ratios:.1%}                                      
    
                 : {(self.weight_invariant + self.weight_stable + self.weight_poses + self.weight_ratios):.1%} {' OK' if self.validate_weights() else ' ERREUR'}                          


Logging             : {self.log_level_sequential}
Save details        : {self.save_validation_details}
"""


# =========================================================================
# CONFIGURATIONS PREDEFINIES (Profils d'usage)
# =========================================================================

def get_config_strict() -> ConfigSequential:
    """
    Configuration STRICT : Securite maximale.
    - Mode: strict_sequential (tous criteres doivent passer)
    - Seuils tres restrictifs (invariant: 0.8, stable: 1.0)
    - Usage: Applications haute securite (banque, militaire)
    """
    config = ConfigSequential()
    config.sequential_mode = 'strict_sequential'
    config.threshold_invariant_group = 0.8
    config.threshold_stable_group = 1.0
    config.threshold_frontal_pose = 1.0
    config.threshold_left_pose = 1.2
    config.threshold_right_pose = 1.2
    config.threshold_ratios = 0.10  # 10% max deviation
    return config


def get_config_balanced() -> ConfigSequential:
    """
    Configuration BALANCED : Equilibre securite/usabilite.
    - Mode: weighted (fusion ponderee)
    - Seuils standards (invariant: 1.0, stable: 1.3)
    - Usage: Applications grand public (smartphones, bureautique)
    """
    config = ConfigSequential()
    config.sequential_mode = 'weighted'
    config.threshold_invariant_group = 1.0
    config.threshold_stable_group = 1.3
    config.threshold_global_weighted = 1.3
    return config


def get_config_permissive() -> ConfigSequential:
    """
    Configuration PERMISSIVE : Usabilite maximale.
    - Mode: adaptive (seuils adaptatifs selon qualite)
    - Seuils plus souples (invariant: 1.2, stable: 1.5)
    - Usage: Demos, prototypes, environnements a faible risque
    """
    config = ConfigSequential()
    config.sequential_mode = 'adaptive'
    config.threshold_invariant_group = 1.2
    config.threshold_stable_group = 1.5
    config.threshold_frontal_pose = 1.5
    config.threshold_left_pose = 1.8
    config.threshold_right_pose = 1.8
    config.threshold_ratios = 0.20  # 20% max deviation
    config.adaptive_quality_factor = 0.4
    return config


# =========================================================================
# EXEMPLE D'USAGE
# =========================================================================

if __name__ == "__main__":
    # Creer config par defaut
    config = ConfigSequential()
    
    # Afficher resume
    print(config.summary())
    
    # Verifier coherence
    if config.validate_weights():
        print(" Configuration valide (poids OK)")
    else:
        print(" Configuration invalide (poids incorrects)")
    
    print("\n" + "="*70)
    print("Configurations predefinies disponibles:")
    print("="*70)
    
    print("\n[1] STRICT (Securite maximale)")
    print(get_config_strict().summary())
    
    print("\n[2] BALANCED (Equilibre)")
    print(get_config_balanced().summary())
    
    print("\n[3] PERMISSIVE (Usabilite)")
    print(get_config_permissive().summary())
