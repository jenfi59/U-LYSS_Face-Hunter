#!/usr/bin/env python3
"""
U-LYSS ARM64 - Configuration centralisée avec système multimodal
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict
import json


@dataclass
class BiometricGroup:
    """Groupe de paramètres biométriques activable/désactivable."""
    name: str
    enabled: bool = False
    weight: float = 1.0
    parameters: Dict = field(default_factory=dict)


@dataclass
class Config:
    """Configuration principale U-LYSS."""
    
    # === PATHS ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    
    # === LANDMARKS ===
    #
    # `num_landmarks` defines the number of facial landmarks used throughout
    # the pipeline. Supported values are 68 (legacy dlib), 98, 194, and 468
    # (full MediaPipe mesh).  This value is duplicated into `n_landmarks` for
    # backward compatibility with existing code.  The `landmark_dims`
    # parameter remains at 3 (x, y, z) because the 3D depth coordinate is
    # synthesised from 2D landmarks when using MediaPipe.
    num_landmarks: int = 468  # Utilise les 468 landmarks bruts de MediaPipe (sans mapping)
    n_landmarks: int = 468
    landmark_dims: int = 3  # 3D (x, y, z) - z computed from depth
    
    # === PCA ===
    pca_n_components: int = 90  # Pour 468 landmarks 3D (1404 features) - conserve ~85% variance
    pca_variance_threshold: float = 0.95
    
    # === ENROLLMENT ===
    enrollment_n_frames: int = 45
    enrollment_min_frames_per_zone: int = 10
    enrollment_zones: List[str] = field(default_factory=lambda: ["FRONTAL", "LEFT", "RIGHT"])
    
    # Angles pour les zones (yaw)
    zone_frontal_range: Tuple[float, float] = (-15.0, 15.0)
    zone_left_range: Tuple[float, float] = (-40.0, -10.0)
    zone_right_range: Tuple[float, float] = (10.0, 40.0)
    
    # Minimum de changement entre frames (degrés)
    # Original: 2.0° pour assurer diversité (comme code JP)
    min_angle_change: float = 2.0
    enrollment_min_angle_change: float = 2.0  # Alias pour compatibilité
    
    # === DTW ===
    dtw_window: int = 10  # Fenêtre Sakoe-Chiba
    dtw_threshold: float = 6.71  # Seuil de vérification calibré
    
    # === LIVENESS ===
    # Eye Aspect Ratio (blink detection)
    ear_threshold: float = 0.21
    liveness_ear_threshold: float = 0.21  # Alias
    ear_consec_frames: int = 2
    liveness_consec_frames: int = 2
    liveness_min_blinks: int = 2
    
    # Motion detection
    motion_threshold: float = 2.0  # pixels
    liveness_motion_threshold: float = 2.0  # Alias
    
    # Texture (LBP variance)
    lbp_variance_threshold: float = 50.0
    liveness_texture_threshold: float = 50.0  # Alias
    
    # === CAMERA ===
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_backend: int = 0  # cv2.CAP_V4L2 sur ARM
    
    # === PERFORMANCE ===
    use_dtw_c_backend: bool = True
    onnx_num_threads: int = 4

    # === MATCHING MODE ===
    # Determines which verification strategy to use. Supported options are:
    #   'temporal'     → classic DTW matching (time-based alignment)
    #   'spatial'      → pose-based matching (compare frames with similar head poses)
    #   'spatiotemporal' → weighted fusion of DTW and pose-based distances
    # Matching mode used for verification.  Supported options:
    # 'temporal' – DTW only
    # 'spatial'  – pose-based matching
    # 'spatiotemporal' – fusion of temporal and spatial
    # 'sequential' – multi‑criteria sequential validation with composite score
    #
    # Set the default to 'sequential' so that the multi‑criteria validator is
    # automatically initialised.  This enables robust identification
    # across multiple sessions of the same user.
    matching_mode: str = "sequential"

    # === POSE‑BASED MATCHING ===
    # Pose tolerance in degrees for yaw (left/right), pitch (up/down) and roll (tilt).
    # Frames from the enrollment sequence whose pose differs by less than these
    # epsilons from the probe pose are considered similar.
    # Pose tolerance in degrees for yaw, pitch and roll.  Frames whose
    # head pose differs by less than these epsilons are considered
    # comparable for pose-based matching and sequential validation.  A
    # wider tolerance (20°) increases the number of matching frames,
    # improving robustness when combining sessions from the same person.
    pose_epsilon_yaw: float = 20.0
    pose_epsilon_pitch: float = 20.0
    pose_epsilon_roll: float = 20.0

    # Threshold on the average landmark distance for pose‑based verification. A probe
    # sequence is accepted if the mean Euclidean distance between matching frames
    # is below this value. Calibration is required to tune this threshold.
    pose_threshold: float = 3.0

    # === SPATIOTEMPORAL FUSION ===
    # Weight applied to the DTW distance when fusing temporal and pose‑based
    # verification. The final fused distance is:
    #   fused = fusion_alpha * dtw_distance + (1 - fusion_alpha) * pose_distance
    # A value of 0.5 gives equal importance to both components.
    fusion_alpha: float = 0.5
    # Threshold on the fused distance for spatiotemporal verification. If the
    # combined score is below this threshold, the probe is considered a match.
    spatiotemporal_threshold: float = 4.0

    # === POSE CALCULATION ===
    # Use PnP pose estimation to compute yaw, pitch and roll. When set to False,
    # the pose fields in the results remain None and pose‑based verification
    # automatically falls back to temporal mode.
    use_pnp_pose: bool = True

    # === SEQUENTIAL / COMPOSITE SCORING PARAMETERS ===
    # The following parameters define how distances from different facial
    # components and pose information are combined into a single composite
    # score for robust 1:N identification.  Each component distance is
    # normalised by its corresponding threshold (defined in
    # ``sequential_validator``) before being weighted.  The weights should
    # sum to 1.0 to maintain interpretability of the final score.
    weight_invariant: float = 0.4
    weight_stable: float = 0.3
    weight_pose: float = 0.2
    weight_ratio: float = 0.1

    # Threshold on the composite score.  A value of 1.0 means that the
    # average normalised distance across all components must not exceed the
    # individual thresholds.  Lower values make the decision stricter.
    # Par défaut, le score composite doit rester inférieur à 0.8 pour qu'une
    # correspondance soit jugée valide.  Cela équivaut à exiger que, en moyenne,
    # chaque distance normalisée reste en dessous de 80 % de son seuil interne.
    # Des valeurs plus basses rendent la décision plus stricte, tandis que des
    # valeurs proches de 1.0 rendent l'algorithme plus permissif.
    composite_threshold: float = 0.8

    # Margin required between the best and second best composite scores in
    # 1:N identification.  For example, ``0.2`` requires the best score to
    # be at least 20 % lower than the second best in order to accept the
    # candidate.  This reduces the risk of misidentification when two
    # candidates have very similar scores.
    # Margin required between the best and second best composite
    # scores in 1:N identification.  A value of 0.0 disables the
    # margin check, allowing acceptance when a single candidate has
    # the lowest score regardless of the next best.  This setting is
    # useful when two enrolment identities refer to the same person.
    # Marge relative minimale entre le meilleur et le second meilleur score
    # composite en 1:N.  Par défaut, un écart d'au moins 20 % est exigé pour
    # considérer qu'un candidat l'emporte clairement.  Un réglage trop faible
    # augmente le risque d'accepter un imposteur lorsque plusieurs scores
    # sont proches ; un réglage trop élevé peut entraîner des rejets injustifiés.
    composite_margin: float = 0.2

    # Coverage parameters for pose‑majority voting.  When poses are
    # available, the coverage is defined as the fraction of probe frames
    # having at least one gallery frame with a similar pose (using
    # ``pose_epsilon_*``).  The best candidate must satisfy the minimum
    # coverage threshold and beat the second best candidate by at least
    # ``coverage_margin`` to be accepted.  These parameters help to ensure
    # that a sufficient portion of the probe sequence contributes to the
    # match and that the match is not due to a small subset of frames.
    # Coverage parameters for pose‑majority voting.  Setting the
    # threshold and margin to 0.0 disables the coverage constraint.
    # This permits matches even when only a few frames share a similar
    # pose between the probe and gallery sequences.  Use with care,
    # as disabling coverage can increase false positives if many
    # impostors are present; however it is appropriate when the
    # gallery contains multiple enrolments of the same individual and
    # coverage may otherwise be low.
    # Seuil minimal de couverture : proportion de frames du probe ayant au moins
    # une correspondance de pose dans la séquence d'enrôlement.  Par défaut,
    # 30 % des frames doivent être comparées pour considérer la validation
    # suffisante.  Un seuil plus élevé diminue le risque de faux positifs.
    coverage_threshold: float = 0.3
    # Marge de couverture : écart minimal de couverture entre le meilleur et
    # le second meilleur candidat.  Par défaut, un écart d'au moins 20 % est
    # requis pour valider une identification en 1:N.  Cela permet d'éviter
    # d'accepter un candidat dont la couverture est trop proche de celle du
    # second meilleur.
    coverage_margin: float = 0.2
    
    # === SYSTÈME MULTIMODAL ===
    multimodal_enabled: bool = False  # Master switch
    
    # Groupes de paramètres biométriques (5 groupes)
    biometric_groups: Dict[str, BiometricGroup] = field(default_factory=lambda: {
        "GROUP_1_SECURITY": BiometricGroup(
            name="Sécurité 1:N",
            enabled=True,  # ACTIVÉ PAR DÉFAUT (critique)
            weight=1.0,
            parameters={
                "use_1_to_n": True,
                "min_margin": 0.20,  # 20% marge minimum
                "use_adaptive_threshold": True,
                "k_std": 3.0,  # μ + 3σ
            }
        ),
        "GROUP_2_GEOMETRIC": BiometricGroup(
            name="Ratios Géométriques",
            enabled=False,  # DÉSACTIVÉ initialement
            weight=0.4,
            parameters={
                "use_facial_ratios": False,
                "n_ratios": 20,  # R1-R20
                "ratio_weight": 0.4,  # Fusion: 0.6*DTW + 0.4*ratios
                "use_asymmetry": True,  # Asymétries L/R
            }
        ),
        "GROUP_3_TEMPORAL": BiometricGroup(
            name="Améliorations Temporelles",
            enabled=False,
            weight=0.3,
            parameters={
                "use_ddtw": False,  # Derivative DTW
                "ddtw_weight": 0.3,
                "use_procrustes": False,  # Normalisation géométrique
            }
        ),
        "GROUP_4_PERFORMANCE": BiometricGroup(
            name="Performance & Préfiltre",
            enabled=True,  # ACTIVÉ (important pour multi-users)
            weight=1.0,
            parameters={
                "use_topk_prefilter": True,
                "topk_k": 10,  # Top-10 candidats
            }
        ),
        "GROUP_5_ADVANCED": BiometricGroup(
            name="Méthodes Avancées",
            enabled=False,
            weight=1.0,
            parameters={
                "use_bootstrap_ci": False,  # Intervalles confiance
                "bootstrap_n": 20,
                "use_quality_gating": False,  # Filtrage qualité
                "quality_threshold": 0.7,
                "use_challenge_response": False,  # Liveness avancé
            }
        ),
    })
    
    def __post_init__(self):
        """Initialization tasks.

        - Ensure paths are `Path` instances
        - Keep `num_landmarks` and `n_landmarks` in sync for legacy code
        """
        # Coerce strings into Path objects
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)

        # Synchronise `n_landmarks` with `num_landmarks` for backward compatibility
        # If either value differs, we propagate `num_landmarks` to `n_landmarks`.
        # This ensures older code referencing `config.n_landmarks` sees the
        # configured number of landmarks while new code can rely on
        # `config.num_landmarks`.
        if hasattr(self, 'num_landmarks') and hasattr(self, 'n_landmarks'):
            if self.n_landmarks != self.num_landmarks:
                self.n_landmarks = self.num_landmarks
    
    @property
    def landmarks_model_path(self) -> Path:
        """Path to dlib shape predictor (68 points)."""
        return self.models_dir / "landmarks" / "shape_predictor_68_face_landmarks.dat"
    
    @property
    def face_detector_path(self) -> Path:
        return self.models_dir / "landmarks" / "face_detector_ultraface.onnx"
    
    @property
    def users_models_dir(self) -> Path:
        return self.models_dir / "users"
    
    def get_active_groups(self) -> List[str]:
        """Retourne les noms des groupes actifs."""
        return [name for name, group in self.biometric_groups.items() if group.enabled]
    
    def enable_group(self, group_name: str):
        """Active un groupe de paramètres."""
        if group_name in self.biometric_groups:
            self.biometric_groups[group_name].enabled = True
            self.multimodal_enabled = True  # Active le système multimodal
    
    def disable_group(self, group_name: str):
        """Désactive un groupe de paramètres."""
        if group_name in self.biometric_groups:
            self.biometric_groups[group_name].enabled = False
    
    def set_group_weight(self, group_name: str, weight: float):
        """Modifie le poids d'un groupe."""
        if group_name in self.biometric_groups:
            self.biometric_groups[group_name].weight = weight

    # ------------------------------------------------------------------
    # Matching mode helpers
    # ------------------------------------------------------------------
    def set_matching_mode(self, mode: str) -> None:
        """Set the verification matching mode.

        Args:
            mode: One of {"temporal", "spatial", "spatiotemporal"}.

        Raises:
            ValueError: If the provided mode is not supported.
        """
        valid_modes = {"temporal", "spatial", "spatiotemporal"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        self.matching_mode = mode

    def get_pose_epsilon(self) -> Tuple[float, float, float]:
        """Return the pose tolerance values for yaw, pitch and roll.

        Returns:
            A tuple (eps_yaw, eps_pitch, eps_roll) in degrees.
        """
        return (self.pose_epsilon_yaw, self.pose_epsilon_pitch, self.pose_epsilon_roll)

    def is_temporal_mode(self) -> bool:
        """Check if the current matching mode is temporal (DTW only)."""
        return self.matching_mode == "temporal"

    def is_spatial_mode(self) -> bool:
        """Check if the current matching mode is spatial (pose‑based)."""
        return self.matching_mode == "spatial"

    def is_spatiotemporal_mode(self) -> bool:
        """Check if the current matching mode is spatiotemporal (fusion)."""
        return self.matching_mode == "spatiotemporal"
    
    def save(self, path: str):
        """Sauvegarde la config en JSON."""
        data = {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Charge la config depuis JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Instance globale
_config: Config = None

def get_config() -> Config:
    """Retourne la configuration globale (singleton)."""
    global _config
    if _config is None:
        _config = Config()
        # Charger les paramètres d'utilisateur personnalisés depuis config/user_config.json
        # Si ce fichier existe, ses valeurs écrasent les attributs par défaut du dataclass.
        try:
            # Le fichier user_config.json est recherché dans le dossier config à la racine du projet
            project_root = _config.project_root  # Racine du projet déterminée par défaut
            user_config_path = project_root / "config" / "user_config.json"
            if user_config_path.exists():
                with open(user_config_path, "r", encoding="utf-8") as f:
                    overrides = json.load(f)
                # Appliquer les paramètres en ignorant ceux qui ne correspondent pas aux attributs existants
                for key, value in overrides.items():
                    if hasattr(_config, key):
                        setattr(_config, key, value)
        except Exception as e:
            # Ne pas interrompre l'exécution si la configuration utilisateur n'est pas lisible
            print(f"[WARNING] Impossible de charger config/user_config.json: {e}")
    return _config

def save_user_config(config: Config) -> None:
    """Enregistre la configuration utilisateur dans config/user_config.json.

    Les champs du dataclass `Config` sont sérialisés dans un dictionnaire
    et écrits dans le fichier JSON. Seuls les champs simples (int, float, str, bool)
    sont enregistrés. Les listes, tuples et objets complexes ne sont pas
    supportés et seront ignorés.

    Args:
        config: instance de Config à sérialiser.
    """
    try:
        data = {}
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            # N'enregistrer que les types simples pour éviter les structures non sérialisables
            if isinstance(value, (int, float, str, bool)):
                data[field_name] = value
        # Assurer l'existence du dossier de configuration
        project_root = config.project_root
        config_dir = project_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        user_config_path = config_dir / "user_config.json"
        with open(user_config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Configuration utilisateur enregistrée dans {user_config_path}")
    except Exception as e:
        print(f"[WARNING] Impossible d'enregistrer la configuration utilisateur: {e}")