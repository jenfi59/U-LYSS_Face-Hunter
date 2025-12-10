"""
Configuration globale pour le systeme de reconnaissance faciale.

Calibre pour modeles LANDMARKS (68 points geometriques).
"""

# ============================================================================
# SEUIL DTW (Dynamic Time Warping)
# ============================================================================

# Seuil de distance DTW pour la verification
# Calibre le 9 decembre 2025 pour modeles landmarks
# Reduction de 90.1% par rapport a l'ancien seuil Gabor+LBP (68.0)
DTW_THRESHOLD = 6.71  # Ancien: 68.0 (Gabor+LBP)

# ============================================================================
# DERIVATIVE DTW (DDTW) - Tier 2 #6
# ============================================================================

# Activer les features dynamiques (velocites/accelerations)
USE_DDTW = True

# Methode DDTW: 'none', 'velocity', 'acceleration'
# - 'none': DTW classique (statiques uniquement)
# - 'velocity': Ajoute derivees 1ere ordre (mouvements)
# - 'acceleration': Ajoute derivees 1ere + 2eme ordre (dynamiques completes)
DDTW_METHOD = 'velocity'  # Recommande: 'velocity' (bon equilibre)

# Normaliser les derivees par leur std
DDTW_NORMALIZE = True

# ============================================================================
# LIVENESS DETECTION / ANTI-SPOOFING - Tier 2 #7
# ============================================================================

# Activer la detection de vivacite
USE_LIVENESS = True

# Methodes de liveness a utiliser (liste)
# Options: 'blink', 'motion', 'texture'
LIVENESS_METHODS = ['blink', 'motion']  # Recommande: blink + motion

# Parametres Blink Detection
LIVENESS_BLINK_MIN = 1  # Nombre minimal de clignements
LIVENESS_BLINK_TIME = 5.0  # Temps maximal (secondes)

# Parametres Motion Analysis
LIVENESS_MOTION_MIN = 2.0  # Mouvement minimal (pixels)
LIVENESS_MOTION_FRAMES = 30  # Nombre de frames

# Parametres Texture Analysis (plus lent, optionnel)
LIVENESS_TEXTURE_THRESHOLD = 50.0  # Seuil de complexite

# Seuil de confiance minimal pour accepter liveness
LIVENESS_CONFIDENCE_THRESHOLD = 0.6  # 60%

# Contexte de calibration:
# - Separation: 1.02 (POSITIVE)
# - FAR: 0.00% (False Accept Rate)
# - FRR: 0.00% (False Reject Rate)
# - Methode: Percentile 75% dans l'intervalle [max_genuine, min_impostor]
# - Donnees: 2 genuine (5.91, 5.95), 2 impostor (6.97, 13.75)

# ============================================================================
# ENROLLMENT (Guided Enrollment)
# ============================================================================

# Zones de pose pour l'enrollment
POSE_ZONES = {
    'FRONTAL': {'yaw_min': -15, 'yaw_max': 15},
    'LEFT': {'yaw_min': -40, 'yaw_max': -10},
    'RIGHT': {'yaw_min': 10, 'yaw_max': 40}
}

# Nombre de frames par zone
FRAMES_PER_ZONE = 15

# Tolerances pour l'acceptance des poses
YAW_TOLERANCE = 15.0
PITCH_TOLERANCE = 15.0
ROLL_TOLERANCE = 15.0

# Changement minimal entre frames (degrees)
MIN_YAW_CHANGE = 2.0
MIN_PITCH_CHANGE = 2.0
MIN_ROLL_CHANGE = 2.0

# ============================================================================
# LANDMARKS (68 points MediaPipe subset)
# ============================================================================

# Nombre de landmarks utilises
N_LANDMARKS = 68

# Nombre de features (x, y coords)
N_LANDMARK_FEATURES = 136  # 68 * 2

# ============================================================================
# PCA (Principal Component Analysis)
# ============================================================================

# Nombre de composantes PCA
PCA_N_COMPONENTS = 45

# Variance expliquee minimale
PCA_MIN_VARIANCE = 0.95

# ============================================================================
# PREPROCESSING
# ============================================================================

# Taille des frames preprocessees
PREPROCESSED_SIZE = (64, 64)

# Normalisation
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# ============================================================================
# VERIFICATION
# ============================================================================

# Nombre de frames pour la verification
VERIFICATION_NUM_FRAMES = 10

# Source video par defaut (0 = webcam)
DEFAULT_VIDEO_SOURCE = 0

# ============================================================================
# QUALITY FILTER
# ============================================================================

# Seuils de qualite pour les frames
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_CONTRAST = 30
MIN_SHARPNESS = 100

# ============================================================================
# LOGGING
# ============================================================================

# Niveau de log
LOG_LEVEL = 'INFO'

# Format de log
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
