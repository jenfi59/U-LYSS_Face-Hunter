# D-Face Hunter ARM64 v1.2.1

**Deterministic Face Hunter** – Système de reconnaissance faciale robuste et transparent, optimisé pour les architectures **ARM64**.  Cette version met à profit l'API **MediaPipe Face Landmarker** pour extraire 478 repères 3D et introduit une validation séquentielle multi‑critères.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)
![Platform](https://img.shields.io/badge/Platform-ARM64-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Présentation

**D‑Face Hunter** est un système déterministe de reconnaissance faciale conçu pour les appareils ARM64.  Il repose sur l’API **MediaPipe Face Landmarker** pour extraire **478 repères 3D** (468 du maillage facial + 10 d’iris) et calculer la pose (yaw, pitch, roll).  La version 1.2.1 introduit un mode de **validation séquentielle multi‑critères** pour l’identification en foule (1 :N) et fournit une documentation complète pour comprendre et modifier l’algorithme.

### Fonctionnalités clés

* ✅ **478 repères 3D** : maillage facial complet + iris grâce à MediaPipe.
* ✅ **Calcul de pose** : extraction d’une matrice 4×4 et conversion en yaw/pitch/roll calibrés.
* ✅ **Modes de vérification modulables** : *temporal* (DTW), *spatial* (filtrage par pose), *spatiotemporel* (fusion DTW/pose) et *séquentiel* (multi‑critères).
* ✅ **Validation séquentielle** : combinaison de distances normalisées sur des groupes de repères, ratios anthropométriques, couverture de pose et marge relative pour réduire les faux positifs en 1 :N.
* ✅ **Enrôlement en deux phases** : capture automatique (frontal/gauche/droite) puis validation manuelle via l’interface interactive.
* ✅ **Scripts interactifs** : outils conviviaux pour l’enrôlement et la vérification.

---

## 🏗️ Architecture

L’architecture complète du projet est détaillée dans `docs/PIPELINE_OVERVIEW.md`.  En résumé, la pipeline comprend :

1. **Capture caméra** via OpenCV.
2. **Détection MediaPipe** et extraction de 478 repères 3D + pose.
3. **Calibrage et normalisation** des repères (PCA, standardisation).
4. **Comparaison** selon quatre modes : temporal, spatial, spatiotemporel ou séquentiel.
5. **Décision** basée sur un seuil et une marge relative (en 1 :N).

Le document `PIPELINE_OVERVIEW.md` fournit des schémas et des explications détaillées sur chaque étape.

---

## 📦 Installation

### Prerequisites

- **Hardware**: ARM64 device (Raspberry Pi 4/5, Jetson Nano, FuriPhone, etc.)
- **OS**: Linux ARM64 (Debian/Ubuntu based)
- **Python**: **3.12.x OBLIGATOIRE** (3.11 possible mais 3.12 recommandé)
  - ⚠️ **Python 3.13+ NON COMPATIBLE** avec MediaPipe 0.10.18
  - Installer via pyenv recommandé (voir guide d'installation)
- **Camera**: USB webcam (ex: /dev/video5, /dev/video6) ou CSI camera
- **Display**: Support Qt/XCB pour interface graphique (QT_QPA_PLATFORM=xcb)

### Quick Install

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/D_Face_Hunter_ARM64.git
cd D_Face_Hunter_ARM64_Vers_1_2_final_release

# IMPORTANT: Installer Python 3.12 via pyenv (si pas déjà installé)
# Voir docs/INSTALLATION.md pour installation complète de pyenv

# Créer environnement virtuel avec Python 3.12
~/.pyenv/versions/3.12.12/bin/python -m venv mp_env
source mp_env/bin/activate

# Installer les dépendances (avec contraintes de version strictes)
pip install --upgrade pip
pip install opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl
pip install "numpy<2.0" mediapipe==0.10.18 scipy scikit-learn dtaidistance

# Télécharger le modèle MediaPipe (si non inclus)
mkdir -p models/mediapipe
wget -O models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**⚠️ ATTENTION**: Ne pas utiliser Python 3.13, MediaPipe n'est pas compatible !

Pour une installation guidée complète : **voir `docs/INSTALLATION.md`**

### Verify Installation

```bash
source mp_env/bin/activate
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
# Expected: MediaPipe: 0.10.18

python -c "import numpy; print('NumPy:', numpy.__version__)"
# Expected: NumPy: 1.26.4 (DOIT être < 2.0)

python -c "import cv2; print('OpenCV:', cv2.__version__)"
# Expected: OpenCV: 4.12.0

python -c "from src.fr_core import VerificationDTW; print('✅ D-Face Hunter ready')"
# Expected: ✅ D-Face Hunter ready
```

---

## 🚀 Quick Start

### 1. Enroll a User

```bash
# Activer l'environnement
source mp_env/bin/activate
export QT_QPA_PLATFORM=xcb

# Interface tactile (recommandé pour smartphone/tablette)
python enroll_touchscreen.py
# OU lancer avec : ./launch_touchscreen.sh

# Interface clavier (pour PC/laptop)
python enroll_interactive.py

# Enrollment direct (ligne de commande)
python scripts/enroll_landmarks.py <username> --camera 5
```

**Enrollment Process:**
1. **Phase 1** (Automatic - 45 frames):
   - Look straight at camera (frontal: 15 frames)
   - Turn head left (left: 15 frames)
   - Turn head right (right: 15 frames)
   - System auto-captures frames when pose changes

2. **Phase 2** (Manual - 5+ frames):
   - Press **SPACE** to capture each frame
   - Vary your pose for robustness
   - Press **'q'** when done (minimum 5 frames)

3. **Validation** (Immediate test):
   - Stay in front of camera
   - System verifies enrollment works
   - Shows distance and coverage

**Output**: `models/users/<username>.npz` (landmarks + poses)

### 2. Verify Identity

```bash
# Activer l'environnement
source mp_env/bin/activate
export QT_QPA_PLATFORM=xcb

# Interactive verification (recommended)
python verify_interactive.py

# Or direct verification
python scripts/verify_mediapipe.py models/users/<username>.npz --camera 5 --seconds 5
```

**Verification Process:**
- Captures 5 seconds of video (~30-45 frames)
- Compares with enrolled user using spatial mode
- Returns: Match (YES/NO), Distance, Coverage

**Expected Output:**
```
✅ User: john_doe
✅ Verified: YES
✅ Distance: 1.234567 (< 3.0 threshold)
✅ Coverage: 45.2%
```

---

## 📁 Project Structure

```
D_Face_Hunter_ARM64_Vers_1_2_sameperson/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup_env.sh                 # Environment setup script
├── enroll_interactive.py        # Interactive enrollment
├── verify_interactive.py        # Interactive verification
│
├── src/
│   └── fr_core/
│       ├── __init__.py
│       ├── config.py            # Configuration
│       ├── verification_dtw.py  # Spatial matching engine
│       └── ...
│
├── scripts/
│   ├── enroll_landmarks.py      # Enrollment script (MediaPipe)
│   ├── verify_mediapipe.py      # Verification script (spatial mode)
│   └── ...
│
├── config/
│   └── camera_calibration.json  # Offsets de pose (optionnel)
│
├── models/
│   ├── mediapipe/
│   │   └── face_landmarker.task
│   └── users/                   # Enrolled users (.npz files)
│
├── docs/
│   ├── PIPELINE_OVERVIEW.md     # Description du pipeline
│   ├── VALIDATION_CRITERIA.md   # Critères de validation et seuils
│   ├── MODES.md                 # Description des modes de comparaison
│   ├── INSTALLATION.md          # Guide d’installation détaillé
│   └── TESTS.md                 # Description des tests
│
└── tests/
    ├── test_imports.py          # Vérification des imports
    ├── test_enrollment_and_verification.py  # Tests synthétiques 1:1 et 1:N
    └── data/                    # Données de test (optionnel)
```

---

## ⚙️ Configuration

Le système n’utilise pas de fichier YAML de configuration : toutes les options sont définies dans les classes Python `Config` et `ConfigSequential` situées dans le module `src/fr_core`. Ces classes sont déclarées sous forme de dataclasses et chargent des paramètres par défaut.

Vous pouvez modifier ces paramètres de deux façons :

1. **En modifiant les attributs du dataclass avant d’instancier le vérificateur**. Par exemple :

```python
from src.fr_core.config import Config

config = Config()
config.matching_mode = "spatial"           # ou "temporal", "spatiotemporal", "sequential"
config.pose_epsilon_yaw = 10.0             # tolérance sur l’angle de lacet (en degrés)
config.pose_epsilon_pitch = 10.0           # tolérance sur l’angle de tangage
config.pose_epsilon_roll = 10.0            # tolérance sur l’angle de roulis
config.pose_threshold = 3.0                # seuil de distance pour accepter un match en mode spatial

# paramètres pour l’algorithme séquentiel
config.weight_invariant = 0.4
config.weight_stable    = 0.3
config.weight_pose      = 0.2
config.weight_ratio     = 0.1
config.composite_threshold = 0.8
config.composite_margin    = 0.2
config.coverage_threshold  = 0.3
config.coverage_margin     = 0.2
```

2. **En passant des arguments au niveau des scripts**. Les scripts `enroll_interactive.py` et `verify_interactive.py` acceptent des options en ligne de commande (par exemple `--matching-mode`, `--pose-epsilon-yaw`, etc.) qui écrasent les valeurs par défaut du dataclass.

3. **Via l’interface tactile (`launch_touchscreen.py`)** : lorsque vous exécutez le script `launch_touchscreen.py`, un bouton **PARAMETRES** s’affiche dans le menu principal.  Il ouvre un écran de réglage qui permet d’ajuster les principaux seuils (DTW, pose, spatiotemporel, composite) ainsi que les marges et la couverture au moyen de boutons `+` et `–`.  Les valeurs sélectionnées sont enregistrées dans le fichier `config/user_config.json` via `save_user_config()` et sont automatiquement réappliquées à chaque lancement.

4. **Via le script en ligne de commande `scripts/settings_cli.py`** : ce script permet de modifier les paramètres depuis le terminal sans passer par l’interface graphique.  Par exemple :

```bash
python scripts/settings_cli.py \
    --composite_threshold 0.8 \
    --composite_margin 0.2 \
    --coverage_threshold 0.3 \
    --coverage_margin 0.2
```

Le script prend en charge plusieurs arguments correspondant aux attributs de la classe `Config` (voir `scripts/settings_cli.py --help` pour la liste complète).  Les modifications sont enregistrées dans `config/user_config.json`.  Utilisez `--reset` pour supprimer ce fichier et revenir aux valeurs par défaut.

Les paramètres disponibles sont décrits en détail dans les fichiers `docs/VALIDATION_CRITERIA.md` et `docs/MODES.md`.

---

## 🧪 Tests

Le dossier `tests/` contient des tests unitaires et fonctionnels basés sur `pytest`. Pour lancer tous les tests :

```bash
cd D_Face_Hunter_ARM64_Vers_1_2_sameperson
pytest -q
```

Les scripts de tests principaux sont :

- **`test_imports.py`** : vérifie la présence des dépendances (MediaPipe, numpy, etc.).
- **`test_enrollment_and_verification.py`** : effectue des tests 1:1 (même personne) et 1:N (galerie) sur des données synthétiques pour valider les différentes méthodes de comparaison.
- **Autres tests** : la plupart des fichiers de test originaux sont conservés pour valider le fonctionnement de MediaPipe, l’alignement des repères et la cohérence de la pose.

Les instructions détaillées pour reproduire les scénarios de test (y compris les cas imposteur) sont décrites dans `docs/TESTS.md`.

---

## 📊 Performances et précision

Les temps de traitement varient selon la plateforme. À titre indicatif, sur une Raspberry Pi 5 (ARM64) :

| Opération                 | Temps approximatif | Notes                            |
|---------------------------|--------------------|----------------------------------|
| Enrôlement (phase 1+2)    | 15–20 s            | 90 images capturées              |
| Vérification              | 0,2–0,5 s          | Séquence probe de 30 frames      |
| Appel du vérificateur     | 0,05 s             | Par comparaison 1:1 ou 1:N       |

Les valeurs de distance et de couverture dépendent de l’utilisateur et du mode :
- **Autovérification** : distance proche de 0–2 (correspondance parfaite).  
- **Même personne, autre session** : distance typiquement entre 1 et 3.  
- **Personnes différentes** : distance supérieure à 3 (rejeter).

Dans le mode séquentiel, on calcule un **score composite** normalisé. Ce score doit être inférieur au seuil (`composite_threshold`) et la différence relative entre le meilleur et le second score doit dépasser `composite_margin` pour valider une identité. La **couverture** (proportion de frames comparables) doit également être supérieure à `coverage_threshold`.

---

## 🔬 Détails techniques

### MediaPipe Integration

```python
# Face detection + 468 landmarks + pose
import mediapipe as mp
from mediapipe.tasks.python import vision

detector = vision.FaceLandmarker.create_from_options(options)
result = detector.detect(image)

# Extract landmarks (468 points)
landmarks = result.face_landmarks[0][:468]  # (x, y, z)

# Extract pose from transformation matrix
pose_matrix = result.facial_transformation_matrixes[0]  # 4×4 matrix
rotation = Rotation.from_matrix(pose_matrix[:3, :3])
yaw, pitch, roll = rotation.as_euler('XZY', degrees=True)  # Euler XZY convention
```

### Spatial Matching Algorithm

```python
def verify_pose_based(probe_landmarks, probe_poses, 
                     gallery_landmarks, gallery_poses):
    """
    Spatial pose-aware matching.
    
    For each probe frame:
      1. Find gallery frames with similar pose (epsilon filtering)
      2. Compute Euclidean distance to each similar frame
      3. Keep minimum distance
    
    Average all per-frame minimum distances → Final score
    """
    distances = []
    for i, probe_frame in enumerate(probe_landmarks):
        # Filter gallery by pose similarity
        similar_frames = find_similar_poses(
            probe_poses[i], gallery_poses,
            epsilon_yaw=10.0, epsilon_pitch=10.0, epsilon_roll=10.0
        )
        
        if len(similar_frames) > 0:
            # Compute distances to similar frames
            dists = [euclidean_distance(probe_frame, gallery_landmarks[j]) 
                     for j in similar_frames]
            distances.append(min(dists))
    
    return np.mean(distances)
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. MediaPipe not found**
```bash
pip3 install mediapipe==0.10.18
```

**2. Camera not opening**
```bash
# Check camera device
ls -l /dev/video*

# Test with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened())"
```

**3. Validation returns `distance: inf`**
- Ensure you're using the latest version with spatial mode
- Check that `config.matching_mode = "spatial"`
- Verify enrollment saved poses: `python3 -c "import numpy as np; d = np.load('models/users/<user>.npz'); print('Poses:', 'poses' in d)"`

**4. Low coverage (<10%)**
- Move your head more during verification
- Ensure good lighting
- Check pose ranges match enrollment

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Jean-Philippe (j-phi)**
- GitHub: [@YOUR_GITHUB_USERNAME]
- Project: D-Face Hunter ARM64

---

## 🙏 Acknowledgments

- **MediaPipe** (Google) - Face mesh and pose estimation
- **scikit-learn** - PCA and machine learning tools
- **OpenCV** - Computer vision library
- **DTAIDistance** - Fast DTW implementation (optional)

---

## 📚 Citation

If you use D-Face Hunter in your research, please cite:

```bibtex
@software{dface_hunter_arm64,
  title={D-Face Hunter ARM64: Deterministic Face Recognition for ARM64 Devices},
  author={Jean-Philippe},
  year={2025},
  version={1.2.1},
  url={https://github.com/YOUR_USERNAME/D_Face_Hunter_ARM64}
}
```

---

## 🔮 Roadmap

- [x] Multi‑user verification (1:N matching) – implémenté dans la v1.2.1 via le mode séquentiel multi‑critères.
- [ ] Anti‑spoofing (liveness detection)
- [ ] GPU acceleration (OpenCL)
- [ ] Real‑time continuous monitoring
- [ ] Web interface
- [ ] Mobile app (Android/iOS)

---

**Version:** 1.2.1  
**Last Updated:** January 01, 2026  
**Status:** Release Candidate ✅
