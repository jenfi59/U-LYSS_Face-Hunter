# D-Face Hunter - Arborescence Classifiée du Projet
**Date**: 2 janvier 2026  
**Version**: 1.2.1 Final Release

## Légende des Classifications

- **[C]** = **CRITIQUE** - Scripts essentiels de l'algorithme, fichiers système, config, bibliothèques utilisées dans main
- **[T]** = **TESTING** - Scripts et fichiers de test
- **[O]** = **OBSOLÈTE** - Anciennes versions, fichiers inutiles ou non utilisés
- **[W]** = **WHEELS** - Bibliothèques, packages, wheels à télécharger/compiler/installer
- **[D]** = **DOCUMENTATION** - Fichiers de documentation
- **[M]** = **MODELS** - Fichiers de modèles (data)

---

## 📁 Racine du Projet

```
D_Face_Hunter_ARM64_Vers_1_2_final_release/
│
├── [C] launch_touchscreen.py         # Interface tactile principale (GUI complète)
├── [C] launch_touchscreen.sh         # Lanceur shell pour interface tactile
├── [C] enroll_interactive.py         # Interface CLI enrollment (alternative terminal)
├── [C] verify_interactive.py         # Interface CLI verification (alternative terminal)
│
├── [C] setup_env.sh                  # Configuration environnement (PYTHONPATH, etc.)
├── [C] install.sh                    # Script d'installation du projet
├── [C] requirements.txt              # Dépendances Python du projet
│
├── [D] README.md                     # Documentation principale
├── [D] TOUCHSCREEN_README.md         # Guide interface tactile
├── [D] CHANGELOG.md                  # Historique des versions
├── [D] LICENSE                       # Licence du projet
├── [C] .gitignore                    # Configuration Git
│
├── 📁 config/                        # → Configuration système
├── 📁 docs/                          # → Documentation complète
├── 📁 models/                        # → Modèles MediaPipe et utilisateurs
├── 📁 opencv_whl_4_12/               # → Wheels OpenCV custom
├── 📁 scripts/                       # → Scripts enrollment/verification
├── 📁 src/                           # → Code source principal (fr_core)
├── 📁 tests/                         # → Suite de tests
├── 📁 utils/                         # → Utilitaires
└── 📁 mp_env/                        # → Environnement virtuel Python
```

---

## 📁 config/ - Configuration Système

```
config/
└── [C] camera_calibration.json.backup   # Backup calibration caméra
```

**Statut**: Configuration optionnelle de calibration caméra.

---

## 📁 docs/ - Documentation

```
docs/
├── [D] INSTALLATION.md                  # Guide installation
├── [D] launch_ts_scripts_call.md        # Architecture navigation touchscreen
├── [D] MODES.md                         # Documentation des modes
├── [D] PIPELINE_OVERVIEW.md             # Vue d'ensemble pipeline
├── [D] PROJECT_FILE_TREE_CLASSIFIED.md  # Arborescence classifiée du projet
├── [D] TESTS.md                         # Documentation tests
└── [D] VALIDATION_CRITERIA.md           # Critères de validation
```

---

## 📁 models/ - Modèles et Data

```
models/
├── mediapipe/
│   └── [M] face_landmarker_v2_with_blendshapes.task   # Modèle MediaPipe 468 landmarks
│
└── users/
    ├── [M] .gitkeep                    # Git placeholder
    ├── [M] jeanphi.npz                 # Profil utilisateur 1
    ├── [M] jp2.npz                     # Profil utilisateur 2
    ├── [M] jp.npz                      # Profil utilisateur 3
    └── [M] test_v1.npz                 # Profil test
```

**Description** :
- `mediapipe/` : Modèle pré-entraîné MediaPipe (fichier .task à ne pas modifier)
- `users/` : Profils d'enrollment (landmarks + poses) au format .npz

---

## 📁 opencv_whl_4_12/ - Wheels OpenCV Custom

```
opencv_whl_4_12/
└── [W] opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl
```

**Description** : Wheel OpenCV 4.12.0 compilé spécifiquement pour ARM64 avec support GTK.  
**Installation** : `pip install opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl`

---

## 📁 scripts/ - Scripts Enrollment & Verification

```
scripts/
├── [C] enroll_landmarks.py             # Script enrollment (phases auto + manuelle)
└── [O] verify_mediapipe.py             # Ancienne vérification externe (obsolète)
```

**Détails** :
- **enroll_landmarks.py** : Script appelé par subprocess depuis `launch_touchscreen.py` pour l'enrollment
- **verify_mediapipe.py** : **OBSOLÈTE** - Remplacé par méthode intégrée `run_validation_capture()` dans launch_touchscreen.py

**Action recommandée** : `verify_mediapipe.py` peut être archivé ou supprimé (non utilisé).

---

## 📁 src/ - Code Source Principal

### src/fr_core/ - Algorithme de Reconnaissance Faciale

```
src/
├── [C] config_sequential.py            # Configuration validation séquentielle
├── [C] sequential_validator.py         # Validateur séquentiel
│
└── fr_core/
    ├── [C] __init__.py                 # Module init
    ├── [C] config.py                   # Configuration générale (seuils, chemins)
    ├── [C] dtw_backend.py              # Backend DTW (Dynamic Time Warping)
    ├── [C] guided_enrollment.py        # Enrollment guidé (3 zones: frontal, gauche, droite)
    ├── [C] landmark_onnx.py            # Détection landmarks via ONNX (non utilisé actuellement)
    ├── [C] liveness.py                 # Détection de vivacité
    ├── [C] pose_matcher.py             # Matching des poses (yaw/pitch/roll)
    ├── [C] preprocessing.py            # Prétraitement des landmarks
    ├── [C] verification_dtw.py         # Vérification DTW principale
    └── [C] verification_multimodal.py  # Vérification multimodale
```

**Description** :
- **Modules critiques** : Tous les fichiers dans `fr_core/` sont essentiels à l'algorithme
- **landmark_onnx.py** : Backend ONNX disponible mais MediaPipe utilisé par défaut

---

## 📁 tests/ - Suite de Tests

```
tests/
├── [D] README.md                               # Documentation tests
│
├── [T] test_imports.py                         # Test imports modules
├── [T] test_system.py                          # Test système complet
├── [T] test_backend_availability.py            # Test disponibilité backends
│
├── [T] test_468_raw.py                         # Test landmarks 468 bruts
├── [T] test_raw_landmarks.py                   # Test landmarks raw
├── [T] test_landmark_indices.py                # Test indices landmarks
├── [T] test_landmark_position.py               # Test positions landmarks
├── [T] test_mediapipe_native_indices.py        # Test indices natifs MediaPipe
├── [T] test_nose_point.py                      # Test point nez
│
├── [T] test_head_pose.py                       # Test pose tête
├── [T] test_pose_468_simple.py                 # Test pose 468 simple
├── [T] test_pose_angles.py                     # Test angles pose
├── [T] test_pose_stability.py                  # Test stabilité pose
├── [T] test_yaw_real_time.py                   # Test yaw temps réel
├── [T] test_rotation_modes.py                  # Test modes rotation
│
├── [T] test_coordinate_system.py               # Test système coordonnées
├── [T] test_real_coordinates.py                # Test coordonnées réelles
│
├── [T] test_camera_gui_468.py                  # Test GUI caméra 468
├── [T] test_facemesh_output.py                 # Test sortie FaceMesh
├── [T] test_mediapipe_integration.py           # Test intégration MediaPipe
├── [T] test_mediapipe_interactive.py           # Test MediaPipe interactif
│
├── [T] test_enrollment_and_verification.py     # Test enrollment + verification
├── [T] test_enrollment_comparison.py           # Test comparaison enrollments
├── [T] test_mediapipe_enrollment_auto.py       # Test enrollment auto MediaPipe
│
├── [T] test_verify_mediapipe.py                # Test vérification MediaPipe
├── [T] test_verify_session.py                  # Test session verification
│
├── [T] test_multi_gallery.py                   # Test galerie multiple
│
├── [T] test_visualize_batch.py                 # Test visualisation batch
├── [T] test_visualize_existing.py              # Test visualisation existants
└── [T] test_visualize_landmarks.py             # Test visualisation landmarks
```

**Catégories de tests** :
1. **Tests système** : imports, backend, système complet
2. **Tests landmarks** : 468 points, indices, positions
3. **Tests pose** : angles, stabilité, rotations
4. **Tests coordonnées** : systèmes de coordonnées
5. **Tests GUI** : interfaces graphiques
6. **Tests enrollment/verification** : workflow complet
7. **Tests visualisation** : affichage résultats

**Total** : 37 scripts de test couvrant tous les aspects du système.

---

## 📁 utils/ - Utilitaires

```
utils/
├── [C] pose_estimation.py              # Estimation pose (yaw/pitch/roll)
└── [C] recalculate.py                  # Recalcul des poses pour modèles existants
```

**Description** :
- **pose_estimation.py** : Calcul angles yaw/pitch/roll depuis landmarks
- **recalculate.py** : Utilitaire pour recalculer poses de profils .npz existants

---

## 📁 mp_env/ - Environnement Virtuel Python

```
mp_env/                                 [W] Environnement virtuel complet
├── bin/                                    Python 3.12.12 + executables
├── include/                                Headers Python
├── lib/
│   └── python3.12/
│       └── site-packages/              [W] Packages installés
│           ├── opencv-contrib-python-4.12.0.88/
│           ├── mediapipe-0.10.18/
│           ├── numpy-1.26.4/
│           ├── scipy-1.16.3/
│           ├── scikit-learn-1.8.0/
│           ├── dtaidistance-2.3.13/
│           └── ... (autres dépendances)
├── pyvenv.cfg                          [C] Config environnement virtuel
└── share/
```

**Description** : Environnement virtuel isolé avec toutes les dépendances installées.

**Packages critiques** :
- `opencv-contrib-python` 4.12.0.88 (wheel custom ARM64)
- `mediapipe` 0.10.18
- `numpy` 1.26.4
- `scipy` 1.16.3
- `scikit-learn` 1.8.0
- `dtaidistance` 2.3.13

---

## 🗑️ Fichiers Obsolètes Identifiés

| Fichier | Raison | Action Recommandée |
|---------|--------|-------------------|
| `scripts/verify_mediapipe.py` | Remplacé par intégration dans launch_touchscreen.py | **Archiver ou supprimer** |

---

## 📊 Statistiques du Projet

### Répartition par Catégorie

| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| **[C] Critique** | 22 | Scripts principaux + config + fr_core |
| **[T] Testing** | 37 | Suite complète de tests |
| **[O] Obsolète** | 1 | Fichier à nettoyer |
| **[W] Wheels** | 1 | Wheel OpenCV + mp_env/ |
| **[D] Documentation** | 8 | Guides et docs |
| **[M] Models** | 5 | Modèles MediaPipe + profils users |

**Total fichiers projet** : 74 fichiers (hors mp_env/)

### Architecture Simplifiée

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE UTILISATEUR                     │
├─────────────────────────────────────────────────────────────┤
│  [C] launch_touchscreen.py  │  [C] enroll_interactive.py   │
│  (Interface tactile)         │  (Interface CLI)              │
└──────────────┬───────────────┴──────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐  ┌─────────────────────────┐
│   SCRIPTS ENROLLMENT/VERIFY  │  │   ALGORITHME (fr_core)  │
├──────────────────────────────┤  ├─────────────────────────┤
│ [C] enroll_landmarks.py      │  │ [C] config.py           │
│ [O] verify_mediapipe.py*     │  │ [C] guided_enrollment.py│
└──────────────┬───────────────┘  │ [C] verification_dtw.py │
               │                  │ [C] pose_matcher.py     │
               │                  │ [C] preprocessing.py    │
               │                  │ [C] dtw_backend.py      │
               │                  │ [C] liveness.py         │
               │                  └────────────┬────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    MODÈLES & DATA                            │
├──────────────────────────────────────────────────────────────┤
│  [M] models/mediapipe/face_landmarker_v2.task               │
│  [M] models/users/*.npz (profils enrollés)                   │
└──────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                 DÉPENDANCES EXTERNES                         │
├──────────────────────────────────────────────────────────────┤
│  [W] opencv-contrib-python 4.12.0 (ARM64 custom)            │
│  [W] mediapipe 0.10.18                                       │
│  [W] numpy, scipy, scikit-learn, dtaidistance                │
└──────────────────────────────────────────────────────────────┘
```

*Note: verify_mediapipe.py marqué obsolète - fonctionnalité intégrée dans launch_touchscreen.py*

---

## 🔍 Dépendances Critiques à Installer

### Ordre d'Installation Recommandé

```bash
# 1. Créer environnement virtuel
python3 -m venv mp_env
source mp_env/bin/activate

# 2. Installer OpenCV custom ARM64 (OBLIGATOIRE)
pip install opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl

# 3. Installer dépendances Python
pip install -r requirements.txt
```

### Contenu requirements.txt

```
mediapipe==0.10.18
numpy==1.26.4
scipy==1.16.3
scikit-learn==1.8.0
dtaidistance==2.3.13
sounddevice==0.5.3
# opencv-contrib-python installé depuis wheel custom
```

---

## 🚀 Commandes de Lancement

### Interface Tactile (Principale)

```bash
cd ~/Develop/D_Face_Hunter_ARM64_Vers_1_2_final_release
source mp_env/bin/activate
python launch_touchscreen.py
```

Ou via script :
```bash
./launch_touchscreen.sh
```

### Interface CLI (Alternative)

**Enrollment** :
```bash
source mp_env/bin/activate
python enroll_interactive.py
```

**Verification** :
```bash
source mp_env/bin/activate
python verify_interactive.py
```

---

## 📝 Notes de Maintenance

### Fichiers à Nettoyer (Optionnel)

1. `scripts/verify_mediapipe.py` → Archivage ou suppression (remplacé par intégration)

### Fichiers à Ne Jamais Modifier

- `models/mediapipe/face_landmarker_v2_with_blendshapes.task` → Modèle pré-entraîné
- `opencv_whl_4_12/*.whl` → Wheel custom ARM64
- `mp_env/` → Environnement virtuel géré par pip

### Fichiers Essentiels au Fonctionnement

**Top 10 fichiers critiques** :
1. `launch_touchscreen.py` - Interface principale
2. `src/fr_core/verification_dtw.py` - Algorithme vérification
3. `src/fr_core/guided_enrollment.py` - Enrollment guidé
4. `scripts/enroll_landmarks.py` - Capture enrollment
5. `src/fr_core/config.py` - Configuration système
6. `src/fr_core/dtw_backend.py` - Backend DTW
7. `src/fr_core/pose_matcher.py` - Matching poses
8. `src/fr_core/preprocessing.py` - Prétraitement
9. `setup_env.sh` - Setup environnement
10. `models/mediapipe/face_landmarker_v2_with_blendshapes.task` - Modèle MediaPipe

---

## 📊 Arborescence Complète Condensée

```
D_Face_Hunter_ARM64_Vers_1_2_final_release/
│
├── [C] Scripts Principaux
│   ├── launch_touchscreen.py (GUI tactile)
│   ├── enroll_interactive.py (CLI enrollment)
│   └── verify_interactive.py (CLI verification)
│
├── [C] Configuration & Setup
│   ├── setup_env.sh
│   ├── install.sh
│   └── requirements.txt
│
├── [D] Documentation (8 fichiers)
│   ├── README.md, CHANGELOG.md, LICENSE
│   └── docs/ (guides installation, modes, tests, etc.)
│
├── [C] Code Source (12 modules fr_core)
│   ├── src/fr_core/ (algorithme reconnaissance)
│   ├── src/config_sequential.py
│   └── src/sequential_validator.py
│
├── [C] Scripts Workflow (2 fichiers)
│   ├── scripts/enroll_landmarks.py
│   └── [O] scripts/verify_mediapipe.py (obsolète)
│
├── [M] Modèles & Data
│   ├── models/mediapipe/face_landmarker_v2.task
│   └── models/users/*.npz (4 profils)
│
├── [T] Tests (37 scripts)
│   └── tests/ (landmarks, pose, GUI, enrollment, etc.)
│
├── [C] Utilitaires
│   └── utils/ (pose_estimation, recalculate)
│
├── [W] Wheels & Environnement
│   ├── opencv_whl_4_12/opencv_contrib_python.whl
│   └── mp_env/ (Python 3.12.12 + packages)
│
└── [C] Configuration
    └── config/camera_calibration.json.backup
```

---

## ✅ Checklist Maintenance

- [x] **Nettoyer** : ~~Supprimer `docs/INSTALLATION.md.old`~~ ✅ **Fait**
- [ ] **Archiver** : `scripts/verify_mediapipe.py` (obsolète)
- [ ] **Backup** : Profils `models/users/*.npz` régulièrement
- [ ] **Vérifier** : Wheel OpenCV présent avant installation
- [ ] **Documenter** : Mettre à jour CHANGELOG.md pour chaque version
- [ ] **Tester** : Exécuter `tests/test_system.py` après modifications

---

**Légende Finale** :
- **[C]** = Critique (22)
- **[T]** = Testing (37)
- **[O]** = Obsolète (1)
- **[W]** = Wheels (1 + mp_env)
- **[D]** = Documentation (8)
- **[M]** = Models (5)

**Total** : 74 fichiers projet + mp_env (environnement virtuel complet)

---

*Document généré le 2 janvier 2026 - Version 1.2.1 Final Release*
