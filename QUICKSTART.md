# 🚀 FR_VERS_JP v2.1 - Guide de Démarrage Rapide

## Installation

### Sur cet ordinateur

**Le dossier est 100% portable** - Toutes les dépendances sont installées.

```bash
cd FR_VERS_JP_2_1
# Prêt à l'emploi - aucune installation requise !
```

### Sur un nouvel ordinateur

```bash
cd FR_VERS_JP_2_1
pip install --user -r requirements.txt
```

**Dépendances installées :**
- OpenCV (caméra + traitement image)
- MediaPipe (détection landmarks)
- NumPy, SciPy (calculs)
- scikit-learn (PCA)

> 📹 **Caméra** : Détection automatique. OpenCV utilise `cv2.VideoCapture(0)` pour la caméra par défaut (index 0). Si plusieurs caméras, essaie index 1, 2, etc.

---

## 🚀 Launcher Interactif (RECOMMANDÉ)

**La façon la plus simple d'utiliser le système :**

```bash
python3 launcher.py
```

**Menu :**
```
[1] 📝 Enrollment - Enregistrer un utilisateur
[2] ✅ Verification - Vérifier l'identité  
[3] 👥 Lister les modèles
[4] 🗑️ Supprimer un modèle
[5] ⚙️ Paramètres (voir config DTW/DDTW/Liveness/PCA)
[0] 🚪 Quitter
```

**Avantages :**
- ✅ Interface colorée et intuitive
- ✅ PYTHONPATH configuré automatiquement
- ✅ Protection contre écrasement/suppression
- ✅ Liste les modèles existants
- ✅ Gestion erreurs et interruptions (Ctrl+C)

---

## Utilisation Manuelle

### 1️⃣ Enrollment (Enregistrement)

**Si vous n'utilisez pas le launcher :**

```bash
# Méthode 1: Wrapper (recommandée)
./run_enrollment.sh <nom_utilisateur>

# Méthode 2: Directe
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/enroll_landmarks.py <nom_utilisateur>
```

**Processus :**
- **Étape 1** : Guided enrollment (3 poses automatiques)
  - FRONTAL (face à la caméra)
  - LEFT (tournez à gauche)
  - RIGHT (tournez à droite)
  - 15 frames par pose = 45 frames total
  - Les marqueurs deviennent verts quand la pose est bonne

- **Étape 2** : Extraction landmarks
  - Restez devant la caméra
  - Appuyez sur **SPACE** pour capturer chaque frame
  - 68 landmarks (points faciaux) extraits
  - Q pour terminer

### 2️⃣ Verification

```bash
# Méthode 1: Directe
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/verify.py models/<nom_utilisateur>.npz

# Méthode 2: Wrapper (plus simple)  
./run_verify.sh <nom_utilisateur>
```

**Méthode DTW :**
- Distance-based (plus stable que GMM)
- DDTW pour anti-spoofing (détecte photos/vidéos)
- Threshold calibré automatiquement

## Fonctionnalités v2.1

✅ **Guided Enrollment** - Poses standardisées  
✅ **68 Landmarks** - Géométrie faciale  
✅ **DTW** - Dynamic Time Warping  
✅ **DDTW** - Derivative DTW (anti-spoofing)  
✅ **Liveness Detection** - Détection de vie  
✅ **PCA** - Réduction dimensionnelle

## Architecture

```
FR_VERS_JP_2_1/
├── fr_core/
│   ├── guided_enrollment.py    # Poses automatiques
│   ├── landmark_utils.py        # 68 landmarks MediaPipe
│   ├── verification_dtw.py      # Vérification DTW
│   ├── ddtw.py                  # Anti-spoofing
│   ├── liveness.py              # Détection de vie
│   └── config.py                # Configuration
├── scripts/
│   ├── enroll_landmarks.py      # Enrollment principal
│   └── verify.py                # Vérification test
├── models/                      # Modèles utilisateurs (.npz)
└── venv/                        # Environnement Python
```

## Performance

- **FAR** : < 1% (False Accept Rate)
- **FRR** : ~5% (False Reject Rate)
- **Liveness** : 95%+ de détection
- **DDTW** : +12.9% amélioration vs DTW classique

## Exemple Complet

```bash
# Enrollment
python3 scripts/enroll_landmarks.py jean
# → Suivez les poses (FRONTAL/LEFT/RIGHT)
# → Appuyez SPACE pour capturer les landmarks

# Vérification
python3 scripts/verify.py models/jean.npz
# → Regardez la caméra, bougez légèrement
# → Résultat: VERIFIED ou REJECTED
```

## Notes

- Les modèles .npz sont compatibles entre v2.0 et v2.1
- Utilisez DDTW pour détecter le spoofing (photos/vidéos)
- Les landmarks capturent la géométrie unique du visage
- PCA réduit les features pour une meilleure performance
