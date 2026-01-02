# Guide d'Installation Complet - D-Face Hunter ARM64 v1.2.1

Ce guide détaille l'installation complète de D-Face Hunter sur un système ARM64.

## 📋 Prérequis Système

### Matériel
- **Processeur** : ARM64/aarch64 (Raspberry Pi 4/5, Jetson Nano, FuriPhone, etc.)
- **RAM** : Minimum 2 GB (4 GB recommandé)
- **Stockage** : Minimum 1 GB d'espace libre
- **Caméra** : USB webcam ou caméra CSI

### Système d'exploitation
- **OS** : Linux ARM64 (Debian/Ubuntu basé)
- **Kernel** : 5.10+ recommandé
- **Display** : Support Qt/XCB pour interface graphique

### Logiciels de base
```bash
sudo apt update
sudo apt install -y build-essential git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    python3-dev pkg-config
```

---

## ⚠️ CONTRAINTES CRITIQUES DE VERSION

### Python : 3.12.x OBLIGATOIRE

**MediaPipe 0.10.18 nécessite Python 3.11 ou 3.12 UNIQUEMENT**

❌ **Python 3.13+ NON COMPATIBLE** : MediaPipe ne compile pas avec Python 3.13+
✅ **Python 3.12.12 RECOMMANDÉ** : Version testée et validée
✅ **Python 3.11.x** : Compatible mais moins testé

### NumPy : < 2.0 OBLIGATOIRE

**MediaPipe nécessite NumPy 1.x**

❌ **NumPy 2.x NON COMPATIBLE** : MediaPipe ne fonctionne pas avec numpy 2.0+
✅ **NumPy 1.26.4** : Version recommandée et testée

### OpenCV : 4.12.0.88

**Utiliser le wheel fourni ou installer depuis PyPI**

✅ **opencv-contrib-python 4.12.0.88** : Version validée ARM64
⚠️ Le wheel `py3-none` nécessite Qt/XCB système

---

## 🔧 Installation Étape par Étape

### Étape 1 : Installation de pyenv (si nécessaire)

pyenv permet d'installer et gérer plusieurs versions de Python.

```bash
# Installer pyenv
curl https://pyenv.run | bash

# Ajouter à ~/.bashrc ou ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Recharger le shell
source ~/.bashrc
```

### Étape 2 : Installation de Python 3.12.12

```bash
# Installer les dépendances de build pour Python
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Installer Python 3.12.12 via pyenv
pyenv install 3.12.12

# Vérifier l'installation
~/.pyenv/versions/3.12.12/bin/python --version
# Doit afficher: Python 3.12.12
```

**⏱️ Temps estimé** : 10-20 minutes selon la puissance du CPU

### Étape 3 : Cloner le Dépôt

```bash
cd ~/Develop  # ou votre dossier de projets
git clone https://github.com/jenfi59/U-LYSS_Face-Hunter.git
cd U-LYSS_Face-Hunter
```

### Étape 4 : Créer l'Environnement Virtuel

```bash
# Créer l'environnement avec Python 3.12.12
~/.pyenv/versions/3.12.12/bin/python -m venv mp_env

# Activer l'environnement
source mp_env/bin/activate

# Vérifier la version
python --version
# Doit afficher: Python 3.12.12
```

### Étape 5 : Installer les Dépendances Python

```bash
# Activer l'environnement (si pas déjà fait)
source mp_env/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer OpenCV depuis le wheel local (si disponible)
pip install opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl

# OU installer depuis PyPI si wheel local indisponible
pip install opencv-contrib-python==4.12.0.88

# Installer MediaPipe avec numpy < 2.0
pip install "numpy<2.0" mediapipe==0.10.18

# Réinstaller numpy 1.x si opencv a installé numpy 2.x
pip install "numpy<2.0"

# Installer les autres dépendances
pip install scipy scikit-learn dtaidistance
```

**⏱️ Temps estimé** : 2-5 minutes

### Étape 6 : Télécharger le Modèle MediaPipe

Le modèle recommandé pour la v1.2.1 est **`face_landmarker_v2_with_blendshapes.task`**, qui
contient la version v2 du Face Landmarker avec raffinements d’iris (478 points).
Si ce fichier n'est pas disponible, vous pouvez utiliser le fichier legacy
`face_landmarker.task`. Le script `install.sh` choisira automatiquement le modèle
disponible.

```bash
mkdir -p models/mediapipe
# Télécharger la version v2 (préféré)
wget -O models/mediapipe/face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker_v2_with_blendshapes.task || \
wget -O models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**📦 Taille** : ~3.7 MB

### Étape 7 : Vérification de l'Installation

```bash
source mp_env/bin/activate

# Test 1: Python version
python --version
# Expected: Python 3.12.12

# Test 2: MediaPipe
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
# Expected: MediaPipe: 0.10.18

# Test 3: NumPy version (CRITIQUE)
python -c "import numpy; print('NumPy:', numpy.__version__)"
# Expected: NumPy: 1.26.4 (DOIT être < 2.0)

# Test 4: OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"
# Expected: OpenCV: 4.12.0

# Test 5: Module D-Face Hunter
python -c "from src.fr_core import VerificationDTW; print('✅ D-Face Hunter ready')"
# Expected: ✅ D-Face Hunter ready

# Test 6: Caméra
python -c "import cv2; cap = cv2.VideoCapture(5); print('Camera:', cap.isOpened()); cap.release()"
# Expected: Camera: True
```

### Étape 8 : Configuration de l'Affichage (Important pour GUI)

```bash
# Ajouter à ~/.bashrc ou ~/.zshrc
echo 'export QT_QPA_PLATFORM=xcb' >> ~/.bashrc
echo 'export DISPLAY=:0' >> ~/.bashrc

# Appliquer immédiatement
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
```

---

## 🚀 Premier Lancement

### Test Rapide de la Caméra

```bash
source mp_env/bin/activate
export QT_QPA_PLATFORM=xcb

# Test avec caméra 5 (arrière)
python test_camera_display.py --camera 5

# Test avec caméra 6 (avant)
python test_camera_display.py --camera 6
```

**Appuyez sur 'q' pour quitter**

### Premier Enrollment

```bash
source mp_env/bin/activate
export QT_QPA_PLATFORM=xcb

# Interface tactile (recommandé pour smartphone/tablette)
./launch_touchscreen.sh

# Interface clavier (pour PC/laptop)
python enroll_interactive.py
```

---

## 🐛 Dépannage

### Problème : "ModuleNotFoundError: No module named 'mediapipe'"

**Solution** : L'environnement virtuel n'est pas activé
```bash
source mp_env/bin/activate
```

### Problème : "ImportError: numpy.core.multiarray failed to import"

**Cause** : NumPy 2.x installé au lieu de 1.x

**Solution** :
```bash
source mp_env/bin/activate
pip uninstall -y numpy
pip install "numpy<2.0"
```

### Problème : "OpenCV loader: missing configuration file"

**Cause** : Le wheel py3-none nécessite des dépendances système

**Solution** : Réinstaller depuis PyPI
```bash
source mp_env/bin/activate
pip uninstall -y opencv-contrib-python
pip install opencv-contrib-python==4.12.0.88
pip install "numpy<2.0"  # Réinstaller numpy 1.x
```

### Problème : "Cannot open camera /dev/video0"

**Cause** : Mauvais ID de caméra

**Solution** : Trouver l'ID correct
```bash
# Lister les caméras
ls -l /dev/video*

# Tester chaque caméra
for i in {0..7}; do
    python -c "import cv2; cap = cv2.VideoCapture($i); print('video$i:', cap.isOpened()); cap.release()"
done
```

Sur FuriPhone : généralement video5 (arrière) et video6 (avant)

### Problème : GUI ne s'affiche pas

**Solution** : Configurer Qt/XCB
```bash
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0

# Vérifier X11
xdpyinfo | grep "number of screens"
```

### Problème : "Python 3.13 detected. Only 3.11 and 3.12 are supported"

**Cause** : Version Python incompatible avec MediaPipe

**Solution** : Installer Python 3.12 via pyenv (voir Étape 2)

### Problème : "unrecognized arguments: 6" lors de l'enrollment

**Cause** : Mauvais format d'argument pour le script

**Solution** : Le script attend `--camera ID` et non `ID` seul. Le script `enroll_touchscreen.py` gère cela automatiquement.

---

## 📦 Versions des Dépendances

### Versions Validées

| Package               | Version       | Contrainte          |
|-----------------------|---------------|---------------------|
| Python                | 3.12.12       | 3.11.x ou 3.12.x    |
| mediapipe             | 0.10.18       | Exacte              |
| numpy                 | 1.26.4        | < 2.0               |
| opencv-contrib-python | 4.12.0.88     | 4.12.x              |
| scipy                 | 1.16.3        | Latest              |
| scikit-learn          | 1.8.0         | Latest              |
| dtaidistance          | 2.3.13        | Latest              |

### Fichier requirements.txt

```txt
numpy<2.0
mediapipe==0.10.18
opencv-contrib-python==4.12.0.88
scipy>=1.16.0
scikit-learn>=1.8.0
dtaidistance>=2.3.0
```

---

## 🔄 Mise à Jour

Pour mettre à jour D-Face Hunter :

```bash
cd U-LYSS_Face-Hunter
git pull origin main

# Réactiver l'environnement
source mp_env/bin/activate

# Réinstaller si nécessaire
pip install -r requirements.txt
```

---

## 🧹 Désinstallation

```bash
# Supprimer l'environnement virtuel
rm -rf mp_env

# Supprimer le dossier du projet
cd ..
rm -rf U-LYSS_Face-Hunter

# (Optionnel) Désinstaller pyenv
rm -rf ~/.pyenv
# Supprimer les lignes pyenv de ~/.bashrc
```

---

## 📞 Support

En cas de problème non résolu :

1. Vérifier les versions avec `python --version` et `pip list`
2. Consulter les logs dans `logs/`
3. Ouvrir une issue sur GitHub avec :
   - Votre OS et architecture (`uname -a`)
   - Version Python (`python --version`)
   - Liste des packages (`pip list`)
   - Message d'erreur complet

---

**Dernière mise à jour** : 1er janvier 2026  
**Version** : 1.2.1
