#!/bin/bash
# D-Face Hunter ARM64 v1.2.1 - Installation Script

set -e

echo "===================================================================="
echo "  D-Face Hunter ARM64 v1.2.1 - Installation"
echo "===================================================================="
echo ""

echo "[1/7] Checking Python version..."
# Chercher Python 3.12 (recommandé) ou 3.11
PYTHON_BIN=""

# Vérifier pyenv Python 3.12.12 d'abord
if [ -f "$HOME/.pyenv/versions/3.12.12/bin/python" ]; then
    PYTHON_BIN="$HOME/.pyenv/versions/3.12.12/bin/python"
    echo "✅ Trouvé Python 3.12.12 via pyenv"
# Chercher python3.12 dans PATH
elif command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
    echo "✅ Trouvé python3.12 dans PATH"
# Chercher python3.11
elif command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
    echo "⚠️  Trouvé python3.11 (3.12 recommandé)"
# Vérifier python3 par défaut
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1-2)
    if [[ "$MAJOR_MINOR" == "3.12" ]]; then
        PYTHON_BIN="python3"
        echo "✅ Python $PYTHON_VERSION détecté"
    elif [[ "$MAJOR_MINOR" == "3.11" ]]; then
        PYTHON_BIN="python3"
        echo "⚠️  Python $PYTHON_VERSION détecté (3.12 recommandé)"
    else
        echo "❌ Python $PYTHON_VERSION détecté"
        echo ""
        echo "ERREUR: Python 3.12 ou 3.11 requis pour MediaPipe 0.10.18"
        echo "Python 3.13+ NON COMPATIBLE avec MediaPipe !"
        echo ""
        echo "Installation recommandée:"
        echo "  1. Installer pyenv: curl https://pyenv.run | bash"
        echo "  2. Installer Python 3.12: pyenv install 3.12.12"
        echo "  3. Utiliser: ~/.pyenv/versions/3.12.12/bin/python"
        echo ""
        echo "Voir docs/INSTALLATION.md pour plus de détails"
        exit 1
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Aucun Python 3.11/3.12 trouvé"
    echo "Veuillez installer Python 3.12 (voir docs/INSTALLATION.md)"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Python utilisé: $PYTHON_BIN ($PYTHON_VERSION)"
echo ""

echo "[2/7] Creating virtual environment..."
if [ -d "mp_env" ]; then
    echo "⚠️  mp_env existe déjà, réutilisation"
else
    $PYTHON_BIN -m venv mp_env
    echo "✅ Environnement virtuel créé"
fi
source mp_env/bin/activate
echo ""

echo "[3/7] Upgrading pip..."
pip install --upgrade pip
echo ""

echo "[4/7] Installing Python dependencies..."
echo "IMPORTANT: Installation avec contraintes de version strictes"
echo "  - NumPy < 2.0 (requis pour MediaPipe)"
echo "  - MediaPipe 0.10.18"
echo "  - OpenCV 4.12.0.88"
echo ""

# Installer OpenCV depuis wheel local si disponible
if [ -f "opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl" ]; then
    echo "Installation OpenCV depuis wheel local..."
    pip install opencv_whl_4_12/opencv_contrib_python-4.12.0-py3-none-linux_aarch64.whl
else
    echo "Installation OpenCV depuis PyPI..."
    pip install opencv-contrib-python==4.12.0.88
fi

# Installer MediaPipe avec NumPy < 2.0
echo "Installation MediaPipe et NumPy < 2.0..."
pip install "numpy<2.0" mediapipe==0.10.18

# Forcer NumPy 1.x si OpenCV a installé 2.x
echo "Vérification version NumPy..."
pip install "numpy<2.0"

# Installer autres dépendances
echo "Installation autres dépendances..."
pip install scipy scikit-learn dtaidistance

echo "✅ Dependencies installed"
echo ""

echo "[3/6] Checking MediaPipe model..."
# Préférer le modèle v2_with_blendshapes s'il existe ou le télécharger.
MODEL_DIR="models/mediapipe"
MODEL_V2="$MODEL_DIR/face_landmarker_v2_with_blendshapes.task"
MODEL_LEGACY="$MODEL_DIR/face_landmarker.task"
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_V2" ]; then
    echo "✅ Modèle MediaPipe v2 déjà présent ($MODEL_V2)"
elif [ -f "$MODEL_LEGACY" ]; then
    echo "✅ Modèle MediaPipe legacy déjà présent ($MODEL_LEGACY)"
else
    echo "Téléchargement du modèle MediaPipe v2_with_blendshapes (~3.7 MB)..."
    # Essayer de télécharger le modèle v2.  Si l'URL échoue, retomber sur le modèle legacy.
    if wget -q --show-progress -O "$MODEL_V2" \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker_v2_with_blendshapes.task"; then
        echo "✅ Modèle v2 téléchargé"
    else
        echo "⚠️  Téléchargement du modèle v2 échoué, téléchargement du modèle legacy..."
        wget -q --show-progress -O "$MODEL_LEGACY" \
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        echo "✅ Modèle legacy téléchargé"
    fi
fi
echo ""

echo "[4/6] Creating directories..."
mkdir -p models/users
mkdir -p logs
echo "✅ Directories created"
echo ""

echo "[5/6] Verifying installation..."
$PYTHON_BIN - <<'PYCODE'
try:
    import mediapipe as mp
    print('  MediaPipe:', mp.__version__)
except Exception as e:
    print('  MediaPipe import error:', e)
try:
    from src.fr_core import VerificationDTW
    print('  D-Face Hunter: Ready')
except Exception as e:
    print('  Verification module error:', e)
PYCODE
echo ""

echo "[6/6] Installation finished!"
echo "===================================================================="
echo "✅ Installation complete!"
echo "===================================================================="
echo ""
echo "Quick Start:"
echo "  1. Enroll a user:    python enroll_interactive.py"
echo "  2. Verify identity:  python verify_interactive.py"
echo ""
echo "Documentation: consultez le dossier docs/ pour plus d'informations"
echo ""
