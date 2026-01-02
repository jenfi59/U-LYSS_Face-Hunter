#!/bin/bash
# Configuration environnement pour enrollment MediaPipe 468 landmarks

# Encodage UTF-8 pour emojis et caractères spéciaux
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Display pour OpenCV
export DISPLAY=:0

# Variables projet
# Détecte automatiquement le répertoire racine du projet en se basant sur
# l'emplacement de ce script.  Cela permet de déplacer librement le dossier
# du projet sans devoir modifier ce fichier.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$SCRIPT_DIR"
export MODELS_DIR="$PROJECT_ROOT/models"
export MEDIAPIPE_MODEL="$MODELS_DIR/mediapipe/face_landmarker.task"

# Ajouter src au PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "✅ Environnement configuré"
echo "   - Encodage : UTF-8"
echo "   - Display : $DISPLAY"
echo "   - Modèle MediaPipe : $MEDIAPIPE_MODEL"

# Vérifier modèle MediaPipe
if [ ! -f "$MEDIAPIPE_MODEL" ]; then
    echo "⚠️  Modèle MediaPipe manquant !"
    echo "   Téléchargez le fichier face_landmarker.task et placez-le dans $MEDIAPIPE_MODEL"
fi

# Vérifier Python et dépendances
echo ""
echo "Vérification Python..."
python --version

echo ""
echo "Vérification MediaPipe..."
python - <<'PYCODE'
try:
    import mediapipe as mp
    print('  MediaPipe:', mp.__version__)
except Exception as e:
    print('  Erreur MediaPipe:', e)
PYCODE

echo ""
echo "Vérification SciPy..."
python - <<'PYCODE'
try:
    from scipy.spatial.transform import Rotation
    print('  SciPy: OK')
except Exception as e:
    print('  Erreur SciPy:', e)
PYCODE

echo ""
echo "Prêt ! Vous pouvez maintenant :"
echo "  - Enrollment : python scripts/enroll_landmarks.py <username>"
echo "  - Vérification : python scripts/verify_mediapipe.py models/users/<username>.npz"
