#!/bin/bash
# Script de lancement de l'enrollment tactile

cd "$(dirname "$0")"

# Activer l'environnement mp_env
if [ -d "$HOME/.pyenv/versions/3.12.12/envs/mp_env" ]; then
    source "$HOME/.pyenv/versions/3.12.12/envs/mp_env/bin/activate"
elif [ -d "mp_env" ]; then
    source mp_env/bin/activate
else
    echo "❌ Environnement mp_env introuvable !"
    echo "Installez-le avec: ./install.sh"
    exit 1
fi

# Configurer l'affichage
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0

echo "=== D-Face Hunter - Enrollment Tactile ==="
echo "Lancement de l'interface..."

python enroll_touchscreen.py
