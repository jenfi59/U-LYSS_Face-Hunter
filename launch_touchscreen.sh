#!/bin/bash
# Script de lancement de l'interface tactile

cd "$(dirname "$0")"

# Activer l'environnement mp_env
if [ -d "$HOME/Develop/Face_Hunter/D_Face_Hunter_ARM64_Vers_1_2_pre_release/mp_env" ]; then
    source "$HOME/Develop/Face_Hunter/D_Face_Hunter_ARM64_Vers_1_2_pre_release/mp_env/bin/activate"
elif [ -d "$HOME/.pyenv/versions/3.12.12/envs/mp_env" ]; then
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

echo "=== D-Face Hunter - Interface Tactile ==="
echo "Lancement de l'interface..."

python launch_touchscreen.py
