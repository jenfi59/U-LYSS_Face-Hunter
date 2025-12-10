#!/bin/bash
# Script de lancement pour enrollment
# Usage: ./run_enrollment.sh <username>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ $# -eq 0 ]; then
    echo "Usage: ./run_enrollment.sh <username>"
    echo ""
    echo "Exemple: ./run_enrollment.sh jean"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 scripts/enroll_landmarks.py "$1"
