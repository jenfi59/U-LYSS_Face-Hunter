#!/bin/bash
# Script de lancement pour verification
# Usage: ./run_verify.sh <username>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ $# -eq 0 ]; then
    echo "Usage: ./run_verify.sh <username>"
    echo ""
    echo "Exemple: ./run_verify.sh jean"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 scripts/verify.py "models/$1.npz"
