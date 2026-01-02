"""Tests de base pour vérifier les imports des dépendances.

Ces tests s'assurent que les principaux modules peuvent être importés sans
erreur.  Ils servent de première ligne de diagnostic lorsque le projet est
installé sur une nouvelle machine.
"""

import importlib


def test_mediapipe_import():
    """Vérifie que mediapipe est importable et expose un numéro de version."""
    mp = importlib.import_module("mediapipe")
    # La plupart des versions de mediapipe définissent __version__
    assert hasattr(mp, "__version__")


def test_opencv_import():
    """Vérifie qu'OpenCV peut être importé."""
    cv2 = importlib.import_module("cv2")
    # Vérifie que VideoCapture est disponible
    assert hasattr(cv2, "VideoCapture")


def test_project_imports():
    """Vérifie que les modules internes du projet se chargent correctement."""
    # Importe la configuration et le vérificateur principal
    config = importlib.import_module("src.fr_core.config")
    verifier_mod = importlib.import_module("src.fr_core.verification_dtw")
    assert hasattr(config, "Config") or hasattr(config, "default_config")
    assert hasattr(verifier_mod, "VerificationDTW")