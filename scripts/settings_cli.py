#!/usr/bin/env python3
"""
Outil en ligne de commande pour ajuster les paramètres de validation de
D‑Face Hunter et les enregistrer dans ``config/user_config.json``.

Utilisation :

.. code-block:: bash

    python scripts/settings_cli.py --composite_threshold 0.8 --composite_margin 0.2 \
           --coverage_threshold 0.3 --coverage_margin 0.2

Chaque argument correspond à un attribut de la classe ``Config``.  Si
plusieurs arguments sont fournis, chacun est converti en float (sauf
booleans) et appliqué sur la configuration courante.  La configuration
modifiée est ensuite enregistrée via ``save_user_config``.  Pour
réinitialiser tous les réglages, utilisez ``--reset``.
"""

import argparse
from src.fr_core.config import get_config, save_user_config

def main():
    parser = argparse.ArgumentParser(description="Modifier les paramètres de validation de D‑Face Hunter")
    parser.add_argument('--reset', action='store_true', help='Réinitialise les paramètres utilisateur (supprime user_config.json)')

    # Ajouter des arguments pour chaque paramètre ajustable
    adjustable_params = [
        'dtw_threshold', 'pose_threshold', 'spatiotemporal_threshold',
        'composite_threshold', 'composite_margin',
        'coverage_threshold', 'coverage_margin',
        'pose_epsilon_yaw', 'pose_epsilon_pitch', 'pose_epsilon_roll',
        'fusion_alpha',
        'weight_invariant', 'weight_stable', 'weight_pose', 'weight_ratio'
    ]
    for param in adjustable_params:
        parser.add_argument(f'--{param}', type=float, help=f'Nouvelle valeur pour {param}')

    args = parser.parse_args()

    # Charger la configuration existante
    config = get_config()

    if args.reset:
        # Supprimer user_config.json si présent
        import os
        user_config_path = config.project_root / 'config' / 'user_config.json'
        if user_config_path.exists():
            user_config_path.unlink()
            print(f"Configuration utilisateur réinitialisée (fichier supprimé : {user_config_path})")
        else:
            print("Aucun fichier user_config.json à supprimer.")
        return

    modified = False
    for param in adjustable_params:
        value = getattr(args, param)
        if value is not None:
            # Définir la nouvelle valeur dans config
            setattr(config, param, value)
            print(f"{param} ← {value}")
            modified = True

    if modified:
        # Enregistrer les modifications
        save_user_config(config)
        print("Configuration utilisateur mise à jour.")
    else:
        print("Aucune modification spécifiée. Use --help pour les options disponibles.")

if __name__ == '__main__':
    main()