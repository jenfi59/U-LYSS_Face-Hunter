#!/usr/bin/env python3
"""
Script interactif de vérification - Mode Spatial
Lance la vérification avec interface utilisateur guidée
"""

import sys
import os
from pathlib import Path
import subprocess

# Détermination du dossier racine du projet.
# Le dossier contenant ce script (verify_interactive.py) est considéré comme
# la racine du projet.  Les chemins sont relatifs afin de rendre le
# projet autonome.
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    """Print header"""
    print("\n" + "="*70)
    print("     VERIFICATION - RECONNAISSANCE FACIALE (MODE SPATIAL)")
    print("="*70 + "\n")

def list_users():
    """List enrolled users"""
    users_dir = PROJECT_DIR / "models" / "users"
    if not users_dir.exists():
        return []
    
    npz_files = sorted(users_dir.glob("*.npz"))
    return [(f.stem, f) for f in npz_files]

def select_user():
    """Select user for verification"""
    users = list_users()
    
    if not users:
        print("❌ Aucun utilisateur enregistré!")
        print("\nVeuillez d'abord effectuer un enrôlement:")
        python_cmd = os.environ.get("PYTHON_BIN", sys.executable)
        print(f"  {python_cmd} enroll_interactive.py")
        print()
        sys.exit(1)
    
    print(f"📋 Utilisateurs enregistres ({len(users)}):\n")
    
    for i, (username, filepath) in enumerate(users, 1):
        # Get file size
        size_kb = filepath.stat().st_size / 1024
        print(f"   {i}. {username:20s} ({size_kb:6.1f} KB)")
    
    print()
    
    while True:
        try:
            choice = input("👤 Selectionnez un numero (ou nom d'utilisateur): ").strip()
            
            if not choice:
                print("❌ Choix vide, reessayez.\n")
                continue
            
            # Try as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(users):
                    return users[idx]
                else:
                    print(f"❌ Numero invalide (1-{len(users)})\n")
                    continue
            
            # Try as username
            for username, filepath in users:
                if username.lower() == choice.lower():
                    return (username, filepath)
            
            print(f"❌ Utilisateur '{choice}' non trouve\n")
            
        except KeyboardInterrupt:
            print("\n\n❌ Annule")
            sys.exit(1)

def show_model_info(username, filepath):
    """Show model information"""
    print(f"\n📊 Informations du modele:")
    print(f"   • Utilisateur: {username}")
    print(f"   • Fichier: {filepath}")
    
    try:
        import numpy as np
        data = np.load(filepath, allow_pickle=True)
        
        landmarks = data['landmarks']
        print(f"   • Frames enrollment: {landmarks.shape[0]}")
        print(f"   • Landmarks: {landmarks.shape[1]} points")
        
        if 'poses' in data and data['poses'] is not None:
            poses = data['poses']
            print(f"   • Yaw range: [{poses[:, 0].min():.1f}° a {poses[:, 0].max():.1f}°]")
            print(f"   • Pitch range: [{poses[:, 1].min():.1f}° a {poses[:, 1].max():.1f}°]")
    except Exception as e:
        print(f"   (Info detaillee non disponible: {e})")

def show_config():
    """Show verification config"""
    try:
        # Importer dynamiquement la classe de configuration
        from src.fr_core.config import Config
        cfg = Config()
        print("\n⚙️  Configuration verification:")
        print(f"   • Mode: {cfg.matching_mode}")
        # Afficher les paramètres selon le mode
        if cfg.matching_mode == "spatial":
            print(f"   • Epsilon pose: +/-{cfg.pose_epsilon_yaw}° yaw, +/-{cfg.pose_epsilon_pitch}° pitch, +/-{cfg.pose_epsilon_roll}° roll")
            print(f"   • Threshold: {cfg.pose_threshold} (distance spatiale)")
        elif cfg.matching_mode == "temporal":
            print(f"   • DTW threshold: {cfg.dtw_threshold}")
        elif cfg.matching_mode == "spatiotemporal":
            print(f"   • Alpha (poids temporel): {cfg.fusion_alpha}")
            print(f"   • DTW threshold: {cfg.dtw_threshold}")
            print(f"   • Pose threshold: {cfg.pose_threshold}")
        elif cfg.matching_mode == "sequential":
            print(f"   • Composite threshold: {cfg.composite_threshold}")
            print(f"   • Composite margin: {cfg.composite_margin}")
            print(f"   • Coverage threshold: {cfg.coverage_threshold}")
        else:
            print(f"   • Paramètres spécifiques pour le mode {cfg.matching_mode} inconnus")
    except Exception as e:
        print(f"\n⚠️  Config non disponible: {e}")

def show_instructions():
    """Show verification instructions"""
    print("\n" + "━"*70)
    print("📋 INSTRUCTIONS DE VERIFICATION")
    print("━"*70 + "\n")
    
    print("CAPTURE:")
    print("  • Positionnez-vous face a la camera")
    print("  • Le systeme capture automatiquement pendant 5 secondes")
    print("  • Bougez legerement la tete (differentes poses)")
    print("  • Environ 30-45 frames seront capturees")
    print()
    
    print("MODE SPATIAL:")
    print("  • Chaque frame de verification est comparee aux frames")
    print("    d'enrollment ayant une pose similaire (+/-10°)")
    print("  • Distance proche de 0 = identite confirmee")
    print("  • Coverage = % de frames avec match de pose")
    print()
    
    print("RESULTAT:")
    print("  • Match: TRUE si distance < threshold (3.0)")
    print("  • Distance: Score de similarite (0 = identique)")
    print("  • Coverage: % frames utilisees pour verification")
    print()
    
    print("━"*70 + "\n")

def run_verification(filepath):
    """Run verification script"""
    script_path = PROJECT_DIR / "scripts" / "verify_mediapipe.py"
    env_script = PROJECT_DIR / "setup_env.sh"
    
    # Déterminer l'interpréteur Python (PYTHON_BIN ou celui en cours)
    python_cmd = os.environ.get("PYTHON_BIN", sys.executable)
    # Construire la commande
    cmd = (
        f"cd {PROJECT_DIR} && "
        f"source {env_script} && "
        f"DISPLAY=:0 {python_cmd} {script_path} {filepath} --camera 0 --seconds 5"
    )
    
    print("📌 Appuyez sur ENTREE pour commencer la verification...")
    input()
    
    print("\n" + "="*70)
    print("VERIFICATION EN COURS...")
    print("="*70 + "\n")
    
    # Run with bash
    result = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash'
    )
    
    return result.returncode

def main():
    """Main function"""
    try:
        clear_screen()
        print_header()
        
        # Select user
        username, filepath = select_user()
        
        # Show model info
        show_model_info(username, filepath)
        
        # Show config
        show_config()
        
        # Show instructions
        show_instructions()
        
        # Run verification
        exit_code = run_verification(filepath)
        
        print("\n" + "="*70)
        if exit_code == 0:
            print("✅ VERIFICATION TERMINEE")
        else:
            print(f"⚠️  Verification terminee avec code: {exit_code}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Verification interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
