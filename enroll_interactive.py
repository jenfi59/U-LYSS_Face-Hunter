#!/usr/bin/env python3
"""
Script interactif d'enrollment - Mode Spatial
Lance l'enrollment avec interface utilisateur guidée
"""

import sys
import os
from pathlib import Path
import subprocess

# Détermination du dossier racine du projet.
# Le dossier contenant ce script (enroll_interactive.py) est considéré
# comme la racine du projet.  Tous les chemins sont calculés de manière
# relative afin de permettre l'utilisation du projet immédiatement après
# extraction de l'archive, sans dépendre d'un chemin absolu de la machine
# d'origine.
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    """Print header"""
    print("\n" + "="*70)
    print("     ENROLLMENT - RECONNAISSANCE FACIALE (MODE SPATIAL)")
    print("="*70 + "\n")

def list_existing_users():
    """List existing enrolled users"""
    users_dir = PROJECT_DIR / "models" / "users"
    if not users_dir.exists():
        return []
    
    npz_files = list(users_dir.glob("*.npz"))
    return [f.stem for f in npz_files]

def get_username():
    """Get username from user input"""
    existing = list_existing_users()
    
    if existing:
        print(f"📋 Utilisateurs deja enregistres ({len(existing)}):")
        for i, user in enumerate(existing, 1):
            print(f"   {i}. {user}")
        print()
    
    while True:
        username = input("👤 Entrez votre nom d'utilisateur (ex: jphi): ").strip()
        
        if not username:
            print("❌ Nom vide, reessayez.\n")
            continue
        
        # Check if exists
        if username in existing:
            print(f"\n⚠️  L'utilisateur '{username}' existe deja!")
            response = input("   Voulez-vous le remplacer? (o/N): ").strip().lower()
            if response == 'o':
                # Delete old file
                old_file = PROJECT_DIR / "models" / "users" / f"{username}.npz"
                old_file.unlink(missing_ok=True)
                print(f"✅ Ancien enrollment supprime\n")
                return username
            else:
                print()
                continue
        
        return username

def show_instructions():
    """Show enrollment instructions"""
    print("\n" + "━"*70)
    print("📋 INSTRUCTIONS D'ENROLLMENT")
    print("━"*70 + "\n")
    
    print("PHASE 1 - Capture Automatique (45 frames):")
    print("  • Placez-vous face a la camera")
    print("  • Le systeme capture automatiquement 3 zones:")
    print("    → 15 frames FRONTAL (tete droite)")
    print("    → 15 frames GAUCHE (tournez la tete a gauche)")
    print("    → 15 frames DROITE (tournez la tete a droite)")
    print("  • Suivez les indications visuelles a l'ecran")
    print()
    
    print("PHASE 2 - Validation Manuelle (minimum 5 frames):")
    print("  • Appuyez sur ESPACE pour capturer chaque frame")
    print("  • Variez les poses pour plus de robustesse")
    print("  • Appuyez sur 'q' quand termine")
    print()
    
    print("VALIDATION IMMEDIATE:")
    print("  • Test automatique de 3 secondes")
    print("  • Verification que l'enrollment fonctionne")
    print()
    
    print("━"*70 + "\n")

def run_enrollment(username: str) -> int:
    """Lance le script d'enrôlement avec l'environnement approprié.

    Cette fonction construit un appel à `enroll_landmarks.py` en
    utilisant le même interpréteur Python que le script courant.  Le
    script `setup_env.sh` est sourcé pour définir les variables
    d'environnement nécessaires (notamment PYTHONPATH et PYTHON_BIN).
    """
    script_path = PROJECT_DIR / "scripts" / "enroll_landmarks.py"
    env_script = PROJECT_DIR / "setup_env.sh"
    # Déterminer l'interpréteur Python à utiliser (PYTHON_BIN dans l'environnement ou sys.executable)
    python_cmd = os.environ.get("PYTHON_BIN", sys.executable)
    # Construire la commande bash
    cmd = (
        f"cd {PROJECT_DIR} && "
        f"source {env_script} && "
        f"DISPLAY=:0 {python_cmd} {script_path} {username} --camera 0"
    )
    print(f"🎥 Lancement de l'enrôlement pour : {username}")
    print()
    input("📌 Appuyez sur ENTREE quand prêt...")
    print("\n" + "=" * 70)
    print("ENROLLMENT EN COURS...")
    print("=" * 70 + "\n")
    result = subprocess.run(cmd, shell=True, executable="/bin/bash")
    return result.returncode

def show_results(username, exit_code):
    """Show enrollment results"""
    print("\n" + "="*70)
    
    if exit_code == 0:
        # Check if file created
        model_file = PROJECT_DIR / "models" / "users" / f"{username}.npz"
        
        if model_file.exists():
            print("✅ ENROLLMENT REUSSI!")
            print("="*70 + "\n")
            
            # Show model info
            print(f"📊 Profil enregistre:")
            print(f"   • Utilisateur: {username}")
            print(f"   • Fichier: {model_file}")
            
            # Get file size
            size_kb = model_file.stat().st_size / 1024
            print(f"   • Taille: {size_kb:.1f} KB")
            
            # Try to load and show stats
            try:
                import numpy as np
                data = np.load(model_file, allow_pickle=True)
                
                landmarks = data['landmarks']
                print(f"   • Frames: {landmarks.shape[0]}")
                print(f"   • Landmarks: {landmarks.shape[1]} points")
                
                if 'poses' in data and data['poses'] is not None:
                    poses = data['poses']
                    print(f"   • Yaw range: [{poses[:, 0].min():.1f}° a {poses[:, 0].max():.1f}°]")
                    print(f"   • Pitch range: [{poses[:, 1].min():.1f}° a {poses[:, 1].max():.1f}°]")
            except Exception as e:
                print(f"   (Info detaillee non disponible: {e})")
            
            print("\n📌 Prochaine étape:")
            # Indiquer comment lancer la vérification interactive avec le même interpréteur Python
            python_cmd = os.environ.get("PYTHON_BIN", sys.executable)
            print(f"   {python_cmd} verify_interactive.py")
            print()
            
        else:
            print("❌ ERREUR: Fichier enrollment non cree")
            print("="*70)
            print("\nLe processus a peut-etre ete interrompu.")
    else:
        print("❌ ENROLLMENT ECHOUE")
        print("="*70)
        print(f"\nCode de sortie: {exit_code}")
        print("L'enrollment n'a pas pu etre complete.")

def main():
    """Main function"""
    try:
        clear_screen()
        print_header()
        
        # Get username
        username = get_username()
        
        # Show instructions
        show_instructions()
        
        # Run enrollment
        exit_code = run_enrollment(username)
        
        # Show results
        show_results(username, exit_code)
        
        print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Enrollment interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
