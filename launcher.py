#!/usr/bin/env python3
"""
FR_VERS_JP v2.1 - Launcher Simple
==================================
Menu interactif pour enrollment, verification et settings.
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration PYTHONPATH
SCRIPT_DIR = Path(__file__).parent.absolute()
os.environ['PYTHONPATH'] = f"{SCRIPT_DIR}:{os.environ.get('PYTHONPATH', '')}"


class Colors:
    """Codes couleur ANSI."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def clear_screen():
    """Efface l'écran."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Affiche l'en-tête."""
    clear_screen()
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  FR_VERS_JP v2.1 - Reconnaissance Faciale{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def show_models():
    """Liste les modèles existants."""
    models_dir = SCRIPT_DIR / "models"
    if not models_dir.exists():
        return []
    
    models = list(models_dir.glob("*.npz"))
    if models:
        print(f"\n{Colors.CYAN}Modèles enregistrés:{Colors.ENDC}")
        for i, model in enumerate(models, 1):
            size_kb = model.stat().st_size / 1024
            print(f"  {Colors.GREEN}{i}.{Colors.ENDC} {model.stem} ({size_kb:.1f} KB)")
    else:
        print(f"\n{Colors.YELLOW}Aucun modèle enregistré.{Colors.ENDC}")
    
    return models


def enrollment():
    """Lance l'enrollment."""
    print_header()
    print(f"{Colors.BOLD}=== ENROLLMENT ==={Colors.ENDC}\n")
    
    show_models()
    
    username = input(f"\n{Colors.CYAN}Nom d'utilisateur:{Colors.ENDC} ").strip()
    if not username:
        print(f"{Colors.RED}✗ Nom requis{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    model_path = SCRIPT_DIR / "models" / f"{username}.npz"
    if model_path.exists():
        overwrite = input(f"{Colors.YELLOW}⚠ Modèle existant. Écraser? (o/N):{Colors.ENDC} ").strip().lower()
        if overwrite != 'o':
            print(f"{Colors.YELLOW}Annulé{Colors.ENDC}")
            input("\nAppuyez sur ENTER pour continuer...")
            return
    
    print(f"\n{Colors.CYAN}Démarrage de l'enrollment...{Colors.ENDC}")
    print(f"{Colors.YELLOW}Étape 1: 3 poses (FRONTAL/LEFT/RIGHT) - 45 frames{Colors.ENDC}")
    print(f"{Colors.YELLOW}Étape 2: Extraction landmarks - Appuyez SPACE{Colors.ENDC}\n")
    
    input("Appuyez sur ENTER pour commencer...")
    
    try:
        result = subprocess.run(
            ["python3", str(SCRIPT_DIR / "scripts" / "enroll_landmarks.py"), username],
            cwd=str(SCRIPT_DIR)
        )
        if result.returncode == 0:
            print(f"\n{Colors.GREEN}✓ Enrollment réussi pour {username}{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}✗ Erreur durant l'enrollment{Colors.ENDC}")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Enrollment interrompu{Colors.ENDC}")
    
    input("\nAppuyez sur ENTER pour continuer...")


def verification():
    """Lance la vérification."""
    print_header()
    print(f"{Colors.BOLD}=== VERIFICATION ==={Colors.ENDC}\n")
    
    models = show_models()
    
    if not models:
        print(f"\n{Colors.RED}Aucun modèle disponible. Faites d'abord un enrollment.{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    username = input(f"\n{Colors.CYAN}Nom d'utilisateur à vérifier:{Colors.ENDC} ").strip()
    if not username:
        print(f"{Colors.RED}✗ Nom requis{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    model_path = SCRIPT_DIR / "models" / f"{username}.npz"
    if not model_path.exists():
        print(f"{Colors.RED}✗ Modèle non trouvé pour {username}{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    print(f"\n{Colors.CYAN}Démarrage de la vérification...{Colors.ENDC}")
    print(f"{Colors.YELLOW}1. Liveness detection (blink + mouvement){Colors.ENDC}")
    print(f"{Colors.YELLOW}2. DTW + DDTW verification{Colors.ENDC}\n")
    
    input("Appuyez sur ENTER pour commencer...")
    
    try:
        result = subprocess.run(
            ["python3", str(SCRIPT_DIR / "scripts" / "verify.py"), str(model_path)],
            cwd=str(SCRIPT_DIR)
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Vérification interrompue{Colors.ENDC}")
    
    input("\nAppuyez sur ENTER pour continuer...")


def settings():
    """Affiche et permet de modifier les paramètres."""
    print_header()
    print(f"{Colors.BOLD}=== PARAMÈTRES ==={Colors.ENDC}\n")
    
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from fr_core import config
        
        print(f"{Colors.CYAN}Configuration DTW:{Colors.ENDC}")
        print(f"  DTW_THRESHOLD: {Colors.GREEN}{config.DTW_THRESHOLD}{Colors.ENDC}")
        print(f"  WINDOW_SIZE: {Colors.GREEN}{config.WINDOW_SIZE}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Configuration DDTW:{Colors.ENDC}")
        print(f"  USE_DDTW: {Colors.GREEN}{config.USE_DDTW}{Colors.ENDC}")
        print(f"  DDTW_METHOD: {Colors.GREEN}{config.DDTW_METHOD}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Configuration Liveness:{Colors.ENDC}")
        print(f"  USE_LIVENESS: {Colors.GREEN}{config.USE_LIVENESS}{Colors.ENDC}")
        print(f"  LIVENESS_THRESHOLD: {Colors.GREEN}{config.LIVENESS_THRESHOLD}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Configuration PCA:{Colors.ENDC}")
        print(f"  N_COMPONENTS: {Colors.GREEN}{config.N_COMPONENTS}{Colors.ENDC}")
        
        print(f"\n{Colors.YELLOW}ℹ  Pour modifier: éditez fr_core/config.py{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Erreur: {e}{Colors.ENDC}")
    
    input("\nAppuyez sur ENTER pour continuer...")


def delete_model():
    """Supprime un modèle."""
    print_header()
    print(f"{Colors.BOLD}=== SUPPRESSION MODÈLE ==={Colors.ENDC}\n")
    
    models = show_models()
    
    if not models:
        print(f"\n{Colors.YELLOW}Aucun modèle à supprimer.{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    username = input(f"\n{Colors.CYAN}Nom d'utilisateur à supprimer:{Colors.ENDC} ").strip()
    if not username:
        print(f"{Colors.RED}✗ Nom requis{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    model_path = SCRIPT_DIR / "models" / f"{username}.npz"
    if not model_path.exists():
        print(f"{Colors.RED}✗ Modèle non trouvé pour {username}{Colors.ENDC}")
        input("\nAppuyez sur ENTER pour continuer...")
        return
    
    confirm = input(f"{Colors.YELLOW}⚠ Confirmer suppression de {username}? (o/N):{Colors.ENDC} ").strip().lower()
    if confirm == 'o':
        try:
            model_path.unlink()
            print(f"{Colors.GREEN}✓ Modèle {username} supprimé{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}✗ Erreur: {e}{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}Suppression annulée{Colors.ENDC}")
    
    input("\nAppuyez sur ENTER pour continuer...")


def main_menu():
    """Affiche le menu principal."""
    while True:
        print_header()
        
        print(f"{Colors.BOLD}Menu Principal:{Colors.ENDC}\n")
        print(f"  {Colors.GREEN}[1]{Colors.ENDC} 📝 Enrollment - Enregistrer un utilisateur")
        print(f"  {Colors.GREEN}[2]{Colors.ENDC} ✅ Verification - Vérifier l'identité")
        print(f"  {Colors.GREEN}[3]{Colors.ENDC} 👥 Lister les modèles")
        print(f"  {Colors.GREEN}[4]{Colors.ENDC} 🗑️  Supprimer un modèle")
        print(f"  {Colors.GREEN}[5]{Colors.ENDC} ⚙️  Paramètres")
        print(f"  {Colors.GREEN}[0]{Colors.ENDC} 🚪 Quitter\n")
        
        choice = input(f"{Colors.CYAN}Votre choix:{Colors.ENDC} ").strip()
        
        if choice == '1':
            enrollment()
        elif choice == '2':
            verification()
        elif choice == '3':
            print_header()
            print(f"{Colors.BOLD}=== MODÈLES ENREGISTRÉS ==={Colors.ENDC}\n")
            show_models()
            input("\nAppuyez sur ENTER pour continuer...")
        elif choice == '4':
            delete_model()
        elif choice == '5':
            settings()
        elif choice == '0':
            print(f"\n{Colors.GREEN}Au revoir! 👋{Colors.ENDC}\n")
            sys.exit(0)
        else:
            print(f"{Colors.RED}✗ Choix invalide{Colors.ENDC}")
            input("\nAppuyez sur ENTER pour continuer...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Au revoir! 👋{Colors.ENDC}\n")
        sys.exit(0)
