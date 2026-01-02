#!/usr/bin/env python3.13
"""
Script pour tester vérification avec sessions pré-enregistrées
Usage: python3.13 scripts/test_verify_session.py <user_id> <session_name> --mode [temporal|spatial|spatiotemporal]
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from src.fr_core import VerificationDTW, get_config
import argparse


def test_session(user_id: str, session_name: str, mode: str = None):
    """Tester vérification avec une session pré-enregistrée"""
    
    # Charger config et ajuster mode si spécifié
    config = get_config()
    if mode:
        config.set_matching_mode(mode)
    
    print("\n" + "="*62)
    print(f"  TEST VERIFICATION - {session_name.upper()}")
    print("="*62)
    print(f"\nUser: {user_id}")
    print(f"Mode: {config.matching_mode}")
    print(f"Session: {session_name}")
    
    # Charger enrollment
    print("\n[INFO] Chargement enrollment...")
    verifier = VerificationDTW()
    enrolled_lm, enrolled_poses = verifier.load_enrollment(user_id, Path('models/users'))
    
    if enrolled_lm is None:
        print(f"[ERROR] Modele enrollment non trouve: models/users/{user_id}.npz")
        return False
    
    print(f"   [OK] Landmarks: {enrolled_lm.shape}")
    print(f"   [OK] Poses: {enrolled_poses.shape if enrolled_poses is not None else 'None'}")
    
    # Charger session vérification
    session_path = Path('datasets') / user_id / 'verify_sessions' / f"{session_name}.npz"
    
    if not session_path.exists():
        print(f"\n[ERROR] Session non trouvee: {session_path}")
        return False
    
    print(f"\n[INFO] Chargement session...")
    session_data = np.load(session_path, allow_pickle=True)
    probe_lm = session_data['landmarks']
    probe_poses = session_data['poses']
    session_meta = session_data['metadata'].item()
    
    print(f"   [OK] Landmarks: {probe_lm.shape}")
    print(f"   [OK] Poses: {probe_poses.shape}")
    print(f"   [INFO] Frames: {session_meta.get('indices', 'N/A')}")
    
    # Statistiques poses
    print(f"\n[POSES] Probe (session):")
    print(f"   Yaw:   mean={probe_poses[:,0].mean():6.1f}deg  std={probe_poses[:,0].std():5.2f}deg  "
          f"range=[{probe_poses[:,0].min():6.1f}deg, {probe_poses[:,0].max():6.1f}deg]")
    print(f"   Pitch: mean={probe_poses[:,1].mean():6.1f}deg  std={probe_poses[:,1].std():5.2f}deg  "
          f"range=[{probe_poses[:,1].min():6.1f}deg, {probe_poses[:,1].max():6.1f}deg]")
    
    print(f"\n[POSES] Gallery (enrollment):")
    if enrolled_poses is not None:
        print(f"   Yaw:   mean={enrolled_poses[:,0].mean():6.1f}deg  std={enrolled_poses[:,0].std():5.2f}deg  "
              f"range=[{enrolled_poses[:,0].min():6.1f}deg, {enrolled_poses[:,0].max():6.1f}deg]")
        print(f"   Pitch: mean={enrolled_poses[:,1].mean():6.1f}deg  std={enrolled_poses[:,1].std():5.2f}deg  "
              f"range=[{enrolled_poses[:,1].min():6.1f}deg, {enrolled_poses[:,1].max():6.1f}deg]")
    
    # Vérification
    print(f"\n[VERIFY] Mode {config.matching_mode}...")
    is_match, distance, details = verifier.verify_auto(
        probe_lm, probe_poses, enrolled_lm, enrolled_poses
    )
    
    # Déterminer seuil
    if config.matching_mode == 'temporal':
        threshold = config.dtw_threshold
    elif config.matching_mode == 'spatial':
        threshold = config.pose_threshold
    elif config.matching_mode == 'spatiotemporal':
        threshold = config.spatiotemporal_threshold
    else:
        threshold = float('inf')
    
    # Afficher résultats
    print("\n" + "="*60)
    print("  RÉSULTATS")
    print("="*60)
    print(f"\n{'Distance:':<20s} {distance:.4f}")
    print(f"{'Seuil:':<20s} {threshold:.4f}")
    print(f"{'Match:':<20s} {'OUI' if is_match else 'NON'}")
    print(f"{'Mode:':<20s} {details.get('mode', 'N/A')}")
    
    if 'coverage' in details:
        print(f"{'Coverage:':<20s} {details['coverage']:.1%}")
    
    if 'distance_temporal' in details:
        print(f"{'Distance temporale:':<20s} {details['distance_temporal']:.4f}")
    
    if 'distance_spatial' in details:
        print(f"{'Distance spatiale:':<20s} {details['distance_spatial']:.4f}")
    
    # Verdict
    print("\n" + "="*60)
    if is_match:
        print("  [OK] AUTHENTIFICATION REUSSIE")
    else:
        margin = distance - threshold
        print(f"  [FAIL] AUTHENTIFICATION ECHOUEE (marge: {margin:+.4f})")
    print("="*60)
    
    return is_match


def test_all_sessions(user_id: str, mode: str = None):
    """Tester toutes les sessions pour un utilisateur"""
    sessions_dir = Path('datasets') / user_id / 'verify_sessions'
    
    if not sessions_dir.exists():
        print(f"[ERROR] Dossier sessions non trouve: {sessions_dir}")
        return
    
    sessions = sorted(sessions_dir.glob('*.npz'))
    
    if not sessions:
        print(f"[ERROR] Aucune session trouvee dans: {sessions_dir}")
        return
    
    print(f"\n" + "="*62)
    print(f"  TEST TOUTES SESSIONS ({len(sessions)} sessions)")
    print("="*62)
    
    results = []
    
    for session_path in sessions:
        session_name = session_path.stem
        success = test_session(user_id, session_name, mode)
        results.append((session_name, success))
        print("\n" + "-"*60 + "\n")
    
    # Résumé
    print("\n" + "="*62)
    print("  RESUME GLOBAL")
    print("="*62)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    success_rate = success_count / total_count if total_count > 0 else 0
    
    print(f"\n[STATS] Authentifications reussies: {success_count}/{total_count} ({success_rate:.1%})")
    print(f"\n{'Session':<25s} {'Resultat'}")
    print("-"*60)
    for session_name, success in results:
        status = "[OK] MATCH" if success else "[FAIL] NO MATCH"
        print(f"{session_name:<25s} {status}")
    
    print(f"\n[SUMMARY] Taux de succes: {success_rate:.1%}")
    
    if success_rate >= 0.9:
        print("   [OK] Excellent (>=90%)")
    elif success_rate >= 0.7:
        print("   [WARN] Bon (>=70%)")
    else:
        print("   [FAIL] Insuffisant (<70%)") 


def main():
    parser = argparse.ArgumentParser(description="Tester vérification avec sessions")
    parser.add_argument("user_id", type=str, help="ID utilisateur")
    parser.add_argument("session_name", type=str, nargs='?', 
                        help="Nom session (ex: session_01_frontal) ou 'all' pour toutes")
    parser.add_argument("--mode", type=str, choices=['temporal', 'spatial', 'spatiotemporal'],
                        help="Mode de vérification (défaut: config actuelle)")
    
    args = parser.parse_args()
    
    if args.session_name == 'all' or args.session_name is None:
        test_all_sessions(args.user_id, args.mode)
    else:
        test_session(args.user_id, args.session_name, args.mode)


if __name__ == '__main__':
    main()
