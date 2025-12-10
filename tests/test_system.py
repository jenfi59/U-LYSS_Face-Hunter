#!/usr/bin/env python3
"""
Test complet du systeme FR_VERS_JP 2.0 avec tous les composants:
- Tier 1: Landmarks + Calibration seuil
- Tier 2 #6: DDTW (dynamiques temporelles)
- Tier 2 #7: Liveness Detection (anti-spoofing)
"""

from fr_core.verification_dtw import verify_dtw
from fr_core import config
import time


def test_full_system(username='jeanphi'):
    """Test du systeme complet avec liveness + DDTW + DTW."""
    
    print("\n" + "="*70)
    print("TEST COMPLET FR_VERS_JP 2.0")
    print("="*70)
    
    model_path = f'models/{username}.npz'
    
    print(f"\nUtilisateur: {username}")
    print(f"Modele: {model_path}")
    
    print("\n" + "="*70)
    print("CONFIGURATION ACTUELLE")
    print("="*70)
    
    # Liveness
    print(f"\n📍 LIVENESS DETECTION:")
    print(f"  USE_LIVENESS: {config.USE_LIVENESS}")
    print(f"  Methods: {config.LIVENESS_METHODS}")
    print(f"  Confidence threshold: {config.LIVENESS_CONFIDENCE_THRESHOLD:.0%}")
    
    if 'blink' in config.LIVENESS_METHODS:
        print(f"  Blink: {config.LIVENESS_BLINK_MIN} clignement(s) en {config.LIVENESS_BLINK_TIME}s")
    if 'motion' in config.LIVENESS_METHODS:
        print(f"  Motion: {config.LIVENESS_MOTION_MIN} pixels sur {config.LIVENESS_MOTION_FRAMES} frames")
    
    # DDTW
    print(f"\n📍 DERIVATIVE DTW:")
    print(f"  USE_DDTW: {config.USE_DDTW}")
    print(f"  Method: {config.DDTW_METHOD}")
    print(f"  Normalize: {config.DDTW_NORMALIZE}")
    
    # DTW
    print(f"\n📍 DTW VERIFICATION:")
    print(f"  Threshold: {config.DTW_THRESHOLD:.2f}")
    print(f"  Window: 10 (Sakoe-Chiba)")
    
    # Landmarks
    print(f"\n📍 FEATURES:")
    print(f"  68 landmarks (geometrie)")
    print(f"  136 features → PCA 45 composantes")
    
    print("\n" + "="*70)
    print("EXECUTION DU PIPELINE COMPLET")
    print("="*70)
    
    print("\nÉtapes:")
    print("  1️⃣  Liveness Detection (clignez et bougez)")
    print("  2️⃣  Landmark Extraction (10 frames)")
    print("  3️⃣  DDTW Augmentation (velocites)")
    print("  4️⃣  DTW Distance Calculation")
    print("  5️⃣  Threshold Decision")
    
    print(f"\nAppuyez sur ENTER pour commencer...")
    input()
    
    start_time = time.time()
    
    print("\n🔄 Verification en cours...")
    print("-" * 70)
    
    # Verification complete
    is_verified, distance = verify_dtw(
        model_path=model_path,
        video_source=0,
        num_frames=10,
        check_liveness=True  # Active liveness
    )
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print("\n" + "="*70)
    print("RÉSULTAT FINAL")
    print("="*70)
    
    if is_verified:
        print("\n✅ VÉRIFIÉ - Accès autorisé")
        print(f"   Distance DTW: {distance:.2f} < {config.DTW_THRESHOLD:.2f}")
    else:
        if distance == float('inf'):
            print("\n❌ REJETÉ - Liveness check échoué (spoof suspect)")
            print(f"   Raison: Anti-spoofing détecté")
        else:
            print("\n❌ REJETÉ - Distance trop élevée")
            print(f"   Distance DTW: {distance:.2f} >= {config.DTW_THRESHOLD:.2f}")
    
    print(f"\n⏱️  Temps total: {elapsed:.2f}s")
    
    print("\n" + "="*70)
    print("DÉTAILS TECHNIQUES")
    print("="*70)
    
    print(f"\n  Verified: {is_verified}")
    print(f"  Distance: {distance:.2f}")
    print(f"  Threshold: {config.DTW_THRESHOLD:.2f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Liveness: {'✓ Passed' if distance != float('inf') else '✗ Failed'}")
    print(f"  DDTW: {'✓ Active' if config.USE_DDTW else '✗ Inactive'} ({config.DDTW_METHOD})")
    
    print("\n" + "="*70 + "\n")
    
    return is_verified, distance


def test_with_without_liveness(username='jeanphi'):
    """Compare avec et sans liveness detection."""
    
    print("\n" + "="*70)
    print("COMPARAISON AVEC/SANS LIVENESS")
    print("="*70)
    
    model_path = f'models/{username}.npz'
    
    # Test 1: Sans liveness
    print("\n" + "="*70)
    print("TEST 1: SANS LIVENESS DETECTION")
    print("="*70)
    print(f"\nAppuyez sur ENTER...")
    input()
    
    start1 = time.time()
    is_verified1, distance1 = verify_dtw(
        model_path=model_path,
        video_source=0,
        num_frames=10,
        check_liveness=False  # Désactivé
    )
    time1 = time.time() - start1
    
    print(f"\nRésultat: {'✓ VÉRIFIÉ' if is_verified1 else '✗ REJETÉ'}")
    print(f"Distance: {distance1:.2f}")
    print(f"Temps: {time1:.2f}s")
    
    # Test 2: Avec liveness
    print("\n" + "="*70)
    print("TEST 2: AVEC LIVENESS DETECTION")
    print("="*70)
    print(f"\nClignez et bougez...")
    print(f"Appuyez sur ENTER...")
    input()
    
    start2 = time.time()
    is_verified2, distance2 = verify_dtw(
        model_path=model_path,
        video_source=0,
        num_frames=10,
        check_liveness=True  # Activé
    )
    time2 = time.time() - start2
    
    print(f"\nRésultat: {'✓ VÉRIFIÉ' if is_verified2 else '✗ REJETÉ'}")
    print(f"Distance: {distance2:.2f}")
    print(f"Temps: {time2:.2f}s")
    
    # Comparaison
    print("\n" + "="*70)
    print("ANALYSE")
    print("="*70)
    
    print(f"\nSans liveness:")
    print(f"  Vérifié: {is_verified1}")
    print(f"  Distance: {distance1:.2f}")
    print(f"  Temps: {time1:.2f}s")
    
    print(f"\nAvec liveness:")
    print(f"  Vérifié: {is_verified2}")
    print(f"  Distance: {distance2:.2f}")
    print(f"  Temps: {time2:.2f}s")
    
    overhead = time2 - time1
    print(f"\nOverhead liveness: +{overhead:.2f}s ({(overhead/time1*100):.0f}%)")
    
    if distance2 == float('inf'):
        print(f"\n⚠️  Liveness a rejeté (spoof suspect)")
    elif is_verified1 == is_verified2:
        print(f"\n✓ Même décision de vérification")
    else:
        print(f"\n⚠️  Décisions différentes!")
    
    print("\n" + "="*70 + "\n")


def test_spoof_attack_simulation():
    """
    Simulation d'une attaque spoof (à faire manuellement).
    Instructions pour tester avec une photo imprimée ou écran.
    """
    print("\n" + "="*70)
    print("TEST D'ATTAQUE SPOOF (SIMULATION)")
    print("="*70)
    
    print("\nPour tester l'anti-spoofing:")
    print("\n1. PHOTO ATTACK:")
    print("   - Prenez une photo de votre visage")
    print("   - Imprimez-la ou affichez sur écran")
    print("   - Présentez à la caméra")
    print("   - Le système devrait REJETER (pas de clignement)")
    
    print("\n2. VIDEO REPLAY:")
    print("   - Enregistrez une vidéo de vous")
    print("   - Rejouez la vidéo devant la caméra")
    print("   - Le système peut détecter via texture/motion patterns")
    
    print("\n3. GENUINE (contrôle):")
    print("   - Vous-même devant la caméra")
    print("   - Clignez et bougez naturellement")
    print("   - Le système devrait ACCEPTER")
    
    print("\nChoisissez le test à effectuer:")
    print("  1. Photo attack")
    print("  2. Video replay")
    print("  3. Genuine (contrôle)")
    print("\nChoix (1-3): ", end='')
    
    choice = input().strip()
    
    if choice in ['1', '2', '3']:
        attack_type = ['photo', 'video', 'genuine'][int(choice)-1]
        print(f"\n🎯 Test: {attack_type}")
        print(f"\nPréparez votre {attack_type}...")
        print(f"Appuyez sur ENTER pour commencer...")
        input()
        
        is_verified, distance = verify_dtw(
            model_path='models/jeanphi.npz',
            video_source=0,
            num_frames=10,
            check_liveness=True
        )
        
        print("\n" + "="*70)
        print(f"RÉSULTAT - TEST {attack_type.upper()}")
        print("="*70)
        
        if attack_type == 'genuine':
            expected = "✓ VÉRIFIÉ"
            success = is_verified
        else:
            expected = "✗ REJETÉ (spoof)"
            success = not is_verified
        
        print(f"\nAttendu: {expected}")
        print(f"Obtenu:  {'✓ VÉRIFIÉ' if is_verified else '✗ REJETÉ'}")
        print(f"Distance: {distance:.2f}")
        
        if success:
            print(f"\n✅ TEST RÉUSSI - Anti-spoofing fonctionne!")
        else:
            print(f"\n❌ TEST ÉCHOUÉ - Vérifier configuration liveness")
        
        print("\n" + "="*70 + "\n")
    else:
        print("Choix invalide")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            test_with_without_liveness()
        elif sys.argv[1] == 'spoof':
            test_spoof_attack_simulation()
        else:
            test_full_system(sys.argv[1])
    else:
        test_full_system()
