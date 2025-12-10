#!/usr/bin/env python3
"""
Test du systeme avec DDTW (Derivative DTW).

Compare les performances:
1. DTW classique (statiques uniquement)
2. DDTW velocity (statiques + derivees 1)
3. DDTW acceleration (statiques + derivees 1+2)
"""

from fr_core.verification_dtw import verify
from fr_core import config
import numpy as np


def test_ddtw_methods(model_path='models/jeanphi.npz'):
    """Test les 3 methodes DDTW."""
    
    print("\n" + "="*70)
    print("TEST DDTW - COMPARAISON DES METHODES")
    print("="*70)
    print(f"\nModele: {model_path}")
    print(f"Utilisateur devant la camera: jeanphi")
    print(f"\nAppuyez sur ENTER pour commencer...")
    input()
    
    methods = [
        ('none', 'DTW Classique (statiques)'),
        ('velocity', 'DDTW Velocity (statiques + derivees 1)'),
        ('acceleration', 'DDTW Acceleration (statiques + derivees 1+2)')
    ]
    
    results = {}
    
    for method, description in methods:
        print(f"\n{'='*70}")
        print(f"METHODE: {description}")
        print(f"{'='*70}")
        
        # Configurer la methode
        original_method = config.DDTW_METHOD
        original_use = config.USE_DDTW
        
        if method == 'none':
            config.USE_DDTW = False
        else:
            config.USE_DDTW = True
            config.DDTW_METHOD = method
        
        print(f"\nConfiguration:")
        print(f"  USE_DDTW: {config.USE_DDTW}")
        print(f"  DDTW_METHOD: {config.DDTW_METHOD}")
        print(f"  DDTW_NORMALIZE: {config.DDTW_NORMALIZE}")
        
        print(f"\nVerification en cours (10 frames)...")
        
        # Verification
        is_verified, distance = verify(
            model_path=model_path,
            video_source=0,
            num_frames=10
        )
        
        results[method] = {
            'verified': is_verified,
            'distance': distance,
            'description': description
        }
        
        print(f"\nResultat:")
        print(f"  Verifie: {'✓ OUI' if is_verified else '✗ NON'}")
        print(f"  Distance: {distance:.2f}")
        print(f"  Seuil: {config.DTW_THRESHOLD:.2f}")
        
        # Restaurer config
        config.DDTW_METHOD = original_method
        config.USE_DDTW = original_use
    
    # Analyse comparative
    print(f"\n{'='*70}")
    print("ANALYSE COMPARATIVE")
    print(f"{'='*70}")
    
    print(f"\n{'Methode':<40} {'Distance':<12} {'Status'}")
    print("-" * 70)
    for method, description in methods:
        r = results[method]
        status = '✓ VERIFIE' if r['verified'] else '✗ REJETE'
        print(f"{description:<40} {r['distance']:>8.2f}     {status}")
    
    # Recommandation
    print(f"\n{'='*70}")
    print("RECOMMANDATION")
    print(f"{'='*70}")
    
    # Comparer les distances
    dist_static = results['none']['distance']
    dist_vel = results['velocity']['distance']
    dist_acc = results['acceleration']['distance']
    
    print(f"\nDistances observees:")
    print(f"  Statique:     {dist_static:.2f}")
    print(f"  Velocity:     {dist_vel:.2f} (ratio: {dist_vel/dist_static:.2f}x)")
    print(f"  Acceleration: {dist_acc:.2f} (ratio: {dist_acc/dist_static:.2f}x)")
    
    if dist_vel < config.DTW_THRESHOLD and dist_static > config.DTW_THRESHOLD:
        print(f"\n✓ DDTW Velocity AMELIORE la verification!")
        print(f"  Statique rejete ({dist_static:.2f} > {config.DTW_THRESHOLD:.2f})")
        print(f"  Velocity accepte ({dist_vel:.2f} < {config.DTW_THRESHOLD:.2f})")
        print(f"\n  Recommandation: ACTIVER DDTW velocity")
    
    elif dist_vel > config.DTW_THRESHOLD and dist_static < config.DTW_THRESHOLD:
        print(f"\n⚠ DDTW Velocity DEGRADE la verification")
        print(f"  Statique accepte ({dist_static:.2f} < {config.DTW_THRESHOLD:.2f})")
        print(f"  Velocity rejete ({dist_vel:.2f} > {config.DTW_THRESHOLD:.2f})")
        print(f"\n  Recommandation: DESACTIVER DDTW (garder statique)")
    
    else:
        print(f"\n→ DDTW ne change pas la decision de verification")
        print(f"  Mais peut ameliorer la separation genuine/impostor")
        print(f"\n  Recommandation: TESTER sur scenarios imposteurs")
    
    print(f"\n{'='*70}\n")
    
    return results


def test_ddtw_impostor():
    """
    Test DDTW sur scenarios imposteurs.
    Necessite jeanphi et lora enrolles.
    """
    print("\n" + "="*70)
    print("TEST DDTW - SCENARIOS IMPOSTEURS")
    print("="*70)
    
    scenarios = [
        ('jeanphi genuine', 'models/jeanphi.npz', 'jeanphi'),
        ('lora genuine', 'models/lora.npz', 'lora'),
        ('lora -> jeanphi impostor', 'models/jeanphi.npz', 'lora'),
        ('jeanphi -> lora impostor', 'models/lora.npz', 'jeanphi'),
    ]
    
    methods = ['none', 'velocity']
    results = {m: {} for m in methods}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"METHODE: {method.upper()}")
        print(f"{'='*70}")
        
        # Configurer
        if method == 'none':
            config.USE_DDTW = False
        else:
            config.USE_DDTW = True
            config.DDTW_METHOD = method
        
        for scenario_name, model_path, person in scenarios:
            print(f"\n{scenario_name}")
            print(f"  Modele: {model_path}")
            print(f"  Personne: {person}")
            print(f"  Appuyez sur ENTER quand pret...")
            input()
            
            is_verified, distance = verify(
                model_path=model_path,
                video_source=0,
                num_frames=10
            )
            
            results[method][scenario_name] = distance
            print(f"  Distance: {distance:.2f} ({'✓ VERIFIE' if is_verified else '✗ REJETE'})")
    
    # Analyse
    print(f"\n{'='*70}")
    print("ANALYSE - SEPARATION GENUINE/IMPOSTOR")
    print(f"{'='*70}")
    
    for method in methods:
        print(f"\n{method.upper()}:")
        
        jp_genuine = results[method].get('jeanphi genuine', 0)
        lora_genuine = results[method].get('lora genuine', 0)
        lora_to_jp = results[method].get('lora -> jeanphi impostor', 0)
        jp_to_lora = results[method].get('jeanphi -> lora impostor', 0)
        
        sep_jp = lora_to_jp - jp_genuine
        sep_lora = jp_to_lora - lora_genuine
        
        print(f"  jeanphi genuine:     {jp_genuine:.2f}")
        print(f"  lora -> jeanphi imp: {lora_to_jp:.2f}")
        print(f"  Separation jeanphi:  {sep_jp:.2f} ({'POSITIF' if sep_jp > 0 else 'NEGATIF'})")
        
        print(f"\n  lora genuine:        {lora_genuine:.2f}")
        print(f"  jeanphi -> lora imp: {jp_to_lora:.2f}")
        print(f"  Separation lora:     {sep_lora:.2f} ({'POSITIF' if sep_lora > 0 else 'NEGATIF'})")
    
    # Comparaison
    sep_static_jp = results['none'].get('lora -> jeanphi impostor', 0) - results['none'].get('jeanphi genuine', 0)
    sep_ddtw_jp = results['velocity'].get('lora -> jeanphi impostor', 0) - results['velocity'].get('jeanphi genuine', 0)
    
    improvement = sep_ddtw_jp - sep_static_jp
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"\nAmelioration de separation (jeanphi):")
    print(f"  Statique: {sep_static_jp:.2f}")
    print(f"  DDTW:     {sep_ddtw_jp:.2f}")
    print(f"  Gain:     {improvement:.2f} ({(improvement/sep_static_jp*100):.1f}%)")
    
    if improvement > 0:
        print(f"\n✅ DDTW AMELIORE la discrimination!")
        print(f"   Recommandation: ACTIVER DDTW velocity")
    else:
        print(f"\n⚠ DDTW n'ameliore pas la discrimination")
        print(f"   Recommandation: Garder DTW classique")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'impostor':
        test_ddtw_impostor()
    else:
        test_ddtw_methods()
