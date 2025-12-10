#!/usr/bin/env python3
"""
Guided Enrollment avec LANDMARKS (géométrie faciale)
====================================================
Retour aux sources : 68 landmarks clés au lieu de Gabor+LBP.

Les landmarks capturent la GÉOMÉTRIE UNIQUE du visage :
- Yeux, nez, bouche, mâchoire, sourcils
- Beaucoup plus discriminant que la texture
- Plus robuste aux variations d'éclairage

Version : 3.0 - Landmarks only (refactored)
"""

import sys
import numpy as np
from fr_core.guided_enrollment import GuidedEnrollment
from fr_core.features import FaceLandmarkDetector
from fr_core.enrollment import save_model
from fr_core.landmark_utils import (
    LANDMARK_INDICES,
    N_LANDMARK_FEATURES,
    extract_landmarks_from_frame
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import logging


def enroll_with_landmarks(username: str):
    """
    Enrôle un utilisateur en utilisant les LANDMARKS (68 points).
    
    Au lieu de Gabor+LBP (features de texture), on utilise la géométrie
    faciale directement. Chaque frame donne 68 points × 2 coords = 136 features.
    
    Args:
        username: Nom de l'utilisateur
    """
    
    print("\n" + "="*70)
    print("ENROLLMENT AVEC LANDMARKS (GÉOMÉTRIE FACIALE)")
    print("="*70)
    print(f"\nUtilisateur: {username}")
    print("\nAvantages des landmarks :")
    print("  ✓ Capturent la géométrie UNIQUE du visage")
    print("  ✓ Robustes aux variations d'éclairage")
    print("  ✓ Plus discriminants que la texture (Gabor/LBP)")
    print(f"  ✓ {len(LANDMARK_INDICES)} points × 2 coords = {len(LANDMARK_INDICES)*2} features")
    print("\n" + "="*70 + "\n")
    
    # Étape 1: Guided enrollment (capture 45 frames avec variabilité)
    enrollment_system = GuidedEnrollment()
    frames = enrollment_system.enroll()
    
    if frames is None or len(frames) == 0:
        raise ValueError("Enrollment échoué : aucune frame capturée")
    
    print(f"\n✓ Captured {len(frames)} frames successfully\n")
    
    # Étape 2: Extraction des landmarks pour chaque frame
    print("📊 Extraction des landmarks...")
    detector = FaceLandmarkDetector()
    
    # Import pour avoir accès aux frames ORIGINALES (pas pré-traitées)
    import cv2
    cap = cv2.VideoCapture(0)
    
    print("⚠️  IMPORTANT: Restez devant la caméra pour extraire les landmarks!")
    print("   Appuyez sur ESPACE pour capturer chaque frame")
    print("   Appuyez sur 'q' pour terminer quand vous avez assez de frames\n")
    
    landmark_features = []
    frame_count = 0
    max_frames = min(len(frames), 45)
    
    try:
        while frame_count < max_frames:
            ret, frame_full = cap.read()
            if not ret:
                break
            
            frame_full = cv2.flip(frame_full, 1)
            display = frame_full.copy()
            
            # Détecte landmarks sur frame complète (utilise fonction centralisée)
            features = extract_landmarks_from_frame(frame_full, detector)
            
            if features is not None:
                cv2.putText(display, "Landmarks OK - Appuyez ESPACE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Frames: {frame_count}/{max_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Pas de landmarks detectes", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Landmark Extraction", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and features is not None:
                landmark_features.append(features)
                frame_count += 1
                print(f"  ✓ Frame {frame_count}/{max_frames} capturée")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    
    if len(landmark_features) < 5:
        raise ValueError(f"Pas assez de landmarks: seulement {len(landmark_features)} frames valides (minimum 5)")
    
    X_landmarks = np.array(landmark_features)
    n_frames, n_features = X_landmarks.shape
    
    print(f"  - Landmark matrix: {X_landmarks.shape}")
    print(f"  - {len(LANDMARK_INDICES)} landmarks × 2 coords = {n_features} features")
    print(f"  - {n_frames} frames with valid landmarks")
    
    # Étape 3: Normalisation avec RobustScaler (robuste aux outliers)
    print(f"  - Applying RobustScaler normalization...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_landmarks)
    
    # Étape 4: PCA pour réduction dimensionnelle
    n_components = min(45, n_frames)
    print(f"  - Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    Variance explained: {variance_explained:.1f}%")
    
    # Étape 5: Sauvegarde du modèle
    model_path = f"models/{username}.npz"
    
    # Paramètres dummy pour compatibilité (pas utilisés en mode DTW landmarks)
    pose_mean = np.array([0.0, 0.0, 0.0])
    R_ref = np.eye(3)
    t_ref = np.zeros((3, 1))
    
    # Save (DTW mode: gmm=None, dtw_template=X_pca)
    save_model(
        model_path,
        pca,           # pca_model
        None,          # gmm_model (DTW mode)
        scaler,        # scaler
        pose_mean,     # pose_mean
        R_ref,         # R_ref
        t_ref,         # t_ref
        X_pca          # dtw_template (séquence PCA pour DTW)
    )
    
    print(f"\n✓ Enrollment complete! Model saved to: {model_path}")
    print(f"  - Template size: {X_pca.shape}")
    print(f"  - Method: 68 landmarks (geometry-based)")
    print(f"  - Features: {n_features} → PCA {n_components} ({variance_explained:.1f}% variance)")
    
    # Validation immédiate
    print("\n" + "="*70)
    print("VALIDATION IMMÉDIATE (ne bougez pas!)")
    print("="*70)
    print(f"\nTest de validation avec {username} toujours devant la caméra...")
    input("Appuyez sur ENTER pour valider l'enrollment...")
    
    from fr_core.verification_dtw import verify
    is_verified, distance = verify(model_path=model_path, video_source=0)
    
    print("\n" + "="*70)
    print("RÉSULTAT VALIDATION GENUINE:")
    print(f"  Utilisateur: {username}")
    print(f"  Vérifié: {'✓ OUI' if is_verified else '✗ NON'}")
    print(f"  Distance DTW: {distance:.2f}")
    print("="*70)
    
    if is_verified:
        print(f"\n✅ Validation réussie pour {username}!\n")
    else:
        print(f"\n⚠️  Attention: validation échouée (distance {distance:.2f} > threshold)\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll_landmarks.py <username>")
        print("\nCette version utilise 68 LANDMARKS au lieu de Gabor+LBP")
        print("Les landmarks capturent la géométrie unique du visage.")
        sys.exit(1)
    
    username = sys.argv[1]
    
    try:
        enroll_with_landmarks(username)
    except Exception as e:
        print(f"\n✗ Erreur: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
