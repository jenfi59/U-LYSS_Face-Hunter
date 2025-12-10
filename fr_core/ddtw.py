"""
Derivative DTW (DDTW) - Capture des dynamiques temporelles
===========================================================

Extension du DTW classique pour inclure les derivees temporelles des landmarks.
Ceci capture les MOUVEMENTS faciaux caracteristiques en plus de la geometrie statique.

Avantages:
- Discrimination supplementaire (mouvements uniques par personne)
- Robustesse aux poses statiques similaires
- Detection de patterns dynamiques (sourire, clignements, etc.)

Methode:
- Features statiques: landmarks(t)
- Features dynamiques: Δlandmarks(t) = landmarks(t) - landmarks(t-1)
- Concatenation: [statiques, dynamiques] pour DTW
"""

import numpy as np
from typing import Tuple, Optional
import logging


def compute_derivatives(
    sequence: np.ndarray,
    order: int = 1,
    delta_t: int = 1
) -> np.ndarray:
    """
    Calcule les derivees temporelles d'une sequence de features.
    
    Parameters
    ----------
    sequence : np.ndarray, shape (n_frames, n_features)
        Sequence de features (ex: landmarks)
    order : int
        Ordre de la derivee (1 = velocite, 2 = acceleration)
    delta_t : int
        Espacement temporel pour le calcul de la derivee
        
    Returns
    -------
    derivatives : np.ndarray, shape (n_frames, n_features)
        Derivees temporelles (premiere frame = 0)
    """
    if order == 1:
        # Premiere derivee (velocite): f'(t) = f(t) - f(t-delta_t)
        derivatives = np.zeros_like(sequence)
        derivatives[delta_t:] = sequence[delta_t:] - sequence[:-delta_t]
        return derivatives
    
    elif order == 2:
        # Deuxieme derivee (acceleration): f''(t) = f'(t) - f'(t-delta_t)
        first_deriv = compute_derivatives(sequence, order=1, delta_t=delta_t)
        derivatives = np.zeros_like(sequence)
        derivatives[delta_t:] = first_deriv[delta_t:] - first_deriv[:-delta_t]
        return derivatives
    
    else:
        raise ValueError(f"Order {order} not supported. Use 1 or 2.")


def compute_delta_features(
    sequence: np.ndarray,
    include_acceleration: bool = False,
    normalize: bool = True
) -> np.ndarray:
    """
    Calcule les delta-features (derivees + optionnel acceleration).
    
    Parameters
    ----------
    sequence : np.ndarray, shape (n_frames, n_features)
        Sequence originale
    include_acceleration : bool
        Inclure la deuxieme derivee (acceleration)
    normalize : bool
        Normaliser les derivees par leur std
        
    Returns
    -------
    augmented : np.ndarray, shape (n_frames, n_features_augmented)
        [statiques, velocites, (accelerations)]
    """
    n_frames, n_features = sequence.shape
    
    # Calcul des velocites (premiere derivee)
    velocities = compute_derivatives(sequence, order=1, delta_t=1)
    
    # Normalisation optionnelle
    if normalize:
        # Normaliser par std non-zero pour eviter division par zero
        std_vel = np.std(velocities[1:], axis=0)  # Ignorer premiere frame
        std_vel[std_vel == 0] = 1.0  # Eviter division par zero
        velocities = velocities / std_vel
    
    if include_acceleration:
        # Calcul des accelerations (deuxieme derivee)
        accelerations = compute_derivatives(sequence, order=2, delta_t=1)
        
        if normalize:
            std_acc = np.std(accelerations[2:], axis=0)  # Ignorer 2 premieres frames
            std_acc[std_acc == 0] = 1.0
            accelerations = accelerations / std_acc
        
        # Concatenation: [statiques, velocites, accelerations]
        augmented = np.concatenate([sequence, velocities, accelerations], axis=1)
        logging.info(f"DDTW: {n_features} statiques + {n_features} velocites + {n_features} accelerations = {augmented.shape[1]} features")
    else:
        # Concatenation: [statiques, velocites]
        augmented = np.concatenate([sequence, velocities], axis=1)
        logging.info(f"DDTW: {n_features} statiques + {n_features} velocites = {augmented.shape[1]} features")
    
    return augmented


def apply_ddtw_augmentation(
    sequence: np.ndarray,
    method: str = 'velocity',
    normalize: bool = True
) -> np.ndarray:
    """
    Applique l'augmentation DDTW a une sequence.
    
    Parameters
    ----------
    sequence : np.ndarray, shape (n_frames, n_features)
        Sequence originale (ex: landmarks PCA)
    method : str
        'none': Pas d'augmentation (DTW classique)
        'velocity': Ajoute velocites (derivee 1)
        'acceleration': Ajoute velocites + accelerations (derivees 1+2)
    normalize : bool
        Normaliser les derivees
        
    Returns
    -------
    augmented : np.ndarray
        Sequence augmentee
    """
    if method == 'none':
        return sequence
    
    elif method == 'velocity':
        return compute_delta_features(
            sequence,
            include_acceleration=False,
            normalize=normalize
        )
    
    elif method == 'acceleration':
        return compute_delta_features(
            sequence,
            include_acceleration=True,
            normalize=normalize
        )
    
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'none', 'velocity', or 'acceleration'.")


def compute_ddtw_distance(
    template: np.ndarray,
    query: np.ndarray,
    method: str = 'velocity',
    normalize: bool = True,
    window: Optional[int] = None,
    use_c: bool = True
) -> Tuple[float, float]:
    """
    Calcule la distance DTW avec augmentation derivative.
    
    Parameters
    ----------
    template : np.ndarray, shape (n_frames1, n_features)
        Sequence template (enrollment)
    query : np.ndarray, shape (n_frames2, n_features)
        Sequence query (verification)
    method : str
        Methode d'augmentation ('none', 'velocity', 'acceleration')
    normalize : bool
        Normaliser les derivees
    window : int, optional
        Fenetre Sakoe-Chiba pour DTW
    use_c : bool
        Utiliser implementation C (plus rapide)
        
    Returns
    -------
    dtw_distance : float
        Distance DTW sur features augmentees
    static_distance : float
        Distance DTW sur features statiques uniquement (pour comparaison)
    """
    from dtaidistance import dtw
    
    # Distance statique (DTW classique)
    static_distance = dtw.distance(
        template.flatten(),
        query.flatten(),
        window=window,
        use_c=use_c
    )
    
    if method == 'none':
        return static_distance, static_distance
    
    # Augmentation DDTW
    template_aug = apply_ddtw_augmentation(template, method=method, normalize=normalize)
    query_aug = apply_ddtw_augmentation(query, method=method, normalize=normalize)
    
    # Distance DDTW
    ddtw_distance = dtw.distance(
        template_aug.flatten(),
        query_aug.flatten(),
        window=window,
        use_c=use_c
    )
    
    logging.info(f"DTW: static={static_distance:.2f}, DDTW({method})={ddtw_distance:.2f}")
    
    return ddtw_distance, static_distance


# =============================================================================
# EXEMPLES D'UTILISATION
# =============================================================================

def demo_velocity_visualization():
    """Visualise les velocites pour comprendre DDTW."""
    import matplotlib.pyplot as plt
    
    # Simulation: 2 utilisateurs avec mouvements differents
    t = np.linspace(0, 2*np.pi, 50)
    
    # User 1: Mouvement sinusoidal lent
    user1 = np.column_stack([
        np.sin(t),
        np.cos(t)
    ])
    
    # User 2: Mouvement sinusoidal rapide
    user2 = np.column_stack([
        np.sin(2*t),
        np.cos(2*t)
    ])
    
    # Calcul des velocites
    vel1 = compute_derivatives(user1, order=1)
    vel2 = compute_derivatives(user2, order=1)
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Positions
    axes[0, 0].plot(user1[:, 0], label='User 1')
    axes[0, 0].plot(user2[:, 0], label='User 2')
    axes[0, 0].set_title('Positions (feature 0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(user1[:, 1], label='User 1')
    axes[0, 1].plot(user2[:, 1], label='User 2')
    axes[0, 1].set_title('Positions (feature 1)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Velocites
    axes[1, 0].plot(vel1[:, 0], label='User 1 velocity')
    axes[1, 0].plot(vel2[:, 0], label='User 2 velocity')
    axes[1, 0].set_title('Velocites (feature 0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(vel1[:, 1], label='User 1 velocity')
    axes[1, 1].plot(vel2[:, 1], label='User 2 velocity')
    axes[1, 1].set_title('Velocites (feature 1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ddtw_velocity_demo.png', dpi=150)
    print("✓ Visualisation sauvegardee: ddtw_velocity_demo.png")
    plt.close()


if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("DERIVATIVE DTW (DDTW) - DEMO")
    print("="*70)
    
    # Test avec donnees simulees
    print("\n1. Generation de sequences test...")
    np.random.seed(42)
    template = np.random.randn(45, 45)  # 45 frames, 45 features PCA
    query_same = template + np.random.randn(45, 45) * 0.1  # Meme utilisateur (bruit)
    query_diff = np.random.randn(45, 45)  # Utilisateur different
    
    print("   Template: (45, 45)")
    print("   Query same user: (45, 45)")
    print("   Query diff user: (45, 45)")
    
    # Test DDTW
    print("\n2. Calcul des distances...")
    
    # Sans DDTW
    from dtaidistance import dtw
    dist_same_static = dtw.distance(template.flatten(), query_same.flatten())
    dist_diff_static = dtw.distance(template.flatten(), query_diff.flatten())
    
    print(f"\n   DTW CLASSIQUE (statique):")
    print(f"   - Same user:  {dist_same_static:.2f}")
    print(f"   - Diff user:  {dist_diff_static:.2f}")
    print(f"   - Separation: {dist_diff_static - dist_same_static:.2f}")
    
    # Avec DDTW velocity
    dist_same_vel, _ = compute_ddtw_distance(template, query_same, method='velocity')
    dist_diff_vel, _ = compute_ddtw_distance(template, query_diff, method='velocity')
    
    print(f"\n   DDTW (velocity):")
    print(f"   - Same user:  {dist_same_vel:.2f}")
    print(f"   - Diff user:  {dist_diff_vel:.2f}")
    print(f"   - Separation: {dist_diff_vel - dist_same_vel:.2f}")
    
    # Avec DDTW acceleration
    dist_same_acc, _ = compute_ddtw_distance(template, query_same, method='acceleration')
    dist_diff_acc, _ = compute_ddtw_distance(template, query_diff, method='acceleration')
    
    print(f"\n   DDTW (acceleration):")
    print(f"   - Same user:  {dist_same_acc:.2f}")
    print(f"   - Diff user:  {dist_diff_acc:.2f}")
    print(f"   - Separation: {dist_diff_acc - dist_same_acc:.2f}")
    
    # Demo visualisation
    print("\n3. Generation de la visualisation...")
    demo_velocity_visualization()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nProchaine etape: Integration dans verification_dtw.py")
    print("="*70 + "\n")
