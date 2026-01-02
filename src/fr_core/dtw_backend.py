#!/usr/bin/env python3
"""
U-LYSS ARM64 - DTW Backend Wrapper
Utilise dtaidistance C si disponible, sinon fallback Python.
"""

import numpy as np
from typing import Union, Optional
import time

# Tenter d'importer dtaidistance avec backend C
try:
    from dtaidistance import dtw as dtw_c
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE_C = True
    print("[DTW] OK dtaidistance C backend available")
except ImportError:
    HAS_DTAIDISTANCE_C = False
    print("[DTW] WARNING dtaidistance not available, using Python fallback")


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: Optional[int] = None,
    use_c: bool = True
) -> float:
    """
    Calcule la distance DTW entre deux séquences.
    
    Args:
        seq1: Première séquence, shape (N, D) ou (N,)
        seq2: Deuxième séquence, shape (M, D) ou (M,)
        window: Fenêtre Sakoe-Chiba (None = pas de contrainte)
        use_c: Utiliser le backend C si disponible
        
    Returns:
        Distance DTW (float)
    """
    seq1 = np.asarray(seq1, dtype=np.float64)
    seq2 = np.asarray(seq2, dtype=np.float64)
    
    # Cas 1D
    if seq1.ndim == 1 and seq2.ndim == 1:
        if HAS_DTAIDISTANCE_C and use_c:
            return float(dtw_c.distance(seq1, seq2, window=window, use_c=True))
        else:
            return _dtw_1d_python(seq1, seq2, window)
    
    # Cas multidimensionnel (N, D)
    if seq1.ndim == 2 and seq2.ndim == 2:
        # IMPORTANT: Compute DTW dimension-by-dimension and sum (like original FR_VERS_JP_2.1)
        # This gives different results than dtw_ndim which uses euclidean distance per frame
        n_features = seq1.shape[1]
        total_distance = 0.0
        
        for i in range(n_features):
            seq1_1d = seq1[:, i].astype(np.float64)
            seq2_1d = seq2[:, i].astype(np.float64)
            
            if HAS_DTAIDISTANCE_C and use_c:
                dist = float(dtw_c.distance(seq1_1d, seq2_1d, window=window, use_c=True))
            else:
                dist = _dtw_1d_python(seq1_1d, seq2_1d, window)
            
            total_distance += dist
        
        return float(total_distance)
    
    raise ValueError(f"Unsupported shapes: seq1={seq1.shape}, seq2={seq2.shape}")


def _dtw_1d_python(seq1: np.ndarray, seq2: np.ndarray, window: Optional[int]) -> float:
    """DTW 1D en Python pur."""
    n, m = len(seq1), len(seq2)
    
    if window is None:
        window = max(n, m)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    return float(dtw_matrix[n, m])


def _dtw_ndim_python(seq1: np.ndarray, seq2: np.ndarray, window: Optional[int]) -> float:
    """DTW multidimensionnel en Python pur."""
    n, m = len(seq1), len(seq2)
    
    if window is None:
        window = max(n, m)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    
    return float(dtw_matrix[n, m])


def dtw_distance_normalized(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: Optional[int] = None
) -> float:
    """
    Distance DTW normalisée par la longueur moyenne du chemin.
    Utilise la méthode 'average' comme l'original FR_VERS_JP_2.1.
    """
    dist = dtw_distance(seq1, seq2, window=window)
    path_length = (len(seq1) + len(seq2)) / 2.0  # Average, not sum
    return dist / path_length


def benchmark_dtw(n_points: int = 100, n_dims: int = 45) -> dict:
    """
    Benchmark du backend DTW.
    
    Returns:
        Dict avec temps d'exécution et backend utilisé
    """
    seq1 = np.random.randn(n_points, n_dims).astype(np.float64)
    seq2 = np.random.randn(n_points, n_dims).astype(np.float64)
    
    # Test avec C backend
    start = time.perf_counter()
    dist_c = dtw_distance(seq1, seq2, window=10, use_c=True)
    time_c = (time.perf_counter() - start) * 1000
    
    # Test avec Python fallback
    start = time.perf_counter()
    dist_py = dtw_distance(seq1, seq2, window=10, use_c=False)
    time_py = (time.perf_counter() - start) * 1000
    
    return {
        'backend_c_available': HAS_DTAIDISTANCE_C,
        'n_points': n_points,
        'n_dims': n_dims,
        'time_c_ms': time_c,
        'time_python_ms': time_py,
        'speedup': time_py / time_c if time_c > 0 else 0,
        'distance_c': dist_c,
        'distance_python': dist_py,
        'distance_match': abs(dist_c - dist_py) < 1e-3
    }


if __name__ == "__main__":
    print("=" * 50)
    print("DTW Backend Benchmark")
    print("=" * 50)
    
    result = benchmark_dtw(n_points=45, n_dims=45)
    
    print(f"Backend C available: {result['backend_c_available']}")
    print(f"Test: {result['n_points']} points × {result['n_dims']} dims")
    print(f"Time C backend: {result['time_c_ms']:.2f} ms")
    print(f"Time Python: {result['time_python_ms']:.2f} ms")
    print(f"Speedup: {result['speedup']:.1f}x")
    print(f"Distances match: {result['distance_match']}")