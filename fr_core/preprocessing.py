"""
Image Preprocessing Module – FR_VERS_JP 2.0
============================================

Advanced facial image preprocessing with texture descriptor extraction.

Features:
---------
- Multi-level wavelet normalization with adaptive noise thresholding
- Bilateral filtering for edge-preserving smoothing
- Gabor filter banks (4 orientations, 2 frequencies)
- Local Binary Patterns (LBP) histograms
- Configurable preprocessing pipeline

Gabor Features (16 dimensions):
-------------------------------
- 4 orientations: 0°, 45°, 90°, 135°
- 2 frequencies: 0.1, 0.3
- 2 statistics per kernel: mean, std
- Total: 4 × 2 × 2 = 16 features

LBP Features (256 dimensions):
------------------------------
- Radius: 1 pixel
- Neighbors: 8 points
- Normalized histogram (256 bins)

Version: 2.0.0
"""

import logging
from typing import Tuple

import numpy as np

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logging.warning(
        "PyWavelets not available. Wavelet-based normalization will be skipped. "
        "Install pywavelets for improved illumination normalization: pip install pywavelets"
    )


def wavelet_normalize(
    image: np.ndarray,
    wavelet: str = 'db1',
    level: int = 2,
    noise_threshold_factor: float = 0.5,
) -> np.ndarray:
    """Apply wavelet-based illumination normalization with multi-level processing.

    This function decomposes the image into low and high frequency
    components using wavelet transform, applies histogram equalization
    to the low frequency component to normalize illumination, applies
    adaptive filtering to high frequency components, and reconstructs.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image.
    wavelet : str
        Wavelet family to use (e.g., 'db1', 'haar', 'sym5').
    level : int
        Decomposition level (2 or 3 recommended).
    noise_threshold_factor : float
        Factor for soft thresholding on detail coefficients (default: 0.5).

    Returns
    -------
    np.ndarray
        Normalized grayscale image.
    """
    if not PYWAVELETS_AVAILABLE:
        # Fall back to simple histogram equalization
        import cv2
        return cv2.equalizeHist(image)

    import cv2

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # coeffs is a list: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    # cA_n is the approximation (low frequency)
    # The tuples contain horizontal, vertical, and diagonal details (high frequency)

    # Extract approximation coefficients (low frequency component)
    approx = coeffs[0]

    # Normalize the approximation to [0, 255] range
    approx_normalized = cv2.normalize(
        approx,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    # Apply histogram equalization to the low frequency component
    approx_equalized = cv2.equalizeHist(approx_normalized)

    # Convert back to float for reconstruction
    approx_equalized = approx_equalized.astype(np.float64)

    # Replace the approximation coefficients
    coeffs[0] = approx_equalized

    # Process detail coefficients (high frequency) with adaptive thresholding
    # This helps reduce noise while preserving edges
    detail_coeffs = []
    for i, detail_tuple in enumerate(coeffs[1:]):
        processed_details = []
        for detail in detail_tuple:
            # Apply soft thresholding to reduce noise
            # Use a more aggressive threshold for higher levels (coarser details)
            level_factor = 1.0 + (i * 0.2)  # Increase threshold for coarser levels
            threshold = np.std(detail) * noise_threshold_factor * level_factor
            detail_processed = pywt.threshold(detail, threshold, mode='soft')
            processed_details.append(detail_processed)
        detail_coeffs.append(tuple(processed_details))

    # Reconstruct with processed coefficients
    coeffs_processed = [coeffs[0]] + detail_coeffs
    reconstructed = pywt.waverec2(coeffs_processed, wavelet)

    # Ensure the reconstructed image has the same shape as input
    # (waverec2 may produce slightly different size due to padding)
    if reconstructed.shape != image.shape:
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]

    # Normalize to [0, 255] range
    reconstructed = cv2.normalize(
        reconstructed,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    return reconstructed


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filtering for edge-preserving noise reduction.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    d : int
        Diameter of each pixel neighborhood.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    import cv2
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def preprocess_face(
    image: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    quantization_levels: int = 8,
    use_wavelet: bool = True,
    wavelet: str = 'db1',
    wavelet_level: int = 2,
    use_bilateral: bool = True,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> np.ndarray:
    """Pre-process a cropped face image with advanced normalization.

    Pipeline:
    1. Convert to grayscale
    2. Apply wavelet-based illumination normalization (optional)
    3. Apply bilateral filtering for noise reduction (optional)
    4. Resize to target size
    5. Quantize to reduce noise

    Parameters
    ----------
    image : np.ndarray
        RGB or BGR face image.
    target_size : tuple of int
        Target size (width, height) for the output image.
    quantization_levels : int
        Number of quantization levels (e.g., 8 means divide by 32).
    use_wavelet : bool
        Whether to use wavelet-based normalization.
    wavelet : str
        Wavelet family to use.
    wavelet_level : int
        Wavelet decomposition level.
    use_bilateral : bool
        Whether to apply bilateral filtering.
    bilateral_d : int
        Bilateral filter diameter.
    bilateral_sigma_color : float
        Bilateral filter sigma in color space.
    bilateral_sigma_space : float
        Bilateral filter sigma in coordinate space.

    Returns
    -------
    np.ndarray
        Pre-processed grayscale image with shape (H, W).
    """
    import cv2

    logging.debug("preprocess_face received image with shape %s", image.shape)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply wavelet-based illumination normalization
    if use_wavelet:
        gray = wavelet_normalize(gray, wavelet=wavelet, level=wavelet_level)
    else:
        # Simple histogram equalization
        gray = cv2.equalizeHist(gray)

    # Apply bilateral filtering for noise reduction
    if use_bilateral:
        gray = bilateral_filter(
            gray,
            d=bilateral_d,
            sigma_color=bilateral_sigma_color,
            sigma_space=bilateral_sigma_space,
        )

    # Resize to standard size
    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)

    # Quantize to reduce noise
    # For 8 levels: divide by (256 / 8 = 32) and multiply back
    quantization_step = 256 // quantization_levels
    quantized = (gray // quantization_step) * quantization_step

    return quantized


def preprocess_face_from_config(image: np.ndarray, config: object = None) -> np.ndarray:
    """Preprocess face using parameters from config module.

    Parameters
    ----------
    image : np.ndarray
        RGB or BGR face image.
    config : module or None
        Configuration module. If None, imports from config.py.

    Returns
    -------
    np.ndarray
        Pre-processed grayscale image.
    """
    if config is None:
        try:
            import config
        except ImportError:
            logging.warning("config.py not found, using default parameters")
            return preprocess_face(image)

    return preprocess_face(
        image,
        target_size=config.FACE_SIZE,
        quantization_levels=config.QUANTIZATION_LEVELS,
        use_wavelet=config.USE_WAVELET_NORMALIZATION,
        wavelet=config.WAVELET_FAMILY,
        wavelet_level=config.WAVELET_LEVEL,
        use_bilateral=config.USE_BILATERAL_FILTER,
        bilateral_d=config.BILATERAL_D,
        bilateral_sigma_color=config.BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=config.BILATERAL_SIGMA_SPACE,
    )


def extract_gabor_features(
    image: np.ndarray,
    orientations: int = 4,
    frequencies: Tuple[float, ...] = (0.1, 0.3),
    ksize: int = 31,
) -> np.ndarray:
    """Extract Gabor filter bank features from an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image.
    orientations : int
        Number of orientations (e.g., 4 for 0°, 45°, 90°, 135°).
    frequencies : tuple of float
        Frequencies to use (e.g., (0.1, 0.3) for 2 frequency bands).
    ksize : int
        Size of the Gabor kernel.

    Returns
    -------
    np.ndarray
        Flattened Gabor feature vector.
    """
    import cv2

    features = []
    theta_step = np.pi / orientations

    for freq in frequencies:
        for theta_idx in range(orientations):
            theta = theta_idx * theta_step
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=4.0,
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            # Filter the image
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            # Compute statistics: mean and std as features
            features.extend([np.mean(filtered), np.std(filtered)])

    return np.array(features, dtype=np.float32)


def extract_lbp_features(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
    method: str = 'uniform',
) -> np.ndarray:
    """Extract Local Binary Pattern (LBP) histogram features.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image.
    radius : int
        Radius of the LBP circle.
    n_points : int
        Number of circularly symmetric neighbor pixels.
    method : str
        LBP method: 'uniform' or 'default'.

    Returns
    -------
    np.ndarray
        Normalized LBP histogram.
    """
    # Simple LBP implementation without scikit-image dependency
    h, w = image.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

    # Simple 8-point LBP for radius=1
    if radius == 1 and n_points == 8:
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                # 8 neighbors in clockwise order
                code |= (image[i - 1, j - 1] >= center) << 7
                code |= (image[i - 1, j] >= center) << 6
                code |= (image[i - 1, j + 1] >= center) << 5
                code |= (image[i, j + 1] >= center) << 4
                code |= (image[i + 1, j + 1] >= center) << 3
                code |= (image[i + 1, j] >= center) << 2
                code |= (image[i + 1, j - 1] >= center) << 1
                code |= (image[i, j - 1] >= center) << 0
                lbp[i - radius, j - radius] = code
    else:
        # Fallback: simple center comparison
        logging.warning(f"LBP with radius={radius}, n_points={n_points} not optimized, using simple version")
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                if i > 0 and j > 0 and image[i - 1, j - 1] >= center:
                    code |= 1 << 7
                if i > 0 and image[i - 1, j] >= center:
                    code |= 1 << 6
                lbp[i - radius, j - radius] = code

    # Compute histogram (256 bins for standard LBP)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

    # Normalize histogram
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist
