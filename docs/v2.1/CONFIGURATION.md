# Configuration Guide - v2.1

This guide explains all configuration options in `fr_core/config.py`.

## DTW Configuration

### `DTW_THRESHOLD`
- **Default**: `6.71`
- **Type**: `float`
- **Description**: Maximum DTW distance for identity verification. Distances below this threshold indicate a match.
- **Tuning**: 
  - Lower values (5.0-6.0): Higher security, more false rejections
  - Higher values (7.0-8.0): Better usability, higher false acceptance risk

### `WINDOW_SIZE`
- **Default**: `10`
- **Type**: `int`
- **Description**: Number of frames to capture during verification
- **Tuning**:
  - Smaller (5-8): Faster verification, less robust
  - Larger (12-15): More robust, slower processing

## DDTW Configuration

### `USE_DDTW`
- **Default**: `True`
- **Type**: `bool`
- **Description**: Enable Derivative Dynamic Time Warping for velocity-based matching

### `DDTW_METHOD`
- **Default**: `'velocity'`
- **Type**: `str`
- **Options**: `'velocity'`, `'acceleration'`, `'combined'`
- **Description**:
  - `velocity`: First derivative (movement speed)
  - `acceleration`: Second derivative (movement acceleration)
  - `combined`: Both derivatives with weighted average

### `DDTW_NORMALIZE`
- **Default**: `True`
- **Type**: `bool`
- **Description**: Normalize DDTW features before distance computation

## Liveness Detection

### `USE_LIVENESS`
- **Default**: `True`
- **Type**: `bool`
- **Description**: Enable anti-spoofing liveness detection

### `LIVENESS_METHODS`
- **Default**: `['blink', 'motion']`
- **Type**: `list[str]`
- **Options**: `'blink'`, `'motion'`
- **Description**:
  - `blink`: Eye blink detection (EAR-based)
  - `motion`: 3D head movement analysis

### `LIVENESS_THRESHOLD`
- **Default**: `0.6`
- **Type**: `float`
- **Description**: Minimum liveness score (0-1) to pass anti-spoofing
- **Tuning**:
  - Lower (0.4-0.5): More lenient, higher spoof risk
  - Higher (0.7-0.8): Stricter, may reject real users

## PCA Configuration

### `N_COMPONENTS`
- **Default**: `20`
- **Type**: `int`
- **Description**: Number of PCA components for dimensionality reduction
- **Tuning**:
  - Lower (10-15): Faster, less information retained
  - Higher (25-30): More information, slower processing

## Enrollment Configuration

### `ENROLLMENT_FRAMES`
- **Default**: `10`
- **Type**: `int`
- **Description**: Number of frames to capture during enrollment
- **Recommendation**: Match with `WINDOW_SIZE` for consistency

### `ENROLLMENT_ZONES`
- **Default**: `['center', 'left', 'right']`
- **Type**: `list[str]`
- **Description**: Face positions to capture during enrollment for robustness

## Performance Tuning

### High Security Profile
```python
DTW_THRESHOLD = 5.5
LIVENESS_THRESHOLD = 0.75
USE_DDTW = True
DDTW_METHOD = 'combined'
```

### Balanced Profile (Default)
```python
DTW_THRESHOLD = 6.71
LIVENESS_THRESHOLD = 0.6
USE_DDTW = True
DDTW_METHOD = 'velocity'
```

### High Usability Profile
```python
DTW_THRESHOLD = 7.5
LIVENESS_THRESHOLD = 0.5
USE_DDTW = True
DDTW_METHOD = 'velocity'
```

### Fast Processing Profile
```python
WINDOW_SIZE = 5
N_COMPONENTS = 15
USE_LIVENESS = False
USE_DDTW = False
```

## Environment Variables

Configuration can be overridden with environment variables:

```bash
export FR_DTW_THRESHOLD=6.0
export FR_USE_LIVENESS=True
export FR_LIVENESS_THRESHOLD=0.7
```

Load with:
```python
import os
DTW_THRESHOLD = float(os.getenv('FR_DTW_THRESHOLD', 6.71))
```

## Validation

After modifying configuration, validate with:

```bash
python tests/test_system.py
```

Monitor metrics:
- **FAR** (False Acceptance Rate): Should be < 1%
- **FRR** (False Rejection Rate): Should be < 5%
- **Processing Time**: Should be < 2s per verification
