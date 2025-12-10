# API Reference - FR_VERS_JP 2.1

Complete API documentation for facial recognition system.

---

## Core Modules

### `fr_core.verification_dtw`

Main verification module.

#### `verify_dtw()`

Verify user identity using DTW matching.

**Signature:**
```python
def verify_dtw(
    model_path: str,
    video_source: int | str = 0,
    num_frames: int = 10,
    check_liveness: bool = True,
    dtw_threshold: Optional[float] = None
) -> Tuple[bool, float]
```

**Parameters:**
- `model_path` (str): Path to user model (.npz file)
- `video_source` (int | str): Video source (0 for webcam, or path to video file)
- `num_frames` (int): Number of frames to capture (default: 10)
- `check_liveness` (bool): Enable liveness detection (default: True)
- `dtw_threshold` (float, optional): Custom DTW threshold (uses config default if None)

**Returns:**
- `is_verified` (bool): True if user verified
- `distance` (float): DTW distance (inf if liveness failed)

**Example:**
```python
from fr_core import verify_dtw

is_verified, distance = verify_dtw(
    model_path='models/jeanphi.npz',
    video_source=0,
    num_frames=10
)

if is_verified:
    print(f"✅ Verified (distance={distance:.2f})")
else:
    print(f"❌ Rejected (distance={distance:.2f})")
```

**Raises:**
- `FileNotFoundError`: Model file not found
- `ValueError`: Invalid model file (not a landmark model)
- `ImportError`: Missing dependencies (dtaidistance)

---

### `fr_core.landmark_utils`

Landmark extraction utilities.

#### `extract_landmarks_from_video()`

Extract landmarks from video and apply PCA.

**Signature:**
```python
def extract_landmarks_from_video(
    video_source: int | str = 0,
    num_frames: int = 10,
    apply_pca: bool = True
) -> Optional[np.ndarray]
```

**Parameters:**
- `video_source` (int | str): Video source
- `num_frames` (int): Number of frames to capture
- `apply_pca` (bool): Apply PCA transformation (default: True)

**Returns:**
- `pca_sequence` (np.ndarray): PCA-transformed landmarks (num_frames, 45)
  - Or None if extraction failed

**Example:**
```python
from fr_core.landmark_utils import extract_landmarks_from_video
import numpy as np

# Capture landmarks
sequence = extract_landmarks_from_video(
    video_source=0,
    num_frames=10
)

if sequence is not None:
    # Save model
    np.savez(
        'models/alice.npz',
        pca_sequence=sequence,
        metadata={'username': 'alice'}
    )
```

---

#### `is_landmark_model()`

Check if model is a landmark model.

**Signature:**
```python
def is_landmark_model(data: np.lib.npyio.NpzFile) -> bool
```

**Parameters:**
- `data` (NpzFile): Loaded .npz file

**Returns:**
- `bool`: True if landmark model

**Example:**
```python
import numpy as np
from fr_core.landmark_utils import is_landmark_model

data = np.load('models/jeanphi.npz')
if is_landmark_model(data):
    print("✓ Landmark model")
else:
    print("✗ Not a landmark model")
```

---

### `fr_core.ddtw`

Derivative DTW (velocity/acceleration features).

#### `compute_ddtw_distance()`

Compute DDTW distance with velocity or acceleration.

**Signature:**
```python
def compute_ddtw_distance(
    template: np.ndarray,
    query: np.ndarray,
    method: str = 'velocity',
    normalize: bool = True,
    window: int = 10
) -> Tuple[float, float]
```

**Parameters:**
- `template` (np.ndarray): Reference sequence (n_frames, n_features)
- `query` (np.ndarray): Query sequence (m_frames, n_features)
- `method` (str): 'none', 'velocity', or 'acceleration'
- `normalize` (bool): Normalize derivatives (default: True)
- `window` (int): Sakoe-Chiba window (default: 10)

**Returns:**
- `ddtw_distance` (float): DDTW distance
- `static_distance` (float): Classic DTW distance (for comparison)

**Example:**
```python
from fr_core.ddtw import compute_ddtw_distance

ddtw_dist, static_dist = compute_ddtw_distance(
    template=template_seq,
    query=query_seq,
    method='velocity'
)

print(f"DDTW: {ddtw_dist:.2f}, Static: {static_dist:.2f}")
```

---

### `fr_core.liveness`

Anti-spoofing / liveness detection.

#### `check_liveness_fusion()`

Multi-method liveness detection with fusion.

**Signature:**
```python
def check_liveness_fusion(
    video_source: int | str = 0,
    use_blink: bool = True,
    use_motion: bool = True,
    use_texture: bool = False,
    show_debug: bool = False
) -> LivenessResult
```

**Parameters:**
- `video_source` (int | str): Video source
- `use_blink` (bool): Enable blink detection
- `use_motion` (bool): Enable motion analysis
- `use_texture` (bool): Enable texture analysis (slow)
- `show_debug` (bool): Show debug visualization

**Returns:**
- `LivenessResult`: Dataclass with:
  - `is_live` (bool): True if live face
  - `confidence` (float): Confidence score [0, 1]
  - `details` (dict): Method-specific details
  - `method` (str): Method used

**Example:**
```python
from fr_core.liveness import check_liveness_fusion

result = check_liveness_fusion(
    video_source=0,
    use_blink=True,
    use_motion=True
)

if result.is_live and result.confidence >= 0.6:
    print(f"✓ Live (confidence={result.confidence:.2%})")
else:
    print(f"✗ Spoof detected")
```

---

### `fr_core.config`

Configuration constants.

**Main Settings:**

```python
# DTW threshold
DTW_THRESHOLD = 6.71

# DDTW
USE_DDTW = True
DDTW_METHOD = 'velocity'  # 'none', 'velocity', 'acceleration'
DDTW_NORMALIZE = True

# Liveness
USE_LIVENESS = True
LIVENESS_METHODS = ['blink', 'motion']  # 'blink', 'motion', 'texture'
LIVENESS_CONFIDENCE_THRESHOLD = 0.6

# Landmarks
N_LANDMARKS = 68
N_LANDMARK_FEATURES = 136

# PCA
PCA_N_COMPONENTS = 45
```

**Usage:**
```python
from fr_core import config

# Get threshold
threshold = config.DTW_THRESHOLD

# Modify (not recommended, edit config.py instead)
config.USE_LIVENESS = False
```

---

## Scripts

### `scripts/enroll.py`

Enroll a new user.

**Usage:**
```bash
python scripts/enroll.py <username>
```

**Example:**
```bash
python scripts/enroll.py alice
```

**Output:**
```
✅ SUCCESS
Model saved: models/alice.npz
Shape: (10, 45)
```

---

### `scripts/verify.py`

Verify user identity.

**Usage:**
```bash
python scripts/verify.py <model_path>
```

**Example:**
```bash
python scripts/verify.py models/alice.npz
```

**Output:**
```
✅ VERIFIED
Distance: 1.96
```

---

## Tests

### `tests/test_system.py`

Complete system test.

**Usage:**
```bash
python tests/test_system.py
```

**Features:**
- Liveness detection test
- DDTW test
- Full pipeline test
- Performance measurement

---

### `tests/test_ddtw.py`

DDTW methods comparison.

**Usage:**
```bash
python tests/test_ddtw.py
```

**Output:**
- Comparison: none vs velocity vs acceleration
- Separation metrics
- Recommendations

---

## Common Patterns

### Custom Threshold

```python
from fr_core import verify_dtw

# Stricter threshold
is_verified, dist = verify_dtw(
    model_path='models/jeanphi.npz',
    dtw_threshold=5.0  # Lower = stricter
)
```

### Disable Liveness

```python
# For testing or specific deployments
is_verified, dist = verify_dtw(
    model_path='models/jeanphi.npz',
    check_liveness=False
)
```

### Video File Input

```python
# Use video file instead of webcam
is_verified, dist = verify_dtw(
    model_path='models/jeanphi.npz',
    video_source='path/to/video.mp4'
)
```

### Error Handling

```python
from fr_core import verify_dtw

try:
    is_verified, distance = verify_dtw('models/jeanphi.npz')
    
    if distance == float('inf'):
        print("❌ Liveness check failed (spoof)")
    elif is_verified:
        print(f"✅ Verified (distance={distance:.2f})")
    else:
        print(f"❌ Rejected (distance={distance:.2f})")
        
except FileNotFoundError:
    print("❌ Model file not found")
except ValueError as e:
    print(f"❌ Invalid model: {e}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
```

---

## Model Format

`.npz` file structure:

```python
{
    'pca_sequence': np.ndarray,  # (num_frames, 45)
    'metadata': {
        'username': str,
        'num_frames': int,
        'features': 'landmarks_68',
        'version': '2.1.0'
    }
}
```

**Example:**
```python
import numpy as np

# Load model
data = np.load('models/jeanphi.npz', allow_pickle=True)
sequence = data['pca_sequence']  # (10, 45)
metadata = data['metadata'].item()  # dict

print(f"User: {metadata['username']}")
print(f"Shape: {sequence.shape}")
```

---

## Return Codes

### `verify_dtw()`

- `(True, distance)` - Verified successfully
- `(False, distance)` - Rejected (distance >= threshold)
- `(False, inf)` - Liveness check failed (spoof detected)

---

## Dependencies

**Required:**
```
mediapipe >= 0.10
opencv-python >= 4.9
numpy >= 1.26
scikit-learn >= 1.4
dtaidistance >= 2.3
scipy >= 1.12
```

**Optional:**
- GPU acceleration: dtaidistance[c] (compile C extension)

---

## Performance Tips

1. **Reduce frames for speed:**
   ```python
   verify_dtw(model_path='...', num_frames=5)  # Faster, less accurate
   ```

2. **Disable DDTW:**
   ```python
   # In config.py
   USE_DDTW = False  # Save ~0.5s
   ```

3. **Simplify liveness:**
   ```python
   # In config.py
   LIVENESS_METHODS = ['blink']  # Faster, less robust
   ```

---

## Thread Safety

**NOT thread-safe:**
- `verify_dtw()` uses OpenCV VideoCapture (not thread-safe)
- Use separate processes for concurrent verifications

**Thread-safe:**
- `load_model()`
- `compute_dtw_distance()`
- `is_landmark_model()`

---

## Version Info

```python
import fr_core
print(fr_core.__version__)  # '2.1.0'
```

---

For more details, see:
- [README.md](../README.md)
- [QUICKSTART.md](../QUICKSTART.md)
- [CONFIGURATION.md](CONFIGURATION.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
