# Deployment Guide - v2.1

Guide for deploying FR_VERS_JP v2.1 in production environments.

## System Requirements

### Minimum
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8+
- **RAM**: 2GB
- **CPU**: 2 cores
- **Webcam**: 720p minimum

### Recommended
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **RAM**: 4GB
- **CPU**: 4 cores (Intel i5+ or equivalent)
- **Webcam**: 1080p with good low-light performance

## Installation

### 1. Create Virtual Environment

```bash
cd FR_VERS_JP_2_1
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "from fr_core import verify_dtw; print('✓ Installation OK')"
```

## Model Preparation

### Enrollment Process

```bash
# Enroll a new user
python scripts/enroll.py username

# The script will:
# 1. Capture 10 frames from webcam
# 2. Extract 68 facial landmarks per frame
# 3. Apply PCA transformation
# 4. Save to models/username.npz
```

**Best Practices**:
- Good lighting (front-facing, diffused)
- Neutral background
- User looks directly at camera
- Slight head movements for robustness

### Model Storage

Models are stored in `models/` directory:
```
models/
├── user1.npz  (71KB typical)
├── user2.npz
└── ...
```

**Security**:
- Models contain PCA-transformed features (non-reversible)
- No raw images or biometric data stored
- Recommend encrypting `models/` directory in production

## Production Configuration

### 1. Security-First Profile

Edit `fr_core/config.py`:

```python
# Strict verification
DTW_THRESHOLD = 5.5

# Strong liveness detection
USE_LIVENESS = True
LIVENESS_THRESHOLD = 0.75
LIVENESS_METHODS = ['blink', 'motion']

# Enhanced DDTW
USE_DDTW = True
DDTW_METHOD = 'combined'
DDTW_NORMALIZE = True
```

### 2. Fast Processing Profile

```python
# Faster verification
WINDOW_SIZE = 5
N_COMPONENTS = 15

# Skip liveness for speed
USE_LIVENESS = False

# Basic DTW
USE_DDTW = False
DTW_THRESHOLD = 6.0
```

### 3. Balanced Profile (Recommended)

Use default `config.py` settings.

## Integration

### Python API

```python
from fr_core import verify_dtw

# Verify user
verified, distance = verify_dtw(
    model_path='models/jeanphi.npz',
    video_source=0,  # Webcam index
    window=10,
    check_liveness=True,
    dtw_threshold=6.71
)

if verified:
    print(f"✓ Access granted (distance: {distance:.2f})")
else:
    print(f"✗ Access denied (distance: {distance:.2f})")
```

### Command Line

```bash
# Verify user
python scripts/verify.py models/jeanphi.npz

# With custom video source
python scripts/verify.py models/jeanphi.npz --video /dev/video1
```

### REST API (Optional)

Create `api_server.py`:

```python
from flask import Flask, request, jsonify
from fr_core import verify_dtw
import os

app = Flask(__name__)

@app.route('/verify/<username>', methods=['POST'])
def verify_user(username):
    model_path = f"models/{username}.npz"
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'User not found'}), 404
    
    # Use video source from request or default
    video_source = request.json.get('video_source', 0)
    
    verified, distance = verify_dtw(model_path, video_source)
    
    return jsonify({
        'verified': bool(verified),
        'distance': float(distance),
        'threshold': 6.71
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run with:
```bash
pip install flask
python api_server.py
```

## Performance Optimization

### 1. Webcam Optimization

```python
import cv2

# Set optimal resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

### 2. GPU Acceleration (Optional)

For high-throughput scenarios:

```bash
pip install opencv-contrib-python-headless
# Ensure CUDA-enabled OpenCV build
```

### 3. Batch Processing

Process multiple users efficiently:

```python
from fr_core import load_model
import numpy as np

# Load all models once
models = {}
for username in os.listdir('models/'):
    if username.endswith('.npz'):
        user = username[:-4]
        models[user] = load_model(f'models/{username}')

# Then verify quickly
def quick_verify(username, features):
    if username not in models:
        return False, np.inf
    
    template, pca, scaler = models[username]
    # ... compute DTW distance ...
    return distance < 6.71, distance
```

## Monitoring

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fr_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('fr_core')
```

### Metrics to Track

1. **Verification Time**: Should be < 2s
2. **False Accept Rate (FAR)**: Target < 1%
3. **False Reject Rate (FRR)**: Target < 5%
4. **Liveness Detection Rate**: Should catch > 95% of spoofs

### Health Check

```bash
# Test system health
python tests/test_system.py

# Expected output:
# ✅ Liveness détectée
# ✅ DDTW activé
# ✅ VÉRIFIÉ
```

## Security Considerations

### 1. Model Protection

- **Encrypt** `models/` directory at rest
- **Restrict** file permissions: `chmod 600 models/*.npz`
- **Backup** models to secure storage

### 2. Anti-Spoofing

- Always enable liveness detection in production
- Use combined methods: `LIVENESS_METHODS = ['blink', 'motion']`
- Set strict threshold: `LIVENESS_THRESHOLD = 0.7`

### 3. Audit Logging

Log all verification attempts:

```python
import json
from datetime import datetime

def log_verification(username, verified, distance):
    with open('audit.log', 'a') as f:
        f.write(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'verified': verified,
            'distance': distance
        }) + '\n')
```

### 4. Rate Limiting

Prevent brute-force attacks:

```python
from collections import defaultdict
from time import time

attempts = defaultdict(list)

def check_rate_limit(username, max_attempts=3, window=60):
    now = time()
    attempts[username] = [t for t in attempts[username] if now - t < window]
    
    if len(attempts[username]) >= max_attempts:
        return False  # Rate limited
    
    attempts[username].append(now)
    return True
```

## Troubleshooting

### Webcam Not Found

```bash
# List available cameras
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAILED')"
```

### Import Errors

```bash
# Verify Python path
python -c "import sys; print(sys.path)"

# Add current directory
export PYTHONPATH="${PYTHONPATH}:/path/to/FR_VERS_JP_2_1"
```

### Performance Issues

- Reduce `WINDOW_SIZE` to 5-7 frames
- Disable liveness: `USE_LIVENESS = False`
- Lower `N_COMPONENTS` to 15

## Scaling

### Multi-User Scenario

For > 100 users:
- Pre-load all models at startup
- Use caching for frequently accessed users
- Consider database storage for model metadata

### Distributed Deployment

For multiple verification stations:
- Share `models/` directory via NFS/S3
- Synchronize configuration across nodes
- Use load balancer for REST API

## Support

For issues or questions:
- Check logs: `fr_system.log`
- Run diagnostics: `python tests/test_system.py`
- Review configuration: `docs/v2.1/CONFIGURATION.md`
