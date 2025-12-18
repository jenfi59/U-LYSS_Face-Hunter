# Models Directory

This directory stores enrolled user models for facial recognition.

## Structure

Each user model is saved as a `.npz` (NumPy archive) file with the following naming convention:
```
<username>.npz
```

## Contents

User models contain:
- PCA transformation model
- Scaler for feature normalization
- Pose reference (mean, rotation, translation)
- DTW template (for DTW-based verification) or GMM model (for legacy mode)

## Usage

### Creating a Model

Use the enrollment script to create a new user model:
```bash
python3 scripts/enroll.py <username>
```

This will save the model to `models/<username>.npz`.

### Using a Model for Verification

Verify a user's identity using:
```bash
python3 scripts/verify.py models/<username>.npz
```

Or programmatically:
```python
from fr_core import verify_dtw

is_verified, distance = verify_dtw(
    model_path='models/username.npz',
    video_source=0
)
```

## Security

⚠️ **Important Security Notes:**
- Model files contain biometric data and should be protected
- In production, encrypt this directory at rest
- Restrict file permissions: `chmod 600 models/*.npz`
- Back up models to secure storage
- Never commit `.npz` files to version control

## gitignore

The `.gitignore` file excludes `models/*.npz` from version control to prevent accidental commits of sensitive biometric data.
