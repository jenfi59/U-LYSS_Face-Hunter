# Face Hunter FR_VERS_JP 2.1

**Facial Recognition System** using landmarks, DTW, and anti-spoofing.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)]()
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![ARM64](https://img.shields.io/badge/ARM64-supported-green.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## Features

- **🎯 68 Facial Landmarks** (MediaPipe)
- **⏱️ DTW Matching** with velocity features (DDTW)
- **🛡️ Anti-Spoofing** (blink + motion detection)
- **⚡ Fast** (~5s verification)
- **🔒 Secure** (2-stage defense: liveness → identity)
- **🐳 Docker & ARM64** (Raspberry Pi, Apple Silicon, AWS Graviton)

---

## Quick Start

### Installation

**Sur cet ordinateur** - Le système est portable et prêt à l'emploi.

```bash
cd FR_VERS_JP_2_1
# Toutes les dépendances sont déjà installées (--user)
```

**Sur un nouvel ordinateur** - Installer les dépendances :

```bash
cd FR_VERS_JP_2_1
pip install --user -r requirements.txt
```

**Build ARM64 (Raspberry Pi, Apple Silicon, AWS Graviton)** :

```bash
# Build pour ARM64
./build-arm64.sh

# Ou avec Docker Compose
docker-compose up --build

# Build multi-architecture (ARM64 + AMD64)
./build-multiarch.sh
```

> 📹 **Caméra** : Détection automatique (USB/intégrée). OpenCV utilise l'index 0 par défaut.  
> 🐳 **Docker** : Voir [BUILD_ARM64.md](BUILD_ARM64.md) pour guide complet ARM64

### Launcher Interactif (Recommandé)

```bash
python3 launcher.py
```

**Menu :**
- `[1]` 📝 Enrollment - Enregistrer un utilisateur
- `[2]` ✅ Verification - Vérifier l'identité
- `[3]` 👥 Lister les modèles
- `[4]` 🗑️ Supprimer un modèle
- `[5]` ⚙️ Paramètres (DTW, DDTW, Liveness, PCA)
- `[0]` 🚪 Quitter

### Enroll User (Méthode Manuelle)

```bash
# Méthode simple avec wrapper
./run_enrollment.sh jeanphi

# Ou directement
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/enroll_landmarks.py jeanphi
```

### Verify User (Méthode Manuelle)

```bash
# Méthode simple avec wrapper
./run_verify.sh jeanphi

# Ou directement
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/verify.py models/jeanphi.npz
```

---

## Architecture

```
Webcam → Liveness Detection → Landmarks → DDTW → DTW → ✓/✗
         (blink + motion)      (68 pts)   (velocity)
```

**2-Stage Pipeline:**
1. **Stage 1:** Liveness (anti-spoofing) → blocks photos/videos
2. **Stage 2:** Identity (landmarks + DTW) → verifies user

---

## Performance

| Metric | Value |
|--------|-------|
| **Verification Time** | ~5s |
| **FAR** | 0% |
| **FRR** | ~5% |
| **Anti-spoofing** | 95%+ (photos) |

---

## Configuration

Edit `fr_core/config.py`:

```python
# DTW threshold
DTW_THRESHOLD = 6.71

# Enable DDTW (velocity features)
USE_DDTW = True
DDTW_METHOD = 'velocity'  # 'none', 'velocity', 'acceleration'

# Enable liveness detection
USE_LIVENESS = True
LIVENESS_METHODS = ['blink', 'motion']  # 'blink', 'motion', 'texture'
```

---

## Project Structure

```
FR_VERS_JP_2_1/
├── fr_core/              # Core modules
│   ├── config.py         # Configuration
│   ├── landmark_utils.py # Landmark extraction
│   ├── ddtw.py           # Derivative DTW
│   ├── liveness.py       # Anti-spoofing
│   └── verification_dtw.py # Main verification
│
├── scripts/              # Utilities
│   ├── enroll.py         # User enrollment
│   └── verify.py         # Verification test
│
├── tests/                # Test suite
│   ├── test_system.py    # Complete system test
│   └── test_ddtw.py      # DDTW test
│
├── docs/                 # Documentation
│   ├── v2.1/             # Current docs
│   └── history/          # Development history
│
└── models/               # User templates
```

---

## Usage Examples

### Python API

```python
from fr_core import verify_dtw

# Verify user
is_verified, distance = verify_dtw(
    model_path='models/jeanphi.npz',
    video_source=0,
    num_frames=10,
    check_liveness=True
)

if is_verified:
    print(f"✅ VERIFIED (distance={distance:.2f})")
else:
    print(f"❌ REJECTED")
```

### Command Line

```bash
# Enroll
python scripts/enroll.py alice

# Verify
python scripts/verify.py models/alice.npz

# Test system
python tests/test_system.py
```

---

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[BUILD_ARM64.md](BUILD_ARM64.md)** - ARM64 build guide (Raspberry Pi, Apple Silicon, AWS Graviton)
- **[docs/v2.1/API.md](docs/v2.1/API.md)** - API reference
- **[docs/v2.1/CONFIGURATION.md](docs/v2.1/CONFIGURATION.md)** - Configuration guide
- **[docs/v2.1/DEPLOYMENT.md](docs/v2.1/DEPLOYMENT.md)** - Deployment guide

---

## Changelog

See **[CHANGELOG.md](CHANGELOG.md)** for version history.

**v2.1.0** (Dec 2024)
- Clean refactoring from v2.0
- Removed legacy Gabor/LBP code
- Simplified architecture
- Lightweight documentation

**v2.0.0** (Dec 2024)
- Landmarks + DTW
- DDTW (velocity features)
- Liveness detection

---

## Requirements

```
python >= 3.10
mediapipe >= 0.10
opencv-python >= 4.9
numpy >= 1.26
scikit-learn >= 1.4
dtaidistance >= 2.3
scipy >= 1.12
```

---

## License

MIT License

---

## Support

- **Issues:** GitHub Issues
- **Docs:** `docs/v2.1/`
- **History:** `docs/history/` (development archives)

---

**Version 2.1.0** - Production Ready ✅
