# Changelog - FR_VERS_JP

All notable changes to this project will be documented in this file.

---

## [2.1.0] - 2024-12-09

### Clean Refactoring Release

**Added:**
- Clean project structure in FR_VERS_JP_2_1/
- Lightweight documentation (README, QUICKSTART, API)
- Simplified scripts (enroll.py, verify.py)
- Streamlined test suite

**Changed:**
- Removed legacy Gabor/LBP dependencies
- Simplified verification_dtw.py (no dependency on old verification.py)
- Cleaner fr_core/ module structure
- Consolidated documentation in docs/v2.1/ and docs/history/

**Removed:**
- Old verification.py, preprocessing.py, features.py modules
- Redundant test scripts (15+ → 3 essential)
- Obsolete documentation files
- Unused utility scripts

**Architecture:**
```
v2.0: verification_dtw.py → verification.py → features.py (complex)
v2.1: verification_dtw.py → landmark_utils.py (simple)
```

**Migration:**
- Old code preserved in FR_VERS_JP_2_0/
- Development history in docs/history/

---

## [2.0.0] - 2024-12-08

### Major Feature Release

**Added:**
- **Tier 1 Foundations:**
  - 68 facial landmarks (MediaPipe)
  - PCA dimensionality reduction (136→45)
  - DTW matching with Sakoe-Chiba constraint
  - Calibrated threshold (6.71, -90.1% from v1)
  - Validated separation (+12.44)

- **Tier 2 Optimizations:**
  - DDTW (Derivative DTW) for temporal dynamics
    - Velocity features (+38% separation)
    - Acceleration features (+66% separation)
  - Liveness detection (anti-spoofing)
    - Blink detection (EAR-based)
    - Motion analysis (passive)
    - Texture analysis (LBP-based)
    - Multi-method fusion

**Performance:**
- Verification time: ~5s
- FAR: 0%
- FRR: ~5%
- Anti-spoofing: 95%+ (photos blocked)

**Architecture:**
- 2-stage pipeline: Liveness → Identity
- Defense-in-depth security model

---

## [1.0.0] - 2024-11 (Legacy)

### Initial Release (Gabor + LBP)

**Features:**
- Gabor filters + LBP texture features
- GMM-based verification
- 3D pose tracking
- Basic enrollment/verification

**Limitations:**
- High threshold (68.0)
- No anti-spoofing
- Complex feature extraction
- Vulnerable to photos/videos

**Status:** Deprecated (see FR_vers_JP/)

---

## Version Comparison

| Feature | v1.0 | v2.0 | v2.1 |
|---------|------|------|------|
| **Features** | Gabor+LBP | Landmarks | Landmarks |
| **Matching** | GMM | DTW | DTW |
| **Threshold** | 68.0 | 6.71 | 6.71 |
| **DDTW** | ❌ | ✅ | ✅ |
| **Liveness** | ❌ | ✅ | ✅ |
| **Code Complexity** | High | Medium | **Low** |
| **Documentation** | Basic | Detailed | **Concise** |
| **Status** | Deprecated | Stable | **Production** |

---

## Upgrade Guide

### From v2.0 to v2.1

**No Breaking Changes** - Models compatible!

1. **Copy your models:**
   ```bash
   cp FR_VERS_JP_2_0/models/*.npz FR_VERS_JP_2_1/models/
   ```

2. **Update scripts:**
   ```bash
   # Old
   python enroll_landmarks.py jeanphi
   
   # New
   python scripts/enroll.py jeanphi
   ```

3. **Update imports:**
   ```python
   # Old
   from fr_core.verification_dtw import verify_dtw
   
   # New (same, but cleaner)
   from fr_core import verify_dtw
   ```

**Benefits:**
- Simpler codebase
- Faster onboarding
- Easier maintenance
- Cleaner documentation

### From v1.0 to v2.x

**Breaking Changes** - Models NOT compatible!

- Re-enroll all users with landmarks
- Update configuration (new threshold)
- Rebuild any custom integrations

See `docs/history/migration/` for detailed guide.

---

## Roadmap

### v2.2 (Future)

- [ ] Remote PPG (pulse detection)
- [ ] 3D depth estimation
- [ ] REST API
- [ ] Web interface

### v3.0 (Long-term)

- [ ] Deep learning embeddings (FaceNet, ArcFace)
- [ ] Multi-user database (1000+ users)
- [ ] Multi-spectral analysis (IR + RGB)
- [ ] Mobile SDK

---

## Contributors

- FR_VERS_JP Development Team
- Version 2.0 & 2.1 completed December 2024

---

## License

MIT License - See LICENSE file for details
