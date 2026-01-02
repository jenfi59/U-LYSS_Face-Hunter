# Changelog

All notable changes to D-Face Hunter ARM64 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-29

### Added
- Initial release of D-Face Hunter ARM64
- MediaPipe Python API integration (468 3D landmarks)
- Spatial pose-aware matching algorithm
- Two-phase enrollment system (guided + manual validation)
- Interactive enrollment and verification scripts
- Real-time verification with pose filtering
- Comprehensive test suite
- Documentation and README

### Features
- 468-point 3D facial landmark detection
- Pose estimation (yaw, pitch, roll) from facial transformation matrix
- Spatial matching mode with epsilon filtering (±10-15°)
- Backward compatibility with legacy models
- Optimized for ARM64 architecture

### Technical Details
- Python 3.11+ support
- MediaPipe 0.10.18
- Euler XZY pose convention
- Distance threshold: 3.0
- Minimum frames: 10 (probe), 45 (enrollment)

---

## Future Releases

### [1.1.0] - Planned
- Multi-user verification (1:N matching)
- Performance optimizations
- Enhanced test coverage

### [1.2.0] - Planned
- Anti-spoofing (liveness detection)
- GPU acceleration support
- Web interface

---

**Note**: This is the first production-ready release of D-Face Hunter ARM64.
