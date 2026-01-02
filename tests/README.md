# Test Scripts

This directory contains test scripts for D-Face Hunter ARM64.

## Test Files

- `test_spatial_mode.py` - Unit tests for spatial matching algorithm
- `test_validation_spatial.py` - Integration tests with real enrollments
- Other test scripts as needed

## Running Tests

```bash
# Run all tests
python3.11 -m pytest

# Run specific test
python3.11 test_spatial_mode.py
```

## Test Coverage

Tests verify:
- Spatial pose-aware matching
- MediaPipe landmark extraction
- Enrollment process
- Verification accuracy
- Edge cases (short sequences, pose variations)
