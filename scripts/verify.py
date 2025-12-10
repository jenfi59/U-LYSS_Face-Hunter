#!/usr/bin/env python3
"""
FR_VERS_JP 2.1 - Verification Test
===================================

Test the complete verification system.

Usage:
    python scripts/verify.py <model_path>

Version: 2.1.0
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fr_core.verification_dtw import verify_dtw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"\n{'='*70}")
    print(f"  FR_VERS_JP 2.1 - VERIFICATION")
    print(f"{'='*70}\n")
    print(f"Model: {model_path}\n")
    print("Instructions:")
    print("  • Look at the camera")
    print("  • Blink your eyes (anti-spoofing)")
    print("  • Move your head slightly\n")
    
    input("Press ENTER to start...")
    
    # Verify
    print("\n🔍 Verifying...")
    is_verified, distance = verify_dtw(
        model_path=model_path,
        video_source=0,
        num_frames=10,
        check_liveness=True
    )
    
    # Result
    print(f"\n{'='*70}")
    print(f"  RESULT")
    print(f"{'='*70}\n")
    
    if is_verified:
        print(f"✅ VERIFIED")
        print(f"Distance: {distance:.2f}\n")
    else:
        if distance == float('inf'):
            print(f"❌ REJECTED - Liveness check failed")
            print(f"(Spoof detected)\n")
        else:
            print(f"❌ REJECTED")
            print(f"Distance: {distance:.2f}\n")


if __name__ == "__main__":
    main()
