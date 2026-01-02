#!/usr/bin/env python3
"""
Simple batch visualization of existing enrollment data.
No camera capture, just visualize what's already saved.
"""

import numpy as np
import cv2
import os
import sys

def draw_68_landmarks(img, landmarks, color=(0, 255, 0)):
    """Draw 68 facial landmarks with connections"""
    # Draw points
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    
    # Face contour: 0-16
    for i in range(16):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Eyebrows: 17-21, 22-26
    for start in [17, 22]:
        for i in range(start, start+4):
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[i+1].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
    
    # Eyes: 36-41, 42-47
    for start in [36, 42]:
        for i in range(start, start+5):
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[i+1].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
        # Close eye
        pt1 = tuple(landmarks[start+5].astype(int))
        pt2 = tuple(landmarks[start].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Nose: 27-30
    for i in range(27, 30):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Mouth outer: 48-59
    for i in range(48, 59):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    # Close mouth
    pt1 = tuple(landmarks[59].astype(int))
    pt2 = tuple(landmarks[48].astype(int))
    cv2.line(img, pt1, pt2, color, 1)
    
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python3.13 test_visualize_batch.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    
    print(f"\n{'='*60}")
    print(f"BATCH VISUALIZATION: {username}")
    print(f"{'='*60}\n")
    
    # Load enrollment data
    npz_path = f"models/users/{username}.npz"
    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)
    
    data = np.load(npz_path, allow_pickle=True)
    landmarks = data['landmarks']
    metadata = data['metadata'].item()
    
    print(f"Loaded: {npz_path}")
    print(f"Landmarks: {landmarks.shape}")
    print(f"Metadata: {metadata}")
    
    # Create output directory
    output_dir = f"tests/frames/{username}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {len(landmarks)} frame visualizations...")
    
    for i, lm in enumerate(landmarks):
        # Create blank canvas
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw landmarks
        img = draw_68_landmarks(img, lm, color=(0, 255, 0))
        
        # Add info
        cv2.putText(img, f"Enrolled Frame {i+1}/{len(landmarks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"User: {username}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Detector: {metadata.get('detector', 'unknown')}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Landmarks: {metadata.get('num_landmarks', 68)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save
        output_path = os.path.join(output_dir, f"enrolled_{i:03d}.jpg")
        cv2.imwrite(output_path, img)
        
        if (i+1) % 10 == 0:
            print(f"  Generated {i+1}/{len(landmarks)} frames...")
    
    print(f"\n✓ Complete! Saved {len(landmarks)} frames to: {output_dir}")
    print(f"\nTo view:")
    print(f"  ls {output_dir}")
    print(f"  eog {output_dir}/enrolled_000.jpg  # View first frame")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
