#!/usr/bin/env python3
"""
Visualize landmarks from existing enrollment and verification data.
Reconstructs DTW alignment from saved .npz files.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def draw_landmarks_on_blank(landmarks, img_size=(640, 480), color=(0, 255, 0), scale=1.0):
    """Draw landmarks on blank canvas"""
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Normalize landmarks to fit canvas
    landmarks_scaled = landmarks * scale
    
    # Draw landmarks
    for i, (x, y) in enumerate(landmarks_scaled):
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    
    # Draw connections (68 landmarks standard)
    if len(landmarks) >= 68:
        # Face contour: 0-16
        for i in range(16):
            pt1 = tuple(landmarks_scaled[i].astype(int))
            pt2 = tuple(landmarks_scaled[i+1].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
        
        # Eyebrows: 17-21, 22-26
        for start in [17, 22]:
            for i in range(start, start+4):
                pt1 = tuple(landmarks_scaled[i].astype(int))
                pt2 = tuple(landmarks_scaled[i+1].astype(int))
                cv2.line(img, pt1, pt2, color, 1)
        
        # Eyes: 36-41, 42-47
        for start in [36, 42]:
            for i in range(start, start+5):
                pt1 = tuple(landmarks_scaled[i].astype(int))
                pt2 = tuple(landmarks_scaled[i+1].astype(int))
                cv2.line(img, pt1, pt2, color, 1)
            # Close eye
            pt1 = tuple(landmarks_scaled[start+5].astype(int))
            pt2 = tuple(landmarks_scaled[start].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
        
        # Nose: 27-35
        for i in range(27, 30):
            pt1 = tuple(landmarks_scaled[i].astype(int))
            pt2 = tuple(landmarks_scaled[i+1].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
        
        # Mouth: 48-59, 60-67
        for i in range(48, 59):
            pt1 = tuple(landmarks_scaled[i].astype(int))
            pt2 = tuple(landmarks_scaled[i+1].astype(int))
            cv2.line(img, pt1, pt2, color, 1)
        pt1 = tuple(landmarks_scaled[59].astype(int))
        pt2 = tuple(landmarks_scaled[48].astype(int))
        cv2.line(img, pt1, pt2, color, 1)
    
    return img

def visualize_enrollment_data(username, output_dir="tests/frames/existing"):
    """Visualize landmarks from saved enrollment .npz file"""
    print(f"\n=== VISUALIZING ENROLLMENT DATA: {username} ===")
    
    # Load enrollment file
    npz_path = f"models/users/{username}.npz"
    if not os.path.exists(npz_path):
        print(f"Error: Enrollment file not found: {npz_path}")
        return None
    
    data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded: {npz_path}")
    print(f"Keys: {list(data.keys())}")
    
    # Get landmarks
    if 'landmarks' in data:
        landmarks = data['landmarks']
    elif 'landmarks_sequence' in data:
        landmarks = data['landmarks_sequence']
    else:
        print("Error: No landmarks found in .npz file")
        return None
    
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Get metadata
    metadata = data.get('metadata', None)
    if metadata is not None:
        metadata = metadata.item()
        print(f"Metadata: {metadata}")
    
    # Create output directories
    enroll_dir = os.path.join(output_dir, "enrollment", username)
    os.makedirs(enroll_dir, exist_ok=True)
    
    # Visualize each frame
    print(f"\nGenerating {len(landmarks)} frames...")
    for i, lm in enumerate(landmarks):
        img = draw_landmarks_on_blank(lm, color=(0, 255, 0))
        
        # Add info
        cv2.putText(img, f"Enrollment Frame {i+1}/{len(landmarks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"User: {username}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if metadata:
            cv2.putText(img, f"Detector: {metadata.get('detector', 'unknown')}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Landmarks: {metadata.get('num_landmarks', len(lm))}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save
        output_path = os.path.join(enroll_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(output_path, img)
    
    print(f"Saved {len(landmarks)} frames to: {enroll_dir}")
    
    return landmarks

def capture_verification_for_existing(username, num_landmarks=68, output_dir="tests/frames/existing"):
    """Capture new verification frames to compare with existing enrollment"""
    print(f"\n=== CAPTURING VERIFICATION FRAMES FOR: {username} ===")
    
    from fr_core import LandmarkDetectorONNX
    
    verify_dir = os.path.join(output_dir, "verification", username)
    os.makedirs(verify_dir, exist_ok=True)
    
    # Initialize detector
    detector = LandmarkDetectorONNX(num_landmarks=num_landmarks)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None
    
    print("\nCapturing verification frames (Q to stop, minimum 30 frames recommended)...")
    print("Press ENTER to start...")
    input()
    
    verify_landmarks = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process
        result = detector.process_frame(frame)
        
        if result is not None:
            landmarks = result['landmarks']
            
            # Visualize on blank canvas
            img = draw_landmarks_on_blank(landmarks, color=(255, 0, 255))
            
            cv2.putText(img, f"Verification Frame {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Q=finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save
            output_path = os.path.join(verify_dir, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(output_path, img)
            
            verify_landmarks.append(landmarks)
            frame_count += 1
            
            cv2.imshow("Verification", img)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Verification", img)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCaptured {len(verify_landmarks)} verification frames")
    print(f"Saved to: {verify_dir}")
    
    if len(verify_landmarks) == 0:
        return None
    
    # Save landmarks
    verify_landmarks = np.array(verify_landmarks, dtype=np.float32)
    np.save(os.path.join(verify_dir, "landmarks.npy"), verify_landmarks)
    
    return verify_landmarks

def visualize_dtw_alignment_existing(enrolled_landmarks, verify_landmarks, username, output_dir="tests/frames/existing/dtw"):
    """Create DTW alignment visualization"""
    print(f"\n=== DTW ALIGNMENT: {username} ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute DTW
    from dtaidistance import dtw
    
    enrolled_seq = enrolled_landmarks.reshape(len(enrolled_landmarks), -1)
    verify_seq = verify_landmarks.reshape(len(verify_landmarks), -1)
    
    distance = dtw.distance(enrolled_seq, verify_seq)
    path = dtw.warping_path(enrolled_seq, verify_seq)
    
    print(f"DTW Distance: {distance:.4f}")
    print(f"Alignment path length: {len(path)}")
    print(f"Enrolled frames: {len(enrolled_landmarks)}")
    print(f"Verification frames: {len(verify_landmarks)}")
    
    # Create alignment visualization
    h = len(enrolled_landmarks)
    w = len(verify_landmarks)
    scale = 5
    
    alignment_img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    
    # Draw grid
    for i in range(0, h*scale, scale):
        cv2.line(alignment_img, (0, i), (w*scale, i), (50, 50, 50), 1)
    for j in range(0, w*scale, scale):
        cv2.line(alignment_img, (j, 0), (j, h*scale), (50, 50, 50), 1)
    
    # Draw path
    for i, j in path:
        x = j * scale + scale//2
        y = i * scale + scale//2
        cv2.circle(alignment_img, (x, y), 2, (0, 255, 255), -1)
    
    # Draw connections
    for k in range(len(path)-1):
        i1, j1 = path[k]
        i2, j2 = path[k+1]
        pt1 = (j1*scale + scale//2, i1*scale + scale//2)
        pt2 = (j2*scale + scale//2, i2*scale + scale//2)
        cv2.line(alignment_img, pt1, pt2, (0, 200, 200), 1)
    
    # Save main alignment
    alignment_path = os.path.join(output_dir, f"{username}_dtw_alignment.jpg")
    cv2.imwrite(alignment_path, alignment_img)
    print(f"Saved alignment map: {alignment_path}")
    
    # Create side-by-side comparisons for sampled pairs
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    
    step = max(1, len(path) // 30)  # Max 30 pairs
    
    print(f"\nGenerating {len(range(0, len(path), step))} aligned pair visualizations...")
    
    for k in range(0, len(path), step):
        i, j = path[k]
        
        # Get landmarks
        enroll_lm = enrolled_landmarks[i]
        verify_lm = verify_landmarks[j]
        
        # Create side-by-side
        img_enroll = draw_landmarks_on_blank(enroll_lm, color=(0, 255, 0))
        img_verify = draw_landmarks_on_blank(verify_lm, color=(255, 0, 255))
        
        # Add labels
        cv2.putText(img_enroll, f"Enrolled[{i}]", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_verify, f"Verify[{j}]", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Combine
        combined = np.hstack([img_enroll, img_verify])
        
        # Add alignment info
        cv2.putText(combined, f"DTW Alignment: step {k}/{len(path)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"Distance: {distance:.4f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Save
        pair_path = os.path.join(pairs_dir, f"pair_{k:04d}_e{i:03d}_v{j:03d}.jpg")
        cv2.imwrite(pair_path, combined)
    
    print(f"Saved pairs to: {pairs_dir}")
    
    # Save path as text
    path_txt = os.path.join(output_dir, f"{username}_dtw_path.txt")
    with open(path_txt, 'w') as f:
        f.write(f"DTW Alignment Path\n")
        f.write(f"User: {username}\n")
        f.write(f"Distance: {distance:.4f}\n")
        f.write(f"Enrolled frames: {len(enrolled_landmarks)}\n")
        f.write(f"Verification frames: {len(verify_landmarks)}\n")
        f.write(f"Path length: {len(path)}\n")
        f.write(f"\nAlignment (enrolled_idx, verify_idx):\n")
        for i, j in path:
            f.write(f"{i:3d}, {j:3d}\n")
    
    print(f"Saved path details: {path_txt}")
    
    return path, distance

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize existing enrollment and new verification data")
    parser.add_argument("username", help="Username to visualize")
    parser.add_argument("--num-landmarks", type=int, default=68, choices=[68, 98, 194, 468],
                       help="Number of landmarks (default: 68)")
    parser.add_argument("--output", default="tests/frames/existing",
                       help="Output directory (default: tests/frames/existing)")
    parser.add_argument("--skip-capture", action="store_true",
                       help="Skip verification capture (use existing landmarks.npy)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"VISUALIZE EXISTING ENROLLMENT DATA")
    print(f"{'='*60}")
    print(f"User: {args.username}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Visualize enrollment
    enrolled_landmarks = visualize_enrollment_data(args.username, output_dir=args.output)
    
    if enrolled_landmarks is None:
        print("Failed to load enrollment data")
        return
    
    # Capture or load verification
    verify_landmarks = None
    
    if args.skip_capture:
        # Try to load existing verification landmarks
        verify_path = os.path.join(args.output, "verification", args.username, "landmarks.npy")
        if os.path.exists(verify_path):
            verify_landmarks = np.load(verify_path)
            print(f"\nLoaded existing verification landmarks: {verify_landmarks.shape}")
        else:
            print(f"\nNo existing verification landmarks found at: {verify_path}")
    else:
        # Capture new verification
        verify_landmarks = capture_verification_for_existing(
            args.username,
            num_landmarks=args.num_landmarks,
            output_dir=args.output
        )
    
    # DTW alignment
    if verify_landmarks is not None:
        dtw_dir = os.path.join(args.output, "dtw", args.username)
        visualize_dtw_alignment_existing(enrolled_landmarks, verify_landmarks, args.username, output_dir=dtw_dir)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"Check folder: {args.output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
