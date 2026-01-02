#!/usr/bin/env python3
"""
Test script to visualize landmarks during enrollment and verification.
Saves frames with landmarks drawn in tests/frames/ folder.
Also visualizes DTW alignment between enrollment and verification sequences.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fr_core import LandmarkDetectorONNX
from fr_core.verification_dtw import VerificationDTW

def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmarks on frame"""
    frame_vis = frame.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame_vis, (int(x), int(y)), radius, color, -1)
        # Draw landmark number for key points
        if i in [30, 33, 36, 39, 45, 48, 54]:  # nose, eyes, mouth corners
            cv2.putText(frame_vis, str(i), (int(x)+3, int(y)-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return frame_vis

def draw_landmark_connections(frame, landmarks, color=(0, 255, 0)):
    """Draw connections between landmarks (face contour, eyes, nose, mouth)"""
    frame_vis = frame.copy()
    
    # Face contour: 0-16
    for i in range(16):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    
    # Eyebrows: 17-21, 22-26
    for start in [17, 22]:
        for i in range(start, start+4):
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[i+1].astype(int))
            cv2.line(frame_vis, pt1, pt2, color, 1)
    
    # Nose: 27-35
    for i in range(27, 30):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    for i in range(31, 35):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    
    # Eyes: 36-41, 42-47
    for start in [36, 42]:
        for i in range(start, start+5):
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[i+1].astype(int))
            cv2.line(frame_vis, pt1, pt2, color, 1)
        # Close eye loop
        pt1 = tuple(landmarks[start+5].astype(int))
        pt2 = tuple(landmarks[start].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    
    # Mouth outer: 48-59
    for i in range(48, 59):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    pt1 = tuple(landmarks[59].astype(int))
    pt2 = tuple(landmarks[48].astype(int))
    cv2.line(frame_vis, pt1, pt2, color, 1)
    
    # Mouth inner: 60-67
    for i in range(60, 67):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(frame_vis, pt1, pt2, color, 1)
    pt1 = tuple(landmarks[67].astype(int))
    pt2 = tuple(landmarks[60].astype(int))
    cv2.line(frame_vis, pt1, pt2, color, 1)
    
    return frame_vis

def capture_and_visualize_enrollment(username, num_landmarks=68, output_dir="tests/frames/enrollment"):
    """Capture frames during enrollment and save with landmarks visualization"""
    print(f"\n=== ENROLLMENT VISUALIZATION: {username} ===")
    print(f"Landmarks: {num_landmarks}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    phase1_dir = os.path.join(output_dir, "phase1")
    phase2_dir = os.path.join(output_dir, "phase2")
    os.makedirs(phase1_dir, exist_ok=True)
    os.makedirs(phase2_dir, exist_ok=True)
    
    # Initialize detector
    detector = LandmarkDetectorONNX(num_landmarks=num_landmarks)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None
    
    print("\n=== PHASE 1: Auto-capture (press ENTER to start) ===")
    input("Press ENTER to start Phase 1...")
    
    phase1_frames = []
    phase1_landmarks = []
    frame_count = 0
    target_frames = 45
    
    while frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        if result is not None:
            landmarks = result['landmarks']
            
            # Save frame with landmarks
            frame_vis = draw_landmark_connections(frame, landmarks, color=(0, 255, 0))
            frame_vis = draw_landmarks(frame_vis, landmarks, color=(0, 255, 0), radius=2)
            
            # Add info text
            cv2.putText(frame_vis, f"Phase 1 - Frame {frame_count+1}/{target_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_vis, f"Landmarks: {num_landmarks}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = os.path.join(phase1_dir, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(output_path, frame_vis)
            
            phase1_frames.append(frame.copy())
            phase1_landmarks.append(landmarks)
            frame_count += 1
            
            # Display
            cv2.imshow("Enrollment Phase 1", frame_vis)
        else:
            # No face detected
            frame_vis = frame.copy()
            cv2.putText(frame_vis, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Enrollment Phase 1", frame_vis)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    print(f"Phase 1 complete: {len(phase1_landmarks)} frames captured")
    
    print("\n=== PHASE 2: Manual validation (press ENTER to start) ===")
    input("Press ENTER to start Phase 2...")
    
    phase2_frames = []
    phase2_landmarks = []
    frame_count = 0
    
    print("Press SPACE to capture, Q to finish (minimum 5 frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        if result is not None:
            landmarks = result['landmarks']
            
            # Visualize
            frame_vis = draw_landmark_connections(frame, landmarks, color=(0, 255, 255))
            frame_vis = draw_landmarks(frame_vis, landmarks, color=(0, 255, 255), radius=2)
            
            cv2.putText(frame_vis, f"Phase 2 - Captured: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_vis, "SPACE=capture, Q=finish (min 5)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Enrollment Phase 2", frame_vis)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                # Capture frame
                output_path = os.path.join(phase2_dir, f"frame_{frame_count:03d}.jpg")
                cv2.imwrite(output_path, frame_vis)
                phase2_frames.append(frame.copy())
                phase2_landmarks.append(landmarks)
                frame_count += 1
                print(f"  Captured frame {frame_count}")
            elif key == ord('q'):
                if frame_count >= 5:
                    break
                else:
                    print(f"  Need at least 5 frames (current: {frame_count})")
        else:
            frame_vis = frame.copy()
            cv2.putText(frame_vis, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Enrollment Phase 2", frame_vis)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                if frame_count >= 5:
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Phase 2 complete: {len(phase2_landmarks)} frames captured")
    
    # Combine phases
    all_landmarks = np.array(phase1_landmarks + phase2_landmarks, dtype=np.float32)
    
    print(f"\nTotal enrollment: {len(all_landmarks)} frames")
    print(f"Saved to: {output_dir}")
    
    return all_landmarks

def capture_and_visualize_verification(username, num_landmarks=68, output_dir="tests/frames/verification"):
    """Capture frames during verification and save with landmarks visualization"""
    print(f"\n=== VERIFICATION VISUALIZATION: {username} ===")
    print(f"Landmarks: {num_landmarks}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = LandmarkDetectorONNX(num_landmarks=num_landmarks)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None
    
    print("\nCapturing verification frames (press ENTER to start, Q to stop)...")
    input("Press ENTER to start verification...")
    
    verify_frames = []
    verify_landmarks = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        if result is not None:
            landmarks = result['landmarks']
            
            # Visualize
            frame_vis = draw_landmark_connections(frame, landmarks, color=(255, 0, 255))
            frame_vis = draw_landmarks(frame_vis, landmarks, color=(255, 0, 255), radius=2)
            
            cv2.putText(frame_vis, f"Verification - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame_vis, "Q=finish", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Save
            output_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(output_path, frame_vis)
            
            verify_frames.append(frame.copy())
            verify_landmarks.append(landmarks)
            frame_count += 1
            
            cv2.imshow("Verification", frame_vis)
        else:
            frame_vis = frame.copy()
            cv2.putText(frame_vis, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Verification", frame_vis)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nVerification complete: {len(verify_landmarks)} frames captured")
    print(f"Saved to: {output_dir}")
    
    if len(verify_landmarks) == 0:
        return None
    
    return np.array(verify_landmarks, dtype=np.float32)

def visualize_dtw_alignment(enrolled_landmarks, verify_landmarks, username, output_dir="tests/frames/dtw_alignment"):
    """Visualize DTW alignment between enrollment and verification sequences"""
    print(f"\n=== DTW ALIGNMENT VISUALIZATION ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DTW verifier
    verifier = VerificationDTW()
    
    # Prepare sequences (flatten landmarks to 1D)
    enrolled_seq = enrolled_landmarks.reshape(len(enrolled_landmarks), -1)
    verify_seq = verify_landmarks.reshape(len(verify_landmarks), -1)
    
    # Compute DTW with path
    from dtaidistance import dtw
    distance = dtw.distance(enrolled_seq, verify_seq)
    path = dtw.warping_path(enrolled_seq, verify_seq)
    
    print(f"DTW Distance: {distance:.4f}")
    print(f"Alignment path length: {len(path)}")
    print(f"Enrolled frames: {len(enrolled_landmarks)}")
    print(f"Verification frames: {len(verify_landmarks)}")
    
    # Save alignment visualization
    # Create alignment map image
    h = max(len(enrolled_landmarks), len(verify_landmarks))
    w = max(len(enrolled_landmarks), len(verify_landmarks))
    alignment_img = np.zeros((h*10, w*10, 3), dtype=np.uint8)
    
    # Draw grid
    for i in range(0, h*10, 10):
        cv2.line(alignment_img, (0, i), (w*10, i), (50, 50, 50), 1)
    for j in range(0, w*10, 10):
        cv2.line(alignment_img, (j, 0), (j, h*10), (50, 50, 50), 1)
    
    # Draw alignment path
    for i, j in path:
        x = j * 10 + 5
        y = i * 10 + 5
        cv2.circle(alignment_img, (x, y), 3, (0, 255, 255), -1)
    
    # Draw lines between consecutive points in path
    for k in range(len(path)-1):
        i1, j1 = path[k]
        i2, j2 = path[k+1]
        pt1 = (j1*10+5, i1*10+5)
        pt2 = (j2*10+5, i2*10+5)
        cv2.line(alignment_img, pt1, pt2, (0, 200, 200), 1)
    
    # Add labels
    cv2.putText(alignment_img, "Enrollment (Y-axis)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(alignment_img, "Verification (X-axis)", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(alignment_img, f"DTW Distance: {distance:.4f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    alignment_path = os.path.join(output_dir, f"{username}_dtw_alignment.jpg")
    cv2.imwrite(alignment_path, alignment_img)
    print(f"Saved alignment map to: {alignment_path}")
    
    # Save aligned frame pairs
    print("\nCreating aligned frame pairs...")
    pairs_dir = os.path.join(output_dir, "aligned_pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    
    # Sample every Nth pair to avoid too many images
    step = max(1, len(path) // 20)  # Max 20 pairs
    
    for k in range(0, len(path), step):
        i, j = path[k]
        
        # Create visualization showing which enrollment frame maps to which verification frame
        info_img = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Alignment pair {k}/{len(path)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_img, f"Enrolled frame: {i}/{len(enrolled_landmarks)-1}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, f"Verification frame: {j}/{len(verify_landmarks)-1}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(info_img, f"DTW stretching: enrollment[{i}] <-> verify[{j}]", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        pair_path = os.path.join(pairs_dir, f"pair_{k:03d}_e{i:03d}_v{j:03d}.jpg")
        cv2.imwrite(pair_path, info_img)
    
    print(f"Saved {len(range(0, len(path), step))} aligned pairs to: {pairs_dir}")
    
    # Save alignment data as text
    alignment_txt = os.path.join(output_dir, f"{username}_dtw_path.txt")
    with open(alignment_txt, 'w') as f:
        f.write(f"DTW Alignment Path\n")
        f.write(f"Distance: {distance:.4f}\n")
        f.write(f"Enrolled frames: {len(enrolled_landmarks)}\n")
        f.write(f"Verification frames: {len(verify_landmarks)}\n")
        f.write(f"\nAlignment path (enrolled_idx, verify_idx):\n")
        for i, j in path:
            f.write(f"{i}, {j}\n")
    
    print(f"Saved alignment path to: {alignment_txt}")
    
    return path, distance

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize landmarks during enrollment and verification")
    parser.add_argument("username", help="Username for enrollment/verification")
    parser.add_argument("--num-landmarks", type=int, default=68, choices=[68, 98, 194, 468],
                       help="Number of landmarks to use (default: 68)")
    parser.add_argument("--mode", choices=["enroll", "verify", "both", "dtw"], default="both",
                       help="Mode: enroll only, verify only, both, or dtw alignment (default: both)")
    parser.add_argument("--output", default="tests/frames",
                       help="Output directory for frames (default: tests/frames)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"LANDMARK VISUALIZATION TEST")
    print(f"{'='*60}")
    print(f"User: {args.username}")
    print(f"Landmarks: {args.num_landmarks}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    enrolled_landmarks = None
    verify_landmarks = None
    
    if args.mode in ["enroll", "both"]:
        enroll_dir = os.path.join(args.output, "enrollment", args.username)
        enrolled_landmarks = capture_and_visualize_enrollment(
            args.username, 
            num_landmarks=args.num_landmarks,
            output_dir=enroll_dir
        )
        
        if enrolled_landmarks is not None:
            # Save enrollment landmarks
            np.save(os.path.join(enroll_dir, "landmarks.npy"), enrolled_landmarks)
            print(f"Saved enrollment landmarks to: {os.path.join(enroll_dir, 'landmarks.npy')}")
    
    if args.mode in ["verify", "both"]:
        verify_dir = os.path.join(args.output, "verification", args.username)
        verify_landmarks = capture_and_visualize_verification(
            args.username,
            num_landmarks=args.num_landmarks,
            output_dir=verify_dir
        )
        
        if verify_landmarks is not None:
            # Save verification landmarks
            np.save(os.path.join(verify_dir, "landmarks.npy"), verify_landmarks)
            print(f"Saved verification landmarks to: {os.path.join(verify_dir, 'landmarks.npy')}")
    
    if args.mode == "dtw":
        # Load saved landmarks
        enroll_dir = os.path.join(args.output, "enrollment", args.username)
        verify_dir = os.path.join(args.output, "verification", args.username)
        
        enroll_path = os.path.join(enroll_dir, "landmarks.npy")
        verify_path = os.path.join(verify_dir, "landmarks.npy")
        
        if os.path.exists(enroll_path) and os.path.exists(verify_path):
            enrolled_landmarks = np.load(enroll_path)
            verify_landmarks = np.load(verify_path)
            print(f"Loaded enrollment landmarks: {enrolled_landmarks.shape}")
            print(f"Loaded verification landmarks: {verify_landmarks.shape}")
        else:
            print(f"Error: Landmarks files not found")
            print(f"  Enrollment: {enroll_path}")
            print(f"  Verification: {verify_path}")
            return
    
    # DTW alignment visualization
    if args.mode in ["both", "dtw"] and enrolled_landmarks is not None and verify_landmarks is not None:
        dtw_dir = os.path.join(args.output, "dtw_alignment", args.username)
        visualize_dtw_alignment(enrolled_landmarks, verify_landmarks, args.username, output_dir=dtw_dir)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
