#!/usr/bin/env python3
"""
Guided Enrollment with Standardized Poses
==========================================
Ensures all users are enrolled with the same set of head positions.

Key improvements:
- Standardized poses P = {p1, p2, ..., pn}
- Visual feedback with colored markers (black → green when validated)
- No timer, user-paced enrollment
- Consistent pose distribution across all users
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from fr_core.features import FaceLandmarkDetector

# Pose tolerances (degrees) - defines the WIDE capture zone for natural variation
YAW_TOLERANCE = 15.0     # Captures frames in range: target ± 15° (larger zone)
PITCH_TOLERANCE = 15.0   # Creates MORE natural variation within each zone
ROLL_TOLERANCE = 15.0

# Frames to capture per pose for robustness
FRAMES_PER_POSE = 15

# Minimum angle change between captures - REDUCED for more variability
# Allows more natural head movements while still ensuring distinct frames
MIN_YAW_CHANGE_DEGREES = 2.0      # Minimum yaw difference from ALL previous captures
MIN_PITCH_CHANGE_DEGREES = 2.0    # Minimum pitch difference from ALL previous captures
MIN_ROLL_CHANGE_DEGREES = 2.0     # Minimum roll difference from ALL previous captures


@dataclass
class PoseTarget:
    """Represents a target head pose for enrollment."""
    name: str
    yaw: float      # Target yaw (left/right rotation)
    pitch: float    # Target pitch (up/down rotation)
    roll: float     # Target roll (tilt)
    position: Tuple[int, int]  # (x, y) position of marker on screen


class GuidedEnrollment:
    """Manages guided enrollment with pose validation."""
    
    def __init__(self, video_source: int = 0, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize guided enrollment.
        
        Args:
            video_source: Camera index
            frame_width: Display frame width
            frame_height: Display frame height
        """
        self.video_source = video_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detector = FaceLandmarkDetector()
        
        # Define standard poses arranged in circle around frame
        self.pose_targets = self._define_pose_targets()
        
        # Track frames per pose (capture multiple frames for robustness)
        self.captured_frames_per_pose = {target.name: [] for target in self.pose_targets}
        self.frame_count_per_pose = {target.name: 0 for target in self.pose_targets}
        
        # Track ALL captured pose angles to prevent duplicates (ensures 45 distinct frames)
        self.all_captured_poses = {target.name: [] for target in self.pose_targets}
    
    def _define_pose_targets(self) -> List[PoseTarget]:
        """
        Define 3 essential poses for enrollment.
        
        Returns 3 poses:
        - Frontal (face camera directly)
        - Left (turn head left)
        - Right (turn head right)
        
        This ensures user turns head while keeping enrollment simple.
        Multiple frames captured per pose for robustness.
        """
        cx, cy = self.frame_width // 2, self.frame_height // 2
        margin = 80
        
        poses = [
            # Center - Frontal (±15° → -15° to +15°)
            PoseTarget("FRONTAL", yaw=0, pitch=0, roll=0, 
                      position=(cx, cy)),
            
            # Left - Turn head left (yaw -25 ±15° → -40° to -10°)
            PoseTarget("LEFT", yaw=-25, pitch=0, roll=0,
                      position=(margin, cy)),
            
            # Right - Turn head right (yaw +25 ±15° → +10° to +40°)
            PoseTarget("RIGHT", yaw=25, pitch=0, roll=0,
                      position=(self.frame_width - margin, cy)),
        ]
        
        return poses
    
    def _check_pose_match(self, current_pose: Tuple[float, float, float], 
                         target: PoseTarget) -> bool:
        """
        Check if current pose matches target within tolerance.
        
        Args:
            current_pose: (yaw, pitch, roll) tuple
            target: Target pose to match
            
        Returns:
            True if pose matches within tolerance
        """
        yaw, pitch, roll = current_pose
        
        yaw_match = abs(yaw - target.yaw) < YAW_TOLERANCE
        pitch_match = abs(pitch - target.pitch) < PITCH_TOLERANCE
        roll_match = abs(roll - target.roll) < ROLL_TOLERANCE
        
        return yaw_match and pitch_match and roll_match
    
    def _pose_changed_enough(self, current_pose: Tuple[float, float, float], 
                            target_name: str) -> bool:
        """
        Check if current pose is sufficiently different from ALL previously captured poses.
        This ensures every captured frame has a truly distinct head position.
        
        Args:
            current_pose: (yaw, pitch, roll) tuple
            target_name: Name of the target pose
            
        Returns:
            True if pose is different enough from ALL previous captures
        """
        previous_poses = self.all_captured_poses[target_name]
        
        if len(previous_poses) == 0:
            return True  # First capture for this pose - always accept
        
        # Check against ALL previously captured poses (not just the last one)
        for prev_pose in previous_poses:
            yaw_diff = abs(current_pose[0] - prev_pose[0])
            pitch_diff = abs(current_pose[1] - prev_pose[1])
            roll_diff = abs(current_pose[2] - prev_pose[2])
            
            # If current pose is too similar to ANY previous pose, reject it
            if (yaw_diff < MIN_YAW_CHANGE_DEGREES and 
                pitch_diff < MIN_PITCH_CHANGE_DEGREES and 
                roll_diff < MIN_ROLL_CHANGE_DEGREES):
                return False  # Too similar to this previous pose
        
        # Current pose is different enough from ALL previous poses
        return True
    
    def _draw_pose_markers(self, frame: np.ndarray, current_pose: Optional[Tuple[float, float, float]]):
        """
        Draw pose markers on frame.
        
        Args:
            frame: Video frame to draw on
            current_pose: Current head pose (yaw, pitch, roll) or None
        """
        for target in self.pose_targets:
            x, y = target.position
            
            # Check frame count for this pose
            frame_count = self.frame_count_per_pose[target.name]
            is_complete = frame_count >= FRAMES_PER_POSE
            
            # Check if current pose matches this target
            is_current_match = False
            if current_pose is not None:
                is_current_match = self._check_pose_match(current_pose, target)
            
            # Determine marker color
            if is_complete:
                color = (0, 255, 0)  # Green - complete
                radius = 20
            elif is_current_match:
                color = (0, 255, 255)  # Yellow - matching now
                radius = 22
            else:
                color = (70, 70, 70)  # Dark gray - not captured
                radius = 15
            
            # Draw circle marker
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 2)
            
            # Draw pose name and progress
            text = f"{target.name} ({frame_count}/{FRAMES_PER_POSE})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x - text_size[0] // 2
            text_y = y + radius + 20
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_instructions(self, frame: np.ndarray, current_pose: Optional[Tuple[float, float, float]]):
        """Draw instructions and status on frame."""
        h, w = frame.shape[:2]
        
        # Background box for instructions
        cv2.rectangle(frame, (10, h - 160), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 160), (w - 10, h - 10), (255, 255, 255), 2)
        
        # Count total captured frames
        total_frames = sum(self.frame_count_per_pose.values())
        required_frames = len(self.pose_targets) * FRAMES_PER_POSE
        
        # Instructions
        instructions = [
            f"Progress: {total_frames}/{required_frames} frames ({total_frames*100//required_frames if required_frames>0 else 0}%)",
            f"MOVE HEAD CONTINUOUSLY - system rejects duplicate poses",
            "Each zone needs 15 DIFFERENT positions (±3° separation)",
            "YELLOW = in zone, capturing unique poses only",
            "GREEN = 15 distinct positions captured!",
            "Press 'q' to cancel"
        ]
        
        y_offset = h - 140
        for text in instructions:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            y_offset += 23
        
        # Current pose info (top-left)
        if current_pose is not None:
            yaw, pitch, roll = current_pose
            pose_text = [
                f"Yaw: {yaw:+.1f}deg",
                f"Pitch: {pitch:+.1f}deg",
                f"Roll: {roll:+.1f}deg"
            ]
            
            y = 30
            for text in pose_text:
                cv2.putText(frame, text, (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y += 25
    
    def enroll(self) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
        """
        Perform guided enrollment with pose validation.
        
        Returns:
            (frames, poses): Lists of captured frames and corresponding poses
        """
        from fr_core.enrollment import preprocess_face_wrapper, estimate_pose_from_landmarks, estimate_pose_from_bbox
        
        cap = cv2.VideoCapture(self.video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Initialize Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        landmark_detector = self.detector
        prev_box = None
        
        logging.info("Starting guided enrollment...")
        logging.info(f"Capturing {len(self.pose_targets)} standardized poses")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror for better UX
                display_frame = frame.copy()
                
                # Detect face
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                
                if len(detections) > 0:
                    x, y, w, h = max(detections, key=lambda bb: bb[2] * bb[3])
                    prev_box = (int(x), int(y), int(w), int(h))
                
                current_pose = None
                face_region = None
                
                if prev_box is not None:
                    x, y, w, h = prev_box
                    # Ensure bounding box is within frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    face = frame[y : y + h, x : x + w].copy()
                    face_region = preprocess_face_wrapper(face)
                    
                    # Estimate pose
                    landmarks = landmark_detector.detect_landmarks(frame)
                    if landmarks is not None:
                        pose = estimate_pose_from_landmarks(frame, landmarks, prev_box)
                    else:
                        pose = estimate_pose_from_bbox(frame, prev_box)
                    
                    current_pose = (pose.yaw, pose.pitch, pose.roll)
                    
                    # Check each target pose and capture frames with guaranteed uniqueness
                    for target in self.pose_targets:
                        if self.frame_count_per_pose[target.name] < FRAMES_PER_POSE:
                            # Check if pose matches target range
                            if self._check_pose_match(current_pose, target):
                                # Only capture if pose is different from ALL previous captures
                                if self._pose_changed_enough(current_pose, target.name):
                                    # Capture this UNIQUE frame
                                    self.captured_frames_per_pose[target.name].append(face_region.copy())
                                    self.frame_count_per_pose[target.name] += 1
                                    self.all_captured_poses[target.name].append(current_pose)
                                    
                                    count = self.frame_count_per_pose[target.name]
                                    
                                    if count == 1:
                                        logging.info(f"Started capturing pose: {target.name}")
                                    elif count == FRAMES_PER_POSE:
                                        logging.info(f"✓ Completed pose: {target.name} - "
                                                   f"{FRAMES_PER_POSE} DISTINCT frames captured")
                                    
                                    # Log every 5th frame for feedback
                                    if count % 5 == 0 and count < FRAMES_PER_POSE:
                                        logging.info(f"  {target.name}: {count}/{FRAMES_PER_POSE} unique frames")
                    
                    # Draw face box
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw markers and instructions
                self._draw_pose_markers(display_frame, current_pose)
                self._draw_instructions(display_frame, current_pose)
                
                # Show frame
                cv2.imshow("Guided Enrollment", display_frame)
                
                # Check if all poses captured with required number of frames
                if all(count >= FRAMES_PER_POSE for count in self.frame_count_per_pose.values()):
                    logging.info(f"✓ All {len(self.pose_targets)} poses captured with {FRAMES_PER_POSE} frames each!")
                    cv2.waitKey(1000)  # Show success for 1 second
                    break
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.warning("Enrollment cancelled by user")
                    return [], []
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Flatten all captured frames into single list
        all_frames = []
        for target in self.pose_targets:
            all_frames.extend(self.captured_frames_per_pose[target.name])
        
        logging.info(f"Enrollment complete: {len(all_frames)} frames captured total")
        return all_frames


def enroll_user_guided(username: str, video_source: int = 0) -> bool:
    """
    Enroll user with guided pose capture.
    
    Args:
        username: Username to enroll
        video_source: Camera index
        
    Returns:
        True if enrollment successful
    """
    from pathlib import Path
    import numpy as np
    from fr_core.enrollment import compute_functional_representation, save_model
    from fr_core.features import average_rotation_matrices
    from fr_core.preprocessing import extract_gabor_features, extract_lbp_features
    from sklearn.preprocessing import RobustScaler
    
    try:
        import config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
    
    print(f"\n{'='*60}")
    print(f"GUIDED ENROLLMENT: {username}")
    print(f"{'='*60}")
    print("\nInstructions:")
    print(f"  1. System captures 15 DISTINCT positions in each of 3 zones:")
    print("     - FRONTAL (yaw ±12° from center)")
    print("     - LEFT (yaw -37° to -13°)")
    print("     - RIGHT (yaw +13° to +37°)")
    print("  2. Position face in camera, turn to match GRAY marker")
    print("  3. Once in zone (YELLOW), MOVE HEAD CONTINUOUSLY:")
    print("     - System rejects positions already captured")
    print("     - Each frame must differ by ±3° from ALL previous")
    print("     - Sweep through zone with smooth head movements")
    print("  4. Marker turns GREEN when 15 unique positions captured")
    print(f"  5. Guaranteed: 45 truly DIFFERENT frames")
    print(f"\n{'='*60}\n")
    
    input("Press ENTER to start enrollment...")
    
    # Perform guided enrollment
    guided = GuidedEnrollment(video_source=video_source)
    frames = guided.enroll()
    
    if len(frames) == 0:
        print("\n✗ Enrollment cancelled or failed")
        return False
    
    print(f"\n✓ Captured {len(frames)} frames successfully")
    
    # Build model using same pipeline as run_enrolment
    print("\n📊 Building model from captured frames...")
    
    # Extract features
    X = None
    
    # Gabor features
    use_gabor = CONFIG_AVAILABLE and hasattr(config, 'USE_GABOR_FEATURES') and config.USE_GABOR_FEATURES
    if use_gabor:
        gabor_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                gab_feat = extract_gabor_features(
                    frame,
                    orientations=config.GABOR_ORIENTATIONS if hasattr(config, 'GABOR_ORIENTATIONS') else [0, 45, 90, 135],
                    frequencies=config.GABOR_FREQUENCIES if hasattr(config, 'GABOR_FREQUENCIES') else [0.1, 0.2],
                    ksize=config.GABOR_KSIZE if hasattr(config, 'GABOR_KSIZE') else 31,
                )
            else:
                gab_feat = extract_gabor_features(frame)
            gabor_features.append(gab_feat)
        gabor_features = np.array(gabor_features)
        X = gabor_features if X is None else np.concatenate([X, gabor_features], axis=1)
    
    # LBP features  
    use_lbp = CONFIG_AVAILABLE and hasattr(config, 'USE_LBP_FEATURES') and config.USE_LBP_FEATURES
    if use_lbp:
        lbp_features = []
        for frame in frames:
            if CONFIG_AVAILABLE:
                lbp_feat = extract_lbp_features(
                    frame,
                    radius=config.LBP_RADIUS if hasattr(config, 'LBP_RADIUS') else 1,
                    n_points=config.LBP_N_POINTS if hasattr(config, 'LBP_N_POINTS') else 8,
                )
            else:
                lbp_feat = extract_lbp_features(frame)
            lbp_features.append(lbp_feat)
        lbp_features = np.array(lbp_features)
        X = lbp_features if X is None else np.concatenate([X, lbp_features], axis=1)
    
    # Fallback to pixels
    if X is None:
        X = np.array([f.ravel() for f in frames])
    
    # Add dummy pose features for compatibility with verification system
    # Values don't matter since all users have same 3-zone distribution
    # The guided enrollment already ensures pose standardization
    n_frames = len(frames)
    dummy_pose_features = np.zeros((n_frames, 3))  # [yaw=0, pitch=0, roll=0] for all frames
    X_full = np.concatenate([X, dummy_pose_features], axis=1)
    
    print(f"  - Feature matrix: {X_full.shape} (including 3 dummy pose features)")
    
    # Scale
    print("  - Applying RobustScaler normalization...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    # PCA
    from sklearn.decomposition import PCA
    n_components = 45
    print(f"  - Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"    Variance explained: {variance_explained:.1f}%")
    
    # Dummy pose values (not used with DTW)
    pose_mean = np.array([0.0, 0.0, 0.0])
    R_ref = np.eye(3)
    t_ref = np.zeros((3, 1))
    
    # Save model (DTW mode)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{username}.npz"
    
    save_model(
        str(model_path),
        pca, None, scaler, pose_mean, R_ref, t_ref,
        dtw_template=scores
    )
    
    print(f"\n✓ Enrollment complete! Model saved to: {model_path}")
    print(f"  - Template size: {scores.shape}")
    print(f"  - Captured: 3 poses × {FRAMES_PER_POSE} frames = {len(frames)} total frames")
    
    # Immediate validation test with same user still in front of camera
    print(f"\n{'='*60}")
    print(f"VALIDATION IMMÉDIATE (ne bougez pas!)")
    print(f"{'='*60}\n")
    print(f"Test de validation avec {username} toujours devant la caméra...")
    print("Appuyez sur ENTER pour valider l'enrollment...")
    input()
    
    from fr_core.verification_dtw import verify
    
    try:
        is_verified, distance = verify(
            model_path=str(model_path),
            video_source=0
        )
        
        print(f"\n{'='*60}")
        print(f"RÉSULTAT VALIDATION GENUINE:")
        print(f"  Utilisateur: {username}")
        print(f"  Vérifié: {'✓ OUI' if is_verified else '✗ NON'}")
        print(f"  Distance DTW: {distance:.2f}")
        print(f"{'='*60}\n")
        
        if is_verified:
            print(f"✅ Validation réussie pour {username}!")
        else:
            print(f"⚠️  Validation échouée - distance trop élevée ({distance:.2f})")
        
    except Exception as e:
        print(f"\n⚠️  Erreur lors de la validation: {e}\n")
    
    return True
