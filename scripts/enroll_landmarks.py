#!/usr/bin/env python3
"""
FR_VERS_JP_2.1 -> ARM64 - Guided Enrollment with LANDMARKS (2 phases)
====================================================================

Goal: reproduce EXACT user workflow of original
original/FR_VERS_JP_2_1/scripts/enroll_landmarks.py

Workflow (must match original):
- Phase 1: guided auto-capture (45 frames) with full GUI guidance
  Original: enrollment_system = GuidedEnrollment(); frames = enrollment_system.enroll()
  ARM64: manual camera loop + detector.process_frame() + enrollment.add_frame()
        auto-add every 5 frames, show overlays via enrollment.draw_guidance()
        stop when enrollment.is_complete == True

- Phase 2: manual validation (like original)
  Re-open camera, show live video, user presses SPACE to validate each frame
  Minimum 5 frames required, 'q' to stop.

Then:
- Fit PCA on collected landmarks sequence
- Save model to ./models/users/<username>.npz using VerificationDTW.save_enrollment()
- Immediate validation: run a short verification against the just-enrolled model
  (kept close to original "validation immediate" behavior)

Constraints:
- Python 3.13
- OpenCV GTK (cv2.imshow)
- No MediaPipe
- ASCII-only console output (no emojis)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow running from workspace root without installation
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

# We no longer rely on the legacy LandmarkDetectorONNX (ONNX/MediaPipe Lite).
# Instead we use the official MediaPipe Python API to obtain 468 facial landmarks
# and head pose directly from the facial transformation matrix. The GuidedEnrollment
# and VerificationDTW classes remain unchanged and are imported from fr_core.
from fr_core import get_config, GuidedEnrollment, VerificationDTW  # type: ignore

# Import MediaPipe Python API and SciPy Rotation to compute Euler angles
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation as R



def _open_camera(camera_id: int, width: int, height: int, backend: int) -> cv2.VideoCapture:
    if backend == 0:
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id, backend)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def _draw_landmarks_points(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    """
    Draw facial landmarks on the frame.

    This helper draws each landmark as a small circle on the BGR image. It
    handles both 2D landmarks (N,2) and 3D landmarks (N,3) by using the first
    two coordinates for drawing. Since MediaPipe returns 468 3D landmarks,
    this function is generic and does not assume a fixed number of points.

    Args:
        frame_bgr: The image in which to draw, in BGR colour space.
        landmarks: An array of shape (N,2) or (N,3) containing landmark
            coordinates in pixel space.
    """
    if landmarks is None:
        return
    for point in landmarks:
        x, y = point[0], point[1]
        # Vert néon (#00FF00) en BGR = (0, 255, 0), taille 2px pour meilleure visibilité
        cv2.circle(frame_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

def _create_mediapipe_detector(model_path: str, num_faces: int = 1,
                               confidence_threshold: float = 0.3) -> vision.FaceLandmarker:
    """
    Create and return a MediaPipe FaceLandmarker detector.

    MediaPipe provides a unified FaceLandmarker task that detects faces,
    returns 478 landmarks (468 face + 10 iris) and provides a 4×4 facial
    transformation matrix for pose estimation. We configure the task to
    output this matrix so that yaw, pitch and roll can be computed via
    SciPy. Only a single face is supported in the current workflow.

    Args:
        model_path: Path to the .task model file (e.g. face_landmarker_v2_with_blendshapes.task).
        num_faces: Maximum number of faces to detect. We set this to 1.
        confidence_threshold: Minimum detection and presence confidence.

    Returns:
        A configured FaceLandmarker instance.
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
        min_face_detection_confidence=confidence_threshold,
        min_face_presence_confidence=confidence_threshold
    )
    return vision.FaceLandmarker.create_from_options(options)

def _process_frame(detector: vision.FaceLandmarker, frame_bgr: np.ndarray) -> Optional[dict]:
    """
    Run the MediaPipe detector on a single frame and extract landmarks and pose.

    The detector returns 478 landmarks; we slice the first 468 landmarks (face
    mesh) and convert them to pixel coordinates. Z coordinates are scaled
    by the image width to keep units consistent with x. Pose is extracted
    from the 4×4 facial transformation matrix using the Euler XZY convention.

    Args:
        detector: A FaceLandmarker instance created by `_create_mediapipe_detector()`.
        frame_bgr: Frame in BGR colour space (H×W×3).

    Returns:
        A dictionary with keys:
            'landmarks': np.ndarray of shape (468, 3)
            'pose': tuple (yaw, pitch, roll) in degrees
        or None if no face is detected.
    """
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        # If no faces detected, return None
        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        # Take first detected face
        landmarks_all = result.face_landmarks[0]
        # Convert to pixel coordinates and keep z scaled by width
        landmarks_468 = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in landmarks_all[:468]
        ], dtype=np.float32)

        # Compute pose using facial_transformation_matrixes
        if result.facial_transformation_matrixes:
            pose_matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            rot_mat = pose_matrix[:3, :3]
            # SciPy Rotation to extract Euler angles in XZY convention
            rot = R.from_matrix(rot_mat)
            angles = rot.as_euler('XZY', degrees=True)
            # Map angles to (yaw, pitch, roll)
            # According to project conventions: yaw = angles[2], pitch = angles[0], roll = angles[1]
            yaw_deg, pitch_deg, roll_deg = angles[2], angles[0], angles[1]
        else:
            yaw_deg, pitch_deg, roll_deg = 0.0, 0.0, 0.0

        return {
            'landmarks': landmarks_468,
            'pose': (yaw_deg, pitch_deg, roll_deg)
        }
    except Exception:
        # On any error, return None to indicate detection failure
        return None


def phase1_auto_capture(
    detector: vision.FaceLandmarker,
    enrollment: GuidedEnrollment,
    camera_id: int,
    width: int,
    height: int,
    backend: int,
    frame_skip_n: int = 5,
    use_preprocessing: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
    """
    Phase 1 : capture automatique avec indication de pose.

    Cette phase fonctionne en boucle jusqu’à ce que le GuidedEnrollment déclare
    que la séquence est complète. À chaque itération nous récupérons
    l’image depuis la webcam, la traitons via MediaPipe pour extraire
    468 points et la pose (yaw, pitch, roll). Si un visage est détecté,
    on affiche les instructions du GuidedEnrollment puis on enregistre
    automatiquement une frame toutes les `frame_skip_n` itérations. La
    fonction renvoie à la fois la liste des landmarks et la liste des poses.

    Args:
        detector: FaceLandmarker instancié par `_create_mediapipe_detector()`.
        enrollment: Instance de GuidedEnrollment gérant la progression des zones.
        camera_id: Index de la caméra.
        width: Largeur désirée de la capture (0 pour laisser par défaut).
        height: Hauteur désirée de la capture (0 pour laisser par défaut).
        backend: Code backend OpenCV (0 pour défaut ou cv2.CAP_V4L2 pour V4L2).
        frame_skip_n: Nombre d’images à ignorer entre deux captures automatiques.
        use_preprocessing: Non utilisé mais gardé pour compatibilité.

    Returns:
        Tuple (phase1_landmarks, phase1_poses) où chaque élément est une liste
        de longueur égale au nombre de frames capturées. Les landmarks ont la
        forme (468, 3) et les poses sont des tuples (yaw, pitch, roll).
    """
    enrollment.reset()
    phase1_landmarks: List[np.ndarray] = []
    phase1_poses: List[Tuple[float, float, float]] = []

    cap = _open_camera(camera_id, width, height, backend)

    window_name = "Phase 1 - Auto Capture (q=quit, r=reset)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    skip = 0
    last_status = ""
    last_status_ts = 0.0

    try:
        while not enrollment.is_complete:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip horizontally so that the user sees a mirror image
            frame = cv2.flip(frame, 1)

            # Run MediaPipe detection on the frame
            result = _process_frame(detector, frame)
            if result is not None:
                landmarks = result["landmarks"]
                pose = result["pose"]
                # Draw guidance based on pose (ceci va créer une nouvelle image)
                display = enrollment.draw_guidance(frame.copy(), pose)
                
                # Dessiner le ROI de détection PAR-DESSUS le guidance
                # (rectangle vert autour du visage détecté)
                h, w = display.shape[:2]
                # Calculer bounding box des landmarks
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x_min, x_max = int(x_coords.min() * w), int(x_coords.max() * w)
                y_min, y_max = int(y_coords.min() * h), int(y_coords.max() * h)
                # Ajouter une marge
                margin = 30
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                # Dessiner rectangle ROI en vert néon ÉPAIS
                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
                # Label en haut à gauche du rectangle
                label_y = max(y_min - 10, 20)
                cv2.putText(display, "ROI DETECTION", (x_min, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                # Auto-capture every N frames when a face is detected
                skip += 1
                if skip >= frame_skip_n:
                    skip = 0
                    enroll_result = enrollment.add_frame(landmarks, pose)
                    # Show last message briefly
                    last_status = enroll_result.get("message", "")
                    last_status_ts = time.time()
                    if enroll_result.get("accepted", False):
                        phase1_landmarks.append(landmarks.copy())
                        phase1_poses.append(pose)
                        summary = enrollment.get_enrollment_summary()
                        print(
                            f"[OK] Frame {len(phase1_landmarks)}/{summary['target_frames']} "
                            f"(FRONTAL:{summary['frontal_frames']} LEFT:{summary['left_frames']} RIGHT:{summary['right_frames']})"
                        )
            else:
                display = frame.copy()
                cv2.putText(
                    display,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Overlay last status message (short-lived)
            if last_status and (time.time() - last_status_ts) < 1.0:
                cv2.putText(
                    display,
                    last_status,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[WARNING] Phase 1 cancelled by user")
                break
            if key == ord("r"):
                print("[INFO] Reset requested")
                enrollment.reset()
                phase1_landmarks.clear()
                phase1_poses.clear()
                skip = 0

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    return phase1_landmarks, phase1_poses


def phase2_manual_validation(
    detector: vision.FaceLandmarker,
    target_count: int,
    camera_id: int,
    width: int,
    height: int,
    backend: int,
    min_required: int = 5,
    use_preprocessing: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
    """
    Phase 2 : validation manuelle (appui sur ESPACE pour valider une image).

    Cette phase demande à l’utilisateur de rester immobile et de valider
    explicitement chaque frame acceptable en appuyant sur la barre
    d’espacement. Les landmarks et les poses sont enregistrés pour chaque
    validation. Le nombre minimal de frames requis est configurable via
    `min_required`.

    Args:
        detector: FaceLandmarker instancié par `_create_mediapipe_detector()`.
        target_count: Nombre de frames à collecter au maximum (issue de Phase 1).
        camera_id: Index de la caméra.
        width: Largeur désirée de la capture.
        height: Hauteur désirée de la capture.
        backend: Code backend OpenCV.
        min_required: Nombre minimal de frames validées pour poursuivre.
        use_preprocessing: Conservé pour compatibilité (non utilisé).

    Returns:
        Tuple (validated_landmarks, validated_poses). Les landmarks ont la
        forme (N, 468, 3) et les poses sont des tuples (yaw, pitch, roll).
    """
    cap = _open_camera(camera_id, width, height, backend)

    window_name = "Phase 2 - Validation (SPACE=accept, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    validated_landmarks: List[np.ndarray] = []
    validated_poses: List[Tuple[float, float, float]] = []
    
    # Variable pour le clic tactile
    button_clicked = False
    
    def mouse_callback(event, x, y, flags, param):
        """Callback pour détection du clic sur bouton"""
        nonlocal button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Zone du bouton en bas de l'écran
            btn_y = param['height'] - 100
            if btn_y <= y <= param['height'] - 20:
                button_clicked = True
    
    # Activer le callback souris pour le tactile
    cv2.setMouseCallback(window_name, mouse_callback, {'height': height})

    try:
        while len(validated_landmarks) < target_count:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Run detection
            result = _process_frame(detector, frame)
            if result is not None:
                landmarks = result["landmarks"]
                pose = result["pose"]
                # Draw landmarks on display frame
                _draw_landmarks_points(display, landmarks)
                cv2.putText(
                    display,
                    "Landmarks OK - Press SPACE to validate",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "No landmarks detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display,
                f"Validated: {len(validated_landmarks)}/{target_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            
            # Bouton tactile en bas de l'écran (format portrait 720x1440)
            btn_height = 80
            btn_y = height - btn_height - 20
            btn_x = 60
            btn_width = 600
            # Dessiner le bouton bleu
            cv2.rectangle(display, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         (168, 9, 9), -1)  # Bleu #0909A8
            cv2.rectangle(display, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         (0, 0, 0), 3)  # Bordure noire
            # Texte centré
            text = "CAPTURER CETTE FRAME"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            text_x = btn_x + (btn_width - text_size[0]) // 2
            text_y = btn_y + (btn_height + text_size[1]) // 2
            cv2.putText(display, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            # Accepter ESPACE ou clic tactile
            if (key == ord(" ") or button_clicked) and result is not None:
                validated_landmarks.append(landmarks.copy())
                validated_poses.append(pose)
                print(f"[OK] Validated frame {len(validated_landmarks)}/{target_count}")
                button_clicked = False  # Reset pour prochain clic
            elif key == ord("q"):
                print("[WARNING] Phase 2 stopped by user")
                break

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    if len(validated_landmarks) < min_required:
        raise ValueError(
            f"Not enough validated frames: {len(validated_landmarks)} (minimum {min_required})"
        )

    return validated_landmarks, validated_poses


def immediate_validation(
    detector: vision.FaceLandmarker,
    verifier: VerificationDTW,
    enrolled_landmarks: np.ndarray,
    enrolled_poses: np.ndarray,
    user_id: str,
    camera_id: int,
    width: int,
    height: int,
    backend: int,
    seconds: float = 3.0,
    min_frames: int = 10,
) -> Tuple[bool, float]:
    """
    Validation immédiate simplifiée : capture quelques secondes de vidéo et
    compare la séquence capturée à la séquence enrolée en utilisant le
    modèle de vérification DTW. Cette version utilise MediaPipe pour
    obtenir 468 landmarks et n’effectue aucune projection supplémentaire.

    Args:
        detector: FaceLandmarker utilisé pour extraire les landmarks et la pose.
        verifier: Instance de VerificationDTW avec PCA pré‑entraînée.
        enrolled_pca_sequence: Séquence enrolée projetée en PCA (n_frames, n_components).
        user_id: Identifiant de l’utilisateur (informational).
        camera_id: Index de la caméra.
        width: Largeur de capture.
        height: Hauteur de capture.
        backend: Backend OpenCV.
        seconds: Durée maximale de capture.
        min_frames: Nombre minimal de frames nécessaires pour valider.

    Returns:
        Tuple (verified, distance) où `verified` est True si la distance est
        inférieure au seuil et `distance` est la distance DTW calculée.
    """
    cap = _open_camera(camera_id, width, height, backend)
    buffer_landmarks: List[np.ndarray] = []
    buffer_poses: List[Tuple[float, float, float]] = []
    start = time.time()
    try:
        while (time.time() - start) < seconds:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            result = _process_frame(detector, frame)
            if result is not None:
                buffer_landmarks.append(result["landmarks"].copy())
                buffer_poses.append(result["pose"])
            time.sleep(0.05)
    finally:
        cap.release()
    
    if len(buffer_landmarks) < min_frames:
        return False, float("inf")
    
    # Prepare probe sequence
    probe_landmarks = np.stack(buffer_landmarks[-min(45, len(buffer_landmarks)):], axis=0).astype(np.float32)
    probe_poses = np.array(buffer_poses[-min(45, len(buffer_poses)):], dtype=np.float32)
    
    print(f"[DEBUG] Captured {len(buffer_landmarks)} frames, using {len(probe_landmarks)} for validation")
    print(f"[DEBUG] Enrolled sequence: {enrolled_landmarks.shape[0]} frames")
    print(f"[DEBUG] Using SPATIAL mode (pose-aware matching)")
    
    # Use spatial mode (verify_auto will route to verify_pose_based)
    is_match, distance, details = verifier.verify_auto(
        probe_landmarks, probe_poses, enrolled_landmarks, enrolled_poses
    )
    
    print(f"[DEBUG] Spatial distance: {distance:.6f}")
    if 'coverage' in details:
        print(f"[DEBUG] Coverage: {details['coverage']:.1f}%")
    
    return is_match, distance


def enroll_with_landmarks(username: str, camera_id: int = 0, use_preprocessing: bool = False) -> bool:
    """
    Procédure d’enrôlement avec 468 landmarks et pose via MediaPipe.

    Cette fonction remplace l’ancien pipeline basé sur 68 points ONNX. Elle
    capture une séquence guidée de frames (Phase 1) puis une séquence
    validée manuellement (Phase 2). Chaque frame fournit 468 points (x, y, z)
    et un triplet (yaw, pitch, roll). Les séquences de landmarks et de poses
    sont sauvegardées dans un fichier .npz via `VerificationDTW.save_enrollment()`.

    Args:
        username: Identifiant de l’utilisateur à enrôler.
        camera_id: Index de la caméra (par défaut 0).
        use_preprocessing: Conservé pour compatibilité (non utilisé avec MediaPipe).

    Returns:
        True si l’enrôlement s’est déroulé jusqu’au bout, False sinon.
    """
    config = get_config()
    print("\n" + "=" * 70)
    print("ENROLLMENT WITH LANDMARKS (POSE-AWARE, 468 points)")
    print("=" * 70)
    print(f"\nUser: {username}")
    print("\nAvantages des 468 landmarks et de la pose :")
    print("  [OK] Capture fine de la géométrie faciale (visage complet)")
    print("  [OK] Robustesse accrue grâce à 468 points 3D")
    print("  [OK] Extraction directe de l’orientation de tête (yaw, pitch, roll)")
    print("\n" + "=" * 70 + "\n")
    # Phase 1
    print("[INFO] Étape 1/2 : capture guidée automatique (45 frames)")
    print("[INFO] Contrôles : 'q' pour annuler, 'r' pour réinitialiser\n")
    time.sleep(1)  # Auto-démarrage pour mode tactile (pas de clavier)
    # Créer le détecteur MediaPipe (chemin modèle potentiellement configurable)
    # Par défaut on cherche le fichier dans config.models_dir/mediapipe
    try:
        models_dir = getattr(config, 'models_dir', Path('.'))
    except Exception:
        models_dir = Path('.')
    # Nom du fichier modèle (modifiez si nécessaire selon votre installation)
    model_path = models_dir / 'mediapipe' / 'face_landmarker_v2_with_blendshapes.task'
    detector = _create_mediapipe_detector(str(model_path), num_faces=1, confidence_threshold=0.3)
    enrollment = GuidedEnrollment()
    # Capture guidée
    phase1_landmarks, phase1_poses = phase1_auto_capture(
        detector=detector,
        enrollment=enrollment,
        camera_id=camera_id,
        width=config.camera_width,
        height=config.camera_height,
        backend=cv2.CAP_V4L2 if config.camera_backend else 0,
        frame_skip_n=5,
        use_preprocessing=use_preprocessing,
    )
    if not enrollment.is_complete:
        print("[ERROR] Enrollment failed or cancelled during Phase 1")
        return False
    if len(phase1_landmarks) == 0:
        print("[ERROR] No frames captured in Phase 1")
        return False
    print(f"\n[OK] Phase 1 complete : {len(phase1_landmarks)} frames captured\n")
    # Phase 2
    print("[INFO] Étape 2/2 : validation manuelle (ESPACE pour accepter chaque frame)")
    print("[WARNING] IMPORTANT : restez en face de la caméra pendant l’extraction des landmarks !")
    print("          Appuyez sur ESPACE pour valider chaque frame")
    print("          Appuyez sur 'q' pour arrêter lorsque vous avez suffisamment de frames\n")
    time.sleep(1)  # Auto-démarrage pour mode tactile
    # Le nombre de frames à valider est limité par l’enrôlement configuré
    target_count = min(len(phase1_landmarks), config.enrollment_n_frames)
    validated_landmarks, validated_poses = phase2_manual_validation(
        detector=detector,
        target_count=target_count,
        camera_id=camera_id,
        width=config.camera_width,
        height=config.camera_height,
        backend=cv2.CAP_V4L2 if config.camera_backend else 0,
        min_required=5,
        use_preprocessing=use_preprocessing,
    )
    # Combiner Phase 1 (diversité guidée) et Phase 2 (qualité manuelle)
    all_landmarks: List[np.ndarray] = phase1_landmarks + validated_landmarks
    all_poses: List[Tuple[float, float, float]] = phase1_poses + validated_poses
    seq = np.stack(all_landmarks, axis=0).astype(np.float32)  # (N, 468, 3)
    poses_arr = np.stack(all_poses, axis=0).astype(np.float32)  # (N, 3)
    print("\n[INFO] Landmark matrix:", seq.shape)
    print(f"[INFO] Phase 1 frames : {len(phase1_landmarks)}")
    print(f"[INFO] Phase 2 frames : {len(validated_landmarks)}")
    print(f"[INFO] Total frames : {seq.shape[0]}")
    # Fit PCA and save model
    verifier = VerificationDTW()
    verifier.fit_pca([seq])
    # Project enrollment sequence to PCA space
    enrolled_pca_sequence = verifier.project_landmarks(seq)
    out_dir = config.users_models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Enregistrer l’enrôlement avec poses
    verifier.save_enrollment(
        user_id=username,
        landmarks_sequence=seq,
        poses_sequence=poses_arr,
        output_dir=out_dir
    )
    model_path = out_dir / f"{username}.npz"
    print(f"\n[OK] Enrollment complete ! Model saved to : {model_path}")
    print(f"[INFO] Template frames : {seq.shape[0]}")
    print("[INFO] Méthode : 468 landmarks + pose-aware")
    # Validation immédiate (optionnelle)
    print("\n" + "=" * 70)
    print("VALIDATION IMMÉDIATE (ne bougez pas !)")
    print("=" * 70)
    print(f"\nValidation test with user '{username}' still in front of the camera.")
    time.sleep(1)  # Auto-démarrage pour mode tactile
    ok, dist = immediate_validation(
        detector=detector,
        verifier=verifier,
        enrolled_landmarks=seq,
        enrolled_poses=poses_arr,
        user_id=username,
        camera_id=camera_id,
        width=config.camera_width,
        height=config.camera_height,
        backend=cv2.CAP_V4L2 if config.camera_backend else 0,
        seconds=3.0,
        min_frames=10,
    )
    print("\n" + "=" * 70)
    print("VALIDATION RESULTAT :")
    print(f"  User : {username}")
    print(f"  Verified : {'YES' if ok else 'NO'}")
    if dist == float("inf"):
        print("  DTW Distance : inf (not enough frames)")
    else:
        print(f"  DTW Distance : {dist:.3f}")
    print("=" * 70)
    if ok:
        print("\n[OK] Validation succeeded.\n")
    else:
        print("\n[WARNING] Validation failed (distance above threshold or insufficient frames).\n")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Guided enrollment with landmarks (2 phases) - ARM64")
    parser.add_argument("username", type=str, help="User ID to enroll")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--use-preprocessing", action="store_true",
                        help="Enable robust preprocessing (ROI+CLAHE+Bilateral) - Reduces variance by 29%%")
    args = parser.parse_args(argv)

    try:
        ok = enroll_with_landmarks(args.username, camera_id=args.camera, use_preprocessing=args.use_preprocessing)
        return 0 if ok else 1
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
