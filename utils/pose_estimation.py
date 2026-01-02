"""
Head Pose Estimation from 2D Landmarks

Calcule la pose 3D de la tête (yaw, pitch, roll) à partir des landmarks 2D.
"""

import numpy as np
import cv2


def calculate_head_pose(landmarks_2d, image_shape):
    """
    Calcule la pose 3D de la tête (yaw, pitch, roll) depuis landmarks 2D.
    
    Utilise la méthode PnP (Perspective-n-Point) pour estimer l'orientation 3D
    de la tête en projetant un modèle 3D canonique sur les landmarks 2D détectés.
    
    Args:
        landmarks_2d: np.ndarray, shape (68, 2) ou (478, 2)
                      Landmarks 2D détectés par MediaPipe
        image_shape: tuple (height, width)
                     Dimensions de l'image pour calibration caméra
    
    Returns:
        dict: {
            'yaw': float,   # Rotation gauche/droite (-90° à +90°)
            'pitch': float, # Rotation haut/bas (-90° à +90°)
            'roll': float   # Inclinaison (-180° à +180°)
        }
        Retourne None si le calcul échoue
    """
    
    height, width = image_shape[:2]
    
    # === ÉTAPE 1 : Définir le modèle 3D canonique de la tête ===
    # Points clés du visage en 3D (coordonnées normalisées)
    # Source : modèle générique de tête humaine
    
    # Indices des landmarks clés (pour 68 landmarks dlib-style)
    if landmarks_2d.shape[0] == 68:
        # Landmarks clés : nez, menton, coins yeux, coins bouche
        model_points_indices = [30, 8, 36, 45, 48, 54]  # Nez, menton, yeux, bouche
        
        model_points = np.array([
            (0.0, 0.0, 0.0),         # 30: Bout du nez (origine)
            (0.0, -330.0, -65.0),    # 8:  Menton
            (-225.0, 170.0, -135.0), # 36: Coin externe œil gauche
            (225.0, 170.0, -135.0),  # 45: Coin externe œil droit
            (-150.0, -150.0, -125.0),# 48: Coin gauche bouche
            (150.0, -150.0, -125.0)  # 54: Coin droit bouche
        ], dtype=np.float64)
        
    elif landmarks_2d.shape[0] in [468, 478]:
        # Pour MediaPipe 468/478 landmarks (468 est le nombre brut, 478 inclut iris)
        # Modèle 3D avec proportions anatomiques réalistes (en mm)
        # Utilisation des coins INTERNES des yeux pour meilleure stabilité
        model_points_indices = [1, 152, 133, 362, 61, 291]
        
        model_points = np.array([
            (0.0, 0.0, 0.0),         # 1:   Nez (origine)
            (0.0, 110.0, -15.0),     # 152: Menton (11cm en bas, 1.5cm en arrière au lieu de 4cm)
            (-32.0, -45.0, -25.0),   # 133: Œil gauche INTERNE (3.2cm à gauche, 4.5cm en haut, 2.5cm en arrière)
            (32.0, -45.0, -25.0),    # 362: Œil droit INTERNE (3.2cm à droite, 4.5cm en haut, 2.5cm en arrière)
            (-50.0, 50.0, -15.0),    # 61:  Bouche gauche (5cm à gauche, 5cm en bas, 1.5cm en arrière)
            (50.0, 50.0, -15.0)      # 291: Bouche droite (5cm à droite, 5cm en bas, 1.5cm en arrière)
        ], dtype=np.float64)
    else:
        print(f"Erreur : nombre de landmarks non supporté ({landmarks_2d.shape[0]})")
        return None
    
    # === ÉTAPE 2 : Extraire les landmarks 2D correspondants ===
    image_points = landmarks_2d[model_points_indices].astype(np.float64)
    
    # === ÉTAPE 2b : Calcul du Roll indépendant à partir de l'axe des yeux ===
    # Roll basé uniquement sur l'inclinaison de l'axe inter-pupillaire (stable)
    if landmarks_2d.shape[0] in [468, 478]:
        eye_left = landmarks_2d[133]   # Œil gauche interne
        eye_right = landmarks_2d[362]  # Œil droit interne
        nose = landmarks_2d[1]         # Nez
    else:
        eye_left = landmarks_2d[36]    # Œil gauche
        eye_right = landmarks_2d[45]   # Œil droit
        nose = landmarks_2d[30]        # Nez
    
    # Vecteur inter-oculaire
    eye_vector = eye_right - eye_left
    # Angle par rapport à l'horizontale (en degrés)
    roll_deg = -np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
    
    # === ÉTAPE 2c : Calcul du Pitch indépendant (géométrie 2D) ===
    # Construction du repère du visage :
    # - B = milieu des yeux
    # - A = nez
    # - C = projection orthogonale de A sur verticale passant par B
    # - Axe Y du visage = vecteur BC
    
    eye_center = (eye_left + eye_right) / 2.0  # Point B
    # Point C : même X que B, même Y que A (nez)
    projection_C = np.array([eye_center[0], nose[1]])
    
    # Axe Y du visage (de C vers B)
    axis_Y = eye_center - projection_C
    
    # Angle de l'axe Y par rapport à la verticale de l'image (0, 1)
    # En coordonnées image : Y+ = vers le bas
    axis_Y_angle = np.degrees(np.arctan2(axis_Y[0], axis_Y[1]))
    
    # Pitch = déviation de l'axe Y corrigée du Roll
    # (le Roll fait pencher l'axe Y même sans Pitch)
    pitch_deg = axis_Y_angle - roll_deg
    
    # === ÉTAPE 3 : Matrice de calibration de la caméra ===
    # Approximation : focal length ≈ largeur de l'image
    focal_length = width
    center = (width / 2, height / 2)
    
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Coefficients de distorsion (assumés nuls pour simplification)
    dist_coeffs = np.zeros((4, 1))
    
    # === ÉTAPE 4 : Résoudre PnP (Perspective-n-Point) ===
    # Trouve la rotation et translation qui projette le modèle 3D sur les points 2D
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None
    
    # === ÉTAPE 5 : Convertir le vecteur de rotation en angles d'Euler ===
    # rotation_vector (3,1) → rotation_matrix (3,3) → angles Euler (yaw, pitch, roll)
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extraire seulement Yaw de la matrice de rotation (Roll et Pitch déjà calculés en 2D)
    yaw_deg, _, _ = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # Utiliser Roll et Pitch calculés indépendamment (plus stables)
    # roll_deg et pitch_deg déjà calculés aux étapes 2b et 2c
    
    return {
        'yaw': yaw_deg,      # Rotation gauche/droite (solvePnP)
        'pitch': pitch_deg,  # Rotation haut/bas (géométrie 2D)
        'roll': roll_deg,    # Inclinaison (géométrie 2D)
        'rvec': rotation_vector,
        'tvec': translation_vector,
        'camera_matrix': camera_matrix
    }


def rotation_matrix_to_euler_angles(R):
    """
    Convertit une matrice de rotation 3×3 en angles d'Euler (yaw, pitch, roll).
    
    Convention adaptée pour MediaPipe + OpenCV (système de coordonnées caméra)
    
    Args:
        R: np.ndarray, shape (3, 3)
           Matrice de rotation
    
    Returns:
        tuple: (yaw, pitch, roll) en degrés
    """
    
    # Convention pour système de coordonnées OpenCV/MediaPipe
    # Y vers le bas, Z vers l'avant (profondeur)
    
    # Vérifier gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        # Yaw (rotation autour de Y - gauche/droite)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        # Pitch (rotation autour de X - haut/bas)  
        pitch = np.arctan2(-R[2, 0], sy)
        
        # Roll (rotation autour de Z - inclinaison)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0
    
    # Conversion en degrés
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)
    
    return yaw_deg, pitch_deg, roll_deg


def find_similar_poses(target_pose, enrolled_poses, epsilon=(10.0, 10.0, 10.0)):
    """
    Trouve les frames enrollment ayant une pose similaire à la pose cible.
    
    Args:
        target_pose: dict {'yaw': float, 'pitch': float, 'roll': float}
                     Pose de la frame verification
        enrolled_poses: list of dict
                        Liste des poses des frames enrollment
        epsilon: tuple (eps_yaw, eps_pitch, eps_roll)
                 Tolérance en degrés pour chaque angle
    
    Returns:
        list of int: Indices des frames enrollment avec pose similaire
    """
    
    eps_yaw, eps_pitch, eps_roll = epsilon
    similar_indices = []
    
    for i, enroll_pose in enumerate(enrolled_poses):
        # Distance normalisée dans l'espace des poses
        dist = np.sqrt(
            ((target_pose['yaw'] - enroll_pose['yaw']) / eps_yaw)**2 +
            ((target_pose['pitch'] - enroll_pose['pitch']) / eps_pitch)**2 +
            ((target_pose['roll'] - enroll_pose['roll']) / eps_roll)**2
        )
        
        # Si distance < 1.0, la pose est considérée similaire
        if dist < 1.0:
            similar_indices.append(i)
    
    return similar_indices


def calculate_pose_distance(pose1, pose2):
    """
    Calcule la distance entre deux poses 3D.
    
    Args:
        pose1, pose2: dict {'yaw': float, 'pitch': float, 'roll': float}
    
    Returns:
        float: Distance euclidienne dans l'espace des poses (en degrés)
    """
    
    dist = np.sqrt(
        (pose1['yaw'] - pose2['yaw'])**2 +
        (pose1['pitch'] - pose2['pitch'])**2 +
        (pose1['roll'] - pose2['roll'])**2
    )
    
    return dist


def visualize_pose(image, pose, landmarks_2d=None):
    """
    Dessine la pose 3D sur l'image (axes XYZ).
    
    Args:
        image: np.ndarray, image BGR
        pose: dict {'yaw': float, 'pitch': float, 'roll': float}
        landmarks_2d: np.ndarray, shape (68, 2), optionnel
                      Pour positionner les axes au niveau du nez
    
    Returns:
        np.ndarray: Image avec visualisation de la pose
    """
    
    img_display = image.copy()
    h, w = image.shape[:2]
    
    # Point de référence (centre du nez ou centre image)
    if landmarks_2d is not None and len(landmarks_2d) > 30:
        nose = landmarks_2d[30].astype(int)  # Landmark 30 = bout du nez
        origin = (int(nose[0]), int(nose[1]))
    else:
        origin = (w // 2, h // 2)
    
    # Texte : afficher les angles
    y_offset = 30
    cv2.putText(img_display, f"Yaw: {pose['yaw']:+.1f}deg", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    cv2.putText(img_display, f"Pitch: {pose['pitch']:+.1f}deg", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    cv2.putText(img_display, f"Roll: {pose['roll']:+.1f}deg", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return img_display


if __name__ == "__main__":
    """Bloc de démonstration pour calculer la pose à partir d'un fichier .npz.

    Lorsque ce module est exécuté directement, il tente de charger un fichier
    d'exemple situé dans le dossier models/users et d'afficher quelques
    estimations d'angles. Les chemins sont construits dynamiquement à partir
    de l'emplacement de ce fichier pour éviter les références absolues.
    """
    import sys
    from pathlib import Path

    # Ajouter le projet à sys.path pour permettre les imports relatifs
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("=== Test Calcul Pose 3D ===\n")

    # Chercher un fichier .npz de test (test_v1.npz par défaut)
    models_dir = project_root / "models" / "users"
    test_files = list(models_dir.glob("*.npz"))
    if not test_files:
        print(f"Aucun fichier .npz trouvé dans {models_dir}, rien à tester.")
        sys.exit(0)

    data_file = test_files[0]
    data = np.load(data_file, allow_pickle=True)
    landmarks = data["landmarks"]
    print(f"Chargé {data_file.name} : {landmarks.shape[0]} frames\n")

    # Utiliser la résolution standard 480x640 par défaut
    image_shape = (480, 640)
    frame_indices = [0, min(10, len(landmarks)-1), min(45, len(landmarks)-1), len(landmarks)-1]
    for frame_idx in frame_indices:
        pose = calculate_head_pose(landmarks[frame_idx], image_shape)
        if pose:
            print(f"Frame {frame_idx:02d}: yaw={pose['yaw']:+6.1f}°  "
                  f"pitch={pose['pitch']:+6.1f}°  roll={pose['roll']:+6.1f}°")
        else:
            print(f"Frame {frame_idx:02d}: Échec calcul pose")
    
    print("\n=== Test Find Similar Poses ===\n")
    
    # Calculer toutes les poses
    enrolled_poses = []
    for i in range(landmarks.shape[0]):
        pose = calculate_head_pose(landmarks[i], image_shape)
        if pose:
            enrolled_poses.append(pose)
        else:
            enrolled_poses.append({'yaw': 0, 'pitch': 0, 'roll': 0})
    
    # Trouver poses similaires à frame 10
    target = enrolled_poses[10]
    similar = find_similar_poses(target, enrolled_poses, epsilon=(5.0, 5.0, 5.0))
    
    print(f"Pose cible (frame 10): yaw={target['yaw']:+.1f}° "
          f"pitch={target['pitch']:+.1f}° roll={target['roll']:+.1f}°")
    print(f"Frames similaires (ε=5°): {similar[:10]} ...")
    print(f"Total: {len(similar)} frames sur 90")
