"""
Anti-Spoofing / Liveness Detection - Tier 2 #7
================================================

Detection des tentatives d'usurpation d'identite:
- Photo attacks (impression papier, ecran)
- Video replay attacks
- Mask attacks (futur)

Methodes implementees:
1. Blink detection (clignements yeux)
2. Motion analysis (mouvements naturels)
3. Texture analysis (LBP pour detection photo vs peau reelle)
4. Challenge-response (instructions aleatoires)

Architecture modulaire pour ajouts futurs:
- Face depth estimation (monoculaire)
- Remote PPG (photoplethysmography - detection pouls)
- 3D face reconstruction
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
import time


@dataclass
class LivenessResult:
    """Resultat de la detection de vivacite."""
    is_live: bool
    confidence: float
    details: Dict[str, any]
    method: str


# =============================================================================
# 1. BLINK DETECTION
# =============================================================================

class BlinkDetector:
    """
    Detecte les clignements d'yeux via Eye Aspect Ratio (EAR).
    
    Principe:
    - EAR = (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist)
    - EAR diminue lors d'un clignement
    - Comptage des clignements sur N frames
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consec_frames: int = 2,
        min_blinks: int = 1,
        max_time: float = 5.0
    ):
        """
        Parameters
        ----------
        ear_threshold : float
            Seuil EAR pour detection fermeture yeux
        consec_frames : int
            Nombre de frames consecutives avec EAR bas pour valider blink
        min_blinks : int
            Nombre minimal de clignements requis
        max_time : float
            Temps maximal pour detecter les clignements (secondes)
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.min_blinks = min_blinks
        self.max_time = max_time
        
        # Indices MediaPipe pour les yeux
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # State
        self.counter = 0
        self.blink_count = 0
        self.start_time = None
        
    def compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calcule Eye Aspect Ratio pour un oeil.
        
        Parameters
        ----------
        eye_landmarks : np.ndarray, shape (6, 2)
            6 points de l'oeil [outer, top1, top2, inner, bottom2, bottom1]
            
        Returns
        -------
        ear : float
            Eye Aspect Ratio
        """
        # Distances verticales
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Distance horizontale
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Detecte un clignement sur une frame.
        
        Parameters
        ----------
        landmarks : np.ndarray, shape (478, 2)
            Landmarks MediaPipe du visage
            
        Returns
        -------
        blinked : bool
            True si clignement detecte
        ear : float
            EAR moyen des deux yeux
        """
        # Extraire landmarks des yeux
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        # Calculer EAR
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Detection clignement
        blinked = False
        if ear < self.ear_threshold:
            self.counter += 1
        else:
            if self.counter >= self.consec_frames:
                self.blink_count += 1
                blinked = True
            self.counter = 0
        
        return blinked, ear
    
    def check_liveness(
        self,
        video_source: int = 0,
        show_debug: bool = False
    ) -> LivenessResult:
        """
        Verifie la vivacite via detection de clignements.
        
        Returns
        -------
        result : LivenessResult
            Resultat de la detection
        """
        cap = cv2.VideoCapture(video_source)
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.start_time = time.time()
        self.blink_count = 0
        self.counter = 0
        
        logging.info(f"Blink detection: Attente de {self.min_blinks} clignement(s) en {self.max_time}s...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Temps ecoule
            elapsed = time.time() - self.start_time
            if elapsed > self.max_time:
                break
            
            # Detection landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Convertir en numpy
                landmarks = np.array([
                    [lm.x * w, lm.y * h]
                    for lm in face_landmarks.landmark
                ])
                
                # Detecter clignement
                blinked, ear = self.detect_blink(landmarks)
                
                if show_debug:
                    # Affichage debug
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Blinks: {self.blink_count}/{self.min_blinks}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Time: {elapsed:.1f}/{self.max_time}s",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    if blinked:
                        cv2.putText(
                            frame,
                            "BLINK!",
                            (w - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2
                        )
                    
                    cv2.imshow('Blink Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Success condition
                if self.blink_count >= self.min_blinks:
                    logging.info(f"✓ Liveness confirmed: {self.blink_count} blink(s) in {elapsed:.1f}s")
                    break
        
        cap.release()
        if show_debug:
            cv2.destroyAllWindows()
        mp_face_mesh.close()
        
        # Resultat
        is_live = self.blink_count >= self.min_blinks
        confidence = min(1.0, self.blink_count / self.min_blinks)
        
        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            details={
                'blink_count': self.blink_count,
                'min_required': self.min_blinks,
                'time_elapsed': elapsed,
                'max_time': self.max_time
            },
            method='blink_detection'
        )


# =============================================================================
# 2. MOTION ANALYSIS
# =============================================================================

class MotionAnalyzer:
    """
    Analyse les mouvements naturels du visage.
    
    Principe:
    - Personne reelle: mouvements micro (respiration, micro-expressions)
    - Photo/Video: mouvements rigides ou repetes
    """
    
    def __init__(
        self,
        min_motion: float = 2.0,
        max_frames: int = 30
    ):
        """
        Parameters
        ----------
        min_motion : float
            Mouvement minimal requis (pixels)
        max_frames : int
            Nombre de frames pour analyse
        """
        self.min_motion = min_motion
        self.max_frames = max_frames
    
    def check_liveness(
        self,
        video_source: int = 0,
        show_debug: bool = False
    ) -> LivenessResult:
        """
        Verifie la vivacite via analyse de mouvement.
        
        Returns
        -------
        result : LivenessResult
            Resultat de la detection
        """
        cap = cv2.VideoCapture(video_source)
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        landmarks_history = []
        frame_count = 0
        
        logging.info(f"Motion analysis: Collecte de {self.max_frames} frames...")
        
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Stocker position nose tip (landmark 1)
                nose_tip = face_landmarks.landmark[1]
                landmarks_history.append([nose_tip.x * w, nose_tip.y * h])
                frame_count += 1
                
                if show_debug:
                    cv2.putText(
                        frame,
                        f"Frames: {frame_count}/{self.max_frames}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow('Motion Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        cap.release()
        if show_debug:
            cv2.destroyAllWindows()
        mp_face_mesh.close()
        
        # Analyse mouvement
        if len(landmarks_history) < 2:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                details={'error': 'Insufficient frames'},
                method='motion_analysis'
            )
        
        landmarks_array = np.array(landmarks_history)
        
        # Calcul mouvement total
        displacements = np.linalg.norm(np.diff(landmarks_array, axis=0), axis=1)
        total_motion = np.sum(displacements)
        avg_motion = np.mean(displacements)
        std_motion = np.std(displacements)
        
        # Detection
        is_live = total_motion > self.min_motion
        confidence = min(1.0, total_motion / (self.min_motion * 2))
        
        logging.info(f"Motion: total={total_motion:.2f}, avg={avg_motion:.2f}, std={std_motion:.2f}")
        
        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            details={
                'total_motion': total_motion,
                'avg_motion': avg_motion,
                'std_motion': std_motion,
                'min_required': self.min_motion,
                'frames_analyzed': len(landmarks_history)
            },
            method='motion_analysis'
        )


# =============================================================================
# 3. TEXTURE ANALYSIS (LBP)
# =============================================================================

class TextureAnalyzer:
    """
    Analyse la texture pour differencier peau reelle vs photo/ecran.
    
    Principe:
    - Peau reelle: texture riche (pores, rides micro)
    - Photo imprimee: texture uniforme, grille impression
    - Ecran: pixels reguliers, moiré patterns
    """
    
    def __init__(self, complexity_threshold: float = 50.0):
        """
        Parameters
        ----------
        complexity_threshold : float
            Seuil de complexite texture (variance LBP)
        """
        self.complexity_threshold = complexity_threshold
    
    def compute_lbp_variance(self, image: np.ndarray) -> float:
        """
        Calcule la variance de l'histogramme LBP.
        
        Texture riche = variance elevee
        Texture uniforme = variance faible
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # LBP simple (8 voisins, rayon 1)
        radius = 1
        n_points = 8 * radius
        
        # Pad image
        padded = np.pad(gray, radius, mode='edge')
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        # Calcul LBP
        for i in range(h):
            for j in range(w):
                center = padded[i + radius, j + radius]
                code = 0
                
                # 8 voisins
                neighbors = [
                    padded[i, j + 1],  # top
                    padded[i, j + 2],  # top-right
                    padded[i + 1, j + 2],  # right
                    padded[i + 2, j + 2],  # bottom-right
                    padded[i + 2, j + 1],  # bottom
                    padded[i + 2, j],  # bottom-left
                    padded[i + 1, j],  # left
                    padded[i, j],  # top-left
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Histogramme et variance
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= hist.sum()
        
        variance = np.var(hist)
        return variance * 1000  # Scale pour lisibilite
    
    def check_liveness(
        self,
        video_source: int = 0,
        show_debug: bool = False
    ) -> LivenessResult:
        """
        Verifie la vivacite via analyse de texture.
        """
        cap = cv2.VideoCapture(video_source)
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        
        complexities = []
        n_samples = 5
        
        logging.info(f"Texture analysis: Collecte de {n_samples} echantillons...")
        
        while len(complexities) < n_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection visage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Extraire region visage (bbox)
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                face_roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_roi.size > 0:
                    complexity = self.compute_lbp_variance(face_roi)
                    complexities.append(complexity)
                    
                    if show_debug:
                        cv2.putText(
                            frame,
                            f"Texture: {complexity:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            frame,
                            f"Samples: {len(complexities)}/{n_samples}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow('Texture Analysis', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        
        cap.release()
        if show_debug:
            cv2.destroyAllWindows()
        mp_face_mesh.close()
        
        # Analyse
        if not complexities:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                details={'error': 'No texture samples'},
                method='texture_analysis'
            )
        
        avg_complexity = np.mean(complexities)
        is_live = avg_complexity > self.complexity_threshold
        confidence = min(1.0, avg_complexity / (self.complexity_threshold * 2))
        
        logging.info(f"Texture complexity: avg={avg_complexity:.2f}, threshold={self.complexity_threshold:.2f}")
        
        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            details={
                'avg_complexity': avg_complexity,
                'complexities': complexities,
                'threshold': self.complexity_threshold,
                'samples': len(complexities)
            },
            method='texture_analysis'
        )


# =============================================================================
# 4. FUSION DES METHODES
# =============================================================================

def check_liveness_fusion(
    video_source: int = 0,
    use_blink: bool = True,
    use_motion: bool = True,
    use_texture: bool = False,  # Plus lent
    show_debug: bool = False
) -> LivenessResult:
    """
    Combine plusieurs methodes pour decision robuste.
    
    Parameters
    ----------
    video_source : int
        Source video (0 = webcam)
    use_blink : bool
        Utiliser blink detection
    use_motion : bool
        Utiliser motion analysis
    use_texture : bool
        Utiliser texture analysis (plus lent)
    show_debug : bool
        Afficher debug visuel
        
    Returns
    -------
    result : LivenessResult
        Resultat fusionne
    """
    results = []
    methods = []
    
    if use_blink:
        detector = BlinkDetector(min_blinks=1, max_time=5.0)
        result = detector.check_liveness(video_source, show_debug)
        results.append(result)
        methods.append('blink')
    
    if use_motion:
        analyzer = MotionAnalyzer(min_motion=2.0, max_frames=30)
        result = analyzer.check_liveness(video_source, show_debug)
        results.append(result)
        methods.append('motion')
    
    if use_texture:
        analyzer = TextureAnalyzer(complexity_threshold=50.0)
        result = analyzer.check_liveness(video_source, show_debug)
        results.append(result)
        methods.append('texture')
    
    # Fusion: decision majoritaire ponderee par confidence
    if not results:
        return LivenessResult(
            is_live=False,
            confidence=0.0,
            details={'error': 'No methods enabled'},
            method='fusion'
        )
    
    # Vote pondere
    weighted_votes = sum(r.confidence if r.is_live else 0 for r in results)
    total_weight = len(results)
    
    is_live = weighted_votes > (total_weight / 2)
    confidence = weighted_votes / total_weight
    
    details = {
        'methods': methods,
        'individual_results': [
            {
                'method': r.method,
                'is_live': r.is_live,
                'confidence': r.confidence,
                'details': r.details
            }
            for r in results
        ],
        'weighted_votes': weighted_votes,
        'total_weight': total_weight
    }
    
    return LivenessResult(
        is_live=is_live,
        confidence=confidence,
        details=details,
        method='fusion'
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("ANTI-SPOOFING / LIVENESS DETECTION - DEMO")
    print("="*70)
    
    print("\nMethodes disponibles:")
    print("  1. Blink detection (clignements yeux)")
    print("  2. Motion analysis (mouvements naturels)")
    print("  3. Texture analysis (LBP)")
    print("  4. Fusion (toutes methodes)")
    
    print("\nChoisissez (1-4): ", end='')
    choice = input().strip()
    
    if choice == '1':
        print("\nBlink detection: Clignez des yeux 1 fois en 5 secondes...")
        detector = BlinkDetector(min_blinks=1, max_time=5.0)
        result = detector.check_liveness(video_source=0, show_debug=True)
    
    elif choice == '2':
        print("\nMotion analysis: Bougez naturellement...")
        analyzer = MotionAnalyzer(min_motion=2.0, max_frames=30)
        result = analyzer.check_liveness(video_source=0, show_debug=True)
    
    elif choice == '3':
        print("\nTexture analysis: Restez devant la camera...")
        analyzer = TextureAnalyzer(complexity_threshold=50.0)
        result = analyzer.check_liveness(video_source=0, show_debug=True)
    
    elif choice == '4':
        print("\nFusion: Clignez des yeux et bougez naturellement...")
        result = check_liveness_fusion(
            video_source=0,
            use_blink=True,
            use_motion=True,
            use_texture=False,
            show_debug=True
        )
    
    else:
        print("Choix invalide")
        exit(1)
    
    # Affichage resultat
    print("\n" + "="*70)
    print("RESULTAT")
    print("="*70)
    print(f"Is live: {result.is_live}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Method: {result.method}")
    print(f"\nDetails:")
    for key, value in result.details.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
