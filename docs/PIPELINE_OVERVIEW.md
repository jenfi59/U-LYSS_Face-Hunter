# Pipeline d'exécution

Ce document décrit le pipeline complet de **D‑Face Hunter** dans sa version 1.2.1.  Le système est conçu pour fonctionner en mode entièrement local (on‑premise) sur des plateformes **ARM64** et utilise la détection de visage et l’extraction de repères 3D fournies par l’API MediaPipe.  Tous les chemins mentionnés sont relatifs à la racine du projet.

## Étapes principales

1. **Capture vidéo** : les scripts interactifs (`enroll_interactive.py` et `verify_interactive.py`) ouvrent un flux vidéo via OpenCV.  La résolution de la caméra et les paramètres d’enrôlement sont définis dans les classes `Config` et `ConfigSequential` (`src/fr_core/config.py`), ou peuvent être passés en ligne de commande.

2. **Détection du visage** : chaque image est passée à la classe `LandmarkDetectorONNX` (dans `src/fr_core/landmark_onnx.py`), laquelle encapsule l’API MediaPipe Face Landmarker.  Le détecteur retourne :
   - un rectangle englobant (`bbox`) si un visage est présent ;
   - un tableau de **478 repères 3D** (468 points du maillage facial + 10 points d’iris) dans l’ordre standard MediaPipe ;
   - la matrice de transformation faciale 4×4 (`facial_transformation_matrixes`) permettant de calculer les angles de pose (yaw, pitch, roll).

3. **Calcul de la pose** : les angles de la tête sont extraits à partir de la matrice de transformation.  Ils sont calibrés en soustrayant les offsets sauvegardés dans `config/camera_calibration.json` afin que la pose neutre (tête droite, caméra bien placée) corresponde à `(0°, 0°, 0°)`.

4. **Séquençage** : lors de l’enrôlement, les repères et poses sont accumulés sur plusieurs images.  Par défaut, 45 images sont capturées automatiquement (15 frontales, 15 gauche, 15 droite), puis au moins 5 images sont validées manuellement.  Ces séquences sont sauvegardées dans `models/users/<username>.npz`.  Les clés du fichier `.npz` sont :
   - `landmarks` : matrice `(N_frames, 478, 3)` des repères 3D ;
   - `poses` : matrice `(N_frames, 3)` des angles (yaw, pitch, roll) ;
   - `metadata` : dictionnaire contenant `version`, `num_landmarks`, `detector`, etc.

5. **Réduction de dimension (PCA)** : lors de la vérification 1:1 ou 1:N, les séquences d’enrôlement sont réduites via une analyse en composantes principales (PCA) afin de projeter les repères dans un espace de dimension réduite.  Les coefficients PCA sont stockés dans chaque fichier `.npz`.

6. **Vérification** : quatre modes de comparaison sont disponibles via la classe `VerificationDTW` :
   - **Temporal** : comparaison via Dynamic Time Warping (DTW) sur l’ensemble des repères réduits (sensible à la longueur de la séquence) ;
   - **Spatial** : comparaison frame‑par‑frame en filtrant les repères par la pose ;
   - **Spatiotemporel** : combinaison linéaire des distances temporelles et spatiales ;
   - **Séquentiel multi‑critères** : nouveau mode introduit en v1.2 qui combine plusieurs groupes de repères, des ratios anthropométriques et une marge relative pour distinguer clairement les meilleurs candidats.

Le mode est choisi via l’attribut `matching_mode` du `Config` ou `ConfigSequential` et peut être modifié sans toucher au code.

## Détails sur la détection

Le détecteur MediaPipe Face Landmarker est initialisé dans `src/fr_core/landmark_onnx.py`.  Par défaut :

```python
from mediapipe.tasks.python import vision
base_options = python.BaseOptions(model_asset_path="models/mediapipe/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3
)
detector = vision.FaceLandmarker.create_from_options(options)
```

Le fichier `face_landmarker.task` (≈3,7 MB) doit être téléchargé depuis les serveurs MediaPipe (voir la section Installation du `README.md`).  Il contient le modèle TensorFlow Lite et les métadonnées nécessaires.

## Sauvegarde et format des modèles

Les utilisateurs sont enregistrés dans le dossier `models/users/` sous forme de fichiers `.npz`.  Chaque fichier contient :

| Clé        | Description                                    |
|------------|-------------------------------------------------|
| `landmarks`| Tableau `(N, 478, 3)` des repères 3D           |
| `poses`    | Tableau `(N, 3)` des angles (yaw, pitch, roll) |
| `pca`      | Objet PCA sérialisé (pour la réduction)        |
| `scaler`   | StandardScaler (normalisation des repères)     |
| `metadata` | Dictionnaire : version, nombre de repères, etc.|

Le script `scripts/enroll_landmarks.py` se charge de créer ce fichier lors de l’enrôlement, en ajoutant également la séquence de poses et les paramètres PCA.

## Fonctionnement de la vérification

Lors de l’appel à `verify_auto()` ou à `verify_multi_gallery()`, les étapes suivantes sont exécutées :

1. **Chargement du modèle** : le fichier `.npz` est lu et ses composantes (`landmarks`, `poses`, `pca`, `scaler`) sont extraites.

2. **Projection** : les repères et poses de la séquence de test sont projetés dans le même espace réduit via `pca.transform()` et normalisés par `scaler` si nécessaire.

3. **Comparaison** : selon le mode :
   - *temporal* : distance DTW sur la séquence projetée ;
   - *spatial* : distances euclidiennes frame‑par‑frame après filtrage par la pose ;
   - *spatiotemporel* : combinaison linéaire (poids configurable) de DTW et Spatial ;
   - *séquentiel* : calcul d’un score composite à partir de groupes de repères et de ratios anthropométriques, suivi d’une marge relative entre les deux meilleurs candidats.

4. **Décision** : si la distance ou le score composite est inférieur au seuil défini dans le `Config` (ou `ConfigSequential`) *et* que la marge relative est suffisante (en mode séquentiel), alors l’utilisateur est reconnu.

Pour plus d’informations sur les critères et les paramètres disponibles, consultez `docs/VALIDATION_CRITERIA.md`.