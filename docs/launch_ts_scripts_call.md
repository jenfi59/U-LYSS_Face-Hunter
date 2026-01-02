# Architecture de Navigation - launch_touchscreen.py

## Vue d'ensemble

Ce document présente l'arbre de navigation complet de l'application tactile D-Face Hunter et les scripts appelés selon les choix utilisateurs.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANCEMENT DU SCRIPT                          │
│                 launch_touchscreen.py                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │     GUI: MENU PRINCIPAL (Portrait)        │
        │  [ENROLLMENT] [VALIDATION]                │
        │  [GESTION] [QUITTER]                      │
        └───────────────────────────────────────────┘
                │           │           │         │
        ────────┴───────────┴───────────┴─────────┴────────
        │                   │           │         │
        ▼                   ▼           ▼         ▼
   ENROLLMENT          VALIDATION   GESTION   QUITTER
        │                   │           │         │
        │                   │           │         └─> sys.exit(0)
        │                   │           │             + enable_sleep()
        │                   │           │
        │                   │           └─> [À IMPLÉMENTER]
        │                   │               └─> Retour MENU PRINCIPAL
        │                   │
        │                   ├─> GUI: Sélection Caméra
        │                   │   [Caméra Arrière (5)] [Caméra Avant (6)]
        │                   │           │
        │                   │           ▼
        │                   ├─> GUI: Sélection Modèle (scrollable)
        │                   │   [Liste des .npz] [ANNULER]
        │                   │           │
        │                   │           ▼
        │                   ├─> GUI: Sélection Mode
        │                   │   [1:1 Verification] [1:N Identification]
        │                   │   [Mode Spatial] [Mode Séquentiel]
        │                   │           │
        │                   │           ▼
        │                   ├─> GUI: Capture Validation
        │                   │   [START VALIDATION] [ANNULER]
        │                   │           │
        │                   │           ▼
        │                   ├─> GUI: Flux Vidéo avec Overlay
        │                   │   • Affichage caméra temps réel
        │                   │   • Compteur frames (X/15)
        │                   │   • Timer (X.Xs restant)
        │                   │   • Barre de progression
        │                   │   • Durée: 4 secondes
        │                   │   │
        │                   │   APPEL INTERNE:
        │                   │   ├─> run_validation_capture()
        │                   │   │   ├─> MediaPipe FaceLandmarker
        │                   │   │   ├─> fr_core.VerificationDTW
        │                   │   │   └─> verify_auto()
        │                   │           │
        │                   │           ▼
        │                   └─> GUI: Résultats Validation
        │                       [VALIDE/REFUSE]
        │                       Distance, Frames, Coverage
        │                       [NOUVELLE VALIDATION] [MENU] [QUITTER]
        │                               │           │         │
        │                               │           │         └─> sys.exit(0)
        │                               │           └─> Retour MENU PRINCIPAL
        │                               └─> Boucle Validation
        │
        ├─> GUI: Sélection Caméra
        │   [Caméra Arrière (5)] [Caméra Avant (6)]
        │           │
        │           ▼
        ├─> GUI: Saisie Nom Utilisateur
        │   Clavier Virtuel QWERTY 4 lignes
        │   [DEL] [OK]
        │           │
        │           ▼
        ├─> GUI: Confirmation
        │   "Utilisateur: XXX"
        │   "Caméra: Arrière/Avant"
        │   [DEMARRER]
        │           │
        │           ▼
        ├─> GUI: Écran de Chargement Animé
        │   "Enrollment en cours"
        │   "Phase 1/3: Capture automatique..."
        │   "Phase 2/3: Validation manuelle..."
        │   "Phase 3/3: Verification DTW..."
        │   Barre de progression animée
        │   │
        │   APPEL EXTERNE (subprocess.Popen):
        │   └─> scripts/enroll_landmarks.py <username> --camera <ID>
        │       ├─> MediaPipe FaceLandmarker (468 landmarks)
        │       ├─> Phase 1: Auto-capture guidée (45 frames)
        │       ├─> Phase 2: Validation manuelle (5+ frames)
        │       └─> Sauvegarde: models/users/<username>.npz
        │               │
        │               ▼
        └─> GUI: Résultats Enrollment
            "Enrollment terminé"
            Distance DTW (si validation immédiate)
            [OK] [RELANCER] [VALIDATION] [GESTION] [QUITTER]
                │      │           │           │         │
                │      │           │           │         └─> sys.exit(0)
                │      │           │           └─> [À IMPLÉMENTER]
                │      │           └─> Workflow VALIDATION
                │      └─> Relance ENROLLMENT
                └─> Retour MENU PRINCIPAL
```

## Scripts Externes

### scripts/enroll_landmarks.py

**Appelé par:** Workflow ENROLLMENT via `subprocess.Popen`

**Arguments:**
```bash
python scripts/enroll_landmarks.py <username> --camera <5|6>
```

**Fonction:**
- Phase 1: GuidedEnrollment (45 frames capture automatique)
- Phase 2: Validation manuelle (5+ frames avec SPACE)
- Fit PCA sur la séquence complète
- Sauvegarde: `models/users/<username>.npz`
- Validation immédiate (optionnelle)

**Modules utilisés:**
- `mediapipe.tasks.python.vision.FaceLandmarker`
- `fr_core.GuidedEnrollment`
- `fr_core.VerificationDTW`

### scripts/verify_mediapipe.py

**Statut:** OBSOLÈTE - Remplacé par intégration interne

**Note:** Anciennement utilisé pour validation via subprocess. Maintenant, la validation utilise `run_validation_capture()` intégrée directement dans `launch_touchscreen.py`.

## Modules Internes

### src/fr_core/

**Composants:**

- **get_config()** - Configuration système
- **GuidedEnrollment** - Guide la capture de 45 frames
- **VerificationDTW** - Moteur de vérification
  - `load_enrollment()` - Charge les modèles .npz
  - `verify_auto()` - Routeur spatial/temporal
  - `save_enrollment()` - Sauvegarde les modèles

## Structure des Fichiers Modèles

### models/users/\<username\>.npz

```python
{
    'landmarks': array (N_frames, 468, 3),  # Coordonnées des landmarks
    'poses': array (N_frames, 3),           # [yaw, pitch, roll]
    'pca_components': array (optionnel)     # Composantes PCA
}
```

### models/mediapipe/face_landmarker_v2_with_blendshapes.task

Modèle MediaPipe pour la détection de 468 landmarks faciaux.

## Résumé des Chemins Utilisateur

### ENROLLMENT
```
Menu → Caméra → Username → Confirm → [enroll_landmarks.py] → Résultats → Menu
```

### VALIDATION
```
Menu → Caméra → Modèle → Mode → START → [Capture intégrée] → Résultats → Menu
```

### GESTION
```
Menu → [À IMPLÉMENTER] → Menu
```

### QUITTER
```
Menu → sys.exit(0) + enable_sleep()
```

## Détails Techniques

### Format d'Écran
- **Portrait:** 720×1440 pixels
- **Tous les GUI** utilisent ce format sauf pendant la capture vidéo de validation

### Gestion de la Mise en Veille
- **Désactivée** au lancement (`disable_sleep()`)
- **Réactivée** à la sortie (`enable_sleep()`)

### Capture Vidéo

#### ENROLLMENT
- Pas de flux vidéo visible pour l'utilisateur
- Écran de chargement animé avec phases
- Subprocess externe (enroll_landmarks.py)

#### VALIDATION
- **Flux vidéo visible** en temps réel
- Overlay avec informations:
  - Nom du modèle
  - Compteur de frames
  - Timer
  - Barre de progression
- Capture intégrée (pas de subprocess)
- Durée: 4 secondes minimum ou 15 frames

## Notes de Développement

### Points Clés
1. **Validation intégrée** - Plus rapide, meilleur UX avec flux vidéo
2. **Enrollment externe** - Utilise le script existant testé
3. **Menu portrait** - Fenêtre recréée à chaque retour pour garantir le bon format
4. **Scrolling** - Implémenté pour liste de modèles si > 8 modèles

### À Implémenter
- **Gestion des modèles** - Interface pour renommer, supprimer, visualiser les détails

---

*Document créé le 2 janvier 2026*  
*Version: 1.0*
