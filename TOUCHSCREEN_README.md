# D-Face Hunter ARM64 - Version 1.2 Final Release

## Interface Tactile (Nouveau !)

### Lancement rapide

```bash
cd ~/Develop/U-LYSS_Face-Hunter
./launch_touchscreen.sh
```

Ou directement :
```bash
python enroll_touchscreen.py
```

### Fonctionnalités

L'interface tactile est **optimisée pour smartphone en mode portrait (720x1440)** :

1. **Sélection de caméra** : Choisir entre caméra arrière (5) ou avant (6)
2. **Clavier virtuel** : Saisir le nom d'utilisateur sans clavier physique
3. **Auto-progression** : Les 3 phases démarrent automatiquement (pas de touches requises)
4. **Bouton tactile Phase 2** : Capturer les frames par toucher

### Améliorations visuelles

- **ROI visible en Phase 1** : Rectangle vert néon autour du visage détecté
- **Landmarks en vert néon** : Points faciaux (468) visibles en vert fluo, taille 2px
- **Interface portrait** : Tous les écrans adaptés au format smartphone

### Phase d'enrollment

**Phase 1** - Auto-capture (45 frames)
- 15 frames frontales
- 15 frames tourné à gauche  
- 15 frames tourné à droite
- Capture automatique toutes les 5 frames
- **ROI** affiché en vert pour montrer la zone de détection

**Phase 2** - Validation manuelle (minimum 5 frames)
- Appuyer sur **ESPACE** ou **toucher le bouton** en bas de l'écran
- Landmarks visibles en **vert néon**
- Compteur : X/target_count

**Phase 3** - Validation DTW
- Vérification automatique de la cohérence
- Distance DTW affichée (seuil recommandé : <6.71)

### Écran de résultats

4 boutons disponibles :
- **OK** : Terminer et quitter
- **RELANCER** : Recommencer l'enrollment
- **GESTION** : Gérer les modèles enregistrés (à implémenter)
- **QUITTER** : Fermer l'application

## Mode clavier (original)

Pour utiliser avec un clavier physique :

```bash
python enroll_interactive.py
```

## Configuration

Camera IDs (par défaut dans `src/fr_core/config.py`) :
- Caméra arrière : ID 5
- Caméra avant : ID 6

Pour changer, modifier `DEFAULT_CAMERA_ID` ou passer l'argument :
```bash
python scripts/enroll_landmarks.py username 6
```

## Dépendances

- Python 3.12.12 (via pyenv)
- MediaPipe 0.10.18
- OpenCV 4.12.0.88
- NumPy 1.26.4
- Environnement : mp_env

Installation :
```bash
./install.sh
```

## Modifications techniques

### enroll_landmarks.py

- Landmarks : Rouge (0,0,255) → **Vert néon (0,255,0)**, taille 1px → **2px**
- `input()` → `time.sleep(1)` pour auto-progression
- ROI ajouté en Phase 1 (rectangle + label)
- Bouton tactile ajouté en Phase 2 (accepte clic souris)

### enroll_touchscreen.py

- Interface complète 720x1440 portrait
- Clavier virtuel QWERTY intégré
- Callbacks souris pour tous les écrans
- Parser DTW pour afficher les résultats
- Gestion des 4 actions post-enrollment

## Fichiers importants

```
U-LYSS_Face-Hunter/
├── enroll_touchscreen.py          # ← Interface tactile (NOUVEAU)
├── launch_touchscreen.sh           # ← Script de lancement (NOUVEAU)
├── enroll_interactive.py           # Interface clavier (original)
├── scripts/
│   ├── enroll_landmarks.py         # ← Modifié : ROI + landmarks verts + auto
│   └── verify_mediapipe.py
├── src/
│   └── fr_core/
│       └── config.py               # Configuration camera_id
├── models/                         # Modèles enregistrés
└── mp_env/                         # Environnement Python
```

## Support

Pour plus d'informations :
- DTW_EXPLANATION.md : Explication du système de validation
- README.md : Documentation complète du projet
