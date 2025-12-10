# FR_VERS_JP_2_1 - Environnement Portable

## Configuration

Ce dossier est **100% portable** et ne nécessite **aucune installation**.

### Dépendances

Toutes les dépendances Python sont installées en mode `--user` :
- Emplacement: `~/.local/lib/python3.10/site-packages/`
- Accessible depuis n'importe quel script Python

### Packages Installés

```
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.7.2
opencv-python==4.12.0.88
mediapipe==0.10.14
pywavelets==1.8.0
fdapy==1.0.1
pytest==9.0.2
pytest-cov==7.0.0
```

## Utilisation

### Méthode 1: Wrappers (Recommandée)

Les scripts wrappers configurent automatiquement le PYTHONPATH :

```bash
# Enrollment
./run_enrollment.sh jean

# Verification
./run_verify.sh jean
```

### Méthode 2: Manuelle

Configurez le PYTHONPATH manuellement :

```bash
# Aller dans le dossier
cd FR_VERS_JP_2_1

# Configurer PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Enrollment
python3 scripts/enroll_landmarks.py jean

# Verification  
python3 scripts/verify.py models/jean.npz
```

## Portabilité

### ✅ Ce qui est portable

- Tous les modules Python (fr_core/)
- Tous les scripts (scripts/)
- Configuration (config.py)
- Documentation
- Scripts wrappers

### ⚠️ Dépendances Système Requises

Pour déplacer sur un autre système, assurez-vous d'avoir :

1. **Python 3.10+**
   ```bash
   python3 --version
   ```

2. **Dépendances système** (une seule fois par système)
   ```bash
   pip3 install --user -r requirements.txt
   ```

3. **Webcam** (pour enrollment/verification)

### Migration vers Nouveau Système

```bash
# 1. Copier le dossier entier
cp -r FR_VERS_JP_2_1 /nouveau/chemin/

# 2. Installer les dépendances (une seule fois)
cd /nouveau/chemin/FR_VERS_JP_2_1
pip3 install --user -r requirements.txt

# 3. Utiliser normalement
./run_enrollment.sh mon_nom
```

## Architecture

```
FR_VERS_JP_2_1/
├── fr_core/               # Modules Python (portable)
│   ├── config.py
│   ├── guided_enrollment.py
│   ├── features.py
│   ├── enrollment.py
│   ├── landmark_utils.py
│   ├── verification_dtw.py
│   ├── verification.py
│   ├── ddtw.py
│   ├── liveness.py
│   └── preprocessing.py
│
├── scripts/               # Scripts d'utilisation
│   ├── enroll_landmarks.py
│   ├── enroll.py
│   └── verify.py
│
├── models/                # Modèles utilisateurs (.npz)
│
├── run_enrollment.sh      # Wrapper enrollment
├── run_verify.sh          # Wrapper verification
├── requirements.txt       # Liste dépendances
├── README.md              # Documentation principale
├── QUICKSTART.md          # Guide rapide
└── CHECK_REPORT.md        # Rapport de vérification

```

## Avantages de cette Configuration

✅ **Pas de venv** - Évite les problèmes de liens symboliques  
✅ **Installation --user** - Dépendances isolées par utilisateur  
✅ **PYTHONPATH auto** - Wrappers configurent tout automatiquement  
✅ **Déplaçable** - Fonctionne depuis n'importe quel chemin  
✅ **Multi-système** - Même dossier sur plusieurs machines  

## Notes

- Les modèles `.npz` sont compatibles entre v2.0 et v2.1
- Le format des landmarks est standard (68 points MediaPipe)
- Les scripts fonctionnent avec chemins relatifs
- Aucune compilation ou build requise
