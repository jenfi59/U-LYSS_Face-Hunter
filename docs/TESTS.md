# Tests et exemples d’utilisation

Le dossier `tests/` contient plusieurs scripts pour vérifier le fonctionnement de **D‑Face Hunter v1.2.1**.  Ils sont destinés à faciliter les validations lors du développement et de la contribution.  Les tests sont écrits pour `pytest` mais peuvent être exécutés individuellement.

## Contenu du dossier `tests/`

| Fichier | Description |
|---|---|
| `test_imports.py` | Vérifie que les modules clés (`mediapipe`, `numpy`, `src.fr_core`) se chargent sans erreur. |
| `test_enrollment_and_verification.py` | Exemple complet d’enrôlement et de vérification 1:1 : crée un utilisateur factice avec des données synthétiques, sauvegarde un fichier `.npz`, puis vérifie qu’une séquence proche est acceptée et qu’un imposteur est rejeté. |
| `test_multi_gallery.py` | Teste la logique 1 :N : simule plusieurs utilisateurs avec des séquences synthétiques, calcule un score composite et vérifie que le meilleur candidat est sélectionné correctement. |
| `data/` | (optionnel) Contient des séquences de repères et de poses réelles ou synthétiques à utiliser pour les tests. |

## Exécution des tests

Installez `pytest` (inclus dans `requirements.txt`) et exécutez :

```bash
cd tests
pytest -q
```

Chaque test est indépendant : si certains requièrent des fichiers `.npz` réels et qu’ils ne sont pas présents, ils seront ignorés ou créeront des données synthétiques.

## Création de données synthétiques

Les tests utilisent des repères aléatoires pour illustrer le fonctionnement général sans dépendre d’un ensemble de données réel.  Les repères sont générés avec `numpy.random.normal` autour de points de référence imaginaires afin de simuler des visages distincts.  Les poses sont générées aléatoirement dans des intervalles plausibles (yaw ±30°, pitch ±20°, roll ±15°).  Ces données ne reflètent pas la variabilité réelle d’un visage mais permettent de tester la logique du code.

Pour des validations plus réalistes, placez des fichiers `.npz` obtenus via `scripts/enroll_landmarks.py` dans le dossier `tests/data/` et modifiez les tests pour les charger.  Les tests sont conçus pour détecter automatiquement ces fichiers.

## Note sur les dépendances

Certains tests importent `mediapipe` uniquement pour vérifier la présence du module.  Si MediaPipe n’est pas installé ou incompatible avec votre version de Python, vous pouvez marquer ces tests comme `xfail` ou les désactiver temporairement.