# Critères de validation

Ce document explique les critères utilisés par **D‑Face Hunter v1.2.1** pour comparer deux séquences de visages.  Contrairement à un simple seuil sur la distance, notre approche combine plusieurs types d’informations pour obtenir une décision robuste, notamment lors d’une identification **1 :N**.

## Groupes de repères

Les 478 repères 3D fournis par MediaPipe sont divisés en trois catégories :

| Groupe    | Description                                               | Poids par défaut |
|----------|-----------------------------------------------------------|------------------|
| Invariants | Points très peu affectés par les expressions (os du visage : arête nasale, menton, contour) | 0.4 |
| Stables   | Points modérément affectés par les expressions (pommettes, sourcils, front) | 0.3 |
| Variables | Points sensibles (lèvres, joues) ; peu utilisés pour la reconnaissance | 0.1 |

Chaque groupe possède un **seuil interne** (ex. : 1.5 pour invariants, 2.0 pour stables, etc.).  Lors de la comparaison, la distance moyenne de chaque groupe est normalisée par son seuil et multipliée par son poids.  La somme de ces valeurs constitue une partie du **score composite**.

## Ratios anthropométriques

Certains rapports de distances entre repères sont relativement constants chez un individu (ex. : largeur du nez / distance inter‑pupillaire).  Ces ratios sont calculés pour les séquences d’enrôlement et de test et la **différence moyenne** est ajoutée au score composite, pondérée par un poids (0.2 par défaut).  Les ratios augmentent la sensibilité de l’algorithme en prenant en compte la forme globale du visage.

## Pose et couverture

En mode **spatial** et **séquentiel**, chaque image est associée à un triplet d’angles `(yaw, pitch, roll)`.  Lors de la comparaison :

1. Pour chaque image de test, on sélectionne les images d’enrôlement dont la pose est suffisamment proche (différences en yaw, pitch et roll inférieures aux valeurs `pose_epsilon_*` définies dans la configuration).  Le nombre minimum de frames sélectionnées est appelé **couverture**.
2. La distance entre les repères est calculée uniquement pour ces correspondances.  Les images sans correspondance ne contribuent pas au score.
3. La couverture minimale requise est définie par `coverage_threshold`.  Si elle n’est pas atteinte, la comparaison est considérée comme insuffisante.

En **1 :N**, la couverture est également utilisée pour départager les candidats : un candidat doit non seulement avoir un bon score composite, mais aussi une couverture significativement supérieure à celle des autres.

## Score composite et marge relative

Le **score composite** est défini comme la somme pondérée :

```
score = w_inv * (d_inv / thr_inv)
       + w_stab * (d_stab / thr_stab)
       + w_var * (d_var / thr_var)
       + w_ratio * (ratio_error / thr_ratio)
```

où :

* `d_inv`, `d_stab`, `d_var` : distances moyennes pour chaque groupe de repères ;
* `ratio_error` : erreur moyenne sur les ratios anthropométriques ;
* `thr_*` : seuils configurables dans la classe `Config` ;
* `w_*` : poids configurables dans la classe `Config` (sommant à 1).

Un **candidat est accepté** si :

1. Son `score` est inférieur à `composite_threshold` (0.8 par défaut).  Un score < 1 signifie qu’en moyenne chaque composante est inférieure à son seuil.
2. La **marge relative** `(score_second - score_best) / score_best` est supérieure à `composite_margin` (20 % par défaut).  Cela évite qu’un candidat soit accepté si un autre a un score très proche, comme cela pouvait se produire avec un simple seuil.  Si la marge est faible, l’identification est jugée **ambiguë** et aucun utilisateur n’est renvoyé.
3. La **couverture** est supérieure à `coverage_threshold` (30 % par défaut) et dépasse la couverture du deuxième meilleur d’au moins `coverage_margin` (20 %).

## Ajustement des paramètres

Les seuils et poids par défaut sont définis dans la dataclass `Config` (fichier `src/fr_core/config.py`).  Ces valeurs ont été calibrées sur un jeu de données interne.  Selon vos propres données, vous pouvez :

* Ajuster les poids `weight_invariant`, `weight_stable`, `weight_variable` et `weight_ratio` pour accorder plus ou moins d’importance à chaque groupe.
* Ajuster `pose_epsilon_yaw`, `pose_epsilon_pitch` et `pose_epsilon_roll` pour élargir ou réduire la plage de correspondance de pose.
* Modifier `composite_threshold` pour rendre l’algorithme plus strict ou plus permissif.
* Modifier `composite_margin` et `coverage_margin` pour contrôler l’ambiguïté autorisée en 1 :N.

## Exemple d’utilisation

Pour tester le mode séquentiel en 1 :N :

```bash
# Enrôler les utilisateurs A, B et C
python enroll_interactive.py --matching-mode sequential

# Vérifier un nouveau visage (qui est un A) contre la galerie
python verify_interactive.py --matching-mode sequential

# Le programme renverra le nom du meilleur candidat, le score composite et la couverture.
```

Vous pouvez également modifier les valeurs directement dans le code :

```python
from src.fr_core.config import Config

cfg = Config()
cfg.matching_mode = "sequential"
cfg.composite_threshold = 1.0
cfg.composite_margin = 0.2
cfg.coverage_threshold = 0.3
cfg.coverage_margin = 0.2
```

Pour plus de détails sur la façon dont chaque mode est implémenté, consultez `docs/MODES.md`.