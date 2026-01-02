# Modes de vérification

Le système **D‑Face Hunter v1.2.1** offre quatre modes de vérification.  Le choix du mode s’effectue via l’attribut `matching_mode` des classes de configuration (`Config` et `ConfigSequential`).  Cette page décrit succinctement chaque mode, ses avantages, ses limites et son usage recommandé.

## 1. Mode **temporal**

### Description

Le mode *temporal* utilise l’algorithme **Dynamic Time Warping (DTW)** pour comparer l’évolution des repères faciaux dans le temps.  Les séquences d’enrôlement et de test sont d’abord réduites par PCA, puis comparées dans leur intégralité.  DTW aligne les séquences même si leur durée diffère légèrement.

### Avantages

* Bonne précision lorsque les séquences sont suffisamment longues (≥ 30 frames).
* Reflète la dynamique complète d’une expression ou d’un mouvement.

### Limites

* Sensible à la longueur de la séquence : performances médiocres avec moins de 30 frames.
* Insensible à la pose : peut comparer une frame de profil à une frame frontale, augmentant le bruit.

### Quand l’utiliser

* Pour des vidéos longues (≥ 2 s) sans contraintes particulières de pose.

## 2. Mode **spatial**

### Description

Le mode *spatial* compare les repères **frame‑par‑frame** en filtrant les images par la pose.  Pour chaque frame de test, seules les frames d’enrôlement ayant un yaw/pitch/roll similaires (selon `pose_epsilon_*`) sont considérées.  La distance finale est la moyenne des distances minimales par frame.

### Avantages

* Efficace avec des séquences courtes (10–30 frames).
* Robuste aux variations d’ordre : l’alignement temporel n’est pas nécessaire.

### Limites

* Ne tient pas compte de la dynamique de l’expression.
* La couverture doit être suffisante ; sinon la vérification échoue.

### Quand l’utiliser

* Pour des captures courtes ou dans des contextes où l’ordre des frames est peu pertinent.
* En combinaison avec d’autres critères (ratios anthropométriques, par exemple).

## 3. Mode **spatiotemporel**

### Description

Le mode *spatiotemporel* est une combinaison linéaire des modes *temporal* et *spatial*.  Les distances résultantes sont pondérées par un facteur `alpha` (défini dans la configuration).  Un `alpha` proche de 1 privilégie la dynamique temporelle ; un `alpha` proche de 0 privilégie la correspondance spatiale.

### Avantages

* Permet d’équilibrer la dynamique et la précision pose‑basée.
* Convient aux captures de longueur intermédiaire.

### Limites

* Nécessite de calibrer le facteur `alpha`.

### Quand l’utiliser

* Lorsque vous souhaitez prendre en compte à la fois la dynamique d’une séquence et la pose.

## 4. Mode **séquentiel** (multi‑critères)

### Description

Introduit en v1.2, le mode *séquentiel* réalise une comparaison en plusieurs étapes :

1. Classe les repères en groupes (invariants, stables, variables) et calcule une distance normalisée pour chaque groupe.
2. Calcule des **ratios anthropométriques** et leur erreur entre les séquences.
3. Combine ces valeurs dans un **score composite** pondéré.
4. Vérifie la **couverture de pose** (pourcentage de frames comparées).
5. Compare le meilleur et le deuxième meilleur score pour appliquer une **marge relative**.

### Avantages

* Très robuste en **1 :N** : réduit les faux positifs en exigeant que le meilleur soit nettement meilleur que les autres.
* Combinaison explicable : chaque composant du score est interprétable.
* Paramètres ajustables : poids, seuils, marges, etc.

### Limites

* Plus complexe à calibrer qu’un simple seuil.
* Un mauvais réglage des poids ou seuils peut réduire la couverture.

### Quand l’utiliser

* Scénarios d’identification en foule (1 :N) ou pour des systèmes de sécurité à faible tolérance aux faux positifs.
* Comparaisons où un simple critère sur la distance n’est pas suffisant.

## Choisir un mode

Le tableau suivant résume les cas d’usage :

| Mode            | Séquence courte | Sensibilité pose | 1 :N | Paramètres clés        |
|-----------------|-----------------|------------------|------|------------------------|
| Temporal        | Non (< 30 frames) | Faible          | Non  | `pca_n_components`     |
| Spatial         | Oui             | Forte           | Oui  | `pose_epsilon_*`, `pose_threshold` |
| Spatiotemporel | Moyen           | Modérée         | Oui  | `alpha`                |
| Séquentiel     | Moyen à long    | Forte           | Oui  | `composite_threshold`, `composite_margin`, `coverage_threshold` |

La configuration de ces modes se trouve dans le fichier `src/fr_core/config.py` (classes `Config` et `ConfigSequential`) et peut être ajustée selon vos tests.  Pour plus d’informations sur les seuils et les paramètres, consultez `docs/VALIDATION_CRITERIA.md`.