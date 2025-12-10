# Historical Documentation Archive

Ce dossier contient la documentation historique du développement de FR_VERS_JP, de la version 1.0 à 2.0.

## Contenu

### Documentation Tier 1 (Foundation)
- **TIER1_COMPLETE_SUMMARY.md**: Résumé complet de la phase Tier 1
  - Implémentation des 68 landmarks (MediaPipe)
  - Système de vérification basique
  - Tests initiaux et calibration

### Documentation Tier 2 (Advanced Features)

#### DDTW (Derivative DTW)
- **TIER2_6_DDTW_COMPLETE.md**: Implémentation complète du DDTW
  - Méthodes: velocity, acceleration, combined
  - Tests de performance
  - Amélioration de 12.9% du taux de vérification

#### Liveness Detection
- **TIER2_7_LIVENESS_COMPLETE.md**: Système anti-spoofing
  - Détection de clignements (EAR)
  - Analyse de mouvement 3D
  - Fusion des méthodes
  - Tests avec photos/vidéos

### Documentation Complète
- **PROJECT_TIER1_TIER2_COMPLETE.md**: Vue d'ensemble du projet complet
  - Architecture globale
  - Résultats finaux Tier 1 + Tier 2
  - Performances et métriques

- **COMPLETION_SUMMARY.md**: Résumé de fin de projet v2.0
  - Récapitulatif de toutes les fonctionnalités
  - État final du système
  - Recommandations pour v2.1

## Organisation Chronologique

```
Version 1.0 (Baseline)
  └─ Gabor + LBP features
  └─ GMM matching

Version 2.0 (Tier 1 + Tier 2)
  ├─ Tier 1: 68 Landmarks + DTW
  │   ├─ MediaPipe face landmarks
  │   ├─ PCA dimensionality reduction
  │   ├─ DTW distance matching
  │   └─ Threshold calibration (6.71)
  │
  └─ Tier 2: Advanced Features
      ├─ DDTW (velocity features)
      │   └─ +12.9% verification improvement
      │
      └─ Liveness Detection
          ├─ Blink detection (EAR < 0.25)
          ├─ Motion analysis (3D movements)
          └─ Fusion score (threshold 0.6)

Version 2.1 (Current - Clean Refactoring)
  └─ Simplification et nettoyage
  └─ Documentation séparée (actuelle vs historique)
```

## Migration vers v2.1

La version 2.1 représente un **refactoring complet** de v2.0 :

### Ce qui a été supprimé
- ❌ Modules legacy (Gabor/LBP, verification.py, preprocessing.py, features.py)
- ❌ Tests redondants (15+ fichiers de tests)
- ❌ Scripts obsolètes (10+ utilitaires de calibration/debug)
- ❌ Documentation verbose intégrée au code

### Ce qui a été conservé
- ✅ 6 modules essentiels (config, landmark_utils, ddtw, liveness, verification_dtw, __init__)
- ✅ 2 scripts (enroll, verify)
- ✅ 3 tests critiques (test_system, test_ddtw, test_far)
- ✅ Documentation concise (README, QUICKSTART, API, CONFIGURATION, DEPLOYMENT)

### Changements architecturaux
1. **verification_dtw.py** maintenant autonome (plus de dépendance à verification.py)
2. **load_model()** intégré directement
3. **Imports simplifiés** (pas de dépendances circulaires)
4. **Documentation séparée** (docs/v2.1/ pour l'actuel, docs/history/ pour l'historique)

## Utilisation de ces documents

Ces documents historiques sont conservés pour :
- 📚 **Référence historique**: Comprendre l'évolution du projet
- 🔍 **Recherche**: Retrouver des décisions techniques passées
- 🎓 **Apprentissage**: Voir comment les problèmes ont été résolus
- 📊 **Métriques**: Comparer les performances entre versions

**Pour la documentation actuelle de v2.1**, consultez :
- `../v2.1/README.md`
- `../v2.1/API.md`
- `../v2.1/CONFIGURATION.md`
- `../v2.1/DEPLOYMENT.md`

## Contexte du Projet

**Objectif**: Système de reconnaissance faciale basé sur les landmarks pour l'authentification utilisateur

**Approches testées**:
1. v1.0: Gabor + LBP + GMM (baseline)
2. v2.0: 68 Landmarks + PCA + DTW + DDTW + Liveness (production)
3. v2.1: Clean refactoring de v2.0 (actuel)

**Résultats finaux v2.0**:
- DTW threshold: 6.71
- FAR (False Accept Rate): < 1%
- FRR (False Reject Rate): ~5%
- Liveness detection: 95%+ spoof detection
- DDTW improvement: +12.9% verification rate

## Auteur et Dates

- **Développeur**: Jean-Philippe (jeanphi)
- **Période**: 2024
- **Dernière mise à jour v2.0**: Décembre 2024
- **Refactoring v2.1**: 9 Décembre 2024

---

*Ces documents sont archivés et ne doivent plus être modifiés. Pour toute mise à jour, créer de nouveaux documents dans `../v2.1/`.*
