# Configuration GitHub pour U-LYSS Face Hunter

Ce répertoire contient les fichiers de configuration pour la gestion du dépôt GitHub.

## 📁 Fichiers de Configuration

### `settings.yml`
Configuration automatique du dépôt via [Probot Settings](https://probot.github.io/apps/settings/).

**Fonctionnalités :**
- Définit la branche par défaut (`main`)
- Configure la protection de la branche `main`
- Active/désactive les fonctionnalités du dépôt (issues, wiki, etc.)

**Utilisation :**
1. Installez l'application Probot Settings sur votre dépôt
2. Le fichier sera automatiquement appliqué lors du prochain push

### `BRANCH_VISIBILITY.md`
Documentation complète sur la gestion de la visibilité des branches.

**Contenu :**
- Instructions pour supprimer les branches inutiles
- Options pour rendre le dépôt privé
- Configuration de la protection des branches
- Limitations techniques de GitHub

### `BRANCH_STRATEGY.md`
Guide de la stratégie de branches pour les contributeurs.

**Contenu :**
- Politique de branche unique (`main` seulement)
- Workflow de développement (feature branches)
- Commandes Git utiles
- FAQ sur la gestion des branches

## 🔧 Workflows (`.github/workflows/`)

### `branch-management.yml`
Workflow GitHub Actions pour gérer les branches.

**Déclencheurs :**
- Push sur une branche non-`main`
- Fermeture d'une Pull Request
- Déclenchement manuel

**Actions :**
- Notifie les pushs sur des branches non-`main`
- Suggère la suppression des branches après fusion

### `build-arm64.yml`
Workflow de build pour l'architecture ARM64 (existant).

## 🛠️ Scripts Utiles

### `scripts/cleanup-branches.sh`
Script pour nettoyer automatiquement les branches distantes.

**Usage :**
```bash
# Mode dry-run (aucune modification)
./scripts/cleanup-branches.sh --dry-run

# Suppression réelle
./scripts/cleanup-branches.sh
```

**Fonctionnalités :**
- Liste toutes les branches distantes
- Identifie les branches à supprimer (tout sauf `main`)
- Demande confirmation avant suppression
- Nettoie les références locales

## 🎯 Objectif Principal

**Seule la branche `main` doit être visible publiquement.**

### Pourquoi ?

1. **Simplicité** : Un seul point de référence pour les utilisateurs
2. **Sécurité** : Évite l'exposition de code expérimental
3. **Clarté** : Les utilisateurs savent où trouver la version stable
4. **Maintenance** : Réduit la confusion et facilite la gestion

## 🚀 Actions Recommandées

### Pour le Propriétaire du Dépôt

1. **Fusionner les branches de développement**
   ```bash
   git checkout main
   git merge arm64-support
   git push origin main
   ```

2. **Supprimer les branches obsolètes**
   ```bash
   # Utiliser le script de nettoyage
   ./scripts/cleanup-branches.sh
   
   # Ou manuellement
   git push origin --delete arm64-support
   git push origin --delete copilot/build-arm64-architecture
   git push origin --delete copilot/restrict-public-branch-access
   ```

3. **Configurer la protection de la branche `main`**
   - Allez sur GitHub → Settings → Branches
   - Ajoutez une règle de protection pour `main`
   - Activez "Require pull request before merging"

### Pour les Contributeurs

1. **Créer une branche de fonctionnalité**
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. **Développer et pousser**
   ```bash
   git push origin feature/ma-fonctionnalite
   ```

3. **Créer une Pull Request vers `main`**

4. **Supprimer la branche après fusion**
   - Utilisez le bouton "Delete branch" sur GitHub
   - Ou : `git push origin --delete feature/ma-fonctionnalite`

## 📋 Checklist de Configuration

- [x] Créer `settings.yml` pour Probot
- [x] Créer documentation `BRANCH_VISIBILITY.md`
- [x] Créer guide `BRANCH_STRATEGY.md`
- [x] Créer workflow `branch-management.yml`
- [x] Créer script `cleanup-branches.sh`
- [ ] Installer Probot Settings (optionnel)
- [ ] Configurer la protection de `main` sur GitHub
- [ ] Supprimer les branches obsolètes
- [ ] Vérifier que seule `main` est visible

## ⚠️ Limitations Techniques

GitHub ne permet **pas** de rendre certaines branches privées dans un dépôt public via des fichiers de configuration.

**Solutions :**
1. ✅ Supprimer les branches non désirées (recommandé)
2. ⚠️ Rendre tout le dépôt privé
3. 💡 Utiliser un dépôt séparé pour le développement

## 📚 Ressources

- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Probot Settings](https://probot.github.io/apps/settings/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Git Branch Management](https://git-scm.com/book/en/v2/Git-Branching-Branch-Management)

## 🆘 Support

Si vous avez des questions ou des problèmes :
1. Consultez la documentation dans `.github/`
2. Ouvrez une issue sur GitHub
3. Contactez le propriétaire du dépôt

---

**Dernière mise à jour** : Décembre 2024  
**Responsable** : jenfi59
