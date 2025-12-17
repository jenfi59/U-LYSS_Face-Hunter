# Configuration de la Visibilité des Branches - Résumé

## ✅ Ce qui a été configuré

Ce PR ajoute la configuration nécessaire pour gérer la visibilité des branches dans le dépôt U-LYSS Face Hunter, avec l'objectif de **ne garder que la branche `main` visible publiquement**.

### 📁 Fichiers ajoutés

1. **`.github/settings.yml`**
   - Configuration automatique du dépôt via Probot Settings
   - Définit `main` comme branche par défaut
   - Active la protection de la branche `main` avec :
     - Approbation requise pour les PRs
     - Pas de force push
     - Pas de suppression

2. **`.github/BRANCH_VISIBILITY.md`**
   - Documentation complète sur la visibilité des branches
   - Instructions pour supprimer les branches inutiles
   - Options pour configurer GitHub manuellement

3. **`.github/BRANCH_STRATEGY.md`**
   - Guide de la stratégie de branches pour contributeurs
   - Workflow de développement avec feature branches
   - Commandes Git utiles

4. **`.github/workflows/branch-management.yml`**
   - Workflow GitHub Actions qui :
     - Notifie lors de push sur branches non-`main`
     - Suggère la suppression après fusion de PR
     - Rappelle la politique de branches

5. **`scripts/cleanup-branches.sh`**
   - Script bash pour nettoyer automatiquement les branches
   - Mode dry-run disponible pour tester sans modifier
   - Supprime toutes les branches sauf `main`

6. **`.github/README.md`**
   - Documentation complète de la configuration GitHub
   - Guide d'utilisation pour propriétaires et contributeurs
   - Checklist de configuration

## 🎯 Branches actuellement présentes

Au moment de cette configuration, ces branches existent :

- ✅ **`main`** - Branche principale (à conserver)
- ❌ **`arm64-support`** - Branche de développement (à supprimer ou fusionner)
- ❌ **`copilot/build-arm64-architecture`** - Branche temporaire Copilot (à supprimer)
- ❌ **`copilot/restrict-public-branch-access`** - Cette branche PR (à supprimer après fusion)

## 🚀 Actions à effectuer après fusion de ce PR

### Étape 1 : Fusionner ce PR dans `main`

```bash
# Cette branche sera fusionnée via GitHub PR
```

### Étape 2 : Vérifier le contenu des autres branches

Avant de supprimer, vérifiez si `arm64-support` ou `copilot/build-arm64-architecture` contiennent du code important :

```bash
# Comparer avec main
git fetch origin
git diff origin/main..origin/arm64-support
git diff origin/main..origin/copilot/build-arm64-architecture
```

### Étape 3 : Fusionner les branches utiles (si nécessaire)

Si `arm64-support` contient des modifications importantes :

```bash
git checkout main
git pull origin main
git merge origin/arm64-support
git push origin main
```

### Étape 4 : Supprimer les branches obsolètes

**Option A : Utiliser le script automatique**

```bash
# Tester d'abord (mode dry-run)
./scripts/cleanup-branches.sh --dry-run

# Exécuter la suppression
./scripts/cleanup-branches.sh
```

**Option B : Suppression manuelle**

```bash
# Supprimer chaque branche
git push origin --delete arm64-support
git push origin --delete copilot/build-arm64-architecture
git push origin --delete copilot/restrict-public-branch-access
```

### Étape 5 : Configurer la protection de `main` (Optionnel)

Si vous n'utilisez pas Probot Settings, configurez manuellement :

1. Allez sur : `https://github.com/jenfi59/U-LYSS_Face-Hunter/settings/branches`
2. Cliquez sur "Add branch protection rule"
3. Branch name pattern : `main`
4. Activez :
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
5. Sauvegardez

### Étape 6 : Installer Probot Settings (Optionnel)

Pour une gestion automatique via `.github/settings.yml` :

1. Allez sur : https://github.com/apps/settings
2. Cliquez sur "Install"
3. Sélectionnez votre dépôt `U-LYSS_Face-Hunter`
4. Les paramètres de `.github/settings.yml` seront automatiquement appliqués

## ⚠️ Important à comprendre

### GitHub ne permet pas de cacher des branches

**Limitation technique** : Dans un dépôt public GitHub, toutes les branches sont visibles publiquement. Il n'existe pas de moyen de rendre certaines branches privées via configuration.

**Solutions :**
1. ✅ **Supprimer les branches** non désirées (recommandé)
2. ⚠️ **Rendre le dépôt privé** (limite l'accès à tout le dépôt)
3. 💡 **Utiliser un dépôt séparé** pour le développement

### Qu'est-ce que cette configuration fait réellement ?

- ✅ **Protège** la branche `main` contre les modifications directes
- ✅ **Encourage** la suppression des branches après fusion
- ✅ **Notifie** lors de push sur branches non-`main`
- ✅ **Fournit des outils** pour nettoyer les branches
- ❌ **Ne cache PAS** les branches du public

## 📋 Checklist post-fusion

- [ ] Fusionner ce PR dans `main`
- [ ] Vérifier le contenu de `arm64-support`
- [ ] Fusionner ou supprimer `arm64-support`
- [ ] Supprimer `copilot/build-arm64-architecture`
- [ ] Supprimer `copilot/restrict-public-branch-access` (cette branche)
- [ ] Vérifier que seule `main` reste : `git ls-remote --heads origin`
- [ ] (Optionnel) Installer Probot Settings
- [ ] (Optionnel) Configurer la protection de `main` manuellement

## 🎓 Pour l'avenir

### Workflow recommandé pour les contributeurs

1. **Créer une feature branch**
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. **Développer et pousser**
   ```bash
   git push origin feature/ma-fonctionnalite
   ```

3. **Créer une Pull Request vers `main`**

4. **Après fusion, supprimer la branche**
   - Utilisez le bouton "Delete branch" sur GitHub
   - Ou : `git push origin --delete feature/ma-fonctionnalite`

### Résultat final attendu

```bash
$ git ls-remote --heads origin
33d1c6ecefabc67cd0b702bf764aec6fd7d80554	refs/heads/main
```

✅ Seule la branche `main` sera visible publiquement !

## 📚 Documentation

- Consultez `.github/README.md` pour la documentation complète
- Lisez `.github/BRANCH_VISIBILITY.md` pour les détails techniques
- Suivez `.github/BRANCH_STRATEGY.md` pour la stratégie de branches

## 🆘 Support

Si vous avez des questions :
1. Consultez la documentation dans `.github/`
2. Ouvrez une issue sur GitHub
3. Vérifiez les logs des workflows Actions

---

**Créé le** : Décembre 2024  
**Objectif** : Ne garder que la branche `main` visible publiquement
