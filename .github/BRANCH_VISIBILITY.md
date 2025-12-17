# Configuration de la Visibilité des Branches

## Objectif
Seule la branche `main` doit être visible publiquement.

## Instructions pour le Propriétaire du Dépôt

### Option 1 : Supprimer les Branches Inutiles (Recommandé)

Si les autres branches ne sont plus nécessaires, la meilleure solution est de les supprimer :

```bash
# Supprimer les branches localement
git branch -d arm64-support
git branch -d copilot/build-arm64-architecture

# Supprimer les branches sur GitHub
git push origin --delete arm64-support
git push origin --delete copilot/build-arm64-architecture
```

### Option 2 : Rendre le Dépôt Privé avec Accès Restreint

Si vous voulez vraiment limiter la visibilité des branches :

1. Allez sur GitHub : `https://github.com/jenfi59/U-LYSS_Face-Hunter/settings`
2. Dans la section "Danger Zone", cliquez sur "Change repository visibility"
3. Sélectionnez "Make private"
4. Ensuite, gérez les accès individuels via "Collaborators"

**Note** : GitHub ne permet pas de rendre certaines branches privées dans un dépôt public.

### Option 3 : Protection des Branches

Pour protéger la branche `main` et contrôler les modifications :

1. Allez sur `https://github.com/jenfi59/U-LYSS_Face-Hunter/settings/branches`
2. Cliquez sur "Add rule" ou "Add branch protection rule"
3. Dans "Branch name pattern", entrez : `main`
4. Cochez les options souhaitées :
   - ✅ Require a pull request before merging
   - ✅ Require approvals (nombre : 1)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Do not allow bypassing the above settings
5. Cliquez sur "Create" ou "Save changes"

### Option 4 : Utiliser Probot Settings (Automatique)

Si vous avez installé l'application Probot Settings :

1. Le fichier `.github/settings.yml` de ce dépôt contient déjà la configuration
2. L'application appliquera automatiquement les paramètres
3. Installez Probot Settings : https://probot.github.io/apps/settings/

## Branches Actuelles

Selon la dernière vérification, ces branches existent :

- ✅ `main` - Branche principale (doit rester visible)
- ❌ `arm64-support` - Branche de développement (à supprimer ou fusionner)
- ❌ `copilot/build-arm64-architecture` - Branche temporaire Copilot (à supprimer)
- ❌ `copilot/restrict-public-branch-access` - Branche temporaire Copilot (à supprimer)

## Recommandation Finale

**Action recommandée** : Fusionnez ou supprimez les branches `arm64-support`, `copilot/build-arm64-architecture`, et `copilot/restrict-public-branch-access` une fois leur contenu intégré dans `main`.

```bash
# Exemple : Fusionner une branche dans main puis la supprimer
git checkout main
git merge arm64-support
git push origin main
git push origin --delete arm64-support
```

## Limitation Technique

⚠️ **Important** : GitHub ne permet pas de cacher des branches spécifiques dans un dépôt public via des fichiers de configuration. Les seules options sont :

1. Supprimer les branches non désirées
2. Rendre le dépôt entièrement privé
3. Utiliser un dépôt séparé pour le développement

La configuration dans `.github/settings.yml` aide à protéger `main`, mais ne cache pas les autres branches du public.
