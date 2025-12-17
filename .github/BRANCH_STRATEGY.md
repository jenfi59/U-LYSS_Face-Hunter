# Stratégie de Branches pour U-LYSS Face Hunter

## Politique de Branches

Ce projet utilise une stratégie de **branche unique publique** :

- ✅ **`main`** : Seule branche publique et permanente
  - Contient le code de production stable
  - Protégée par des règles de protection
  - Toutes les fonctionnalités doivent être fusionnées ici

- ❌ **Branches de fonctionnalités** : Temporaires uniquement
  - Créées pour le développement de nouvelles fonctionnalités
  - **DOIVENT** être supprimées après fusion dans `main`
  - Ne doivent pas rester visibles publiquement

## Workflow de Développement

### 1. Créer une branche de fonctionnalité

```bash
git checkout main
git pull origin main
git checkout -b feature/ma-nouvelle-fonctionnalite
```

### 2. Développer et commiter

```bash
git add .
git commit -m "feat: ajout de ma nouvelle fonctionnalité"
git push origin feature/ma-nouvelle-fonctionnalite
```

### 3. Créer une Pull Request

- Allez sur GitHub
- Créez une PR depuis votre branche vers `main`
- Demandez une revue de code si nécessaire
- Une fois approuvée, fusionnez la PR

### 4. Supprimer la branche après fusion

```bash
# Supprimer localement
git checkout main
git branch -d feature/ma-nouvelle-fonctionnalite

# Supprimer sur GitHub
git push origin --delete feature/ma-nouvelle-fonctionnalite
```

Ou utilisez le bouton "Delete branch" sur GitHub après avoir fusionné la PR.

## Branches Protégées

La branche `main` est protégée avec les règles suivantes :

- ✅ Pull Request requise avant fusion
- ✅ Au moins 1 approbation requise
- ✅ Les approbations périmées sont rejetées lors de nouveaux commits
- ❌ Force push interdit
- ❌ Suppression interdite

## Commandes Utiles

### Lister toutes les branches

```bash
# Branches locales
git branch

# Branches distantes
git branch -r

# Toutes les branches
git branch -a
```

### Supprimer des branches

```bash
# Supprimer une branche locale (déjà fusionnée)
git branch -d nom-de-la-branche

# Forcer la suppression d'une branche locale
git branch -D nom-de-la-branche

# Supprimer une branche distante
git push origin --delete nom-de-la-branche
```

### Nettoyer les références de branches supprimées

```bash
# Nettoyer les références locales vers des branches distantes supprimées
git fetch --prune
```

## Pourquoi Une Seule Branche Publique ?

1. **Simplicité** : Un seul point de vérité pour le code de production
2. **Sécurité** : Évite l'exposition de code expérimental ou non testé
3. **Clarté** : Les utilisateurs savent toujours où trouver la dernière version stable
4. **Maintenance** : Réduit la confusion sur quelle branche utiliser

## Questions Fréquentes

### Q: Que faire si j'ai besoin d'une branche de développement à long terme ?

**R:** Utilisez un dépôt privé séparé pour le développement, puis fusionnez les fonctionnalités finalisées dans le dépôt public.

### Q: Comment gérer les releases ?

**R:** Utilisez des tags Git pour marquer les versions :

```bash
git tag -a v2.1.0 -m "Version 2.1.0"
git push origin v2.1.0
```

### Q: Puis-je créer des branches temporaires ?

**R:** Oui, mais supprimez-les immédiatement après fusion dans `main`.

## Voir Aussi

- [BRANCH_VISIBILITY.md](.github/BRANCH_VISIBILITY.md) - Instructions détaillées sur la configuration
- [GitHub Flow](https://guides.github.com/introduction/flow/) - Guide officiel GitHub
