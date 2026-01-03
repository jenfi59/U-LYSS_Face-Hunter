# Fix: Problème de syntaxe Docker Hub Login Action

## 📋 Le Problème (The Problem)

La syntaxe suivante dans un workflow GitHub Actions est **incorrecte** :

```yaml
- name: Login to Docker Hub
  if: github.event_name != 'pull_request' && secrets.DOCKERHUB_USERNAME != ''
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

### ❌ Pourquoi c'est incorrect ?

1. **Accès au contexte `secrets` dans les conditions `if`** : 
   - Dans GitHub Actions, vous ne pouvez pas directement vérifier si un secret existe ou est vide en utilisant `secrets.DOCKERHUB_USERNAME != ''` dans une expression `if`.
   - Le contexte `secrets` n'est pas directement accessible pour les comparaisons dans les conditions.
   - Cette syntaxe provoquera une erreur ou un comportement inattendu.

2. **Problème de sécurité** :
   - GitHub Actions ne permet pas de comparer directement les secrets dans les conditions pour éviter les fuites potentielles de secrets.

## ✅ La Solution (The Solution)

Il existe plusieurs façons correctes de résoudre ce problème :

### Option 1 : Utiliser une condition simplifiée (Recommandée)

```yaml
- name: Login to Docker Hub
  if: github.event_name != 'pull_request'
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

**Explication** : L'action `docker/login-action` gère automatiquement les cas où les secrets sont vides ou non définis. Si le secret n'existe pas, l'action échouera proprement avec un message d'erreur clair.

### Option 2 : Utiliser une variable d'environnement intermédiaire

```yaml
- name: Check Docker Hub credentials
  id: check_dockerhub
  run: |
    if [ -n "${{ secrets.DOCKERHUB_USERNAME }}" ]; then
      echo "has_credentials=true" >> $GITHUB_OUTPUT
    else
      echo "has_credentials=false" >> $GITHUB_OUTPUT
    fi

- name: Login to Docker Hub
  if: github.event_name != 'pull_request' && steps.check_dockerhub.outputs.has_credentials == 'true'
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

**Explication** : Cette approche crée d'abord un step qui vérifie l'existence du secret et stocke le résultat dans une sortie. Ensuite, on peut utiliser cette sortie dans la condition `if`.

### Option 3 : Utiliser les jobs conditionnels

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t myapp:latest .
  
  push:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request' && vars.DOCKERHUB_ENABLED == 'true'
    steps:
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Push to Docker Hub
        run: docker push myapp:latest
```

**Explication** : Séparer le build et le push en deux jobs distincts, avec des conditions au niveau du job. Utilisez une variable de configuration (`vars.DOCKERHUB_ENABLED`) pour activer/désactiver la fonctionnalité.

## 🎯 Solution Recommandée pour ce Projet

Pour le projet **D-Face Hunter ARM64**, voici la solution recommandée :

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: docker.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        with:
          platforms: linux/arm64

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/arm64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Points clés de cette solution :

1. ✅ **Condition correcte** : `if: github.event_name != 'pull_request'` sans référence directe aux secrets
2. ✅ **Gestion automatique des erreurs** : Si les secrets ne sont pas définis, l'action échouera avec un message clair
3. ✅ **Support ARM64** : Configuration QEMU pour la plateforme ARM64 (Raspberry Pi, etc.)
4. ✅ **Métadonnées automatiques** : Génération automatique des tags Docker
5. ✅ **Cache optimisé** : Utilisation du cache GitHub Actions pour accélérer les builds
6. ✅ **Push conditionnel** : Push uniquement sur les événements non-PR

## 🔒 Configuration des Secrets

Pour utiliser ce workflow, vous devez configurer les secrets suivants dans votre repository GitHub :

1. Aller dans **Settings** → **Secrets and variables** → **Actions**
2. Ajouter les secrets suivants :
   - `DOCKERHUB_USERNAME` : Votre nom d'utilisateur Docker Hub
   - `DOCKERHUB_TOKEN` : Votre Personal Access Token Docker Hub (recommandé au lieu du mot de passe)

### Création d'un Personal Access Token Docker Hub :

1. Se connecter à [Docker Hub](https://hub.docker.com/)
2. Aller dans **Account Settings** → **Security** → **Personal Access Tokens**
3. Cliquer sur **New Access Token**
4. Donner un nom (ex: "GitHub Actions")
5. Sélectionner les permissions nécessaires (Read, Write, Delete)
6. Copier le token généré et l'ajouter comme secret dans GitHub

## 📚 Références

- [GitHub Actions: Contexts](https://docs.github.com/en/actions/learn-github-actions/contexts)
- [Docker Login Action](https://github.com/docker/login-action)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [GitHub Actions: Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

## 🎓 Résumé

**Le problème** : On ne peut pas utiliser `secrets.DOCKERHUB_USERNAME != ''` dans une condition `if` de GitHub Actions.

**La solution** : Utiliser simplement `if: github.event_name != 'pull_request'` et laisser l'action `docker/login-action` gérer les secrets manquants.

**Résultat** : Un workflow qui fonctionne correctement, est sécurisé, et qui échoue proprement si les secrets ne sont pas configurés.
