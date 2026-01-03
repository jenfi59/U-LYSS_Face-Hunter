# GitHub Actions Setup Guide

This guide explains how to set up GitHub Actions for the D-Face Hunter ARM64 project, including Docker Hub integration.

## 📋 Overview

The project includes a GitHub Actions workflow (`docker-build-push.yml`) that:
- Builds Docker images for ARM64 architecture
- Pushes images to Docker Hub on push events (not on pull requests)
- Uses proper secret handling for Docker Hub credentials

## 🔧 Initial Setup

### 1. Configure Docker Hub Secrets

To enable Docker Hub integration, you need to add two secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

#### DOCKERHUB_USERNAME
- **Name**: `DOCKERHUB_USERNAME`
- **Value**: Your Docker Hub username

#### DOCKERHUB_TOKEN
- **Name**: `DOCKERHUB_TOKEN`
- **Value**: Your Docker Hub access token (see below for creation)

### 2. Create a Docker Hub Access Token

**Important**: Use an access token instead of your Docker Hub password for better security.

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Click on your username → **Account Settings**
3. Navigate to **Security** → **Personal Access Tokens**
4. Click **New Access Token**
5. Configure the token:
   - **Token Description**: `GitHub Actions - D-Face Hunter`
   - **Access permissions**: Select `Read, Write, Delete` (or just `Read, Write` if you don't need delete)
6. Click **Generate**
7. **Copy the token** - you won't be able to see it again!
8. Add this token as `DOCKERHUB_TOKEN` secret in GitHub

### 3. Create a Dockerfile (Optional)

If you want to enable Docker builds, you need to create a `Dockerfile` in the repository root. An example is provided in `Dockerfile.example`.

To use the example:
```bash
cp Dockerfile.example Dockerfile
# Edit Dockerfile as needed for your specific requirements
```

## 📝 Workflow Configuration

### Current Workflow: `docker-build-push.yml`

The workflow is triggered on:
- **Push** to `main` or `develop` branches
- **Push** of version tags (e.g., `v1.0.0`)
- **Pull requests** to `main`
- **Manual trigger** via workflow_dispatch

### Key Features

#### ✅ Correct Secret Handling

The workflow uses the correct syntax for checking secrets:

```yaml
- name: Log in to Docker Hub
  if: github.event_name != 'pull_request'
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

**Why this is correct:**
- The condition only checks the event type, not the secret directly
- The `docker/login-action` handles missing secrets gracefully
- Secrets are only used in the `with:` parameters, not in conditions

#### ❌ Incorrect Syntax (DO NOT USE)

```yaml
# WRONG - This will cause errors!
if: github.event_name != 'pull_request' && secrets.DOCKERHUB_USERNAME != ''
```

**Why this is wrong:**
- GitHub Actions doesn't allow direct secret comparisons in `if` conditions
- This syntax will cause the workflow to fail
- It's a potential security risk

### ARM64 Support

The workflow includes QEMU setup for ARM64 builds:

```yaml
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3
  with:
    platforms: linux/arm64
```

This allows building ARM64 images even on x86_64 GitHub runners.

## 🚀 Usage

### Automatic Builds

The workflow runs automatically on:
- Push to `main` or `develop` → builds and pushes with branch name as tag
- Push of version tag (e.g., `v1.2.1`) → builds and pushes with semantic version tags
- Pull request → builds only (no push to registry)

### Manual Trigger

You can also trigger the workflow manually:
1. Go to **Actions** tab in GitHub
2. Select **Docker Build and Push** workflow
3. Click **Run workflow**
4. Select the branch
5. Click **Run workflow**

## 📦 Docker Image Tags

The workflow automatically generates tags based on the event:

| Event | Example Tag(s) |
|-------|---------------|
| Push to `main` | `latest`, `main` |
| Push to `develop` | `develop` |
| Tag `v1.2.1` | `1.2.1`, `1.2`, `latest` |
| Pull request #42 | `pr-42` |
| Commit SHA | `sha-abc1234` |

## 🧪 Testing

### Test Without Docker Hub

To test the workflow without pushing to Docker Hub:
1. Comment out or remove the Docker Hub login step
2. Set `push: false` in the build step
3. Push to a test branch

### Verify Secret Configuration

To verify secrets are configured correctly:
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. You should see:
   - `DOCKERHUB_USERNAME` (Hidden)
   - `DOCKERHUB_TOKEN` (Hidden)
3. Note: You cannot view secret values after creation

### Check Workflow Status

After pushing code:
1. Go to the **Actions** tab
2. Find your workflow run
3. Check each step's status
4. Review logs for any errors

## 🔍 Troubleshooting

### Error: "Error logging in to Docker Hub"

**Cause**: Missing or incorrect Docker Hub secrets

**Solution**:
1. Verify secrets are set in repository settings
2. Ensure `DOCKERHUB_TOKEN` is a valid access token (not password)
3. Check token hasn't expired
4. Verify token has necessary permissions

### Error: "Error parsing workflow file"

**Cause**: YAML syntax error

**Solution**:
1. Check YAML indentation (use spaces, not tabs)
2. Validate YAML syntax using an online validator
3. Review the error message in the Actions tab

### Error: "No Dockerfile found"

**Cause**: Dockerfile not present in repository

**Solution**:
1. Create a Dockerfile (use `Dockerfile.example` as template)
2. Uncomment the build step in the workflow
3. Commit and push the Dockerfile

### Pull Request Builds Fail

**Cause**: Trying to push images during PR (should be prevented by condition)

**Solution**:
- Verify the `if: github.event_name != 'pull_request'` condition is present
- Check that `push: ${{ github.event_name != 'pull_request' }}` is set correctly

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Login Action](https://github.com/docker/login-action)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Docker Hub Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [Working with Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

## 🎓 Key Takeaways

1. **Never compare secrets directly in `if` conditions**
2. **Use access tokens instead of passwords for Docker Hub**
3. **Test workflows on feature branches before merging**
4. **Keep secrets secure - never commit them to code**
5. **Use conditional push to avoid pushing on pull requests**

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review workflow logs in the Actions tab
3. Consult the [Docker Hub Login Fix documentation](DOCKER_HUB_LOGIN_FIX.md)
4. Open an issue in the repository

---

**Last Updated**: January 3, 2026
