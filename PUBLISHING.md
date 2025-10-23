# Publishing to PyPI

This project uses GitHub Actions to automatically publish to PyPI when you create a release.

## Setup (One-time only)

### Step 1: Create a GitHub Environment

1. Go to your repository settings: https://github.com/AratoAi/arato-python-sdk/settings/environments
2. Click "New environment"
3. Name it: `pypi`
4. (Optional) Add protection rules:
   - Required reviewers: Add maintainers who should approve releases
   - Wait timer: Add a delay before deployment if desired
5. Click "Configure environment"

### Step 2: Set up Trusted Publishing on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `arato-client`
   - **Owner**: `AratoAi`
   - **Repository name**: `arato-python-sdk`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi` (must match the GitHub environment you created)
4. Save

## How to Publish a New Version

1. **Update the version** in `pyproject.toml`:
   ```toml
   version = "1.0.4"  # Increment as needed
   ```

2. **Commit and push** your changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 1.0.4"
   git push origin main
   ```

3. **Create a GitHub Release**:
   - Go to https://github.com/AratoAi/arato-python-sdk/releases
   - Click "Draft a new release"
   - Choose a tag: `v1.0.4` (must match the version in pyproject.toml)
   - Release title: `v1.0.4`
   - Description: Add release notes describing what's new
   - Click "Publish release"

4. **GitHub Actions will automatically**:
   - Build the package
   - Publish to PyPI
   - The package will be available at https://pypi.org/project/arato-client/

## Verifying the Publication

After the release:
- Check the Actions tab: https://github.com/AratoAi/arato-python-sdk/actions
- Verify on PyPI: https://pypi.org/project/arato-client/
- Test installation: `pip install arato-client==1.0.4`

## Version Numbering

Follow semantic versioning (https://semver.org/):
- **MAJOR** version (1.x.x): Breaking changes
- **MINOR** version (x.1.x): New features, backwards compatible
- **PATCH** version (x.x.1): Bug fixes, backwards compatible
