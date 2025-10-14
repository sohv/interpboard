# PyPI Deployment Guide for InterpBoard

This guide provides step-by-step instructions for deploying **InterpBoard** to PyPI.

## 🎯 Quick Deployment Summary

To deploy InterpBoard to PyPI:

1. **Build**: `python -m build`
2. **Check**: `twine check dist/*`
3. **Test Upload**: `twine upload --repository testpypi dist/*`
4. **Production Upload**: `twine upload dist/*`
5. **Verify**: `pip install interpboard`

## 📋 Prerequisites

### 1. Create PyPI Accounts
- **Production PyPI**: https://pypi.org/account/register/
- **Test PyPI**: https://test.pypi.org/account/register/ (for testing)

### 2. Install Build Tools
```bash
pip install --upgrade pip setuptools wheel build twine
```

### 3. Configure API Tokens (Recommended)

#### Create API Tokens:
1. Go to PyPI Account Settings → API Tokens
2. Create token with scope "Entire account" or specific project
3. Copy the token (starts with `pypi-`)

#### Configure tokens:
```bash
# Create/edit ~/.pypirc
cat > ~/.pypirc << EOF
[distutils]
index-servers = pypi testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
EOF

# Secure the file
chmod 600 ~/.pypirc
```

## 🚀 Deployment Steps

### Step 1: Pre-deployment Checklist

```bash
cd /home/azureuser/interpret-llm

# 1. Update version in setup.py if needed
# 2. Ensure all files are committed to git
git add .
git commit -m "Prepare for PyPI release v0.1.0"
git tag v0.1.0
git push origin main --tags

# 3. Clean previous builds
rm -rf dist/ build/ *.egg-info/

# 4. Test installation locally
pip install -e .
python -c "import interpboard; print('Import successful!')"
```

### Step 2: Build the Package

```bash
# Build source distribution and wheel
python -m build

# This creates:
# dist/interpboard-0.1.0.tar.gz (source distribution)
# dist/interpboard-0.1.0-py3-none-any.whl (wheel)
```

### Step 3: Check the Package

```bash
# Verify the package
twine check dist/*

# Should show:
# Checking dist/interpboard-0.1.0.tar.gz: PASSED
# Checking dist/interpboard-0.1.0-py3-none-any.whl: PASSED
```

### Step 4: Test Upload (Recommended)

```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ interpboard

# Test the installation
python -c "
from interpboard.dashboards import create_unified_dashboard
print('Test PyPI installation successful!')
"

# Uninstall test version
pip uninstall interpboard -y
```

### Step 5: Production Upload

```bash
# Upload to production PyPI
twine upload dist/*

# You should see:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading interpboard-0.1.0.tar.gz
# Uploading interpboard-0.1.0-py3-none-any.whl
```

### Step 6: Verify Production Installation

```bash
# Test installation from production PyPI
pip install interpboard

# Verify it works
python -c "
from interpboard.dashboards import create_unified_dashboard
print('Production PyPI installation successful!')
print('Package available at: https://pypi.org/project/interpboard/')
"
```

## 🔄 Updating Your Package

### For Bug Fixes (Patch Version: 0.1.0 → 0.1.1)

```bash
# 1. Update version in setup.py
sed -i 's/version="0.1.0"/version="0.1.1"/' setup.py

# 2. Clean and rebuild
rm -rf dist/ build/ *.egg-info/
python -m build

# 3. Upload
twine upload dist/*
```

### For New Features (Minor Version: 0.1.1 → 0.2.0)

```bash
# 1. Update version
sed -i 's/version="0.1.1"/version="0.2.0"/' setup.py

# 2. Update changelog/docs
# 3. Follow build and upload process
```

### For Breaking Changes (Major Version: 0.2.0 → 1.0.0)

```bash
# 1. Update version
sed -i 's/version="0.2.0"/version="1.0.0"/' setup.py

# 2. Update classifiers if moving from Alpha to Stable
sed -i 's/Development Status :: 3 - Alpha/Development Status :: 5 - Production\/Stable/' setup.py

# 3. Follow build and upload process
```

## 🛠️ Automation with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Publish to Test PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
      run: twine upload --repository testpypi dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## 📊 Package Statistics

After deployment, you can track your package:

- **PyPI Page**: https://pypi.org/project/interpboard/
- **Download Stats**: Use `pypistats` package
- **Security Scanning**: Automatic vulnerability scanning

```bash
# Track downloads
pip install pypistats
pypistats recent interpboard
```

## 🔍 Troubleshooting

### Common Issues:

1. **Package name already exists**
   ```bash
   # Change name in setup.py
   name="interpboard-yourname"
   ```

2. **Version already exists**
   ```bash
   # Increment version number
   version="0.1.1"
   ```

3. **Dependencies issues**
   ```bash
   # Test in clean environment
   python -m venv test_env
   source test_env/bin/activate
   pip install interpboard
   ```

4. **Large package size**
   ```bash
   # Check what's included
   tar -tzf dist/interpboard-0.1.0.tar.gz
   
   # Update MANIFEST.in to exclude large files
   ```

## ✅ Success Checklist

- [ ] Package builds without errors
- [ ] All checks pass (`twine check`)
- [ ] Test PyPI upload successful
- [ ] Test installation from Test PyPI works
- [ ] Production PyPI upload successful  
- [ ] Final installation test from production PyPI
- [ ] Package appears on https://pypi.org/project/interpboard/
- [ ] CLI command works: `interpboard --help`

Your package is now live on PyPI! Users can install it with:
```bash
pip install interpboard
```

## 🚀 InterpBoard-Specific Deployment Instructions

### Current Package Status
- **Package Name**: `interpboard`
- **CLI Command**: `interpboard`
- **Python Import**: `from interpboard.dashboards import create_unified_dashboard`
- **Current Version**: 0.1.0

### Ready-to-Deploy Commands

```bash
# 1. Navigate to project directory
cd /home/azureuser/interpret-llm

# 2. Clean previous builds
rm -rf dist/ build/ *.egg-info/

# 3. Build the package
python -m build

# 4. Check the package
twine check dist/*

# 5. Upload to Test PyPI (optional but recommended)
twine upload --repository testpypi dist/*

# 6. Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ interpboard

# 7. Test the CLI
interpboard --help

# 8. Test Python imports
python -c "from interpboard.dashboards import create_unified_dashboard; print('✅ InterpBoard ready!')"

# 9. Upload to Production PyPI
twine upload dist/*

# 10. Final verification
pip uninstall interpboard -y
pip install interpboard
interpboard --help
```

### Post-Deployment Verification

After successful deployment, verify that:

1. **Package is available**: https://pypi.org/project/interpboard/
2. **Installation works**: `pip install interpboard`
3. **CLI is functional**: `interpboard analyze --help`
4. **Python imports work**: `from interpboard.dashboards import create_unified_dashboard`
5. **Dependencies are resolved**: Core ML libraries (torch, transformers) install correctly

### Marketing Your Package

Once deployed, promote InterpBoard:

1. **GitHub README**: Ensure the installation instructions show `pip install interpboard`
2. **Documentation**: Update all docs to reference the PyPI package
3. **Social Media**: Share the release with `#InterpBoard #LLMInterpretability #PyPI`
4. **Research Community**: Post in ML/AI forums and communities

### Version Management

InterpBoard follows semantic versioning:
- **Patch** (0.1.0 → 0.1.1): Bug fixes, documentation updates
- **Minor** (0.1.1 → 0.2.0): New features, backward-compatible changes  
- **Major** (0.2.0 → 1.0.0): Breaking changes, API redesign

Update version in `setup.py` before each release:
```python
version="0.1.1"  # Increment appropriately
```