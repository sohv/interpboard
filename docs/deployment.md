# Deployment Guide

This guide covers how to deploy the LLM Interpretability Dashboard as a library or package for different use cases.

## 📦 Package Deployment Options

### 1. PyPI Package (Recommended for Public Release)

#### Prerequisites
```bash
# Install build and upload tools
pip install build twine

# Ensure you have accounts on:
# - PyPI: https://pypi.org/account/register/
# - Test PyPI: https://test.pypi.org/account/register/
```

#### Build and Upload Process
```bash
# 1. Clean previous builds
rm -rf dist/ build/ *.egg-info/

# 2. Build the package
python -m build

# 3. Check the package
twine check dist/*

# 4. Upload to Test PyPI first
twine upload --repository testpypi dist/*

# 5. Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ interpret-llm

# 6. Upload to production PyPI
twine upload dist/*
```

#### Configuration for PyPI
Update `setup.py` for production:

```python
setup(
    name="interpret-llm",
    version="1.0.0",  # Update version
    author="Your Name",
    author_email="your-email@example.com",
    description="A comprehensive toolkit for interpreting large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/interpret-llm",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/interpret-llm/issues",
        "Documentation": "https://interpret-llm.readthedocs.io/",
        "Source Code": "https://github.com/your-username/interpret-llm",
    },
    # ... rest of setup configuration
)
```

### 2. Conda Package

#### Create conda-forge recipe
```bash
# 1. Fork conda-forge/staged-recipes
# 2. Create recipe in recipes/interpret-llm/meta.yaml
```

#### meta.yaml example:
```yaml
{% set name = "interpret-llm" %}
{% set version = "1.0.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/interpret-llm-{{ version }}.tar.gz
  sha256: <SHA256_HASH>

build:
  noarch: python
  script: {{ PYTHON }} -m pip install . -vv
  number: 0

requirements:
  host:
    - python >=3.8
    - pip
  run:
    - python >=3.8
    - torch >=1.13.0
    - transformers >=4.20.0
    - matplotlib >=3.5.0
    - plotly >=5.0.0
    - rich >=12.0.0
    # ... other dependencies

test:
  imports:
    - interpboard
  commands:
    - pip check
  requires:
    - pip

about:
  home: https://github.com/your-username/interpret-llm
  summary: A comprehensive toolkit for interpreting large language models
  license: MIT
  license_file: LICENSE
```

### 3. Docker Container

#### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Expose port for web interfaces
EXPOSE 8501 7860

# Default command
CMD ["python", "-m", "interpboard.cli"]
```

#### Build and Deploy Docker Image
```bash
# Build the image
docker build -t interpret-llm:latest .

# Run locally
docker run -it --gpus all -p 8501:8501 interpret-llm:latest

# Push to Docker Hub
docker tag interpret-llm:latest your-username/interpret-llm:latest
docker push your-username/interpret-llm:latest

# Push to GitHub Container Registry
docker tag interpret-llm:latest ghcr.io/your-username/interpret-llm:latest
docker push ghcr.io/your-username/interpret-llm:latest
```

### 4. GitHub Releases

#### Automated Release with GitHub Actions
Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
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
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## 🚀 Specialized Deployment Scenarios

### 1. Enterprise/Internal Package

#### Private PyPI Server
```bash
# Set up private package index
pip install devpi-server

# Upload to private server
twine upload --repository-url https://your-private-pypi.com dist/*

# Install from private server
pip install --index-url https://your-private-pypi.com interpret-llm
```

#### Git-based Installation
```bash
# Direct from GitHub
pip install git+https://github.com/your-username/interpret-llm.git

# Specific branch/tag
pip install git+https://github.com/your-username/interpret-llm.git@v1.0.0

# With SSH
pip install git+ssh://git@github.com/your-username/interpret-llm.git
```

### 2. Web Application Deployment

#### Streamlit App
Create `streamlit_app.py`:
```python
import streamlit as st
from interpret_llm.dashboards import create_unified_dashboard

st.title("LLM Interpretability Dashboard")

model_name = st.selectbox("Choose Model", ["gpt2", "gpt2-medium", "EleutherAI/gpt-neo-125M"])
text = st.text_input("Enter text to analyze")

if st.button("Analyze"):
    with st.spinner("Loading model..."):
        attribution_dashboard, _ = create_unified_dashboard(model_name)
    
    with st.spinner("Running analysis..."):
        result = attribution_dashboard.analyze(text, visualize=False)
    
    st.plotly_chart(result.visualizations["integrated_gradients_heatmap"])
```

Deploy with:
```bash
# Local
streamlit run streamlit_app.py

# Streamlit Cloud
# Push to GitHub and deploy via streamlit.io

# Docker
docker run -p 8501:8501 interpret-llm:latest streamlit run streamlit_app.py
```

#### Gradio Interface
Create `gradio_app.py`:
```python
import gradio as gr
from interpret_llm.dashboards import create_unified_dashboard

def analyze_text(text, model_name):
    attribution_dashboard, _ = create_unified_dashboard(model_name)
    result = attribution_dashboard.analyze(text, visualize=False)
    return result.visualizations["integrated_gradients_heatmap"]

iface = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(label="Text to analyze"),
        gr.Dropdown(["gpt2", "gpt2-medium"], label="Model")
    ],
    outputs=gr.Plot(label="Attribution Heatmap"),
    title="LLM Interpretability Dashboard"
)

if __name__ == "__main__":
    iface.launch()
```

### 3. Cloud Deployment

#### AWS Lambda (for lightweight analysis)
```python
# lambda_function.py
import json
from interpret_llm.dashboards import create_unified_dashboard

def lambda_handler(event, context):
    text = event['text']
    model_name = event.get('model', 'gpt2')
    
    attribution_dashboard, _ = create_unified_dashboard(model_name)
    result = attribution_dashboard.analyze(text, visualize=False)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'attributions': result.attribution_results
        })
    }
```

#### Google Cloud Run
```dockerfile
# Dockerfile for Cloud Run
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8080
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app
```

#### Azure Container Instances
```bash
# Deploy to Azure
az container create \
    --resource-group myResourceGroup \
    --name interpret-llm \
    --image interpret-llm:latest \
    --dns-name-label interpret-llm-demo \
    --ports 8501
```

## 📋 Pre-Deployment Checklist

### Code Quality
- [ ] All tests pass: `pytest tests/`
- [ ] Code formatting: `black interpret_llm/`
- [ ] Linting: `flake8 interpret_llm/`
- [ ] Type checking: `mypy interpret_llm/`

### Documentation
- [ ] README.md is complete and accurate
- [ ] API documentation is generated
- [ ] Examples run without errors
- [ ] Installation instructions are tested

### Security
- [ ] No hardcoded secrets or API keys
- [ ] Dependencies are up to date
- [ ] Security scan passed: `safety check`

### Performance
- [ ] Memory usage is reasonable
- [ ] GPU memory is properly managed
- [ ] Large model support is tested

### Legal/Compliance
- [ ] License is included
- [ ] Dependencies licenses are compatible
- [ ] Attribution is correct

## 🔧 Maintenance and Updates

### Version Management
```bash
# Semantic versioning
# MAJOR.MINOR.PATCH
# 1.0.0 -> 1.0.1 (patch)
# 1.0.1 -> 1.1.0 (minor)
# 1.1.0 -> 2.0.0 (major)

# Update version in setup.py
# Create git tag
git tag v1.0.1
git push origin v1.0.1
```

### Automated Testing
Set up CI/CD with GitHub Actions (`.github/workflows/ci.yml`):

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest tests/ --cov=interpret_llm
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

This comprehensive deployment guide covers all major scenarios for distributing your LLM interpretability library!