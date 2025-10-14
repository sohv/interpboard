"""
LLM Interpretability Dashboard - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="interpboard",
    version="0.1.0",
    author="LLM Interpretability Team",
    author_email="contact@example.com",  # Replace with your email
    description="A comprehensive toolkit for interpreting and understanding large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/interpboard",  # Replace with your GitHub URL
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/interpboard/issues",
        "Documentation": "https://github.com/yourusername/interpboard/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/interpboard",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core ML dependencies
        "torch>=1.13.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # Text processing and display
        "rich>=12.0.0",
        "termcolor>=1.1.0",
        
        # Data handling
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        
        # Optional: web interfaces
        "streamlit>=1.20.0",
        "gradio>=3.0.0",
        
        # Jupyter support
        "ipywidgets>=7.6.0",
        "jupyter>=1.0.0",
        
        # Utils
        "tqdm>=4.62.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "interpboard=interpboard.cli:main",
        ],
    },
)