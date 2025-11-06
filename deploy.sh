#!/bin/bash
# PyPI Deployment Script for InterpBoard

set -e  # Exit on any error

echo "Starting PyPI deployment for InterpBoard..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found. Please run this script from the project root."
    exit 1
fi

# Check if required tools are installed
if ! command -v twine &> /dev/null; then
    echo "ERROR: twine not found. Please install with: pip install twine"
    exit 1
fi

if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found."
    exit 1
fi

# Use python3 if python is not available
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Pre-deployment checklist:"
echo "  - Checking setup.py..."
echo "  - Using Python: $($PYTHON_CMD --version)"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build the package
echo "Building package..."
$PYTHON_CMD -m build

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Check the package
echo "Checking package quality..."
twine check dist/*

if [ $? -ne 0 ]; then
    echo "Package check failed!"
    exit 1
fi

echo "Package quality check passed!"

# Show what was built
echo "Built packages:"
ls -la dist/

# Ask user about deployment target
echo ""
echo "Choose deployment target:"
echo "1) Test PyPI (recommended first)"
echo "2) Production PyPI"
echo "3) Both (Test first, then Production)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Uploading to Test PyPI..."
        twine upload --repository testpypi dist/*
        echo "Uploaded to Test PyPI!"
        echo "View at: https://test.pypi.org/project/interpboard/"
        echo "Test install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ interpboard"
        ;;
    2)
        echo "Uploading to Production PyPI..."
        twine upload dist/*
        echo "Uploaded to Production PyPI!"
        echo "View at: https://pypi.org/project/interpboard/"
        echo "Install with: pip install interpboard"
        ;;
    3)
        echo "Uploading to Test PyPI first..."
        twine upload --repository testpypi dist/*
        echo "Test upload successful!"
        
        read -p "Test installation successful? Continue to production? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "Uploading to Production PyPI..."
            twine upload dist/*
            echo "Uploaded to Production PyPI!"
            echo "View at: https://pypi.org/project/interpboard/"
            echo "Install with: pip install interpboard"
        else
            echo "Production upload cancelled."
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "The InterpBoard is ready to use!"
echo ""
echo "Your package statistics:"
echo "  • Package name: interpboard"
echo "  • Version: $(grep 'version=' setup.py | cut -d'"' -f2)"
echo "  • Python support: $(grep 'python_requires=' setup.py | cut -d'"' -f2)"
echo ""
echo "To update your package:"
echo "  1. Update version in setup.py"
echo "  2. Run this script again"
echo ""
echo "Track downloads with: pip install pypistats && pypistats recent interpboard"