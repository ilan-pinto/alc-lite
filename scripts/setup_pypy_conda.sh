#!/bin/bash
# PyPy Environment Setup Script for alc-lite
# Creates conda environment with PyPy for 2-10x performance improvements

set -e  # Exit on any error

echo "🏎️ Setting up PyPy environment for alc-lite..."
echo "This will create a conda environment 'alc-pypy' with PyPy 3.10"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "alc-pypy"; then
    echo "⚠️  Environment 'alc-pypy' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n alc-pypy -y
    else
        echo "ℹ️  Using existing environment"
        conda activate alc-pypy
        exit 0
    fi
fi

echo "📦 Creating conda environment with PyPy..."
conda create -n alc-pypy -c conda-forge pypy3.10 -y

echo "🔧 Activating environment and setting up pip..."
eval "$(conda shell.bash hook)"
conda activate alc-pypy

# Ensure pip is available and up to date
pypy3 -m ensurepip --default-pip
pypy3 -m pip install --upgrade pip setuptools wheel

echo "📋 Installing PyPy-compatible dependencies..."
if [ -f "requirements-pypy.txt" ]; then
    pypy3 -m pip install -r requirements-pypy.txt
else
    echo "⚠️  requirements-pypy.txt not found, installing core dependencies..."
    # Install critical dependencies that we know work with PyPy
    pypy3 -m pip install \
        ib_async==1.0.1 \
        pandas \
        rich \
        asyncpg \
        pyfiglet \
        tenacity \
        tabulate
fi

echo
echo "✅ PyPy environment setup complete!"
echo
echo "🚀 To use PyPy with alc-lite:"
echo "   conda activate alc-pypy"
echo "   pypy3 alchimest.py sfr --symbols SPY --debug"
echo
echo "🧪 To run performance benchmarks:"
echo "   conda activate alc-pypy"
echo "   ./benchmarks/compare_runtimes.sh"
echo
echo "📊 Expected performance improvements:"
echo "   • Options chain processing: 3-5x faster"
echo "   • Arbitrage detection: 2-4x faster"
echo "   • Memory efficiency: Improved for long scans"
echo

# Test PyPy installation
echo "🔍 Testing PyPy installation..."
if pypy3 -c "import sys; print(f'PyPy {sys.pypy_version_info} ready!')"; then
    echo "✅ PyPy test successful"
else
    echo "❌ PyPy test failed"
    exit 1
fi

echo
echo "🎯 Next steps:"
echo "1. Activate the environment: conda activate alc-pypy"
echo "2. Test with a simple scan: pypy3 alchimest.py sfr --symbols SPY"
echo "3. Run benchmarks to see performance gains"
