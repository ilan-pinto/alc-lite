#!/bin/bash
# alc-pypy wrapper script - runs alchimest.py with PyPy for enhanced performance
# This script is designed to be symlinked to /usr/local/bin/alc-pypy
# Usage: alc-pypy [alchimest.py arguments...]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the actual location of this script (resolve symlinks)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Function to check and activate conda environment
activate_pypy_env() {
    # Try micromamba first, then conda
    if command -v micromamba &> /dev/null; then
        if ! micromamba env list | grep -q "alc-pypy"; then
            echo -e "${RED}❌ Error: alc-pypy environment not found${NC}"
            echo "Please run ./scripts/setup_pypy_conda.sh first"
            exit 1
        fi

        # Source micromamba initialization
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate alc-pypy

    elif command -v conda &> /dev/null; then
        if ! conda env list | grep -q "alc-pypy"; then
            echo -e "${RED}❌ Error: alc-pypy environment not found${NC}"
            echo "Please run ./scripts/setup_pypy_conda.sh first"
            exit 1
        fi

        # Source conda initialization
        eval "$(conda shell.bash hook)"
        conda activate alc-pypy

    else
        echo -e "${RED}❌ Error: Neither conda nor micromamba found${NC}"
        echo "Please install conda/micromamba and run ./scripts/setup_pypy_conda.sh"
        exit 1
    fi
}

# Function to verify PyPy is working
verify_pypy() {
    if ! pypy3 --version &> /dev/null; then
        echo -e "${RED}❌ Error: PyPy not working in alc-pypy environment${NC}"
        echo "Please run ./scripts/setup_pypy_conda.sh to repair the environment"
        exit 1
    fi
}

# Activate PyPy environment
activate_pypy_env

# Verify PyPy is working
verify_pypy

# Display PyPy version info (only if verbose or debug mode)
if [[ "$*" == *"--debug"* ]] || [[ "$*" == *"-v"* ]] || [[ "$*" == *"--verbose"* ]]; then
    PYPY_VERSION=$(pypy3 -c "import sys; print(f'PyPy {sys.pypy_version_info.major}.{sys.pypy_version_info.minor}.{sys.pypy_version_info.micro}')" 2>/dev/null || echo "PyPy")
    echo -e "${GREEN}🚀 Using $PYPY_VERSION${NC}"
fi

# Run alchimest with PyPy, passing all arguments
exec pypy3 alchimest.py "$@"
