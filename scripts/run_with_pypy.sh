#!/bin/bash
# Convenience script to run alc-lite with PyPy for enhanced performance
# Usage: ./scripts/run_with_pypy.sh [alchimest.py arguments...]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏎️ Running alc-lite with PyPy for enhanced performance${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: conda is not installed or not in PATH${NC}"
    echo "Please run ./scripts/setup_pypy_conda.sh first"
    exit 1
fi

# Check if alc-pypy environment exists
if ! conda env list | grep -q "alc-pypy"; then
    echo -e "${RED}❌ Error: alc-pypy conda environment not found${NC}"
    echo "Please run ./scripts/setup_pypy_conda.sh first to set up PyPy"
    exit 1
fi

# Activate the PyPy environment
echo -e "${YELLOW}🔄 Activating alc-pypy conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate alc-pypy

# Verify PyPy is working
if ! pypy3 --version &> /dev/null; then
    echo -e "${RED}❌ Error: PyPy not working in alc-pypy environment${NC}"
    echo "Please run ./scripts/setup_pypy_conda.sh to repair the environment"
    exit 1
fi

# Display PyPy version info
PYPY_VERSION=$(pypy3 -c "import sys; print(f'PyPy {sys.pypy_version_info.major}.{sys.pypy_version_info.minor}.{sys.pypy_version_info.micro}')")
echo -e "${GREEN}🚀 Using $PYPY_VERSION${NC}"

# Check if this is a performance-critical operation
PERF_HINT=""
if [[ "$*" == *"sfr"* ]] || [[ "$*" == *"syn"* ]]; then
    PERF_HINT="\n${GREEN}💡 Performance Tip: PyPy provides the biggest speedup for options chain processing and arbitrage detection${NC}"
fi

# JIT warmup message for longer scans
if [[ "$*" == *"--symbols"* ]]; then
    SYMBOL_COUNT=$(echo "$*" | grep -o '\--symbols[^-]*' | wc -w)
    if [ "$SYMBOL_COUNT" -gt 3 ]; then
        echo -e "${YELLOW}⏱️  Note: PyPy JIT will warm up during the first few symbols for optimal performance${NC}"
    fi
fi

echo -e "${BLUE}🎯 Running: pypy3 alchimest.py $*${NC}"
echo

# Record start time for performance tracking
START_TIME=$(date +%s)

# Run alchimest with PyPy, passing all arguments
pypy3 alchimest.py "$@"

# Calculate and display runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo
echo -e "${GREEN}✅ Execution completed in ${RUNTIME} seconds with PyPy${NC}"

if [ "$RUNTIME" -gt 30 ]; then
    echo -e "${BLUE}📊 For performance comparison, try running the same command with CPython:${NC}"
    echo -e "${BLUE}   python alchimest.py $*${NC}"
fi

if [ -n "$PERF_HINT" ]; then
    echo -e "$PERF_HINT"
fi

echo -e "${BLUE}🔧 To benchmark PyPy vs CPython performance:${NC}"
echo -e "${BLUE}   ./benchmarks/compare_runtimes.sh${NC}"
