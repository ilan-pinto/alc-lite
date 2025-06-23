#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Alchimist Project installation...${NC}"

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher first.${NC}"
    echo "You can download it from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

# Function to compare version numbers
version_compare() {
    local version1=$1
    local version2=$2
    local IFS=.
    local i ver1=($version1) ver2=($version2)

    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=${#ver2[@]}; i<${#ver1[@]}; i++)); do
        ver2[i]=0
    done

    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ ${ver1[i]} -gt ${ver2[i]} ]]; then
            return 0
        elif [[ ${ver1[i]} -lt ${ver2[i]} ]]; then
            return 1
        fi
    done
    return 0
}

if ! version_compare "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
    echo -e "${RED}Python version $PYTHON_VERSION detected. Please upgrade to Python 3.8 or higher.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${RED}requirements.txt not found!${NC}"
    exit 1
fi

# Create alias
echo -e "${YELLOW}Creating global alias...${NC}"
SCRIPT_PATH="$(pwd)/alchimest.py"
if [ -f "$SCRIPT_PATH" ]; then
    # Add alias to .zshrc (macOS default shell)
    echo "alias alc-lite=\"python $SCRIPT_PATH\"" >> ~/.zshrc
    source ~/.zshrc
    echo -e "${GREEN}Alias 'alc-lite' has been created and added to your shell configuration.${NC}"
else
    echo -e "${RED}alchimest.py not found in the current directory!${NC}"
    exit 1
fi

echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${YELLOW}To start using the software:${NC}"
echo "1. Open a new terminal window or run: source ~/.zshrc"
echo "2. Type 'alc-lite' to run the program"
echo -e "${YELLOW}Note:${NC} Make sure to activate the virtual environment when working on the project:"
echo "source venv/bin/activate"
