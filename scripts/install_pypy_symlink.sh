#!/bin/bash
# Install alc-pypy symbolic link for system-wide access
# This script creates a symlink to the alc-pypy wrapper script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Installing alc-pypy symbolic link${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WRAPPER_SCRIPT="$SCRIPT_DIR/alc-pypy-wrapper.sh"

# Verify wrapper script exists
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
    echo -e "${RED}❌ Error: Wrapper script not found at $WRAPPER_SCRIPT${NC}"
    exit 1
fi

# Make wrapper script executable
chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}✅ Made wrapper script executable${NC}"

# Determine installation directory
INSTALL_DIR=""
SYMLINK_PATH=""

# Check common installation directories
if [[ -d "/usr/local/bin" ]] && [[ -w "/usr/local/bin" ]]; then
    INSTALL_DIR="/usr/local/bin"
elif [[ -d "$HOME/.local/bin" ]]; then
    INSTALL_DIR="$HOME/.local/bin"
    # Create directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
else
    # Fallback to creating ~/.local/bin
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    echo -e "${YELLOW}📁 Created directory: $INSTALL_DIR${NC}"
fi

SYMLINK_PATH="$INSTALL_DIR/alc-pypy"

# Function to install with sudo if needed
install_symlink() {
    local use_sudo=false

    # Check if we need sudo for the installation directory
    if [[ "$INSTALL_DIR" == "/usr/local/bin" ]] && [[ ! -w "/usr/local/bin" ]]; then
        use_sudo=true
        echo -e "${YELLOW}🔐 Need sudo access to install in $INSTALL_DIR${NC}"
    fi

    # Remove existing symlink if it exists
    if [[ -L "$SYMLINK_PATH" ]] || [[ -f "$SYMLINK_PATH" ]]; then
        echo -e "${YELLOW}🗑️  Removing existing alc-pypy at $SYMLINK_PATH${NC}"
        if [[ "$use_sudo" == true ]]; then
            sudo rm -f "$SYMLINK_PATH"
        else
            rm -f "$SYMLINK_PATH"
        fi
    fi

    # Create the symlink
    echo -e "${BLUE}🔗 Creating symlink: $SYMLINK_PATH -> $WRAPPER_SCRIPT${NC}"
    if [[ "$use_sudo" == true ]]; then
        sudo ln -s "$WRAPPER_SCRIPT" "$SYMLINK_PATH"
    else
        ln -s "$WRAPPER_SCRIPT" "$SYMLINK_PATH"
    fi

    # Make sure the symlink is executable
    if [[ "$use_sudo" == true ]]; then
        sudo chmod +x "$SYMLINK_PATH"
    else
        chmod +x "$SYMLINK_PATH"
    fi
}

# Install the symlink
install_symlink

# Verify the installation
if [[ -L "$SYMLINK_PATH" ]] && [[ -x "$SYMLINK_PATH" ]]; then
    echo -e "${GREEN}✅ Successfully installed alc-pypy symlink${NC}"

    # Check if installation directory is in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo -e "${YELLOW}⚠️  Warning: $INSTALL_DIR is not in your PATH${NC}"
        echo -e "${YELLOW}   Add this to your ~/.bashrc or ~/.zshrc:${NC}"
        echo -e "${YELLOW}   export PATH=\"$INSTALL_DIR:\$PATH\"${NC}"
        echo
    fi

    # Display usage information
    echo -e "${GREEN}🎉 Installation complete! Usage examples:${NC}"
    echo
    echo -e "${BLUE}  alc-pypy sfr --symbols SPY QQQ --debug${NC}"
    echo -e "${BLUE}  alc-pypy syn --symbols TSLA --cost-limit 100${NC}"
    echo -e "${BLUE}  alc-pypy calendar --symbols AAPL MSFT${NC}"
    echo
    echo -e "${BLUE}📍 Symlink location: $SYMLINK_PATH${NC}"
    echo -e "${BLUE}🎯 Target script: $WRAPPER_SCRIPT${NC}"

    # Test basic functionality (check if alc-pypy environment exists)
    echo
    echo -e "${BLUE}🧪 Testing installation...${NC}"
    if "$SYMLINK_PATH" --help >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Test passed: alc-pypy command works${NC}"
    else
        echo -e "${YELLOW}⚠️  Note: PyPy environment may need setup. Run:${NC}"
        echo -e "${YELLOW}   cd $PROJECT_DIR && ./scripts/setup_pypy_conda.sh${NC}"
    fi

else
    echo -e "${RED}❌ Error: Failed to create symlink${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🚀 alc-pypy is now available system-wide!${NC}"
