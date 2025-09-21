#!/bin/bash
# Uninstall alc-pypy symbolic link
# This script removes the alc-pypy symlink from the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗑️  Uninstalling alc-pypy symbolic link${NC}"

# Common installation locations
LOCATIONS=(
    "/usr/local/bin/alc-pypy"
    "$HOME/.local/bin/alc-pypy"
    "/usr/bin/alc-pypy"
)

found_symlinks=0

for location in "${LOCATIONS[@]}"; do
    if [[ -L "$location" ]] || [[ -f "$location" ]]; then
        echo -e "${YELLOW}🔍 Found alc-pypy at: $location${NC}"

        # Check if we need sudo
        dir_name=$(dirname "$location")
        if [[ "$dir_name" == "/usr/local/bin" ]] && [[ ! -w "/usr/local/bin" ]]; then
            echo -e "${YELLOW}🔐 Need sudo access to remove from $dir_name${NC}"
            sudo rm -f "$location"
        else
            rm -f "$location"
        fi

        if [[ ! -e "$location" ]]; then
            echo -e "${GREEN}✅ Removed: $location${NC}"
            ((found_symlinks++))
        else
            echo -e "${RED}❌ Failed to remove: $location${NC}"
        fi
    fi
done

if [[ $found_symlinks -eq 0 ]]; then
    echo -e "${YELLOW}ℹ️  No alc-pypy symlinks found${NC}"
else
    echo -e "${GREEN}🎉 Successfully uninstalled $found_symlinks alc-pypy symlink(s)${NC}"
fi

echo -e "${BLUE}✨ Uninstallation complete${NC}"
