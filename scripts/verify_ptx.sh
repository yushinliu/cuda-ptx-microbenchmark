#!/bin/bash
# PTX assembly verification script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PTX_DIR="${PROJECT_DIR}/src/kernels/ptx"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Architecture target (RTX 4070 = sm_89)
ARCH="sm_89"

echo "========================================"
echo "PTX Verification"
echo "Architecture: ${ARCH}"
echo "========================================"

# Check for ptxas
if ! command -v ptxas &> /dev/null; then
    echo -e "${YELLOW}Warning: ptxas not found in PATH, trying CUDA path${NC}"
    CUDA_PATH="/usr/local/cuda/bin"
    if [[ -x "${CUDA_PATH}/ptxas" ]]; then
        export PATH="${CUDA_PATH}:${PATH}"
    else
        echo -e "${RED}Error: ptxas not found${NC}"
        exit 1
    fi
fi

# Verify each PTX file
FAILED=0
VERIFIED=0

if [[ -d "$PTX_DIR" ]]; then
    for ptx_file in "$PTX_DIR"/*.ptx; do
        if [[ -f "$ptx_file" ]]; then
            filename=$(basename "$ptx_file")
            echo -n "Verifying: $filename ... "

            if ptxas -arch=$ARCH "$ptx_file" -o /dev/null 2>&1; then
                echo -e "${GREEN}OK${NC}"
                ((VERIFIED++))
            else
                echo -e "${RED}FAILED${NC}"
                ((FAILED++))
            fi
        fi
    done
fi

echo "========================================"
echo "Verified: $VERIFIED"
if [[ $FAILED -gt 0 ]]; then
    echo -e "Failed: ${RED}$FAILED${NC}"
    exit 1
else
    echo -e "Failed: ${GREEN}0${NC}"
    echo -e "${GREEN}All PTX files verified successfully!${NC}"
    exit 0
fi
