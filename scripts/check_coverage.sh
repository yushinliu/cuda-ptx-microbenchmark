#!/bin/bash
# Coverage check script with 80% threshold

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# Thresholds
LINE_THRESHOLD=80
FUNCTION_THRESHOLD=80
BRANCH_THRESHOLD=80

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if coverage data exists
if [[ ! -f "${BUILD_DIR}/coverage_filtered.info" ]]; then
    echo "Coverage data not found. Running tests with coverage..."
    "${SCRIPT_DIR}/run_tests.sh" --coverage
fi

cd "$BUILD_DIR"

# Extract coverage percentages
LINE_PCT=$(lcov --summary coverage_filtered.info 2>&1 | grep "lines" | grep -oP '\d+\.?\d*' | head -1)
FUNC_PCT=$(lcov --summary coverage_filtered.info 2>&1 | grep "functions" | grep -oP '\d+\.?\d*' | head -1)
BRANCH_PCT=$(lcov --summary coverage_filtered.info 2>&1 | grep "branches" | grep -oP '\d+\.?\d*' | head -1)

echo "========================================"
echo "Coverage Report"
echo "========================================"
echo "Lines:     ${LINE_PCT}% (target: ${LINE_THRESHOLD}%)"
echo "Functions: ${FUNC_PCT}% (target: ${FUNCTION_THRESHOLD}%)"
echo "Branches:  ${BRANCH_PCT}% (target: ${BRANCH_THRESHOLD}%)"
echo "========================================"

# Check thresholds
PASS=1

if (( $(echo "$LINE_PCT < $LINE_THRESHOLD" | bc -l) )); then
    echo -e "${RED}✗ Line coverage below threshold${NC}"
    PASS=0
else
    echo -e "${GREEN}✓ Line coverage meets threshold${NC}"
fi

if (( $(echo "$FUNC_PCT < $FUNCTION_THRESHOLD" | bc -l) )); then
    echo -e "${RED}✗ Function coverage below threshold${NC}"
    PASS=0
else
    echo -e "${GREEN}✓ Function coverage meets threshold${NC}"
fi

if (( $(echo "$BRANCH_PCT < $BRANCH_THRESHOLD" | bc -l) )); then
    echo -e "${RED}✗ Branch coverage below threshold${NC}"
    PASS=0
else
    echo -e "${GREEN}✓ Branch coverage meets threshold${NC}"
fi

if [[ $PASS -eq 1 ]]; then
    echo ""
    echo -e "${GREEN}All coverage thresholds met!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some coverage thresholds not met.${NC}"
    exit 1
fi
