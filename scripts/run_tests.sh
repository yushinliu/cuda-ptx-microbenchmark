#!/bin/bash
# Test runner script for CUDA+PTX Microbenchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
TEST_TYPE="all"
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --coverage)
            COVERAGE="1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --e2e           Run only E2E tests"
            echo "  --verbose, -v   Verbose output"
            echo "  --coverage      Enable coverage reporting"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build if necessary
if [[ ! -d "$BUILD_DIR" ]]; then
    print_header "Building project"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    CMAKE_ARGS="-DENABLE_TESTING=ON"
    if [[ -n "$COVERAGE" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=ON"
    fi

    cmake .. $CMAKE_ARGS
    make -j$(nproc)
else
    cd "$BUILD_DIR"
    print_header "Using existing build"
fi

# Run tests
print_header "Running Tests"

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "unit" ]]; then
    echo ""
    echo "Running Unit Tests..."
    if ./tests/unit_tests $VERBOSE; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        exit 1
    fi
fi

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "integration" ]]; then
    echo ""
    echo "Running Integration Tests..."
    if ./tests/integration_tests $VERBOSE; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        exit 1
    fi
fi

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "e2e" ]]; then
    echo ""
    echo "Running E2E Tests..."
    if ./tests/e2e_tests $VERBOSE; then
        print_success "E2E tests passed"
    else
        print_error "E2E tests failed"
        exit 1
    fi
fi

# Coverage report
if [[ -n "$COVERAGE" ]]; then
    print_header "Generating Coverage Report"

    if command -v lcov &> /dev/null; then
        lcov --capture --directory . --output-file coverage.info
        lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage_filtered.info

        echo ""
        echo "Coverage Summary:"
        lcov --summary coverage_filtered.info 2>&1 | grep -E "(lines|functions|branches)"

        if command -v genhtml &> /dev/null; then
            genhtml coverage_filtered.info --output-directory coverage_report
            print_success "Coverage report generated: ${BUILD_DIR}/coverage_report/index.html"
        fi
    else
        print_error "lcov not found, skipping coverage report"
    fi
fi

print_header "All Tests Passed!"
