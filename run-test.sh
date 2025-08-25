#!/bin/bash

# Friend-Lite Local Test Runner
# Runs the same tests as GitHub CI but configured for local development
# Usage: ./run-test.sh [advanced-backend|speaker-recognition|all]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run advanced backend tests
run_advanced_backend_tests() {
    print_info "Running Advanced Backend Integration Tests..."
    
    if [ ! -f "backends/advanced/run-test.sh" ]; then
        print_error "backends/advanced/run-test.sh not found!"
        return 1
    fi
    
    cd backends/advanced
    ./run-test.sh
    cd ../..
    
    print_success "Advanced Backend tests completed"
}

# Function to run speaker recognition tests
run_speaker_recognition_tests() {
    print_info "Running Speaker Recognition Tests..."
    
    if [ ! -f "extras/speaker-recognition/run-test.sh" ]; then
        print_error "extras/speaker-recognition/run-test.sh not found!"
        return 1
    fi
    
    cd extras/speaker-recognition
    ./run-test.sh
    cd ../..
    
    print_success "Speaker Recognition tests completed"
}

# Main execution
print_info "Friend-Lite Local Test Runner"
print_info "=============================="

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    print_error "Please run this script from the friend-lite root directory"
    exit 1
fi

# Parse command line argument
TEST_SUITE="${1:-all}"

case "$TEST_SUITE" in
    "advanced-backend")
        run_advanced_backend_tests
        ;;
    "speaker-recognition")
        run_speaker_recognition_tests
        ;;
    "all")
        print_info "Running all test suites..."
        
        # Run advanced backend tests
        if run_advanced_backend_tests; then
            print_success "Advanced Backend tests: PASSED"
        else
            print_error "Advanced Backend tests: FAILED"
            exit 1
        fi
        
        # Run speaker recognition tests
        if run_speaker_recognition_tests; then
            print_success "Speaker Recognition tests: PASSED"
        else
            print_error "Speaker Recognition tests: FAILED"
            exit 1
        fi
        
        print_success "All test suites completed successfully!"
        ;;
    *)
        print_error "Unknown test suite: $TEST_SUITE"
        echo "Usage: $0 [advanced-backend|speaker-recognition|all]"
        exit 1
        ;;
esac

print_success "Test execution completed!"