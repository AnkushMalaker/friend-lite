#!/bin/bash

# Wrapper script to purge unused images
# Calls both registry and container image purge scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🧹 Starting comprehensive image purge..."
echo ""

echo "1️⃣  Purging unused registry images..."
echo "=================================="
"$SCRIPT_DIR/purge-registry-images.sh"

echo ""
echo "2️⃣  Purging unused container images..."
echo "====================================="
"$SCRIPT_DIR/purge-container-images.sh"

echo ""
echo "🎉 All purge operations completed!"