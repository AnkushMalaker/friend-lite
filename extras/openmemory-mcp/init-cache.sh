#!/bin/bash
# Initialize or update local cached mem0 from Ankush's fork

CACHE_DIR="./cache/mem0"
FORK_REPO="https://github.com/AnkushMalaker/mem0.git"
BRANCH="fix/get-endpoint"

echo "🔄 Updating OpenMemory cache from fork..."

if [ ! -d "$CACHE_DIR/.git" ]; then
    echo "📥 Initializing cache from fork..."
    rm -rf "$CACHE_DIR"
    git clone "$FORK_REPO" "$CACHE_DIR"
    cd "$CACHE_DIR"
    git checkout "$BRANCH"
    echo "✅ Cache initialized from $FORK_REPO ($BRANCH)"
else
    echo "🔄 Updating existing cache..."
    cd "$CACHE_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    echo "✅ Cache updated from $FORK_REPO ($BRANCH)"
fi

echo ""
echo "📂 Cache directory: $(pwd)"
echo "🌿 Current branch: $(git branch --show-current)"
echo "📝 Latest commit: $(git log --oneline -1)"
echo ""
echo "🚀 Ready to build! Run: docker compose build openmemory-mcp --no-cache"