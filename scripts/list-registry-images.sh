#!/bin/bash

REGISTRY="192.168.1.42:32000"

echo "📋 Images in registry $REGISTRY:"
echo

# Get all repositories
REPOS=$(curl -s http://$REGISTRY/v2/_catalog | jq -r '.repositories[]')

if [ -z "$REPOS" ]; then
    echo "No repositories found in registry"
    exit 1
fi

# For each repository, get its tags
for repo in $REPOS; do
    echo "🏷️  Repository: $repo"
    tags=$(curl -s http://$REGISTRY/v2/$repo/tags/list | jq -r '.tags[]?' 2>/dev/null)
    
    if [ -z "$tags" ]; then
        echo "   No tags found"
    else
        for tag in $tags; do
            echo "   📦 $REGISTRY/$repo:$tag"
        done
    fi
    echo
done

