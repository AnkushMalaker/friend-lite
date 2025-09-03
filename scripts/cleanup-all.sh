#!/bin/bash

# Clean up all Friend-Lite deployments and resources
# This script removes all services, namespaces, and resources
#
# Usage: ./scripts/cleanup-all.sh [--force]
# Example: ./scripts/cleanup-all.sh
# Example: ./scripts/cleanup-all.sh --force

set -e

FORCE=false
if [ "$1" = "--force" ]; then
    FORCE=true
fi

echo "🧹 Cleaning up all Friend-Lite deployments..."
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "✅ Connected to Kubernetes cluster"

# Check if skaffold.env exists for better cleanup information
if [ ! -f "skaffold.env" ]; then
    echo "⚠️  Warning: skaffold.env not found - cleanup will proceed with default namespaces"
fi
echo ""

# Confirmation prompt
if [ "$FORCE" = false ]; then
    echo "⚠️  WARNING: This will remove ALL Friend-Lite deployments and data!"
    echo "This includes:"
    echo "  - All application pods and services"
    echo "  - MongoDB database and data"
    echo "  - Qdrant vector database and data"
    echo "  - Speaker recognition services"
    echo "  - ASR services"
    echo "  - All persistent volumes and claims"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "❌ Cleanup cancelled"
        exit 0
    fi
fi

echo "🚀 Starting cleanup..."

# 1. Remove application deployments
echo "📦 Removing application deployments..."
skaffold delete --profile=advanced-backend 2>/dev/null || echo "No advanced-backend profile found"
skaffold delete --profile=speaker-recognition 2>/dev/null || echo "No speaker-recognition profile found"
skaffold delete --profile=moonshine-asr 2>/dev/null || echo "No moonshine-asr profile found"
echo "✅ Application deployments removed"

# 2. Remove infrastructure deployments
echo "🏗️  Removing infrastructure deployments..."
skaffold delete --profile=infrastructure 2>/dev/null || echo "No infrastructure profile found"
echo "✅ Infrastructure deployments removed"

# 3. Remove namespaces (this will remove all resources in them)
echo "🗑️  Removing namespaces..."
kubectl delete namespace friend-lite --ignore-not-found=true
kubectl delete namespace root --ignore-not-found=true
kubectl delete namespace speech --ignore-not-found=true
kubectl delete namespace asr --ignore-not-found=true
echo "✅ Namespaces removed"

# 4. Remove persistent volumes
echo "💾 Removing persistent volumes..."
kubectl delete pv --all --ignore-not-found=true
echo "✅ Persistent volumes removed"

# 5. Remove storage classes
echo "📁 Removing storage classes..."
kubectl delete storageclass openebs-hostpath --ignore-not-found=true
echo "✅ Storage classes removed"

# 6. Remove any remaining resources
echo "🔍 Removing any remaining resources..."
kubectl delete all --all --all-namespaces --ignore-not-found=true
kubectl delete ingress --all --all-namespaces --ignore-not-found=true
kubectl delete pvc --all --all-namespaces --ignore-not-found=true
echo "✅ Remaining resources removed"

# 7. Clean up Skaffold artifacts
echo "🧹 Cleaning Skaffold artifacts..."
skaffold clean 2>/dev/null || echo "No Skaffold artifacts to clean"
echo "✅ Skaffold artifacts cleaned"

# 8. Verify cleanup
echo ""
echo "🔍 Verifying cleanup..."
echo "Remaining namespaces:"
kubectl get namespaces
echo ""
echo "Remaining pods:"
kubectl get pods -A
echo ""
echo "Remaining services:"
kubectl get services -A
echo ""

echo "🎯 Cleanup complete!"
echo ""
echo "Note: Some system namespaces and resources may remain (this is normal)"
echo "To completely reset your cluster, you may need to:"
echo "  - Restart MicroK8s: sudo snap restart microk8s"
echo "  - Or reinstall MicroK8s: sudo snap remove microk8s && sudo snap install microk8s --classic"
