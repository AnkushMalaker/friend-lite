#!/bin/bash

# Check Kubernetes cluster status and health
# This script provides a comprehensive overview of your cluster
#
# Usage: ./scripts/cluster-status.sh [namespace]
# Example: ./scripts/cluster-status.sh
# Example: ./scripts/cluster-status.sh friend-lite

set -e

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load-env.sh"

NAMESPACE="${1:-all}"

echo "🔍 Kubernetes Cluster Status Check"
echo "=================================="
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "✅ Connected to Kubernetes cluster"

# Check if skaffold.env exists for better registry information
if [ ! -f "skaffold.env" ]; then
    echo "⚠️  Warning: skaffold.env not found - some information may be limited"
fi
echo ""

# 1. Node Status
echo "🏗️  NODE STATUS"
echo "---------------"
kubectl get nodes -o wide
echo ""

# 2. Namespace Overview
echo "📁 NAMESPACES"
echo "-------------"
kubectl get namespaces
echo ""

# 3. Pod Status (all or specific namespace)
echo "📦 POD STATUS"
echo "-------------"
if [ "$NAMESPACE" = "all" ]; then
    echo "All namespaces:"
    kubectl get pods -A --sort-by=.metadata.namespace
else
    echo "Namespace: $NAMESPACE"
    kubectl get pods -n "$NAMESPACE"
fi
echo ""

# 4. Service Status
echo "🔌 SERVICES"
echo "-----------"
if [ "$NAMESPACE" = "all" ]; then
    kubectl get services -A --sort-by=.metadata.namespace
else
    kubectl get services -n "$NAMESPACE"
fi
echo ""

# 5. Ingress Status
echo "🌐 INGRESS"
echo "----------"
kubectl get ingress -A
echo ""

# 6. Persistent Volumes
echo "💾 STORAGE"
echo "----------"
echo "Storage Classes:"
kubectl get storageclass
echo ""
echo "Persistent Volumes:"
kubectl get pv
echo ""
echo "Persistent Volume Claims:"
kubectl get pvc -A
echo ""

# 7. Resource Usage
echo "📊 RESOURCE USAGE"
echo "-----------------"
echo "Node resources:"
kubectl top nodes 2>/dev/null || echo "Metrics server not available"
echo ""
echo "Pod resources:"
if [ "$NAMESPACE" = "all" ]; then
    kubectl top pods -A 2>/dev/null || echo "Metrics server not available"
else
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics server not available"
fi
echo ""

# 8. Recent Events
echo "📝 RECENT EVENTS"
echo "----------------"
if [ "$NAMESPACE" = "all" ]; then
    kubectl get events -A --sort-by=.lastTimestamp | tail -10
else
    kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -10
fi
echo ""

# 9. GPU Status (if available)
if kubectl get nodes -o json | grep -q "nvidia.com/gpu"; then
    echo "🎮 GPU STATUS"
    echo "-------------"
    kubectl get nodes -o json | jq -r '.items[] | "\(.metadata.name): \(.status.allocatable."nvidia.com/gpu" // "0") GPU(s)"' 2>/dev/null || echo "GPU information not available"
    echo ""
fi

# 10. Registry Status
echo "🐳 REGISTRY STATUS"
echo "------------------"
# Get registry from config.env
REGISTRY="${CONTAINER_REGISTRY:-localhost:32000}"
REGISTRY_IP=$(echo "$REGISTRY" | cut -d':' -f1)
REGISTRY_PORT=$(echo "$REGISTRY" | cut -d':' -f2)

echo "Registry: $REGISTRY"
if curl -s "http://$REGISTRY/v2/" > /dev/null 2>&1; then
    echo "✅ Registry accessible"
else
    echo "❌ Registry not accessible"
fi
echo ""

echo "🎯 Status check complete!"
echo ""
echo "For detailed information about specific resources:"
echo "  kubectl describe <resource> <name> -n <namespace>"
echo "  kubectl logs <pod-name> -n <namespace>"
echo "  kubectl exec -it <pod-name> -n <namespace> -- /bin/bash"
