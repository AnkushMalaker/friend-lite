#!/bin/bash

# Setup NVIDIA GPU Operator for Kubernetes
# This script installs and configures the NVIDIA GPU operator
#
# Usage: ./scripts/setup-nvidia-operator.sh
# Prerequisites: kubectl configured, NVIDIA GPU(s) installed on nodes

set -e

echo "🚀 Setting up NVIDIA GPU Operator..."

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

# Add NVIDIA Helm repository
echo "📦 Adding NVIDIA Helm repository..."
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install NVIDIA GPU Operator
echo "🔧 Installing NVIDIA GPU Operator..."
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=false

# Wait for GPU operator to be ready
echo "⏳ Waiting for GPU operator to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=gpu-operator -n gpu-operator --timeout=300s

echo "✅ GPU operator installed successfully!"

# Verify GPU detection
echo "🔍 Checking GPU detection..."
echo "Nodes and GPU resources:"
kubectl get nodes -o json | jq -r '.items[] | "\(.metadata.name): \(.status.allocatable."nvidia.com/gpu" // "0") GPU(s)"' 2>/dev/null || echo "No GPUs detected or jq not available"

# Check GPU operator pods
echo ""
echo "📋 GPU operator pods status:"
kubectl get pods -n gpu-operator

echo ""
echo "🎯 GPU operator setup complete!"
echo ""
echo "To verify GPU access in a pod, use this resource specification:"
echo "resources:"
echo "  limits:"
echo "    nvidia.com/gpu: 1"
echo ""
echo "Example test pod:"
echo "kubectl run gpu-test --image=nvidia/cuda:11.8-base-ubuntu20.04 --rm -it --restart=Never -- nvidia-smi"
