#!/bin/bash

# Test GPU access in Kubernetes
# This script creates a temporary pod to verify GPU functionality
#
# Usage: ./scripts/test-gpu-pod.sh [gpu_count]
# Example: ./scripts/test-gpu-pod.sh
# Example: ./scripts/test-gpu-pod.sh 2

set -e

GPU_COUNT="${1:-1}"

echo "🎮 Testing GPU access in Kubernetes..."
echo "GPU count: $GPU_COUNT"
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

# Check if GPU operator is installed
if ! kubectl get namespace gpu-operator &> /dev/null; then
    echo "❌ Error: GPU operator not installed"
    echo "Please run: ./scripts/setup-nvidia-operator.sh"
    exit 1
fi

# Check GPU operator status
echo "🔍 Checking GPU operator status..."
if ! kubectl get pods -n gpu-operator | grep -q "Running"; then
    echo "❌ Error: GPU operator not running"
    echo "Please check: kubectl get pods -n gpu-operator"
    exit 1
fi

echo "✅ GPU operator is running"

# Check available GPU resources
echo "🔍 Checking available GPU resources..."
GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable."nvidia.com/gpu" != null) | .metadata.name' 2>/dev/null || echo "")

if [ -z "$GPU_NODES" ]; then
    echo "❌ Error: No nodes with GPU resources found"
    echo "Please check: kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.allocatable.\"nvidia.com/gpu\"}'"
    exit 1
fi

echo "✅ Found GPU nodes:"
echo "$GPU_NODES"
echo ""

# Get total available GPUs
TOTAL_GPUS=$(kubectl get nodes -o json | jq -r '.items[] | .status.allocatable."nvidia.com/gpu" // "0"' | awk '{sum += $1} END {print sum}' 2>/dev/null || echo "0")

if [ "$TOTAL_GPUS" -lt "$GPU_COUNT" ]; then
    echo "❌ Error: Requested $GPU_COUNT GPUs, but only $TOTAL_GPUS available"
    exit 1
fi

echo "✅ Available GPUs: $TOTAL_GPUS"
echo ""

# Create test pod
echo "🚀 Creating GPU test pod..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-$(date +%s)
  labels:
    app: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-base-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: $GPU_COUNT
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: all
    - name: NVIDIA_DRIVER_CAPABILITIES
      value: compute,utility
EOF

# Wait for pod to be ready
echo "⏳ Waiting for pod to be ready..."
POD_NAME=$(kubectl get pods -l app=gpu-test --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')

kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=60s

echo "✅ Pod is ready: $POD_NAME"
echo ""

# Show pod details
echo "📋 Pod details:"
kubectl get pod $POD_NAME -o wide
echo ""

# Execute nvidia-smi
echo "🎮 Running nvidia-smi..."
kubectl exec $POD_NAME -- nvidia-smi

# Show pod logs
echo ""
echo "📝 Pod logs:"
kubectl logs $POD_NAME

# Clean up
echo ""
echo "🧹 Cleaning up test pod..."
kubectl delete pod $POD_NAME

echo ""
echo "🎯 GPU test completed successfully!"
echo "✅ GPU access is working correctly"
echo "✅ $GPU_COUNT GPU(s) allocated successfully"
echo ""
echo "You can now deploy GPU-enabled applications with:"
echo "resources:"
echo "  limits:"
echo "    nvidia.com/gpu: $GPU_COUNT"
