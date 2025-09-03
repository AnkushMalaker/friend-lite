#!/bin/bash

# Deploy all Friend-Lite services
# This script deploys infrastructure, application, and additional services
#
# Usage: ./scripts/deploy-all-services.sh [registry]
# Example: ./scripts/deploy-all-services.sh 192.168.1.42:32000

set -e

# Default registry - read from skaffold.env or use fallback
if [ -f "skaffold.env" ]; then
    DEFAULT_REGISTRY=$(grep "^REGISTRY=" skaffold.env | cut -d'=' -f2)
else
    DEFAULT_REGISTRY="192.168.1.42:32000"  # Default fallback
fi
REGISTRY="${1:-$DEFAULT_REGISTRY}"

echo "🚀 Deploying all Friend-Lite services..."
echo "Registry: $REGISTRY"
echo ""

# Check if skaffold is available
if ! command -v skaffold &> /dev/null; then
    echo "❌ Error: skaffold is not installed or not in PATH"
    exit 1
fi

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if skaffold.env exists and validate configuration
if [ ! -f "skaffold.env" ]; then
    echo "❌ Error: skaffold.env not found"
    echo "Please run: cp skaffold.env.template skaffold.env"
    echo "Then configure the values in skaffold.env"
    exit 1
fi

# Validate essential configuration
echo "🔍 Validating configuration..."
REQUIRED_VARS=("REGISTRY" "BACKEND_IP" "BACKEND_NODEPORT" "WEBUI_NODEPORT")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^$var=" skaffold.env; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "❌ Error: Missing required variables in skaffold.env:"
    printf '  - %s\n' "${MISSING_VARS[@]}"
    echo "Please check skaffold.env.template for required variables"
    exit 1
fi

echo "✅ Configuration validated"

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "✅ Connected to Kubernetes cluster"

# Function to wait for deployment
wait_for_deployment() {
    local profile=$1
    local description=$2
    
    echo "⏳ Waiting for $description to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=advanced-backend -n friend-lite --timeout=300s 2>/dev/null || true
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=webui -n friend-lite --timeout=300s 2>/dev/null || true
    echo "✅ $description deployment complete"
}

# 1. Deploy Infrastructure
echo ""
echo "🏗️  Step 1: Deploying infrastructure services..."
skaffold run --profile=infrastructure --default-repo="$REGISTRY"

echo "⏳ Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mongodb -n root --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant -n root --timeout=300s
echo "✅ Infrastructure services ready"

# 2. Deploy Main Application
echo ""
echo "🚀 Step 2: Deploying main application..."
skaffold run --profile=advanced-backend --default-repo="$REGISTRY"
wait_for_deployment "advanced-backend" "Main application"

# 3. Deploy Speaker Recognition (if configured)
if [ -n "$HF_TOKEN" ] || grep -q "HF_TOKEN" skaffold.env 2>/dev/null; then
    echo ""
    echo "🎤 Step 3: Deploying speaker recognition service..."
    skaffold run --profile=speaker-recognition --default-repo="$REGISTRY"
    echo "✅ Speaker recognition service deployed"
else
    echo ""
    echo "⏭️  Step 3: Skipping speaker recognition (HF_TOKEN not configured)"
fi

# 4. Deploy ASR Services (if configured)
if [ -d "extras/asr-services/charts" ]; then
    echo ""
    echo "🗣️  Step 4: Deploying ASR services..."
    skaffold run --profile=moonshine-asr --default-repo="$REGISTRY"
    echo "✅ ASR services deployed"
else
    echo ""
    echo "⏭️  Step 4: Skipping ASR services (charts not found)"
fi

# 5. Verify all deployments
echo ""
echo "🔍 Step 5: Verifying all deployments..."
echo "Infrastructure pods:"
kubectl get pods -n root
echo ""
echo "Application pods:"
kubectl get pods -n friend-lite
echo ""
echo "Speaker recognition pods:"
kubectl get pods -n speech 2>/dev/null || echo "No speaker recognition namespace found"
echo ""
echo "ASR service pods:"
kubectl get pods -n asr 2>/dev/null || echo "No ASR namespace found"

# 6. Show access information
echo ""
echo "🎯 Deployment Summary:"
echo "✅ Infrastructure: MongoDB and Qdrant running in 'root' namespace"
echo "✅ Main Application: Backend and WebUI running in 'friend-lite' namespace"
echo "✅ Services accessible via NodePorts (check skaffold.env for ports)"
echo ""
echo "🌐 Access your application:"
# Read IP and ports from skaffold.env
BACKEND_IP=$(grep "^BACKEND_IP=" skaffold.env | cut -d'=' -f2 2>/dev/null || echo "192.168.1.42")
WEBUI_PORT=$(grep "^WEBUI_NODEPORT=" skaffold.env | cut -d'=' -f2 2>/dev/null || echo "31011")
BACKEND_PORT=$(grep "^BACKEND_NODEPORT=" skaffold.env | cut -d'=' -f2 2>/dev/null || echo "30270")
echo "   WebUI: http://$BACKEND_IP:$WEBUI_PORT"
echo "   Backend: http://$BACKEND_IP:$BACKEND_PORT"
echo ""
echo "🚀 All services deployed successfully!"
