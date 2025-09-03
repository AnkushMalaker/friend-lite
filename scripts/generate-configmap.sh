#!/bin/bash

# Generate ConfigMap data from .env file for Helm template
ENV_FILE="backends/advanced/.env"
OUTPUT_FILE="backends/charts/advanced-backend/templates/env-configmap.yaml"

# Check if we're in the right directory
if [ ! -f "skaffold.env" ]; then
    echo "Error: Please run this script from the friend-lite root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    echo "Please create the .env file first:"
    echo "  cd backends/advanced"
    echo "  cp .env.template .env"
    echo "  # Edit .env with your values"
    echo "  cd ../.."
    exit 1
fi

echo "Generating ConfigMap from $ENV_FILE..."

# Create the complete ConfigMap structure
cat > "$OUTPUT_FILE" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: advanced-backend-env
  labels:
    {{- include "advanced-backend.labels" . | nindent 4 }}
data:
EOF

# Process .env file and convert to YAML format
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Split on first = sign
        key="${line%%=*}"
        value="${line#*=}"
        # Remove any trailing comments from value
        value="${value%% #*}"
        # Skip Kubernetes-specific values that will be set by Skaffold
        if [[ "$key" =~ ^(MONGODB_URI|QDRANT_BASE_URL|CORS_ORIGINS)$ ]]; then
            # Don't add anything for these - they'll be set by Skaffold
            continue
        else
            # Clean up trailing spaces and put the actual value from .env directly into the ConfigMap
            value=$(echo "$value" | sed 's/[[:space:]]*$//')
            echo "  $key: \"$value\"" >> "$OUTPUT_FILE"
        fi
    fi
done < "$ENV_FILE"

# Add Kubernetes-specific values section
cat >> "$OUTPUT_FILE" << 'EOF'

  # Kubernetes-specific values (set by Skaffold)
  MONGODB_URI: "{{ .Values.env.MONGODB_URI | default "mongo" }}"
  QDRANT_BASE_URL: "{{ .Values.env.QDRANT_BASE_URL | default "qdrant.root.svc.cluster.local" }}"
  CORS_ORIGINS: "{{ .Values.env.CORS_ORIGINS }}"
EOF

echo "ConfigMap generated at $OUTPUT_FILE"
echo ""
echo "The script has created a complete env-configmap.yaml file with:"
echo "- All values from your .env file"
echo "- Kubernetes-specific values that use Helm templating"
echo ""
echo "You can now deploy with: skaffold run --profile advanced-backend --default-repo=anubis:32000"
