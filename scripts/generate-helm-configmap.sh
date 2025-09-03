#!/bin/bash

# Generate a complete ConfigMap Helm template from .env file
ENV_FILE="backends/advanced/.env"
OUTPUT_FILE="backends/charts/advanced-backend/templates/env-configmap.yaml"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

echo "Generating ConfigMap template from $ENV_FILE..."

# Create the complete ConfigMap template
cat > "$OUTPUT_FILE" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: advanced-backend-env
  labels:
    {{- include "advanced-backend.labels" . | nindent 4 }}
data:
EOF

# Process .env file and add to ConfigMap
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Split on first = sign
        key="${line%%=*}"
        value="${line#*=}"
        # Remove any trailing comments from value
        value="${value%% #*}"
        # Add properly indented key-value pair
        echo "  $key: \"$value\"" >> "$OUTPUT_FILE"
    fi
done < "$ENV_FILE"

echo "ConfigMap template generated at $OUTPUT_FILE"

