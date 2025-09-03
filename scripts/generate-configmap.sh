#!/bin/bash

# Generate ConfigMap data from .env file for Helm template
ENV_FILE="backends/advanced/.env"
OUTPUT_FILE="backends/charts/advanced-backend/templates/env-data.yaml"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

echo "Generating ConfigMap data from $ENV_FILE..."

# Create a YAML snippet that Helm can include
cat > "$OUTPUT_FILE" << 'EOF'
{{- define "advanced-backend.envData" -}}
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
        # Properly quote the value for YAML
        echo "  $key: \"$value\"" >> "$OUTPUT_FILE"
    fi
done < "$ENV_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
{{- end -}}
EOF

echo "ConfigMap data template generated at $OUTPUT_FILE"
