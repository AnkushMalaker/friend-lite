#!/bin/bash
set -Eeuo pipefail
IFS=$'\n\t'
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

# Process .env file and convert to YAML format (robust)
SECRET_KEY_PATTERN='(SECRET|PASSWORD|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY|CLIENT_SECRET)'
valid_key_re='^[A-Za-z_][A-Za-z0-9_]*$'
while IFS= read -r line || [[ -n "$line" ]]; do
  # strip CR from CRLF
  line="${line%$'\r'}"
  # skip empty/comment lines
  [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
  # allow leading 'export '
  line="${line#export }"
  # must contain '='
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  value="${line#*=}"
  # trim key/value whitespace
  key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
  value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"

  # strip surrounding quotes if present (single or double)
  if [[ "$value" =~ ^\".*\"$ ]]; then value="${value:1:${#value}-2}"; fi
  if [[ "$value" =~ ^\'.*\'$ ]]; then value="${value:1:${#value}-2}"; fi
  # remove trailing inline comment only if preceded by space
  value="${value%% #*}"

  # Skip values injected via Helm values
  if [[ "$key" =~ ^(MONGODB_URI|QDRANT_BASE_URL|CORS_ORIGINS)$ ]]; then
    continue
  fi
  # Guard against leaking secrets into a ConfigMap
  if [[ "$key" =~ $SECRET_KEY_PATTERN ]]; then
    echo "WARN: Skipping potential secret key '$key'. Manage via Kubernetes Secret instead." >&2
    continue
  fi
  # Validate key
  if ! [[ "$key" =~ $valid_key_re ]]; then
    echo "WARN: Skipping invalid env key '$key' (must match $valid_key_re)" >&2
    continue
  fi
  # YAML-escape value for double-quoted string
  value="${value//\\/\\\\}"; value="${value//\"/\\\"}"
  printf '  %s: "%s"\n' "$key" "$value" >> "$OUTPUT_FILE"
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
