#!/bin/bash

# Script to create Kubernetes Ingress for speaker-recognition service
# Reads configuration from nginx.conf instead of hardcoding values

set -e

# Default values
NAMESPACE=${NAMESPACE:-"speech"}
HOSTNAME=${HOSTNAME:-"speaker.friend-lite.192-168-1-42.nip.io"}
NGINX_CONF=${NGINX_CONF:-"extras/speaker-recognition/nginx.conf"}
OUTPUT_FILE=${OUTPUT_FILE:-"extras/speaker-recognition/charts/speaker-recognition-ingress.yaml"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Creating Kubernetes Ingress for speaker-recognition service${NC}"
echo -e "${YELLOW}Namespace: ${NAMESPACE}${NC}"
echo -e "${YELLOW}Hostname: ${HOSTNAME}${NC}"
echo -e "${YELLOW}Nginx config: ${NGINX_CONF}${NC}"
echo -e "${YELLOW}Output: ${OUTPUT_FILE}${NC}"
echo

# Check if nginx.conf exists
if [ ! -f "${NGINX_CONF}" ]; then
    echo -e "${RED}Error: nginx.conf not found at ${NGINX_CONF}${NC}"
    exit 1
fi

# Extract values from nginx.conf
CLIENT_MAX_BODY_SIZE=$(grep "client_max_body_size" "${NGINX_CONF}" | awk '{print $2}' | sed 's/;//' | sed 's/M/m/')
PROXY_READ_TIMEOUT=$(grep "proxy_read_timeout" "${NGINX_CONF}" | awk '{print $2}' | sed 's/;//')
PROXY_SEND_TIMEOUT=$(grep "proxy_send_timeout" "${NGINX_CONF}" | awk '{print $2}' | sed 's/;//')
PROXY_CONNECT_TIMEOUT=$(grep "proxy_connect_timeout" "${NGINX_CONF}" | awk '{print $2}' | sed 's/;//')

# Set defaults if not found in nginx.conf
CLIENT_MAX_BODY_SIZE=${CLIENT_MAX_BODY_SIZE:-"100m"}
PROXY_READ_TIMEOUT=${PROXY_READ_TIMEOUT:-"86400"}
PROXY_SEND_TIMEOUT=${PROXY_SEND_TIMEOUT:-"86400"}
PROXY_CONNECT_TIMEOUT=${PROXY_CONNECT_TIMEOUT:-"60s"}

echo -e "${GREEN}Extracted configuration from nginx.conf:${NC}"
echo -e "  • client_max_body_size: ${CLIENT_MAX_BODY_SIZE}"
echo -e "  • proxy_read_timeout: ${PROXY_READ_TIMEOUT}"
echo -e "  • proxy_send_timeout: ${PROXY_SEND_TIMEOUT}"
echo -e "  • proxy_connect_timeout: ${PROXY_CONNECT_TIMEOUT}"
echo

# Ensure the output directory exists
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# Create the ingress YAML
cat > "${OUTPUT_FILE}" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: speaker-recognition-ingress
  namespace: ${NAMESPACE}
  annotations:
    # Use configuration snippet for specific path rewrites
    nginx.ingress.kubernetes.io/configuration-snippet: |
      if (\$request_uri ~ ^/api/(.*)\$) {
        rewrite ^/api/(.*)\$ /\$1 break;
      }
    
    # SSL and security
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
    
    # File upload settings (from nginx.conf)
    nginx.ingress.kubernetes.io/client-max-body-size: "${CLIENT_MAX_BODY_SIZE}"
    nginx.ingress.kubernetes.io/proxy-body-size: "${CLIENT_MAX_BODY_SIZE}"
    
    # Timeout settings (from nginx.conf)
    nginx.ingress.kubernetes.io/proxy-read-timeout: "${PROXY_READ_TIMEOUT}"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "${PROXY_SEND_TIMEOUT}"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "${PROXY_CONNECT_TIMEOUT}"
    
    # WebSocket support
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-request-buffering: "off"
    
    # CORS headers
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization"
  
spec:
  ingressClassName: nginx
  rules:
  - host: ${HOSTNAME}
    http:
      paths:
      # Health check endpoint - proxy to speaker service (no rewrite)
      - path: /health
        pathType: Exact
        backend:
          service:
            name: speaker-recognition-speaker
            port:
              number: 8085
      
      # API endpoints - proxy to speaker service and rewrite /api prefix
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: speaker-recognition-speaker
            port:
              number: 8085
      
      # WebSocket endpoints - proxy to speaker service
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: speaker-recognition-speaker
            port:
              number: 8085
      
      # Frontend application - proxy to webui service
      - path: /
        pathType: Prefix
        backend:
          service:
            name: speaker-recognition-webui
            port:
              number: 5173
EOF

echo -e "${GREEN}✅ Ingress created: ${OUTPUT_FILE}${NC}"
echo
echo -e "${YELLOW}To deploy the ingress with the Helm chart:${NC}"
echo -e "  kubectl apply -f ${OUTPUT_FILE}"
echo -e "  helm upgrade --install speaker-recognition ./extras/speaker-recognition/charts -n ${NAMESPACE}"
echo
echo -e "${YELLOW}Or deploy everything with Skaffold:${NC}"
echo -e "  skaffold run --profile speaker-recognition --default-repo=anubis:32000"
echo
echo -e "${YELLOW}To check the ingress status:${NC}"
echo -e "  kubectl get ingress -n ${NAMESPACE}"
echo -e "  kubectl describe ingress speaker-recognition-ingress -n ${NAMESPACE}"
echo
echo -e "${YELLOW}To test the API endpoints:${NC}"
echo -e "  curl http://${HOSTNAME}/health"
echo -e "  curl http://${HOSTNAME}/api/health"
echo
echo -e "${GREEN}Ingress routing configuration:${NC}"
echo -e "  • ${YELLOW}/api/*${NC} → speaker-recognition-speaker:8085 (with /api prefix removed)"
echo -e "  • ${YELLOW}/ws/*${NC} → speaker-recognition-speaker:8085 (WebSocket support)"
echo -e "  • ${YELLOW}/health${NC} → speaker-recognition-speaker:8085"
echo -e "  • ${YELLOW}/*${NC} → speaker-recognition-webui:5173 (frontend)"
echo
echo -e "${GREEN}Configuration values extracted from: ${NGINX_CONF}${NC}"