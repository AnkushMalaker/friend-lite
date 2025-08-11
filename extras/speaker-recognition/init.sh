#!/bin/bash
set -e

# Initialize speaker recognition with custom Tailscale IP
# Usage: ./init.sh <tailscale-ip>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tailscale-ip>"
    echo "Example: $0 100.83.66.30"
    exit 1
fi

TAILSCALE_IP="$1"

# Validate IP format (basic check)
if ! echo "$TAILSCALE_IP" | grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' > /dev/null; then
    echo "Error: Invalid IP format. Expected format: xxx.xxx.xxx.xxx"
    exit 1
fi

echo "Initializing speaker recognition with Tailscale IP: $TAILSCALE_IP"

# Create nginx.conf from template
if [ ! -f "nginx.conf.template" ]; then
    echo "Error: nginx.conf.template not found"
    exit 1
fi

echo "Creating nginx.conf with IP: $TAILSCALE_IP"
sed "s/TAILSCALE_IP/$TAILSCALE_IP/g" nginx.conf.template > nginx.conf

echo "✅ nginx.conf created successfully"
echo "✅ Ready to start with: docker compose up --build -d"
echo ""
echo "Access URLs:"
echo "  - Web UI: https://localhost/ or https://$TAILSCALE_IP/"
echo "  - API: https://localhost/api/ or https://$TAILSCALE_IP/api/"
echo ""
echo "Next steps:"
echo "  1. docker compose up --build -d"
echo "  2. Visit https://localhost/ (accept SSL certificate)"
echo "  3. Follow quickstart.md for usage instructions"