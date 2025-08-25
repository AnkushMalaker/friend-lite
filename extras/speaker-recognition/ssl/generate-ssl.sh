#!/bin/bash
set -e

# Generate self-signed SSL certificate for Friend-Lite Advanced Backend
# Supports localhost and custom Tailscale IP

TAILSCALE_IP="$1"

if [ -z "$TAILSCALE_IP" ]; then
    echo "Usage: $0 <tailscale-ip>"
    echo "Example: $0 100.83.66.30"
    exit 1
fi

# Validate IP format
if ! echo "$TAILSCALE_IP" | grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' > /dev/null; then
    echo "Error: Invalid IP format. Expected format: xxx.xxx.xxx.xxx"
    exit 1
fi

echo "�� Generating SSL certificate for localhost and $TAILSCALE_IP"

# Determine the output directory - we should be in backends/advanced when running
SSL_DIR="ssl"
if [ -d "$SSL_DIR" ]; then
    # We're in the parent directory
    OUTPUT_DIR="$SSL_DIR"
else
    # We're already in the ssl directory
    OUTPUT_DIR="."
fi

# Create certificate configuration with Subject Alternative Names
cat > $OUTPUT_DIR/server.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = SF
O = Dev
CN = localhost

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = $TAILSCALE_IP
EOF

# Generate private key
openssl genrsa -out $OUTPUT_DIR/server.key 2048

# Generate certificate signing request
openssl req -new -key $OUTPUT_DIR/server.key -out $OUTPUT_DIR/server.csr -config $OUTPUT_DIR/server.conf

# Generate self-signed certificate
openssl x509 -req -in $OUTPUT_DIR/server.csr -signkey $OUTPUT_DIR/server.key -out $OUTPUT_DIR/server.crt -days 365 -extensions v3_req -extfile $OUTPUT_DIR/server.conf

# Clean up
rm $OUTPUT_DIR/server.csr $OUTPUT_DIR/server.conf

# Set appropriate permissions
chmod 600 $OUTPUT_DIR/server.key
chmod 644 $OUTPUT_DIR/server.crt

echo "✅ SSL certificate generated successfully"
echo "   - Certificate: $OUTPUT_DIR/server.crt"
echo "   - Private key: $OUTPUT_DIR/server.key"
echo "   - Valid for: localhost, *.localhost, 127.0.0.1, $TAILSCALE_IP"
echo ""
echo "Certificate Details:"
openssl x509 -in $OUTPUT_DIR/server.crt -text -noout | grep -A 1 "Subject Alternative Name" || echo "   (Certificate generated successfully)"
