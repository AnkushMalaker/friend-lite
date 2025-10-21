#!/bin/bash
set -e

# Generate self-signed SSL certificate for Friend-Lite Advanced Backend
# Supports localhost, IP addresses, and domain names

SERVER_ADDRESS="$1"

if [ -z "$SERVER_ADDRESS" ]; then
    echo "Usage: $0 <ip-or-domain>"
    echo "Example: $0 100.83.66.30"
    echo "Example: $0 myserver.tailxxxxx.ts.net"
    exit 1
fi

# Detect if it's an IP address or domain name
if echo "$SERVER_ADDRESS" | grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' > /dev/null; then
    IS_IP=true
    echo "🔐 Generating SSL certificate for localhost and IP: $SERVER_ADDRESS"
else
    IS_IP=false
    echo "🔐 Generating SSL certificate for localhost and domain: $SERVER_ADDRESS"
fi

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
EOF

# Add custom address as either IP or DNS
if [ "$IS_IP" = true ]; then
    echo "IP.2 = $SERVER_ADDRESS" >> $OUTPUT_DIR/server.conf
else
    echo "DNS.3 = $SERVER_ADDRESS" >> $OUTPUT_DIR/server.conf
fi

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
echo "   - Valid for: localhost, *.localhost, 127.0.0.1, $SERVER_ADDRESS"
echo ""
echo "Certificate Details:"
openssl x509 -in $OUTPUT_DIR/server.crt -text -noout | grep -A 1 "Subject Alternative Name" || echo "   (Certificate generated successfully)"