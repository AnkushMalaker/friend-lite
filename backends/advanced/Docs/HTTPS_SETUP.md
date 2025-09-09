# HTTPS Setup for Friend-Lite Advanced Backend

This guide explains how to set up HTTPS/SSL access for Friend-Lite Advanced Backend, enabling secure microphone access and network connectivity.

## Why HTTPS is Needed

Modern browsers require HTTPS for:
- **Microphone access** over network connections (not localhost)
- **Secure WebSocket connections** (WSS)
- **Tailscale/VPN access** with audio features
- **Production deployments**

## Quick Setup

### 1. Initialize HTTPS with Your IP

Run the initialization script with your Tailscale or network IP:

```bash
cd backends/advanced
./init.sh 100.83.66.30  # Replace with your actual IP
```

This script will:
- Generate SSL certificates for localhost and your IP
- Create nginx configuration files
- Update CORS settings for HTTPS origins

### 2. Start with HTTPS Proxy

```bash
# HTTPS with nginx proxy (REQUIRED for network microphone access)
docker compose up --build -d

# HTTP only (no nginx, localhost microphone access only)
docker compose up --build -d
```

**NOTE**: The nginx service now starts automatically with the standard docker compose command, providing immediate HTTPS access when SSL certificates are configured.

### 3. Access the Services

#### Friend-Lite Advanced Backend (Primary - ports 80/443)
- **HTTPS:** https://localhost/ or https://your-ip/ (accept SSL certificate)
- **HTTP:** http://localhost/ (redirects to HTTPS)
- **Features:** Dashboard, Live Recording, Conversations, Memories

#### Speaker Recognition Service (Secondary - ports 8081/8444)  
- **HTTPS:** https://localhost:8444/ or https://your-ip:8444/ (accept SSL certificate)
- **HTTP:** http://localhost:8081/ (redirects to HTTPS)
- **Features:** Speaker enrollment, audio analysis, live inference

## Port Allocation

### Advanced Backend (Primary Service)
- **Port 80:** HTTP (redirects to HTTPS)
- **Port 443:** HTTPS with nginx proxy
- **Port 5173:** Direct Vite dev server (development only)
- **Port 8000:** Direct backend API (development only)

### Speaker Recognition (Secondary Service)
- **Port 8081:** HTTP (redirects to HTTPS)
- **Port 8444:** HTTPS with nginx proxy
- **Port 5175:** Direct React dev server (internal)
- **Port 8085:** Direct API service (internal)

## Manual Setup

### SSL Certificate Generation

If you need to regenerate certificates:

```bash
cd ssl
./generate-ssl.sh 100.83.66.30  # Your IP address
```

### Environment Configuration

Update your `.env` file to include HTTPS origins:

```bash
CORS_ORIGINS=https://localhost,https://127.0.0.1,https://100.83.66.30
```

## Docker Compose Profiles

### With HTTPS Configuration (Network Access)
**Services started:**
- ✅ nginx (ports 443/80) - SSL termination and proxy
- ✅ webui (port 5173, internal) - Vite dev server  
- ✅ friend-backend (port 8000, internal)
- ✅ mongo, qdrant (databases)

**Access:** https://localhost/ or https://your-ip/  
**Microphone:** Works over network with HTTPS

### Without HTTPS Configuration (Default - Localhost Only)
**Services started:**
- ✅ nginx (ports 443/80) - but without SSL certificates
- ✅ webui (port 5173, direct access) - Vite dev server
- ✅ friend-backend (port 8000)
- ✅ mongo, qdrant (databases)

**Access:** http://localhost:5173  
**Microphone:** Only works on localhost (browser security)

## Nginx Configuration

The setup uses a single nginx configuration:

### Single Config (`nginx.conf.template`)
- Proxies to `webui:5173` for the Vite dev server
- Handles WebSocket connections for audio streaming
- SSL termination with proper headers  
- Supports Vite HMR (Hot Module Replacement) over WSS
- Always provides development experience with hot reload

## WebSocket Endpoints

All WebSocket endpoints are proxied through nginx with SSL:

- **`wss://your-ip/ws_pcm`** - Primary audio streaming (Wyoming protocol + PCM)
- **`wss://your-ip/ws_omi`** - OMI device audio streaming (Wyoming protocol + Opus)  
- **`wss://your-ip/ws`** - Legacy audio streaming (Opus packets)

**Note:** When accessed through HTTPS proxy, all API calls use relative URLs automatically.

## Browser Certificate Trust

Since we use self-signed certificates, browsers will show security warnings:

### Chrome/Edge
1. Visit https://localhost/
2. Click "Advanced" → "Proceed to localhost (unsafe)"
3. Or add certificate to trusted store

### Firefox
1. Visit https://localhost/
2. Click "Advanced" → "Accept the Risk and Continue"

### Safari
1. Visit https://localhost/
2. Click "Show Details" → "visit this website"

## Troubleshooting

### Certificate Issues

**Problem:** "SSL certificate problem: self signed certificate"
**Solution:** 
```bash
# Regenerate certificates
cd ssl
./generate-ssl.sh your-ip
docker compose restart nginx
```

### WebSocket Connection Fails

**Problem:** WSS connection refused
**Solution:** 
1. Check nginx is running: `docker compose ps nginx`
2. Verify certificate: `curl -k https://localhost/health`
3. Check logs: `docker compose logs nginx`

### CORS Errors

**Problem:** "Cross-Origin Request Blocked"
**Solution:**
1. Update CORS_ORIGINS in `.env` to include your HTTPS origin
2. Restart backend: `docker compose restart friend-backend`

### Microphone Access Denied

**Problem:** Browser blocks microphone access
**Solution:**
1. Ensure you're using HTTPS (not HTTP)
2. Accept SSL certificate warnings
3. Grant microphone permissions when prompted

## Port Reference

### HTTPS Setup (Production)
- **443** - HTTPS (nginx → webui:80)
- **80** - HTTP redirect to HTTPS

### HTTPS Setup (Development)
- **8443** - HTTPS (nginx-dev → webui-dev:5173)
- **8080** - HTTP redirect to HTTPS

### Standard Setup
- **3000** - HTTP (webui production)
- **5173** - HTTP (webui development)
- **8000** - HTTP (friend-backend)

## Live Recording Feature

The Live Recording feature automatically adapts to your connection:

- **HTTP + localhost:** Uses `ws://localhost:8000/ws_pcm`
- **HTTPS:** Uses `wss://your-domain/ws_pcm`
- **Microphone access:** Requires HTTPS for network connections

Access at:
- Local: https://localhost/live-record
- Network: https://your-ip/live-record

## Security Considerations

### Self-Signed Certificates
- Only for development and local network use
- Use proper CA certificates for production
- Consider Let's Encrypt for public deployments

### Network Security
- HTTPS encrypts all traffic including WebSocket data
- Nginx handles SSL termination
- Backend services remain on internal Docker network

### Browser Security
- Modern browsers block microphone access over HTTP (except localhost)
- WSS required for secure WebSocket connections over network
- CORS properly configured for cross-origin requests

## Production Deployment

For production deployments:

1. **Use proper SSL certificates** (Let's Encrypt, commercial CA)
2. **Update nginx configuration** with your domain name
3. **Configure DNS** to point to your server
4. **Use production docker compose profile**:
   ```bash
   docker compose up -d
   ```

## Integration with Other Services

### Speaker Recognition
If using the speaker recognition service alongside Friend-Lite:

```bash
# Use different HTTPS ports to avoid conflicts
# Speaker Recognition: 443/80
# Friend-Lite: 8443/8080
docker compose up -d
```

### Tailscale Integration
The setup is optimized for Tailscale usage:

- SSL certificates include your Tailscale IP
- CORS automatically supports 100.x.x.x IP range
- WebSocket connections work over Tailscale network