## iOS Installation

### Method 1: Using Xcode (Development)
1. Navigate to the `friend-lite` folder
2. Run prebuild command:
   ```bash
   npx expo prebuild --clean
   ```
3. Navigate to the `ios` folder
4. Install iOS dependencies:
   ```bash
   pod install
   ```
5. Open `friendlite.xcworkspace` in Xcode
6. Build and run the app from Xcode

### Method 2: Using Expo CLI (Recommended)
1. Navigate to the `friend-lite` folder
2. Clean and prebuild the project:
   ```bash
   npx expo prebuild --clean
   ```
3. Install Expo development client:
   ```bash
   npx expo install expo-dev-client
   ```
4. Start the development server:
   ```bash
   npx expo start --dev-client
   ```
5. Run on iOS device:
   ```bash
   npx expo run:ios --device
   ```

## Android Installation

Follow the same Expo CLI procedure as iOS Method 2 above, but use the Android run command:
```bash
npx expo run:android --device
```

### Android Network Configuration

After building, you'll need to configure network permissions for local development:

1. The build process generates `android/app/src/main/AndroidManifest.xml`
2. Add `usesCleartextTraffic="true"` to the application tag for local development:
   ```xml
   <application
       android:usesCleartextTraffic="true"
       ... >
   ```
   **Security Note**: This allows unencrypted HTTP traffic and should only be used for development with local servers.

### Production Network Security (Android)

For production deployments, use a more secure network configuration:

1. Create `android/app/src/main/res/xml/network_security_config.xml`:
   ```xml
   <?xml version="1.0" encoding="utf-8"?>
   <network-security-config>
       <domain-config cleartextTrafficPermitted="true">
           <domain includeSubdomains="true">your-domain.com</domain>
       </domain-config>
   </network-security-config>
   ```

2. Reference the config in your `AndroidManifest.xml`:
   ```xml
   <application
       android:networkSecurityConfig="@xml/network_security_config"
       ... >
   ```

Replace `your-domain.com` with your actual domain name.


# Backend Configuration

## Available Backends

The Friend Lite app supports two backend options:

1. **`backends/simple-backend`** - Basic audio storage for testing and verification
2. **`backends/advanced-backend`** - Full-featured backend with authentication, transcription, and memory storage

## Backend URL Configuration

### For Simple Backend
Enter the WebSocket URL directly:
```
ws://your-server-ip:8000/ws
```

### For Advanced Backend (Recommended)
The advanced backend requires authentication. Enter the WebSocket URL in the app:
```
ws://your-server-ip:8000/ws_pcm
```

#### Exposing Your Backend

**Option 1: Direct IP Access**
- Use your machine's local/public IP address
- Example: `ws://192.168.1.100:8000/ws_pcm`

**Option 2: Using ngrok (Recommended for remote access)**
```bash
ngrok http 8000
```
This gives you a public URL. Use the WebSocket format:
```
wss://your-ngrok-id.ngrok.app/ws_pcm
```

## Authentication Setup (Advanced Backend)

The Friend Lite app now includes built-in authentication for the advanced backend:

### 1. Backend Authentication

In the app's **Backend Authentication** section, enter your credentials:

- **Username (email)**: Your registered email address
- **Password**: Your account password  
- **JWT Token** (optional): Direct token paste for advanced users

### 2. Authentication Methods

**Method A: Username & Password**
1. Enter your email and password
2. Tap "Test Auth" to validate credentials
3. App automatically retrieves and saves JWT token

**Method B: Direct Token**
1. Get a JWT token from the backend API:
   ```bash
   curl -X POST "http://your-server:8000/auth/jwt/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=your-email@example.com&password=your-password"
   ```
2. Paste the `access_token` into the JWT Token field
3. Tap "Test Auth" to validate

### 3. User Account Creation

Create user accounts through the backend:

**Option A: Web Dashboard**
1. Open `http://your-server:8501`
2. Login as admin
3. Create users through the interface

**Option B: API (Admin Token Required)**
```bash
export ADMIN_TOKEN="your-admin-jwt-token"

curl -X POST "http://your-server:8000/api/create_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "userpass", "display_name": "John Doe"}'
```

### 4. Admin Credentials

Admin credentials are set in the backend's `.env` file:
```bash
ADMIN_PASSWORD=your-secure-admin-password
ADMIN_EMAIL=admin@example.com
```

## Authentication Features

### Secure Credential Storage
- **Automatic Storage**: App saves credentials locally using AsyncStorage
- **Token Management**: JWT tokens are automatically refreshed
- **Password Security**: Passwords stored locally with show/hide toggle

### Data Isolation
- **User-Specific Data**: Each user can only access their own conversations and memories
- **Multi-Device Support**: Single user can connect multiple devices
- **Client ID Format**: Auto-generated as `user_id_suffix-device_name`

### WebSocket Authentication
The app automatically appends authentication to WebSocket URLs:
```
ws://your-server:8000/ws_pcm?token=JWT_TOKEN&device_name=phone&user_id=optional_user_id
```

## Troubleshooting Authentication

### Common Issues

**"Authentication failed"**
- Verify email/password combination
- Check backend is running and accessible
- Ensure user account exists

**"Could not connect to authentication server"**
- Verify WebSocket URL is correct
- Check network connectivity
- Ensure backend authentication endpoints are accessible

**"No access token received"**
- Check backend logs for authentication errors
- Verify `AUTH_SECRET_KEY` is set in backend `.env`
- Ensure user credentials are correct

### Testing Authentication

1. **Test Backend Connection**:
   ```bash
   curl http://your-server:8000/health
   ```

2. **Test Manual Login**:
   ```bash
   curl -X POST "http://your-server:8000/auth/jwt/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=your-email@example.com&password=your-password"
   ```

3. **Verify Token**:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://your-server:8000/api/users
   ```

## Security Best Practices

- **HTTPS in Production**: Always use `wss://` (secure WebSocket) in production
- **Strong Passwords**: Use strong, unique passwords for user accounts
- **Token Expiry**: JWT tokens expire after 1 hour for security
- **Network Security**: Configure proper firewall rules for backend access

For complete authentication documentation, see [`backends/advanced-backend/Docs/auth.md`](../backends/advanced-backend/Docs/auth.md).

## Getting Started

1. **Set up the backend** (see backend documentation)
2. **Install the app** using one of the methods above  
3. **Configure WebSocket URL** in the app
4. **Set up authentication** (for advanced backend)
5. **Connect to your OMI device** and start streaming audio 

