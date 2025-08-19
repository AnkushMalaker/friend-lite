# Friend-Lite Mobile App

React Native mobile application for connecting OMI devices and streaming audio to Friend-Lite backends. Supports cross-platform deployment on iOS and Android with Bluetooth integration.

## Features

- **OMI Device Integration**: Connect via Bluetooth and stream audio
- **Cross-Platform**: iOS and Android support using React Native
- **Real-time Audio Streaming**: OPUS audio transmission to backend services
- **WebSocket Communication**: Efficient real-time data transfer
- **Backend Selection**: Configure connection to any compatible backend

## Quick Start

### Prerequisites

- Node.js 18+ and npm installed
- Expo CLI: `npm install -g @expo/cli`
- **iOS**: Xcode and iOS Simulator or physical iOS device
- **Android**: Android Studio and Android device/emulator

### Installation

```bash
# Navigate to app directory
cd app

# Install dependencies
npm install

# Start development server
npm start
```

## Platform-Specific Setup

### iOS Development

#### Method 1: Expo Development Build (Recommended)

```bash
# Clean and prebuild
npx expo prebuild --clean

# Install development client
npx expo install expo-dev-client

# Start development server
npx expo start --dev-client

# Run on iOS device
npx expo run:ios --device
```

#### Method 2: Xcode Development

```bash
# Prebuild for iOS
npx expo prebuild --clean

# Install iOS dependencies
cd ios && pod install && cd ..

# Open in Xcode
open ios/friendlite.xcworkspace
```

Build and run from Xcode interface.

### Android Development

```bash
# Build and run on Android device
npx expo run:android --device
```

#### Android Network Configuration

For local development, configure network permissions:

**Development (Local Backend):**
Add to `android/app/src/main/AndroidManifest.xml`:
```xml
<application
    android:usesCleartextTraffic="true"
    ... >
```

**Production (HTTPS Backend):**
Create `android/app/src/main/res/xml/network_security_config.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">your-backend-domain.com</domain>
    </domain-config>
</network-security-config>
```

Reference in `AndroidManifest.xml`:
```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... >
```

## Backend Configuration

### Supported Backends

The app connects to any backend that accepts OPUS audio streams:

1. **Simple Backend** (`backends/simple/`)
   - Basic audio capture and storage
   - Good for testing and development
   - WebSocket endpoint: `/ws`

2. **Advanced Backend** (`backends/advanced-backend/`)
   - Full transcription and memory features
   - Real-time processing with speaker recognition
   - WebSocket endpoint: `/ws_pcm`

### Connection Setup

#### Local Development
```
Backend URL: ws://[machine-ip]:8000/ws_pcm
Example: ws://192.168.1.100:8000/ws_pcm
```

#### Public Access (Production)
Use ngrok or similar tunneling service:

```bash
# Start ngrok tunnel
ngrok http 8000

# Use provided URL in app
Backend URL: wss://[ngrok-subdomain].ngrok.io/ws_pcm
```

### Configuration Steps

1. **Start your chosen backend** (see backend-specific README)
2. **Open the mobile app**
3. **Navigate to Settings**
4. **Enter Backend URL**:
   - Local: `ws://[your-ip]:8000/ws_pcm`
   - Public: `wss://[your-domain]/ws_pcm`
5. **Save configuration**

## User Workflow

### Device Connection

1. **Enable Bluetooth** on your mobile device
2. **Open Friend-Lite app**
3. **Pair OMI device**:
   - Go to Device Settings
   - Scan for nearby OMI devices
   - Select your device from the list
   - Complete pairing process

### Audio Streaming

1. **Configure backend connection** (see Configuration Steps above)
2. **Test connection**:
   - Tap "Test Connection" in settings
   - Verify green status indicator
3. **Start recording**:
   - Press record button in main interface
   - Speak into OMI device
   - Audio streams to backend in real-time

### Monitoring

1. **Check connection status** in app header
2. **View real-time indicators**:
   - Audio level meters
   - Connection status
   - Battery level (if supported)
3. **Access backend dashboard** for processed results

## Troubleshooting

### Common Issues

**Bluetooth Connection Problems:**
- Ensure OMI device is in pairing mode
- Reset Bluetooth on mobile device
- Clear app cache and restart
- Check device compatibility

**Audio Streaming Issues:**
- Verify backend URL format (include `ws://` or `wss://`)
- Check network connectivity
- Test with simple backend first
- Monitor backend logs for connection attempts

**Build Errors:**
- Clear Expo cache: `npx expo start --clear`
- Clean prebuild: `npx expo prebuild --clean`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Debug Mode

Enable detailed logging:
1. Go to app Settings
2. Enable "Debug Mode"
3. View console logs for connection details

### Network Testing

Test backend connectivity:
```bash
# Test WebSocket endpoint
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: test" \
     -H "Sec-WebSocket-Version: 13" \
     http://[backend-ip]:8000/ws_pcm
```

## Development

### Project Structure
```
app/
├── src/
│   ├── components/     # React Native components
│   ├── screens/        # App screens
│   ├── services/       # WebSocket and Bluetooth services
│   └── utils/          # Helper utilities
├── app.json           # Expo configuration
└── package.json       # Dependencies
```

### Key Dependencies
- **React Native**: Cross-platform mobile framework
- **Expo**: Development and build toolchain
- **React Native Bluetooth**: OMI device communication
- **WebSocket**: Real-time backend communication

### Building for Production

#### iOS App Store
```bash
# Build for iOS
npx expo build:ios

# Follow Expo documentation for App Store submission
```

#### Android Play Store
```bash
# Build for Android
npx expo build:android

# Generate signed APK for distribution
```

## Integration Examples

### WebSocket Communication
```javascript
// Connect to backend
const ws = new WebSocket('ws://backend-url:8000/ws_pcm');

// Send audio data
ws.send(audioBuffer);

// Handle responses
ws.onmessage = (event) => {
  // Process transcription or acknowledgment
};
```

### Bluetooth Audio Capture
```javascript
// Start audio streaming from OMI device
await BluetoothService.startAudioStream();

// Handle audio data
BluetoothService.onAudioData = (audioBuffer) => {
  websocket.send(audioBuffer);
};
```

## Related Documentation

- **[Backend Setup](../backends/)**: Choose and configure backend services
- **[Quick Start Guide](../quickstart.md)**: Complete system setup
- **[Advanced Backend](../backends/advanced-backend/)**: Full-featured backend option
- **[Simple Backend](../backends/simple/)**: Basic backend for testing