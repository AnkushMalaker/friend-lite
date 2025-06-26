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
