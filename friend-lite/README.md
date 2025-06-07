## IOS install with Xcode (gives errors)
- Go to friend-lite folder
- npx expo prebuild --clean
- to to ios folder
- pod install
- Open the friendlite.workplacrc in xcode and create the app.

## IOS install with expo
- Go to friend-lite folder
- npx expo prebuild --clean
- npx expo install expo-dev-client
- npx expo start --dev-client
- npx expo run:ios --device