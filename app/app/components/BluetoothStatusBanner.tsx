import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Linking, Platform } from 'react-native';
import { State as BluetoothState } from 'react-native-ble-plx';

interface BluetoothStatusBannerProps {
  bluetoothState: BluetoothState;
  isPermissionsLoading: boolean;
  permissionGranted: boolean;
  onRequestPermission: () => void;
}

export const BluetoothStatusBanner: React.FC<BluetoothStatusBannerProps> = ({ 
  bluetoothState, 
  isPermissionsLoading, 
  permissionGranted, 
  onRequestPermission 
}) => {
  if (isPermissionsLoading && bluetoothState === BluetoothState.Unknown) {
    return (
      <View style={[styles.statusBanner, styles.bannerInfo]}>
        <Text style={styles.statusText}>Initializing Bluetooth...</Text>
      </View>
    );
  }

  if (bluetoothState === BluetoothState.PoweredOn && permissionGranted) {
    return null; // All good, don't show banner
  }

  let bannerMessage = 'Bluetooth status is unknown.';
  let buttonText = 'Check Status';
  let onButtonPress: (() => void) | undefined = undefined;

  switch (bluetoothState) {
    case BluetoothState.PoweredOff:
      bannerMessage = 'Bluetooth is turned off. Please enable Bluetooth to use this app.';
      buttonText = 'Open Settings';
      onButtonPress = () => Linking.openSettings().catch(err => console.warn("Couldn't open settings:", err));
      break;
    case BluetoothState.Unauthorized:
      bannerMessage = 'Bluetooth permission not granted. Please allow Bluetooth access.';
      buttonText = 'Grant Permission';
      onButtonPress = onRequestPermission;
      break;
    case BluetoothState.Unsupported:
      bannerMessage = 'Bluetooth is not supported on this device.';
      buttonText = 'OK';
      onButtonPress = undefined;
      break;
    case BluetoothState.Resetting:
      bannerMessage = 'Bluetooth is resetting. Please wait.';
      buttonText = 'OK';
      onButtonPress = undefined;
      break;
    case BluetoothState.PoweredOn:
      if (!permissionGranted) {
        bannerMessage = 'Bluetooth is on, but permission is needed.';
        buttonText = 'Grant Permission';
        onButtonPress = onRequestPermission;
      } else {
         // This case should be caught by the early return null
      }
      break;
    default:
      bannerMessage = `Bluetooth state: ${bluetoothState}. Please ensure it is enabled and permissions are granted.`;
      buttonText = 'Request Permissions';
      onButtonPress = onRequestPermission;
      break;
  }

  return (
    <View style={[styles.statusBanner, bluetoothState === BluetoothState.PoweredOff || bluetoothState === BluetoothState.Unauthorized ? styles.bannerWarning : styles.bannerInfo]}>
      <Text style={styles.statusText}>{bannerMessage}</Text>
      {onButtonPress && (
        <TouchableOpacity style={styles.statusButton} onPress={onButtonPress}>
          <Text style={styles.statusButtonText}>{buttonText}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  statusBanner: {
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  bannerWarning: {
    backgroundColor: '#FF9500', // Orange for warnings
  },
  bannerInfo: {
    backgroundColor: '#007AFF', // Blue for info
  },
  statusText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
    marginRight: 10,
  },
  statusButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  statusButtonText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 12,
  },
});

// Exporting default to be easily consumable if this is the only export
export default BluetoothStatusBanner; 