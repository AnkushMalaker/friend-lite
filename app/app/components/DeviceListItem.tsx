import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { OmiDevice } from 'friend-lite-react-native';

interface DeviceListItemProps {
  device: OmiDevice;
  onConnect: (deviceId: string) => void;
  onDisconnect: () => void;
  isConnecting: boolean;
  connectedDeviceId: string | null;
}

export const DeviceListItem: React.FC<DeviceListItemProps> = ({ 
  device, 
  onConnect, 
  onDisconnect,
  isConnecting,
  connectedDeviceId
}) => {
  const isThisDeviceConnected = connectedDeviceId === device.id;
  const isAnotherDeviceConnected = connectedDeviceId !== null && connectedDeviceId !== device.id;

  return (
    <View style={styles.deviceItem}>
      <View style={styles.deviceInfoContainer}>
        <Text style={styles.deviceName}>{device.name || 'Unknown Device'}</Text>
        <Text style={styles.deviceInfo}>ID: {device.id}</Text>
        {device.rssi != null && <Text style={styles.deviceInfo}>RSSI: {device.rssi} dBm</Text>}
      </View>
      {
        isThisDeviceConnected ? (
          <TouchableOpacity
            style={[styles.button, styles.smallButton, styles.buttonDanger]}
            onPress={onDisconnect}
            disabled={isConnecting} // Disable if any connection process is ongoing
          >
            <Text style={styles.buttonText}>{isConnecting ? 'Disconnecting...' : 'Disconnect'}</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[
              styles.button, 
              styles.smallButton, 
              (isConnecting || isAnotherDeviceConnected) ? styles.buttonDisabled : null
            ]}
            onPress={() => onConnect(device.id)}
            disabled={isConnecting || isAnotherDeviceConnected} // Disable if connecting to this/another device or another device is connected
          >
            <Text style={styles.buttonText}>{isConnecting && connectedDeviceId === device.id ? 'Connecting...' : 'Connect'}</Text>
          </TouchableOpacity>
        )
      }
    </View>
  );
};

const styles = StyleSheet.create({
  deviceItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 5, // Added some horizontal padding
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  deviceInfoContainer: {
    flex: 1, // Allow text to take available space and wrap if needed
    marginRight: 10, // Space between text and button
  },
  deviceName: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  deviceInfo: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    elevation: 1,
  },
  smallButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  buttonDanger: {
    backgroundColor: '#FF3B30',
  },
  buttonDisabled: {
    backgroundColor: '#A0A0A0',
    opacity: 0.7,
  },
  buttonText: {
    color: 'white',
    fontSize: 14, // Slightly smaller for small buttons
    fontWeight: '600',
  },
});

export default DeviceListItem; 