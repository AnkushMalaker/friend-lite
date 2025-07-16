import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';

interface ScanControlsProps {
  scanning: boolean;
  onScanPress: () => void;
  onStopScanPress: () => void;
  canScan: boolean; // To disable button if permissions not granted or BT is off
}

export const ScanControls: React.FC<ScanControlsProps> = ({
  scanning,
  onScanPress,
  onStopScanPress,
  canScan,
}) => {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Bluetooth Connection</Text>
      <TouchableOpacity
        style={[
          styles.button,
          scanning ? styles.buttonWarning : null,
          !canScan && !scanning ? styles.buttonDisabled : null, // Disable if cannot scan and not already scanning
        ]}
        onPress={scanning ? onStopScanPress : onScanPress}
        disabled={!canScan && !scanning}
      >
        <Text style={styles.buttonText}>{scanning ? "Stop Scan" : "Scan for Devices"}</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  section: {
    marginBottom: 25,
    padding: 15,
    backgroundColor: 'white',
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#333',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  buttonWarning: {
    backgroundColor: '#FF9500',
  },
  buttonDisabled: {
    backgroundColor: '#A0A0A0',
    opacity: 0.7,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ScanControls; 