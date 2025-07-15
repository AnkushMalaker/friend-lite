import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';

interface AudioModeSelectorProps {
  phoneAudioMode: boolean;
  onModeChange: (phoneMode: boolean) => void;
  disabled?: boolean;
}

export const AudioModeSelector: React.FC<AudioModeSelectorProps> = ({
  phoneAudioMode,
  onModeChange,
  disabled = false,
}) => {
  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Audio Source</Text>
      
      <View style={styles.audioModeContainer}>
        <TouchableOpacity
          style={[
            styles.audioModeButton,
            !phoneAudioMode && styles.audioModeButtonActive,
            disabled && styles.audioModeButtonDisabled,
          ]}
          onPress={() => !disabled && onModeChange(false)}
          disabled={disabled}
        >
          <Text style={[
            styles.audioModeButtonText,
            !phoneAudioMode && styles.audioModeButtonTextActive
          ]}>
            🎧 OMI Device
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[
            styles.audioModeButton,
            phoneAudioMode && styles.audioModeButtonActive,
            disabled && styles.audioModeButtonDisabled,
          ]}
          onPress={() => !disabled && onModeChange(true)}
          disabled={disabled}
        >
          <Text style={[
            styles.audioModeButtonText,
            phoneAudioMode && styles.audioModeButtonTextActive
          ]}>
            📱 Phone Audio
          </Text>
        </TouchableOpacity>
      </View>
      
      <Text style={styles.descriptionText}>
        {phoneAudioMode 
          ? "Record audio directly from your phone's microphone" 
          : "Connect to an OMI/Friend device via Bluetooth"
        }
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    margin: 15,
    marginBottom: 0,
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
  audioModeContainer: {
    flexDirection: 'row',
    marginBottom: 15,
  },
  audioModeButton: {
    flex: 1,
    paddingVertical: 15,
    paddingHorizontal: 10,
    marginHorizontal: 5,
    borderRadius: 10,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  audioModeButtonActive: {
    backgroundColor: '#007AFF',
    borderColor: '#0056B3',
  },
  audioModeButtonDisabled: {
    opacity: 0.5,
  },
  audioModeButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
  },
  audioModeButtonTextActive: {
    color: 'white',
  },
  descriptionText: {
    fontSize: 13,
    color: '#666',
    textAlign: 'center',
    lineHeight: 18,
  },
});

export default AudioModeSelector;