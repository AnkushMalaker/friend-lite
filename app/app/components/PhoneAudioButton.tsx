// PhoneAudioButton.tsx
import React from 'react';
import {
  TouchableOpacity,
  Text,
  View,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';

interface PhoneAudioButtonProps {
  isRecording: boolean;
  isInitializing: boolean;
  isDisabled: boolean;
  audioLevel: number;
  error: string | null;
  onPress: () => void;
}

const PhoneAudioButton: React.FC<PhoneAudioButtonProps> = ({
  isRecording,
  isInitializing,
  isDisabled,
  audioLevel,
  error,
  onPress,
}) => {

  const getButtonStyle = () => {
    if (isDisabled && !isRecording) {
      return [styles.button, styles.buttonDisabled];
    }
    if (isRecording) {
      return [styles.button, styles.buttonRecording];
    }
    if (error) {
      return [styles.button, styles.buttonError];
    }
    return [styles.button, styles.buttonIdle];
  };

  const getButtonText = () => {
    if (isInitializing) {
      return 'Initializing...';
    }
    if (isRecording) {
      return 'Stop Phone Audio';
    }
    return 'Stream Phone Audio';
  };

  const getMicrophoneIcon = () => {
    if (isRecording) {
      return '🎤'; // Recording microphone
    }
    return '🎙️'; // Idle microphone
  };

  return (
    <View style={styles.container}>
      <View style={styles.buttonWrapper}>
        <TouchableOpacity
          style={getButtonStyle()}
          onPress={onPress}
          disabled={isDisabled || isInitializing}
          activeOpacity={0.7}
        >
          {isInitializing ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <View style={styles.buttonContent}>
              <Text style={styles.icon}>{getMicrophoneIcon()}</Text>
              <Text style={styles.buttonText}>{getButtonText()}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      {/* Audio Level Indicator */}
      {isRecording && (
        <View style={styles.audioLevelContainer}>
          <View style={styles.audioLevelBackground}>
            <View
              style={[
                styles.audioLevelBar,
                { width: `${Math.min(audioLevel * 100, 100)}%` },
              ]}
            />
          </View>
          <Text style={styles.audioLevelText}>Audio Level</Text>
        </View>
      )}

      {/* Status Message */}
      {isRecording && (
        <Text style={styles.statusText}>
          Streaming audio to backend...
        </Text>
      )}

      {/* Error Message */}
      {error && !isRecording && (
        <Text style={styles.errorText}>{error}</Text>
      )}

      {/* Disabled Message */}
      {isDisabled && !isRecording && (
        <Text style={styles.disabledText}>
          Disconnect Bluetooth device to use phone audio
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 10,
    paddingHorizontal: 20,
  },
  buttonWrapper: {
    alignSelf: 'stretch',
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    minHeight: 48,
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonIdle: {
    backgroundColor: '#007AFF',
  },
  buttonRecording: {
    backgroundColor: '#FF3B30',
  },
  buttonDisabled: {
    backgroundColor: '#C7C7CC',
  },
  buttonError: {
    backgroundColor: '#FF9500',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  icon: {
    fontSize: 20,
  },
  statusText: {
    textAlign: 'center',
    marginTop: 8,
    fontSize: 12,
    color: '#8E8E93',
  },
  errorText: {
    textAlign: 'center',
    marginTop: 8,
    fontSize: 12,
    color: '#FF3B30',
  },
  disabledText: {
    textAlign: 'center',
    marginTop: 8,
    fontSize: 12,
    color: '#8E8E93',
    fontStyle: 'italic',
  },
  audioLevelContainer: {
    marginTop: 12,
    alignItems: 'center',
  },
  audioLevelBackground: {
    width: '100%',
    height: 4,
    backgroundColor: '#E5E5EA',
    borderRadius: 2,
    overflow: 'hidden',
  },
  audioLevelBar: {
    height: '100%',
    backgroundColor: '#34C759',
    borderRadius: 2,
  },
  audioLevelText: {
    marginTop: 4,
    fontSize: 10,
    color: '#8E8E93',
  },
});

export default PhoneAudioButton;