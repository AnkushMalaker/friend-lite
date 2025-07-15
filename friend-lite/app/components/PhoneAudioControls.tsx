import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { usePhoneAudioRecorder } from '../hooks/usePhoneAudioRecorder';
import { useAudioStreamer } from '../hooks/useAudioStreamer';

interface PhoneAudioControlsProps {
  backendUrl: string;
  authToken: string;
  userId: string;
  disabled: boolean;
}

export const PhoneAudioControls: React.FC<PhoneAudioControlsProps> = ({
  backendUrl,
  authToken,
  userId,
  disabled,
}) => {
  const [isActive, setIsActive] = useState(false);
  const isActiveRef = useRef(false);
  
  const phoneRecorder = usePhoneAudioRecorder();
  const audioStreamer = useAudioStreamer();

  // Keep ref in sync with state
  useEffect(() => {
    isActiveRef.current = isActive;
  }, [isActive]);

  const buildPhoneAudioWebSocketUrl = useCallback(() => {
    if (!backendUrl || !authToken) return '';
    
    // Convert HTTP URL to WebSocket URL and add /ws_pcm for phone audio
    let wsUrl = backendUrl;
    if (wsUrl.startsWith('http://')) {
      wsUrl = wsUrl.replace('http://', 'ws://');
    } else if (wsUrl.startsWith('https://')) {
      wsUrl = wsUrl.replace('https://', 'wss://');
    } else if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
      // Assume http if no protocol specified
      wsUrl = 'ws://' + wsUrl;
    }
    
    // Remove any existing WebSocket paths and add /ws_pcm for phone audio (raw PCM data)
    wsUrl = wsUrl.replace(/\/(ws|ws_pcm|ws_omi).*$/, '');
    wsUrl += '/ws_pcm';
    
    // Add authentication and device info
    wsUrl += `?token=${encodeURIComponent(authToken)}&device_name=phone`;
    
    // Add user_id if provided
    if (userId && userId.trim() !== '') {
      wsUrl += `&user_id=${encodeURIComponent(userId.trim())}`;
    }
    
    return wsUrl;
  }, [backendUrl, authToken, userId]);

  const handleStartPhoneRecording = useCallback(async () => {
    if (!backendUrl || backendUrl.trim() === '') {
      Alert.alert('Backend URL Required', 'Please enter the backend URL for streaming.');
      return;
    }
    
    if (!authToken || authToken.trim() === '') {
      Alert.alert('Authentication Required', 'Please authenticate first to record phone audio.');
      return;
    }

    try {
      const phoneAudioUrl = buildPhoneAudioWebSocketUrl();
      console.log('[PhoneAudioControls] Starting new recording session');
      console.log('[PhoneAudioControls] Using WebSocket URL:', phoneAudioUrl);

      // Check if already connected to avoid multiple connections
      const currentReadyState = audioStreamer.getWebSocketReadyState();
      if (currentReadyState !== WebSocket.OPEN) {
        // Start WebSocket streaming first
        console.log('[PhoneAudioControls] Starting WebSocket connection...');
        await audioStreamer.startStreaming(phoneAudioUrl);
        console.log('[PhoneAudioControls] WebSocket connection established');
      } else {
        console.log('[PhoneAudioControls] WebSocket already connected, reusing connection');
      }

      // Then start phone recording with connection check
      await phoneRecorder.startRecording((audioBytes) => {
        const wsReadyState = audioStreamer.getWebSocketReadyState();
        if (wsReadyState === WebSocket.OPEN && audioBytes.length > 0) {
          audioStreamer.sendAudio(audioBytes);
        } else {
          console.log('[PhoneAudioControls] WebSocket not ready for audio data, skipping chunk');
        }
      });

      setIsActive(true);
      console.log('[PhoneAudioControls] Phone audio recording and streaming started');

    } catch (error) {
      console.error('[PhoneAudioControls] Error starting phone recording:', error);
      const errorMessage = error instanceof Error ? error.message : 'Could not start phone audio recording.';
      
      if (errorMessage.includes('permission')) {
        Alert.alert(
          'Microphone Permission Required', 
          'This app needs microphone access to record audio. Please enable microphone permission in your device settings and try again.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => {
              // On React Native, you can't directly open settings, but this provides user guidance
              Alert.alert('Settings Help', 'Go to Settings > Apps > Friend Lite > Permissions > Microphone and enable access.');
            }}
          ]
        );
      } else {
        Alert.alert('Error', errorMessage);
      }
      
      // Cleanup on error
      if (audioStreamer.isStreaming) audioStreamer.stopStreaming();
      if (phoneRecorder.isRecording) await phoneRecorder.stopRecording();
    }
  }, [backendUrl, authToken, buildPhoneAudioWebSocketUrl, phoneRecorder, audioStreamer]);

  const handleStopPhoneRecording = useCallback(async () => {
    try {
      console.log('[PhoneAudioControls] Stopping phone audio recording and streaming');
      
      // Stop recording first
      await phoneRecorder.stopRecording();
      
      // Then stop streaming
      audioStreamer.stopStreaming();
      
      setIsActive(false);
      console.log('[PhoneAudioControls] Phone audio recording and streaming stopped');

    } catch (error) {
      console.error('[PhoneAudioControls] Error stopping phone recording:', error);
      Alert.alert('Error', 'Could not stop phone audio recording.');
      
      // Ensure we clean up state even if there's an error
      setIsActive(false);
    }
  }, [phoneRecorder, audioStreamer]);

  const getStatusText = () => {
    if (phoneRecorder.isProcessing) return 'Initializing...';
    if (phoneRecorder.isRecording && audioStreamer.isStreaming) return 'Recording & Streaming';
    if (phoneRecorder.isRecording) return 'Recording...';
    if (audioStreamer.isConnecting) return 'Connecting...';
    return 'Ready';
  };

  const getStatusColor = () => {
    if (phoneRecorder.error || audioStreamer.error) return '#FF3B30';
    if (isActive && phoneRecorder.isRecording && audioStreamer.isStreaming) return '#34C759';
    if (phoneRecorder.isProcessing || audioStreamer.isConnecting) return '#FF9500';
    return '#666';
  };

  // Cleanup on component unmount only
  useEffect(() => {
    return () => {
      if (isActiveRef.current) {
        console.log('[PhoneAudioControls] Component unmounting, cleaning up connections');
        phoneRecorder.stopRecording().catch(console.error);
        audioStreamer.stopStreaming();
      }
    };
  }, []); // Empty dependency array - only run on mount/unmount

  const isLoading = phoneRecorder.isProcessing || audioStreamer.isConnecting;
  const hasError = phoneRecorder.error || audioStreamer.error;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>📱 Phone Audio Recording</Text>
        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
          <Text style={[styles.statusText, { color: getStatusColor() }]}>
            {getStatusText()}
          </Text>
        </View>
      </View>

      {hasError && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            {phoneRecorder.error || audioStreamer.error}
          </Text>
        </View>
      )}

      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isActive ? styles.recordButtonActive : styles.recordButtonInactive,
            (disabled || isLoading) && styles.recordButtonDisabled,
          ]}
          onPress={isActive ? handleStopPhoneRecording : handleStartPhoneRecording}
          disabled={disabled || isLoading}
        >
          {isLoading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.recordButtonText}>
              {isActive ? '⏹ Stop Recording' : '🎤 Start Recording'}
            </Text>
          )}
        </TouchableOpacity>
      </View>

      {(phoneRecorder.isRecording || phoneRecorder.audioPacketsStreamed > 0) && (
        <View style={styles.statsContainer}>
          <Text style={styles.statsText}>
            Audio packets streamed: {phoneRecorder.audioPacketsStreamed}
          </Text>
          {audioStreamer.isStreaming && (
            <Text style={styles.statsText}>
              ✅ Connected to backend
            </Text>
          )}
        </View>
      )}

      <Text style={styles.descriptionText}>
        Records audio from your phone's microphone and streams it directly to the backend for real-time transcription and processing.
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    margin: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  errorContainer: {
    backgroundColor: '#FFF2F2',
    borderColor: '#FF3B30',
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    marginBottom: 15,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    textAlign: 'center',
  },
  controlsContainer: {
    alignItems: 'center',
    marginBottom: 15,
  },
  recordButton: {
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    minWidth: 200,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButtonInactive: {
    backgroundColor: '#007AFF',
  },
  recordButtonActive: {
    backgroundColor: '#FF3B30',
  },
  recordButtonDisabled: {
    backgroundColor: '#A0A0A0',
    opacity: 0.7,
  },
  recordButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  statsContainer: {
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
  },
  statsText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 4,
  },
  descriptionText: {
    fontSize: 13,
    color: '#888',
    textAlign: 'center',
    lineHeight: 18,
  },
});

export default PhoneAudioControls;