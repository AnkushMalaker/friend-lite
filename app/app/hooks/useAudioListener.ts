import { useState, useRef, useCallback } from 'react';
import { Alert } from 'react-native';
import { OmiConnection } from 'friend-lite-react-native';
import { Subscription, ConnectionPriority } from 'react-native-ble-plx'; // OmiConnection might use this type for subscriptions

interface UseAudioListener {
  isListeningAudio: boolean;
  audioPacketsReceived: number;
  startAudioListener: (onAudioData: (bytes: Uint8Array) => void) => Promise<void>;
  stopAudioListener: () => Promise<void>;
}

export const useAudioListener = (
  omiConnection: OmiConnection,
  isConnected: () => boolean // Function to check current connection status
): UseAudioListener => {
  const [isListeningAudio, setIsListeningAudio] = useState<boolean>(false);
  const [audioPacketsReceived, setAudioPacketsReceived] = useState<number>(0);
  
  const audioSubscriptionRef = useRef<Subscription | null>(null);
  const uiUpdateIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const localPacketCounterRef = useRef<number>(0);

  const stopAudioListener = useCallback(async () => {
    console.log('Attempting to stop audio listener...');
    if (uiUpdateIntervalRef.current) {
      clearInterval(uiUpdateIntervalRef.current);
      uiUpdateIntervalRef.current = null;
    }
    if (audioSubscriptionRef.current) {
      try {
        await omiConnection.stopAudioBytesListener(audioSubscriptionRef.current);
        audioSubscriptionRef.current = null;
        setIsListeningAudio(false);
        localPacketCounterRef.current = 0; // Reset local counter
        // setAudioPacketsReceived(0); // Optionally reset global counter on stop, or keep cumulative
        console.log('Audio listener stopped.');
      } catch (error) {
        console.error('Stop audio listener error:', error);
        Alert.alert('Error', `Failed to stop audio listener: ${error}`);
      }
    } else {
      console.log('Audio listener was not active.');
    }
    setIsListeningAudio(false); // Ensure state is false even if no subscription was found
  }, [omiConnection]);

  const startAudioListener = useCallback(async (onAudioData: (bytes: Uint8Array) => void) => {
    if (!isConnected()) {
      Alert.alert('Not Connected', 'Please connect to a device first to start audio listener.');
      return;
    }
    if (isListeningAudio) { 
        console.log('Audio listener is already active. Stopping first.');
        await stopAudioListener();
    }

    setAudioPacketsReceived(0); // Reset counter on start
    localPacketCounterRef.current = 0;
    console.log('Starting audio bytes listener...');

    // Request high connection priority before starting audio listener
    try {
      await omiConnection.requestConnectionPriority(ConnectionPriority.High); // 1 for ConnectionPriority.High
      console.log('Requested high connection priority.');
    } catch (error) {
      console.error('Failed to request high connection priority:', error);
      Alert.alert('Error', `Failed to request high connection priority: ${error}`);
    }
    
    // Batch UI updates for packet counter
    if (uiUpdateIntervalRef.current) clearInterval(uiUpdateIntervalRef.current);
    uiUpdateIntervalRef.current = setInterval(() => {
      if (localPacketCounterRef.current > 0) {
        setAudioPacketsReceived(prev => prev + localPacketCounterRef.current);
        localPacketCounterRef.current = 0;
      }
    }, 500); // Update UI every 500ms

    try {
      const subscription = await omiConnection.startAudioBytesListener((bytes) => {
        localPacketCounterRef.current++;
        // If bytes is number[], convert to Uint8Array for the callback
        // If bytes is already Uint8Array, this is redundant but harmless
        // If bytes is ArrayBuffer, this is also valid.
        if (bytes && bytes.length > 0) {
            onAudioData(new Uint8Array(bytes)); 
        }
      });

      if (subscription) {
        audioSubscriptionRef.current = subscription;
        setIsListeningAudio(true);
        console.log('Audio listener started successfully.');
      } else {
        Alert.alert('Error', 'Failed to start audio listener. No subscription returned.');
        if (uiUpdateIntervalRef.current) clearInterval(uiUpdateIntervalRef.current);
        setIsListeningAudio(false); // Ensure state consistency
      }
    } catch (error) {
      console.error('Start audio listener error:', error);
      Alert.alert('Error', `Failed to start audio listener: ${error}`);
      if (uiUpdateIntervalRef.current) clearInterval(uiUpdateIntervalRef.current);
      setIsListeningAudio(false);
    }
  }, [omiConnection, isConnected, stopAudioListener]); // Added stopAudioListener dependency

  return {
    isListeningAudio,
    audioPacketsReceived,
    startAudioListener,
    stopAudioListener,
  };
}; 