import { useState, useRef, useCallback } from 'react';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Alert, Platform, PermissionsAndroid } from 'react-native';

interface UsePhoneAudioRecorder {
  isRecording: boolean;
  isProcessing: boolean;
  error: string | null;
  startRecording: (onAudioData: (audioBytes: Uint8Array) => void) => Promise<void>;
  stopRecording: () => Promise<void>;
  audioPacketsStreamed: number;
}

export const usePhoneAudioRecorder = (): UsePhoneAudioRecorder => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioPacketsStreamed, setAudioPacketsStreamed] = useState(0);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const onAudioDataRef = useRef<((audioBytes: Uint8Array) => void) | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const shouldContinueRecordingRef = useRef<boolean>(false);

  const startRecording = useCallback(async (onAudioData: (audioBytes: Uint8Array) => void) => {
    try {
      setError(null);
      setIsProcessing(true);
      setAudioPacketsStreamed(0);
      onAudioDataRef.current = onAudioData;
      shouldContinueRecordingRef.current = true;

      // Platform-specific permission checking
      if (Platform.OS === 'android') {
        console.log('[PhoneAudioRecorder] Checking Android permissions...');
        
        // Check if permission is already granted
        const hasPermission = await PermissionsAndroid.check(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
        console.log('[PhoneAudioRecorder] Android RECORD_AUDIO permission granted:', hasPermission);
        
        if (!hasPermission) {
          console.log('[PhoneAudioRecorder] Requesting Android RECORD_AUDIO permission...');
          const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
            {
              title: 'Microphone Permission',
              message: 'Friend Lite needs access to your microphone to record audio for transcription.',
              buttonNeutral: 'Ask Me Later',
              buttonNegative: 'Cancel',
              buttonPositive: 'OK',
            }
          );
          console.log('[PhoneAudioRecorder] Android permission request result:', granted);
          
          if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
            throw new Error('Android microphone permission not granted. Please enable it in Settings > Apps > Friend Lite > Permissions > Microphone.');
          }
        }
      }

      // Check current permissions first with expo-av
      const currentPermission = await Audio.getPermissionsAsync();
      console.log('[PhoneAudioRecorder] Current expo-av permission status:', currentPermission);
      
      let permission = currentPermission;
      if (currentPermission.status !== 'granted') {
        console.log('[PhoneAudioRecorder] Requesting expo-av permissions...');
        permission = await Audio.requestPermissionsAsync();
        console.log('[PhoneAudioRecorder] expo-av permission request result:', permission);
      }
      
      if (permission.status !== 'granted') {
        console.error('[PhoneAudioRecorder] expo-av permission denied:', permission);
        throw new Error(`Audio recording permission not granted. Status: ${permission.status}. Please enable microphone access in your device settings.`);
      }

      // Configure audio mode for recording (platform-specific)
      console.log('[PhoneAudioRecorder] Platform.OS:', Platform.OS);
      
      if (Platform.OS === 'ios') {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
          staysActiveInBackground: true,
          interruptionModeIOS: Audio.INTERRUPTION_MODE_IOS_DO_NOT_MIX,
        });
      } else {
        // Android - use minimal config to avoid invalid values
        console.log('[PhoneAudioRecorder] Setting Android audio mode...');
        try {
          await Audio.setAudioModeAsync({
            staysActiveInBackground: true,
            shouldDuckAndroid: false,
            playThroughEarpieceAndroid: false,
          });
          console.log('[PhoneAudioRecorder] Android audio mode set successfully');
        } catch (audioError) {
          console.warn('[PhoneAudioRecorder] Android audio mode failed, using default:', audioError);
          // Fall back to default/empty config
          await Audio.setAudioModeAsync({});
        }
      }

      // Recording options optimized for backend requirements
      const recordingOptions: Audio.RecordingOptions = {
        android: {
          extension: '.wav',
          outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_DEFAULT,
          audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_DEFAULT,
          sampleRate: 16000, // 16kHz as required by backend
          numberOfChannels: 1, // Mono as required
          bitRate: 256000,
        },
        ios: {
          extension: '.wav',
          outputFormat: Audio.RECORDING_OPTION_IOS_OUTPUT_FORMAT_LINEARPCM,
          audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
          sampleRate: 16000, // 16kHz as required by backend
          numberOfChannels: 1, // Mono as required
          bitRate: 256000,
          linearPCMBitDepth: 16, // 16-bit as required
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {
          mimeType: 'audio/wav',
          bitsPerSecond: 256000,
        },
      };

      // Create recording
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(recordingOptions);
      recordingRef.current = recording;

      // Start recording
      await recording.startAsync();
      recordingStartTimeRef.current = Date.now();
      setIsRecording(true);
      setIsProcessing(false);

      console.log('[PhoneAudioRecorder] Recording started - continuous mode');

      // Send first chunk after 10 seconds for immediate feedback
      setTimeout(async () => {
        console.log('[PhoneAudioRecorder] 10-second timeout triggered');
        if (shouldContinueRecordingRef.current) {
          console.log('[PhoneAudioRecorder] Processing first 10-second chunk...');
          try {
            await processLargeAudioChunk();
          } catch (error) {
            console.error('[PhoneAudioRecorder] Error in first chunk processing:', error);
          }
        } else {
          console.log('[PhoneAudioRecorder] Should continue recording is false, skipping first chunk');
        }
      }, 10000);

      // Then send chunks every 60 seconds for regular streaming
      intervalRef.current = setInterval(async () => {
        await processLargeAudioChunk();
      }, 60000); // Process every 60 seconds after first chunk

    } catch (err) {
      console.error('[PhoneAudioRecorder] Error starting recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to start recording');
      setIsProcessing(false);
      setIsRecording(false);
      shouldContinueRecordingRef.current = false;
    }
  }, []);

  const processLargeAudioChunk = useCallback(async () => {
    console.log('[PhoneAudioRecorder] processLargeAudioChunk called');
    console.log('[PhoneAudioRecorder] Checking conditions:', {
      hasRecording: !!recordingRef.current,
      hasCallback: !!onAudioDataRef.current,
      shouldContinue: shouldContinueRecordingRef.current
    });
    
    if (!recordingRef.current || !onAudioDataRef.current || !shouldContinueRecordingRef.current) {
      console.log('[PhoneAudioRecorder] Conditions not met, exiting processLargeAudioChunk');
      return;
    }

    try {
      const recordingDuration = Date.now() - recordingStartTimeRef.current;
      console.log(`[PhoneAudioRecorder] Processing large audio chunk after ${Math.round(recordingDuration/1000)} seconds...`);
      
      // Stop current recording to get the audio file
      console.log('[PhoneAudioRecorder] Stopping and unloading recording...');
      await recordingRef.current.stopAndUnloadAsync();
      console.log('[PhoneAudioRecorder] Recording stopped, getting URI...');
      const uri = recordingRef.current.getURI();
      console.log('[PhoneAudioRecorder] URI obtained:', uri);
      
      if (uri) {
        console.log('[PhoneAudioRecorder] Reading large audio file from:', uri);
        
        // Read the audio file as binary data
        const audioData = await FileSystem.readAsStringAsync(uri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        
        // Convert base64 to Uint8Array
        const binaryString = atob(audioData);
        const audioBytes = new Uint8Array(binaryString.length);
        
        for (let i = 0; i < binaryString.length; i++) {
          audioBytes[i] = binaryString.charCodeAt(i);
        }

        // Extract PCM data (skip WAV header - first 44 bytes)
        const pcmData = audioBytes.slice(44);
        
        console.log('[PhoneAudioRecorder] Sending large PCM data:', pcmData.length, 'bytes');
        
        // Send audio data to callback
        onAudioDataRef.current(pcmData);
        setAudioPacketsStreamed(prev => {
          const newCount = prev + 1;
          console.log(`[PhoneAudioRecorder] Audio packets count updated to: ${newCount}`);
          return newCount;
        });
        
        // Clean up the temporary file
        await FileSystem.deleteAsync(uri, { idempotent: true });
      }

      // Restart recording for next 2-minute chunk
      if (shouldContinueRecordingRef.current) {
        console.log('[PhoneAudioRecorder] Restarting recording for next 2-minute chunk...');
        recordingStartTimeRef.current = Date.now();
        
        const recording = new Audio.Recording();
        
        const recordingOptions: Audio.RecordingOptions = {
          android: {
            extension: '.wav',
            outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_DEFAULT,
            audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_DEFAULT,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 256000,
          },
          ios: {
            extension: '.wav',
            outputFormat: Audio.RECORDING_OPTION_IOS_OUTPUT_FORMAT_LINEARPCM,
            audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 256000,
            linearPCMBitDepth: 16,
            linearPCMIsBigEndian: false,
            linearPCMIsFloat: false,
          },
          web: {
            mimeType: 'audio/wav',
            bitsPerSecond: 256000,
          },
        };
        
        await recording.prepareToRecordAsync(recordingOptions);
        await recording.startAsync();
        recordingRef.current = recording;
        
        console.log('[PhoneAudioRecorder] Recording restarted for next 2-minute chunk');
      }

    } catch (err) {
      console.error('[PhoneAudioRecorder] Error processing large audio chunk:', err);
    }
  }, []);

  const stopRecording = useCallback(async () => {
    try {
      setIsProcessing(true);
      shouldContinueRecordingRef.current = false;

      // Clear interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      // Stop recording and process the final audio data
      if (recordingRef.current) {
        const status = await recordingRef.current.getStatusAsync();
        if (status.isRecording) {
          await recordingRef.current.stopAndUnloadAsync();
        }
        
        // Process the final recording
        const uri = recordingRef.current.getURI();
        if (uri && onAudioDataRef.current) {
          console.log('[PhoneAudioRecorder] Processing final recording from:', uri);
          
          try {
            // Read the audio file as binary data
            const audioData = await FileSystem.readAsStringAsync(uri, {
              encoding: FileSystem.EncodingType.Base64,
            });
            
            // Convert base64 to Uint8Array
            const binaryString = atob(audioData);
            const audioBytes = new Uint8Array(binaryString.length);
            
            for (let i = 0; i < binaryString.length; i++) {
              audioBytes[i] = binaryString.charCodeAt(i);
            }

            // Extract PCM data (skip WAV header - first 44 bytes)
            const pcmData = audioBytes.slice(44);
            
            console.log('[PhoneAudioRecorder] Sending final PCM data:', pcmData.length, 'bytes');
            
            // Send audio data to callback
            onAudioDataRef.current(pcmData);
            setAudioPacketsStreamed(prev => {
              const newCount = prev + 1;
              console.log(`[PhoneAudioRecorder] Final audio packet count updated to: ${newCount}`);
              return newCount;
            });
          } catch (audioError) {
            console.error('[PhoneAudioRecorder] Error processing final audio:', audioError);
          }
        }
        
        // Clean up final recording file
        if (uri) {
          await FileSystem.deleteAsync(uri, { idempotent: true });
        }
        
        recordingRef.current = null;
      }

      setIsRecording(false);
      setIsProcessing(false);
      onAudioDataRef.current = null;
      shouldContinueRecordingRef.current = false;

      console.log('[PhoneAudioRecorder] Recording stopped');
      console.log(`[PhoneAudioRecorder] Total audio packets streamed: ${audioPacketsStreamed}`);

    } catch (err) {
      console.error('[PhoneAudioRecorder] Error stopping recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to stop recording');
      setIsRecording(false);
      setIsProcessing(false);
    }
  }, [audioPacketsStreamed]);

  return {
    isRecording,
    isProcessing,
    error,
    startRecording,
    stopRecording,
    audioPacketsStreamed,
  };
};