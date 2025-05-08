import { useState, useRef, useCallback, useEffect } from 'react';
import { Alert } from 'react-native';
import NetInfo from "@react-native-community/netinfo";

interface UseTranscription {
  enableTranscription: boolean;
  setEnableTranscription: (value: boolean) => void;
  deepgramApiKey: string;
  setDeepgramApiKey: (apiKey: string) => void;
  transcription: string;
  isTranscribing: boolean;
  initializeWebSocket: (customWebSocketUrl?: string) => void;
  closeWebSocket: () => void;
  processAudioForTranscription: (audioBytes: Uint8Array) => void;
  clearTranscription: () => void;
}

const MAX_TRANSCRIPTION_LINES = 5;

export const useTranscription = (): UseTranscription => {
  const [enableTranscription, setEnableTranscriptionState] = useState<boolean>(false);
  const [deepgramApiKey, setDeepgramApiKey] = useState<string>('');
  const [transcription, setTranscription] = useState<string>('');
  
  const websocketRef = useRef<WebSocket | null>(null);
  const isTranscribingRef = useRef<boolean>(false); // Using ref for synchronous checks within WebSocket callbacks
  const audioBufferRef = useRef<Uint8Array[]>([]);
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const clearTranscription = useCallback(() => {
    setTranscription('');
  }, []);

  const sendAudioToWebSocket = useCallback(() => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN || !isTranscribingRef.current || audioBufferRef.current.length === 0) {
      return;
    }
    try {
      for (const chunk of audioBufferRef.current) {
        websocketRef.current.send(chunk);
      }
      audioBufferRef.current = []; // Clear buffer after sending
    } catch (error) {
      console.error('Error sending audio to Deepgram WebSocket:', error);
    }
  }, []);

  const closeWebSocket = useCallback(() => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    if (websocketRef.current) {
      console.log('Closing Deepgram WebSocket connection.');
      websocketRef.current.close();
      websocketRef.current = null;
    }
    isTranscribingRef.current = false;
    audioBufferRef.current = []; // Clear buffer on close
  }, []);

  const initializeWebSocket = useCallback(async (customWebSocketUrl?: string) => {
    if (!enableTranscription) {
      console.log('Transcription disabled.');
      closeWebSocket();
      return;
    }

    let targetWsUrl = '';
    let headers: { [key: string]: string } = {};

    if (customWebSocketUrl && customWebSocketUrl.trim() !== '') {
      console.log('Using custom WebSocket URL for transcription:', customWebSocketUrl);
      targetWsUrl = customWebSocketUrl.trim();
      // Custom backend handles its own auth; no Deepgram token header here.
    } else if (deepgramApiKey) {
      console.log('Using Deepgram WebSocket for transcription.');
      const params = new URLSearchParams({
        sample_rate: '16000',
        encoding: 'opus',
        channels: '1',
        model: 'nova-2',
        language: 'en-US',
        smart_format: 'true',
        interim_results: 'false',
        punctuate: 'true',
        diarize: 'true'
      });
      targetWsUrl = `wss://api.deepgram.com/v1/listen?${params.toString()}`;
      headers['Authorization'] = `Token ${deepgramApiKey}`;
    } else {
      console.log('Transcription enabled, but no API key for Deepgram and no custom WebSocket URL provided.');
      Alert.alert("Configuration Missing", "Please provide a Deepgram API Key or a custom WebSocket URL for transcription.");
      closeWebSocket();
      return;
    }

    const netState = await NetInfo.fetch();
    if (!netState.isConnected || !netState.isInternetReachable) {
      Alert.alert("No Internet", "Please check your internet connection to use transcription.");
      return;
    }

    console.log('Initializing Deepgram WebSocket transcription...');
    closeWebSocket(); // Close any existing connection first

    try {
      // Use React Native's WebSocket constructor with custom headers
      // @ts-ignore - React Native WebSocket may support headers as a third argument
      const ws = new WebSocket(targetWsUrl, [], {
        headers: Object.keys(headers).length > 0 ? headers : undefined
      });

      ws.onopen = () => {
        console.log('Deepgram WebSocket connection established.');
        isTranscribingRef.current = true;
        // Start processing interval to send accumulated audio
        if (processingIntervalRef.current) clearInterval(processingIntervalRef.current);
        processingIntervalRef.current = setInterval(() => {
          if (audioBufferRef.current.length > 0 && isTranscribingRef.current) {
            sendAudioToWebSocket();
          }
        }, 250); // Send audio every 250ms
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string);
          console.log("Deepgram transcript received:", data);

          if (data.channel?.alternatives?.[0]?.transcript) {
            const transcriptText = data.channel.alternatives[0].transcript.trim();
            if (transcriptText) {
              setTranscription((prev) => {
                const lines = prev ? prev.split('\n') : [];
                if (lines.length >= MAX_TRANSCRIPTION_LINES) {
                  lines.shift();
                }
                const now = new Date();
                const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
                const speakerInfo = data.channel.alternatives[0].words?.[0]?.speaker
                  ? `[Speaker ${data.channel.alternatives[0].words[0].speaker}]`
                  : '';
                lines.push(`[${timestamp}] ${speakerInfo} ${transcriptText}`);
                return lines.join('\n');
              });
            }
          }
        } catch (parseError) {
          console.error('Error parsing WebSocket message:', parseError);
        }
      };

      ws.onerror = (error) => {
        console.error('Deepgram WebSocket error:', error);
        Alert.alert("Transcription Error", "WebSocket connection error. Please check your API key and internet connection.");
        isTranscribingRef.current = false; // Ensure transcribing is set to false
      };

      ws.onclose = (event) => {
        console.log('Deepgram WebSocket connection closed. Code:', event.code, 'Reason:', event.reason);
        isTranscribingRef.current = false;
        if (processingIntervalRef.current) {
            clearInterval(processingIntervalRef.current);
            processingIntervalRef.current = null;
        }
        // Optionally attempt to re-initialize if closed unexpectedly and transcription is still enabled
        // if (enableTranscription && !event.wasClean) { initializeWebSocket(); }
      };

      websocketRef.current = ws;
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      Alert.alert("WebSocket Error", "Could not establish transcription service.");
      isTranscribingRef.current = false;
    }
  }, [deepgramApiKey, closeWebSocket, sendAudioToWebSocket, enableTranscription]);

  const setEnableTranscription = useCallback((value: boolean) => {
    setEnableTranscriptionState(value);
    if (!value) {
      closeWebSocket();
      clearTranscription(); // Clear transcription text when disabled
    } else {
        // Auto-initialize if API key exists or custom URL might be set later by UI
        // We won't auto-init here anymore; let user click "Start Transcription"
        // which will call initializeWebSocket with the appropriate URL from App.tsx state.
    }
  }, [closeWebSocket, clearTranscription]);
  
  // Removed the useEffect that attempted to auto-initialize WebSocket on API key change,
  // as initialization is now explicitly driven by UI calls to initializeWebSocket
  // with the appropriate URL (custom or implying Deepgram).

  const processAudioForTranscription = useCallback((audioBytes: Uint8Array) => {
    if (isTranscribingRef.current && websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN && audioBytes.length > 0) {
      audioBufferRef.current.push(audioBytes);
    }
  }, []);

  return {
    enableTranscription,
    setEnableTranscription,
    deepgramApiKey,
    setDeepgramApiKey,
    transcription,
    isTranscribing: isTranscribingRef.current, // Return the ref's current value
    initializeWebSocket,
    closeWebSocket,
    processAudioForTranscription,
    clearTranscription
  };
}; 