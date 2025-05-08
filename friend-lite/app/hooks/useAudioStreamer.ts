import { useState, useRef, useCallback } from 'react';
import { Alert } from 'react-native';
import NetInfo from "@react-native-community/netinfo";

interface UseAudioStreamer {
  isStreaming: boolean;
  isConnecting: boolean;
  error: string | null;
  startStreaming: (url: string) => Promise<void>;
  getWebSocketReadyState: () => number | undefined;
  stopStreaming: () => void;
  sendAudio: (audioBytes: Uint8Array) => void;
}

export const useAudioStreamer = (): UseAudioStreamer => {
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);

  const stopStreaming = useCallback(() => {
    if (websocketRef.current) {
      console.log('[AudioStreamer] Closing WebSocket connection.');
      websocketRef.current.close();
      websocketRef.current = null;
    }
    setIsStreaming(false);
    setIsConnecting(false);
  }, []);

  const startStreaming = useCallback(async (url: string): Promise<void> => {
    if (!url || url.trim() === '') {
      Alert.alert('WebSocket URL Missing', 'Please provide a valid WebSocket URL.');
      const errorMsg = 'WebSocket URL is required.';
      setError(errorMsg);
      return Promise.reject(new Error(errorMsg));
    }

    const netState = await NetInfo.fetch();
    if (!netState.isConnected || !netState.isInternetReachable) {
      Alert.alert("No Internet", "Please check your internet connection to stream audio.");
      const errorMsg = 'No internet connection.';
      setError(errorMsg);
      return Promise.reject(new Error(errorMsg));
    }

    console.log(`[AudioStreamer] Initializing WebSocket connection to: ${url}`);
    if (websocketRef.current) {
      console.log('[AudioStreamer] Found existing WebSocket. Closing it before creating a new one.');
      stopStreaming(); // Close any existing connection
    }

    setIsConnecting(true);
    setError(null);

    return new Promise<void>((resolve, reject) => {
      try {
        const ws = new WebSocket(url.trim());

        ws.onopen = () => {
          console.log('[AudioStreamer] WebSocket connection established.');
          setIsConnecting(false);
          setIsStreaming(true);
          setError(null);
          websocketRef.current = ws; // Assign ref only on successful open
          resolve();
        };

        ws.onmessage = (event) => {
          console.log('[AudioStreamer] Received message:', event.data);
        };

        ws.onerror = (e) => {
          const errorMessage = (e as any).message || 'WebSocket connection error.';
          console.error('[AudioStreamer] WebSocket error:', errorMessage);
          Alert.alert("Streaming Error", `WebSocket error: ${errorMessage}`);
          setError(errorMessage);
          setIsConnecting(false);
          setIsStreaming(false);
          if (websocketRef.current === ws) { // Ensure we only nullify if it's this instance
            websocketRef.current = null;
          }
          reject(new Error(errorMessage));
        };

        ws.onclose = (event) => {
          console.log('[AudioStreamer] WebSocket connection closed. Code:', event.code, 'Reason:', event.reason);
          const wasSuccessfullyOpened = websocketRef.current === ws;

          setIsConnecting(false); // Always ensure connecting is false
          setIsStreaming(false); // Always ensure streaming is false

          if (websocketRef.current === ws) { // If this is the instance that was active
            websocketRef.current = null;
          }

          if (!wasSuccessfullyOpened) {
            // If onopen never fired for this instance, it's a failure of startStreaming.
            const closeErrorMsg = `WebSocket closed before opening. Code: ${event.code}, Reason: ${event.reason || 'Unknown'}`;
            // Only set error if not already set by ws.onerror
            if (error === null) setError(closeErrorMsg);
            reject(new Error(closeErrorMsg));
          } else {
            // If it was open and then closed.
            if (!event.wasClean && error === null) { // And it was not a clean closure and no prior error.
              setError('WebSocket connection closed unexpectedly.');
            }
          }
        };
      } catch (e) {
        const errorMessage = (e as any).message || 'Failed to create WebSocket.';
        console.error('[AudioStreamer] Error creating WebSocket:', errorMessage);
        Alert.alert("WebSocket Error", `Could not establish connection: ${errorMessage}`);
        setError(errorMessage);
        setIsConnecting(false);
        setIsStreaming(false);
        reject(new Error(errorMessage));
      }
    });
  }, [stopStreaming]);

  const sendAudio = useCallback((audioBytes: Uint8Array) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN && audioBytes.length > 0) {
      try {
        console.log(`[AudioStreamer] Attempting to send audio: ${audioBytes.length} bytes. WebSocket readyState: ${websocketRef.current?.readyState}`);
        websocketRef.current.send(audioBytes);
      } catch (e) {
        const errorMessage = (e as any).message || 'Error sending audio data.';
        console.error('[AudioStreamer] Error sending audio:', errorMessage);
        // Optionally set error state or attempt to stop/restart streaming if send fails repeatedly
        setError(errorMessage);
      }
    } else {
      // Log why it didn't send
      console.log(`[AudioStreamer] NOT sending audio. Conditions check: websocketRef.current exists: ${!!websocketRef.current}, readyState === OPEN: ${websocketRef.current?.readyState === WebSocket.OPEN}, audioBytes.length > 0: ${audioBytes.length > 0}. Actual readyState: ${websocketRef.current?.readyState}`);
    }
  }, []);

  const getWebSocketReadyState = useCallback(() => {
    return websocketRef.current?.readyState;
  }, []);

  return {
    isStreaming,
    isConnecting,
    error,
    startStreaming,
    getWebSocketReadyState,
    stopStreaming,
    sendAudio,
  };
}; 