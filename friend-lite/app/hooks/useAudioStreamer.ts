import { useState, useRef, useCallback, useEffect } from 'react';
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
  const manuallyStoppedRef = useRef<boolean>(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const currentUrlRef = useRef<string>('');
  const reconnectAttemptsRef = useRef<number>(0);
  const MAX_RECONNECT_ATTEMPTS = 5;
  const RECONNECT_DELAY = 3000; // 3 seconds between reconnects

  const stopStreaming = useCallback(() => {
    if (websocketRef.current) {
      console.log('[AudioStreamer] Closing WebSocket connection.');
      // Mark that we're manually stopping the connection
      manuallyStoppedRef.current = true;
      
      // Clear any pending reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      websocketRef.current.close();
      websocketRef.current = null;
    }
    setIsStreaming(false);
    setIsConnecting(false);
  }, []);

  const attemptReconnect = useCallback(() => {
    if (manuallyStoppedRef.current || !currentUrlRef.current) {
      console.log('[AudioStreamer] Not reconnecting: connection was manually stopped or no URL available');
      return;
    }

    if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
      console.log(`[AudioStreamer] Maximum reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached`);
      Alert.alert("Connection Failed", "Failed to reconnect to the server after multiple attempts.");
      manuallyStoppedRef.current = true; // Stop trying to reconnect
      return;
    }

    console.log(`[AudioStreamer] Attempting to reconnect (attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
    reconnectAttemptsRef.current += 1;
    setIsConnecting(true);
    
    // Clear any previous reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    // Use the stored URL to reconnect
    reconnectTimeoutRef.current = setTimeout(() => {
      if (!manuallyStoppedRef.current) {
        startStreaming(currentUrlRef.current)
          .catch(error => {
            console.error('[AudioStreamer] Reconnection attempt failed:', error);
            // Schedule next reconnect attempt
            reconnectTimeoutRef.current = setTimeout(attemptReconnect, RECONNECT_DELAY);
          });
      }
    }, RECONNECT_DELAY);
  }, []);

  const startStreaming = useCallback(async (url: string): Promise<void> => {
    if (!url || url.trim() === '') {
      Alert.alert('WebSocket URL Missing', 'Please provide a valid WebSocket URL.');
      const errorMsg = 'WebSocket URL is required.';
      setError(errorMsg);
      return Promise.reject(new Error(errorMsg));
    }

    // Store the URL for reconnection attempts
    currentUrlRef.current = url.trim();
    
    // Reset the manually stopped flag when starting a new connection
    manuallyStoppedRef.current = false;
    
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
          // Reset reconnect attempts on successful connection
          reconnectAttemptsRef.current = 0;
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
              
              // If not manually stopped, try to reconnect
              if (!manuallyStoppedRef.current) {
                console.log('[AudioStreamer] Connection closed unexpectedly. Attempting to reconnect...');
                attemptReconnect();
              }
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
  }, [stopStreaming, attemptReconnect]);

  const sendAudio = useCallback((audioBytes: Uint8Array) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN && audioBytes.length > 0) {
      try {
        // console.debug(`[AudioStreamer] Attempting to send audio: ${audioBytes.length} bytes. WebSocket readyState: ${websocketRef.current?.readyState}`);
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopStreaming();
    };
  }, [stopStreaming]);

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