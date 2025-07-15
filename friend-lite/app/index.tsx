import React, { useRef, useCallback, useEffect, useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, ScrollView, Platform, FlatList, ActivityIndicator, Alert, Switch, Button, TouchableOpacity, KeyboardAvoidingView, Modal } from 'react-native';
import { OmiConnection } from 'friend-lite-react-native'; // OmiDevice also comes from here
import { State as BluetoothState } from 'react-native-ble-plx'; // Import State from ble-plx

// Hooks
import { useBluetoothManager } from './hooks/useBluetoothManager';
import { useDeviceScanning } from './hooks/useDeviceScanning';
import { useDeviceConnection } from './hooks/useDeviceConnection';
import {
  saveLastConnectedDeviceId,
  getLastConnectedDeviceId,
  saveWebSocketUrl,
  getWebSocketUrl,
  saveUserId,
  getUserId,
  saveAuthUsername,
  getAuthUsername,
  saveAuthPassword,
  getAuthPassword,
  saveAuthToken,
  getAuthToken,
  clearAuthData,
} from './utils/storage';
import { useAudioListener } from './hooks/useAudioListener';
import { useAudioStreamer } from './hooks/useAudioStreamer';

// Components
import BluetoothStatusBanner from './components/BluetoothStatusBanner';
import ScanControls from './components/ScanControls';
import DeviceListItem from './components/DeviceListItem';
import DeviceDetails from './components/DeviceDetails';
import ConversationsView from './components/ConversationsView';
import BackendConfiguration from './components/BackendConfiguration';
import AudioModeSelector from './components/AudioModeSelector';
import PhoneAudioControls from './components/PhoneAudioControls';

export default function App() {
  // Initialize OmiConnection with error handling
  const omiConnection = useRef<any>(null);
  
  if (!omiConnection.current) {
    try {
      omiConnection.current = new OmiConnection();
    } catch (error) {
      console.error('[App.tsx] Error initializing OmiConnection:', error);
      // Return a mock object to prevent crashes
      omiConnection.current = {
        isConnected: () => false,
        connectedDeviceId: null,
        connect: () => Promise.reject(new Error('OmiConnection failed to initialize')),
        disconnect: () => Promise.resolve(),
      };
    }
  }

  // Filter state
  const [showOnlyOmi, setShowOnlyOmi] = useState(false);

  // State for remembering the last connected device
  const [lastKnownDeviceId, setLastKnownDeviceId] = useState<string | null>(null);
  const [isAttemptingAutoReconnect, setIsAttemptingAutoReconnect] = useState(false);
  const [triedAutoReconnectForCurrentId, setTriedAutoReconnectForCurrentId] = useState(false);

  // State for Backend URL for custom audio streaming
  const [backendUrl, setBackendUrl] = useState<string>('http://100.99.62.5:8000');
  
  // State for User ID
  const [userId, setUserId] = useState<string>('');
  
  // State for Authentication
  const [authUsername, setAuthUsername] = useState<string>('admin@example.com');
  const [authPassword, setAuthPassword] = useState<string>('Getaway-Unplowed-Flaxseed6-Roundup');
  const [authToken, setAuthToken] = useState<string>('');
  
  // State for Conversations View
  const [showConversations, setShowConversations] = useState<boolean>(false);
  
  // State for Phone Audio Mode
  const [phoneAudioMode, setPhoneAudioMode] = useState<boolean>(false);
  
  // Bluetooth Management Hook
  const {
    bleManager,
    bluetoothState,
    permissionGranted,
    requestBluetoothPermission,
    isPermissionsLoading,
  } = useBluetoothManager();

  // Custom Audio Streamer Hook
  const audioStreamer = useAudioStreamer();


  const {
    isListeningAudio: isOmiAudioListenerActive,
    audioPacketsReceived,
    startAudioListener: originalStartAudioListener,
    stopAudioListener: originalStopAudioListener,
  } = useAudioListener(
    omiConnection.current,
    () => !!deviceConnection.connectedDeviceId
  );

  // Refs to hold the current state for onDeviceDisconnect without causing re-memoization
  const isOmiAudioListenerActiveRef = useRef(isOmiAudioListenerActive);
  const isAudioStreamingRef = useRef(audioStreamer.isStreaming);

  useEffect(() => {
    isOmiAudioListenerActiveRef.current = isOmiAudioListenerActive;
  }, [isOmiAudioListenerActive]);

  useEffect(() => {
    isAudioStreamingRef.current = audioStreamer.isStreaming;
  }, [audioStreamer.isStreaming]);

  // Now define the stable onDeviceConnect and onDeviceDisconnect callbacks
  const onDeviceConnect = useCallback(async () => {
    console.log('[App.tsx] Device connected callback.');
    const deviceIdToSave = omiConnection.current.connectedDeviceId; // Corrected: Use property from OmiConnection instance

    if (deviceIdToSave) {
      console.log('[App.tsx] Saving connected device ID to storage:', deviceIdToSave);
      await saveLastConnectedDeviceId(deviceIdToSave);
      setLastKnownDeviceId(deviceIdToSave); // Update state for consistency
      setTriedAutoReconnectForCurrentId(false); // Reset if a new device connects successfully
    } else {
      console.warn('[App.tsx] onDeviceConnect: Could not determine connected device ID to save. omiConnection.current.connectedDeviceId was null/undefined.');
    }
    // Actions on connect (e.g., auto-fetch codec/battery)
  }, [omiConnection]); // saveLastConnectedDeviceId is stable, omiConnection is stable ref

  const onDeviceDisconnect = useCallback(async () => {
    console.log('[App.tsx] Device disconnected callback.');
    if (isOmiAudioListenerActiveRef.current) {
      console.log('[App.tsx] Disconnect: Stopping audio listener.');
      await originalStopAudioListener();
    }
    if (isAudioStreamingRef.current) {
      console.log('[App.tsx] Disconnect: Stopping custom audio streaming.');
      audioStreamer.stopStreaming();
    }
  }, [originalStopAudioListener, audioStreamer.stopStreaming]);

  // Initialize Device Connection hook, passing the memoized callbacks
  const deviceConnection = useDeviceConnection(
    omiConnection.current,
    onDeviceDisconnect,
    onDeviceConnect
  );

  // Effect to load settings on app startup
  useEffect(() => {
    const loadSettings = async () => {
      const deviceId = await getLastConnectedDeviceId();
      if (deviceId) {
        console.log('[App.tsx] Loaded last known device ID from storage:', deviceId);
        setLastKnownDeviceId(deviceId);
        setTriedAutoReconnectForCurrentId(false);
      } else {
        console.log('[App.tsx] No last known device ID found in storage. Auto-reconnect will not be attempted.');
        setLastKnownDeviceId(null); // Explicitly ensure it's null
        setTriedAutoReconnectForCurrentId(true); // Mark that we shouldn't try (as no ID is known)
      }

      const storedBackendUrl = await getWebSocketUrl(); // Keep same storage key for compatibility
      if (storedBackendUrl) {
        console.log('[App.tsx] Loaded Backend URL from storage:', storedBackendUrl);
        setBackendUrl(storedBackendUrl);
      }

      const storedUserId = await getUserId();
      if (storedUserId) {
        console.log('[App.tsx] Loaded User ID from storage:', storedUserId);
        setUserId(storedUserId);
      }

      const storedAuthUsername = await getAuthUsername();
      if (storedAuthUsername) {
        console.log('[App.tsx] Loaded auth username from storage:', storedAuthUsername);
        setAuthUsername(storedAuthUsername);
      }

      const storedAuthPassword = await getAuthPassword();
      if (storedAuthPassword) {
        console.log('[App.tsx] Loaded auth password from storage.');
        setAuthPassword(storedAuthPassword);
      }

      const storedAuthToken = await getAuthToken();
      if (storedAuthToken) {
        console.log('[App.tsx] Loaded auth token from storage.');
        setAuthToken(storedAuthToken);
      }
    };
    loadSettings();
  }, []);

  // Device Connection hook is now available for audio readiness checks

  // Device Scanning Hook
  const {
    devices: scannedDevices,
    scanning,
    startScan,
    stopScan: stopDeviceScanAction,
  } = useDeviceScanning(
    bleManager, // From useBluetoothManager
    omiConnection,
    permissionGranted, // From useBluetoothManager
    bluetoothState === BluetoothState.PoweredOn, // Derived from useBluetoothManager
    requestBluetoothPermission // From useBluetoothManager, should be stable
  );
  
  // Effect for attempting auto-reconnection
  useEffect(() => {
    if (
      bluetoothState === BluetoothState.PoweredOn &&
      permissionGranted &&
      lastKnownDeviceId &&
      !deviceConnection.connectedDeviceId && // Only if not already connected
      !deviceConnection.isConnecting &&        // Only if not currently trying to connect by other means
      !scanning &&                             // Only if not currently scanning
      !isAttemptingAutoReconnect &&            // Only if not already attempting auto-reconnect
      !triedAutoReconnectForCurrentId          // Only try once per loaded/set lastKnownDeviceId
    ) {
      const attemptAutoConnect = async () => {
        console.log(`[App.tsx] Attempting to auto-reconnect to device: ${lastKnownDeviceId}`);
        setIsAttemptingAutoReconnect(true);
        setTriedAutoReconnectForCurrentId(true); // Mark that we've initiated an attempt for this ID
        try {
          // useDeviceConnection.connectToDevice can take a device ID string directly
          await deviceConnection.connectToDevice(lastKnownDeviceId);
          // If connectToDevice throws, catch block handles it.
          // If it resolves, the connection attempt was made.
          // The onDeviceConnect callback will be triggered if successful.
          console.log(`[App.tsx] Auto-reconnect attempt initiated for ${lastKnownDeviceId}. Waiting for connection event.`);
          // Removed the if(success) block as connectToDevice is void
        } catch (error) {
          console.error(`[App.tsx] Error auto-reconnecting to ${lastKnownDeviceId}:`, error);
          // Clear the problematic device ID from storage and state
          if (lastKnownDeviceId) { // Ensure we have an ID to clear
            console.log(`[App.tsx] Clearing problematic device ID ${lastKnownDeviceId} from storage due to auto-reconnect failure.`);
            await saveLastConnectedDeviceId(null); // Clears from AsyncStorage
            setLastKnownDeviceId(null); // Clears from current app state
          }
        } finally {
          setIsAttemptingAutoReconnect(false);
        }
      };
      attemptAutoConnect();
    }
  }, [
    bluetoothState,
    permissionGranted,
    lastKnownDeviceId,
    deviceConnection.connectedDeviceId,
    deviceConnection.isConnecting,
    scanning,
    deviceConnection.connectToDevice, // Stable function from the hook
    triedAutoReconnectForCurrentId,
    isAttemptingAutoReconnect, // Added to prevent re-triggering while one is in progress
    // Added saveLastConnectedDeviceId and setLastKnownDeviceId to dependency array if they were not already implicitly covered
    // saveLastConnectedDeviceId is an import, setLastKnownDeviceId is a state setter - typically stable
  ]);

  const handleStartAudioListeningAndStreaming = useCallback(async () => {
    if (!backendUrl || backendUrl.trim() === '') {
      Alert.alert('Backend URL Required', 'Please enter the Backend URL for streaming.');
      return;
    }
    if (!omiConnection.current.isConnected() || !deviceConnection.connectedDeviceId) {
      Alert.alert('Device Not Connected', 'Please connect to an OMI device first.');
      return;
    }

    try {
      // Build WebSocket URL for OMI device (use /ws_omi endpoint)
      let wsUrl = backendUrl;
      if (wsUrl.startsWith('http://')) {
        wsUrl = wsUrl.replace('http://', 'ws://');
      } else if (wsUrl.startsWith('https://')) {
        wsUrl = wsUrl.replace('https://', 'wss://');
      } else if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
        wsUrl = 'ws://' + wsUrl;
      }
      
      // Remove any existing WebSocket paths and add /ws_omi for OMI device (Opus audio)
      wsUrl = wsUrl.replace(/\/(ws_pcm|ws_omi)($|\?.*$)/, '');
      wsUrl += '/ws_omi';
      
      // Add user_id if provided
      if (userId && userId.trim() !== '') {
        wsUrl += `?user_id=${encodeURIComponent(userId.trim())}`;
        console.log('[App.tsx] Using WebSocket URL with user_id:', wsUrl);
      } else {
        console.log('[App.tsx] Using WebSocket URL without user_id:', wsUrl);
      }

      // Start custom WebSocket streaming first
      await audioStreamer.startStreaming(wsUrl);

      // Then start OMI audio listener
      await originalStartAudioListener((audioBytes) => {
        const wsReadyState = audioStreamer.getWebSocketReadyState();
        if (wsReadyState === WebSocket.OPEN && audioBytes.length > 0) {
          audioStreamer.sendAudio(audioBytes);
        }
      });
    } catch (error) {
      console.error('[App.tsx] Error starting audio listening/streaming:', error);
      Alert.alert('Error', 'Could not start audio listening or streaming.');
      // Ensure cleanup if one part started but the other failed
      if (audioStreamer.isStreaming) audioStreamer.stopStreaming();
    }
  }, [originalStartAudioListener, audioStreamer, backendUrl, userId, omiConnection, deviceConnection.connectedDeviceId]);

  const handleStopAudioListeningAndStreaming = useCallback(async () => {
    console.log('[App.tsx] Stopping audio listening and streaming.');
    await originalStopAudioListener();
    audioStreamer.stopStreaming();
  }, [originalStopAudioListener, audioStreamer]);

  // Cleanup OmiConnection and BleManager when App unmounts
  useEffect(() => {
    const disconnectFunc = deviceConnection.disconnectFromDevice;
    const currentStopStreaming = audioStreamer.stopStreaming;

    return () => {
      console.log('App unmounting - cleaning up OmiConnection, BleManager, and AudioStreamer');
      if (omiConnection.current.isConnected()) {
        disconnectFunc().catch(err => console.error("Error disconnecting in cleanup:", err));
      }
      if (bleManager) {
        bleManager.destroy();
      }
      currentStopStreaming();
    };
  }, [omiConnection, bleManager, deviceConnection.disconnectFromDevice, audioStreamer.stopStreaming]);

  const canScan = React.useMemo(() => (
    permissionGranted &&
    bluetoothState === BluetoothState.PoweredOn &&
    !isAttemptingAutoReconnect &&
    !deviceConnection.isConnecting &&
    !deviceConnection.connectedDeviceId &&
    (triedAutoReconnectForCurrentId || !lastKnownDeviceId)
  ), [
    permissionGranted,
    bluetoothState,
    isAttemptingAutoReconnect,
    deviceConnection.isConnecting,
    deviceConnection.connectedDeviceId,
    triedAutoReconnectForCurrentId,
    lastKnownDeviceId,
  ]);

  const filteredDevices = React.useMemo(() => {
    if (!showOnlyOmi) {
      return scannedDevices;
    }
    return scannedDevices.filter(device => {
      const name = device.name?.toLowerCase() || '';
      return name.includes('omi') || name.includes('friend');
    });
  }, [scannedDevices, showOnlyOmi]);

  const handleSetAndSaveBackendUrl = useCallback(async (url: string) => {
    setBackendUrl(url);
    await saveWebSocketUrl(url); // Keep same storage key for compatibility
  }, []);

  const handleSetAndSaveUserId = useCallback(async (id: string) => {
    setUserId(id);
    await saveUserId(id || null);
  }, []);

  const handleSetAndSaveAuthUsername = useCallback(async (username: string) => {
    setAuthUsername(username);
    await saveAuthUsername(username || null);
  }, []);

  const handleSetAndSaveAuthPassword = useCallback(async (password: string) => {
    setAuthPassword(password);
    await saveAuthPassword(password || null);
  }, []);

  const handleSetAndSaveAuthToken = useCallback(async (token: string) => {
    setAuthToken(token);
    await saveAuthToken(token || null);
  }, []);

  const handleClearAuthData = useCallback(async () => {
    setAuthUsername('');
    setAuthPassword('');
    setAuthToken('');
    await clearAuthData();
    console.log('[App.tsx] Authentication data cleared.');
  }, []);

  const handleTestAuth = useCallback(async (): Promise<boolean> => {
    if (!backendUrl) {
      throw new Error('Backend URL not set');
    }

    // Ensure baseUrl is HTTP/HTTPS format
    let baseUrl = backendUrl;
    if (baseUrl.startsWith('ws://')) {
      baseUrl = baseUrl.replace('ws://', 'http://');
    } else if (baseUrl.startsWith('wss://')) {
      baseUrl = baseUrl.replace('wss://', 'https://');
    } else if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
      // Assume http if no protocol specified
      baseUrl = 'http://' + baseUrl;
    }
    // Remove any trailing slash and WebSocket paths (but preserve port numbers)
    baseUrl = baseUrl.replace(/\/(ws_pcm|ws_omi)($|\?.*$)/, '').replace(/\/$/, '');
    
    try {
      console.log(`[App.tsx] Testing authentication against: ${baseUrl}`);
      
      // If we already have a token, test it first
      if (authToken && authToken.trim() !== '') {
        console.log('[App.tsx] Testing existing token...');
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
          
          const response = await fetch(`${baseUrl}/api/users`, {
            headers: {
              'Authorization': `Bearer ${authToken.trim()}`,
            },
            signal: controller.signal,
          });
          
          clearTimeout(timeoutId);
          
          if (response.ok) {
            console.log('[App.tsx] Token authentication successful.');
            return true;
          } else {
            console.log(`[App.tsx] Token authentication failed with status: ${response.status}`);
            // Clear the invalid token
            await handleSetAndSaveAuthToken('');
          }
        } catch (tokenError) {
          console.log('[App.tsx] Token test failed with error:', tokenError);
          // Clear the problematic token
          await handleSetAndSaveAuthToken('');
        }
      }

      // If no token or token failed, try username/password authentication
      if (authUsername && authPassword) {
        console.log(`[App.tsx] Attempting username/password authentication to: ${baseUrl}/auth/jwt/login`);
        // Use URL-encoded string for body, not FormData
        const body = `username=${encodeURIComponent(authUsername)}&password=${encodeURIComponent(authPassword)}`;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
        
        const response = await fetch(`${baseUrl}/auth/jwt/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        console.log(`[App.tsx] Auth response status: ${response.status}`);
        
        if (response.ok) {
          const result = await response.json();
          const token = result.access_token;
          if (token) {
            await handleSetAndSaveAuthToken(token);
            console.log('[App.tsx] Username/password authentication successful, token saved.');
            return true;
          }
        } else {
          const errorText = await response.text();
          console.log(`[App.tsx] Auth failed with status ${response.status}: ${errorText}`);
        }
      }

      console.log('[App.tsx] Authentication failed.');
      return false;
    } catch (error) {
      console.error('[App.tsx] Error testing authentication:', error);
      throw error;
    }
  }, [backendUrl, authUsername, authPassword, authToken, handleSetAndSaveAuthToken]);

  const handleShowConversations = useCallback(() => {
    if (!authToken) {
      Alert.alert('Authentication Required', 'Please authenticate first to view conversations.');
      return;
    }
    setShowConversations(true);
  }, [authToken]);

  const handleCloseConversations = useCallback(() => {
    setShowConversations(false);
  }, []);

  const fetchUsers = useCallback(async (): Promise<string[]> => {
    if (!backendUrl) {
      throw new Error('Backend URL not set');
    }
    
    // Ensure baseUrl is HTTP/HTTPS format
    let baseUrl = backendUrl;
    if (baseUrl.startsWith('ws://')) {
      baseUrl = baseUrl.replace('ws://', 'http://');
    } else if (baseUrl.startsWith('wss://')) {
      baseUrl = baseUrl.replace('wss://', 'https://');
    } else if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
      baseUrl = 'http://' + baseUrl;
    }
    baseUrl = baseUrl.replace(/\/(ws_pcm|ws_omi)($|\?.*$)/, '').replace(/\/$/, '');
    
    try {
      const response = await fetch(`${baseUrl}/api/users`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const users = await response.json();
      return users.map((user: any) => user.user_id);
    } catch (error) {
      console.error('[App.tsx] Error fetching users:', error);
      throw error;
    }
  }, [backendUrl]);

  const handleCancelAutoReconnect = useCallback(async () => {
    console.log('[App.tsx] Cancelling auto-reconnection attempt.');
    if (lastKnownDeviceId) {
      // Clear the last known device ID to prevent further auto-reconnect attempts in this session
      await saveLastConnectedDeviceId(null);
      setLastKnownDeviceId(null);
      setTriedAutoReconnectForCurrentId(true); // Mark as tried to prevent immediate re-trigger if conditions meet again
    }
    // Attempt to stop any ongoing connection process
    // disconnectFromDevice also sets isConnecting to false internally.
    await deviceConnection.disconnectFromDevice(); 
    setIsAttemptingAutoReconnect(false); // Explicitly set to false to hide the auto-reconnect screen
  }, [deviceConnection, lastKnownDeviceId, saveLastConnectedDeviceId, setLastKnownDeviceId, setTriedAutoReconnectForCurrentId, setIsAttemptingAutoReconnect]);

  if (isPermissionsLoading && bluetoothState === BluetoothState.Unknown) {
    return (
      <View style={styles.centeredMessageContainer}>
        <ActivityIndicator size="large" />
        <Text style={styles.centeredMessageText}>
          {isAttemptingAutoReconnect 
            ? `Attempting to reconnect to the last device (${lastKnownDeviceId ? lastKnownDeviceId.substring(0, 10) + '...' : ''})...` 
            : 'Initializing Bluetooth...'}
        </Text>
      </View>
    );
  }

  if (isAttemptingAutoReconnect) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centeredMessageContainer}>
          <ActivityIndicator size="large" />
          <Text style={styles.centeredMessageText}>
            Attempting to reconnect to the last device ({lastKnownDeviceId ? lastKnownDeviceId.substring(0, 10) + '...' : ''})...
          </Text>
          <Button title="Cancel" onPress={handleCancelAutoReconnect} color="#FF6347" />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}
      >
        <ScrollView 
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={styles.title}>Friend Lite</Text>

          {/* Backend Configuration - Always Visible */}
          <BackendConfiguration
            backendUrl={backendUrl}
            onSetBackendUrl={handleSetAndSaveBackendUrl}
            authUsername={authUsername}
            authPassword={authPassword}
            authToken={authToken}
            onSetAuthUsername={handleSetAndSaveAuthUsername}
            onSetAuthPassword={handleSetAndSaveAuthPassword}
            onSetAuthToken={handleSetAndSaveAuthToken}
            onClearAuthData={handleClearAuthData}
            onTestAuth={handleTestAuth}
            onShowConversations={handleShowConversations}
          />

          {/* Audio Source Selection */}
          <AudioModeSelector
            phoneAudioMode={phoneAudioMode}
            onModeChange={setPhoneAudioMode}
            disabled={deviceConnection.connectedDeviceId !== null || scanning || isAttemptingAutoReconnect}
          />

          {phoneAudioMode ? (
            /* Phone Audio Recording */
            <PhoneAudioControls
              backendUrl={backendUrl}
              authToken={authToken}
              userId={userId}
              disabled={false}
            />
          ) : (
            /* OMI Device Flow */
            <>
              <BluetoothStatusBanner
                bluetoothState={bluetoothState}
                isPermissionsLoading={isPermissionsLoading}
                permissionGranted={permissionGranted}
                onRequestPermission={requestBluetoothPermission}
              />

              <ScanControls
                scanning={scanning}
                onScanPress={startScan}
                onStopScanPress={stopDeviceScanAction}
                canScan={canScan}
              />

              {scannedDevices.length > 0 && !deviceConnection.connectedDeviceId && !isAttemptingAutoReconnect && (
                <View style={styles.section}>
              <View style={styles.sectionHeaderWithFilter}>
                <Text style={styles.sectionTitle}>Found Devices</Text>
                <View style={styles.filterContainer}>
                  <Text style={styles.filterText}>Show only OMI/Friend</Text>
                  <Switch
                    trackColor={{ false: "#767577", true: "#81b0ff" }}
                    thumbColor={showOnlyOmi ? "#f5dd4b" : "#f4f3f4"}
                    ios_backgroundColor="#3e3e3e"
                    onValueChange={setShowOnlyOmi}
                    value={showOnlyOmi}
                  />
                </View>
              </View>
              {filteredDevices.length > 0 ? (
                <FlatList
                  data={filteredDevices}
                  renderItem={({ item }) => (
                    <DeviceListItem
                      device={item}
                      onConnect={deviceConnection.connectToDevice}
                      onDisconnect={deviceConnection.disconnectFromDevice}
                      isConnecting={deviceConnection.isConnecting}
                      connectedDeviceId={deviceConnection.connectedDeviceId}
                    />
                  )}
                  keyExtractor={(item) => item.id}
                  style={{ maxHeight: 200 }}
                />
              ) : (
                <View style={styles.noDevicesContainer}>
                  <Text style={styles.noDevicesText}>
                    {showOnlyOmi 
                      ? `No OMI/Friend devices found. ${scannedDevices.length} other device(s) hidden by filter.`
                      : 'No devices found.'
                    }
                  </Text>
                </View>
              )}
            </View>
          )}
          
              {deviceConnection.connectedDeviceId && filteredDevices.find(d => d.id === deviceConnection.connectedDeviceId) && (
                   <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Connected Device</Text>
                  <DeviceListItem
                      device={filteredDevices.find(d => d.id === deviceConnection.connectedDeviceId)!}
                      onConnect={() => {}}
                      onDisconnect={async () => {
                        console.log('[App.tsx] Manual disconnect initiated via DeviceListItem.');
                        // Prevent auto-reconnection by clearing the last known device ID *before* disconnecting.
                        await saveLastConnectedDeviceId(null);
                        setLastKnownDeviceId(null); 
                        setTriedAutoReconnectForCurrentId(true); 
                        
                        // TODO: Consider adding setIsDisconnecting(true) here if a visual indicator is needed
                        // and a finally block to set it to false, similar to the old handleDisconnectPress.
                        // For now, focusing on the core logic.

                        try {
                          await deviceConnection.disconnectFromDevice();
                          console.log('[App.tsx] Manual disconnect from device successful.');
                        } catch (error) {
                          console.error('[App.tsx] Error during manual disconnect call:', error);
                          Alert.alert('Error', 'Failed to disconnect from the device.');
                        }
                      }}
                      isConnecting={deviceConnection.isConnecting}
                      connectedDeviceId={deviceConnection.connectedDeviceId}
                  />
              </View>
          )}
          
              {/* Show disconnect button when connected but scan list isn't visible */}
              {deviceConnection.connectedDeviceId && !filteredDevices.find(d => d.id === deviceConnection.connectedDeviceId) && (
                <View style={styles.section}>
              <View style={styles.disconnectContainer}>
                <Text style={styles.connectedText}>
                  Connected to device: {deviceConnection.connectedDeviceId.substring(0, 15)}...
                </Text>
                <TouchableOpacity
                  style={[styles.button, styles.buttonDanger]}
                  onPress={async () => {
                    console.log('[App.tsx] Manual disconnect initiated via standalone disconnect button.');
                    await saveLastConnectedDeviceId(null);
                    setLastKnownDeviceId(null); 
                    setTriedAutoReconnectForCurrentId(true);
                    
                    try {
                      await deviceConnection.disconnectFromDevice();
                      console.log('[App.tsx] Manual disconnect from device successful.');
                    } catch (error) {
                      console.error('[App.tsx] Error during manual disconnect call:', error);
                      Alert.alert('Error', 'Failed to disconnect from the device.');
                    }
                  }}
                  disabled={deviceConnection.isConnecting}
                >
                  <Text style={styles.buttonText}>{deviceConnection.isConnecting ? 'Disconnecting...' : 'Disconnect'}</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

              {deviceConnection.connectedDeviceId && (
                <DeviceDetails
              connectedDeviceId={deviceConnection.connectedDeviceId}
              onGetAudioCodec={deviceConnection.getAudioCodec}
              currentCodec={deviceConnection.currentCodec}
              onGetBatteryLevel={deviceConnection.getBatteryLevel}
              batteryLevel={deviceConnection.batteryLevel}
              isListeningAudio={isOmiAudioListenerActive}
              onStartAudioListener={handleStartAudioListeningAndStreaming}
              onStopAudioListener={handleStopAudioListeningAndStreaming}
              audioPacketsReceived={audioPacketsReceived}
              webSocketUrl={backendUrl}
              onSetWebSocketUrl={handleSetAndSaveBackendUrl}
              isAudioStreaming={audioStreamer.isStreaming}
              isConnectingAudioStreamer={audioStreamer.isConnecting}
              audioStreamerError={audioStreamer.error}
              userId={userId}
              onSetUserId={handleSetAndSaveUserId}
              onFetchUsers={fetchUsers}
              authUsername={authUsername}
              authPassword={authPassword}
              authToken={authToken}
              onSetAuthUsername={handleSetAndSaveAuthUsername}
              onSetAuthPassword={handleSetAndSaveAuthPassword}
              onSetAuthToken={handleSetAndSaveAuthToken}
              onClearAuthData={handleClearAuthData}
              onTestAuth={handleTestAuth}
              onShowConversations={handleShowConversations}
                />
              )}
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Conversations Modal */}
      <Modal
        visible={showConversations}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={handleCloseConversations}
      >
        {authToken ? (
          <ConversationsView
            webSocketUrl={backendUrl}
            authToken={authToken}
            onClose={handleCloseConversations}
          />
        ) : (
          <View style={styles.centeredMessageContainer}>
            <Text style={styles.centeredMessageText}>Authentication required</Text>
            <Button title="Close" onPress={handleCloseConversations} />
          </View>
        )}
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 20,
    paddingTop: Platform.OS === 'android' ? 30 : 10,
    paddingBottom: 50,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
    textAlign: 'center',
  },
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
  sectionHeaderWithFilter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  filterContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  filterText: {
    marginRight: 8,
    fontSize: 14,
    color: '#333',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  centeredMessageContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  centeredMessageText: {
    marginTop: 10,
    fontSize: 16,
    color: '#555',
    textAlign: 'center',
  },
  disconnectContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 5,
  },
  connectedText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
    marginRight: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDanger: {
    backgroundColor: '#FF3B30',
  },
  buttonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  noDevicesContainer: {
    padding: 20,
    alignItems: 'center',
  },
  noDevicesText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
