import React, { useRef, useCallback, useEffect, useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, ScrollView, Platform, FlatList, ActivityIndicator, Alert, Switch, Button, TouchableOpacity, KeyboardAvoidingView } from 'react-native';
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
} from './utils/storage';
import { useAudioListener } from './hooks/useAudioListener';
import { useAudioStreamer } from './hooks/useAudioStreamer';

// Components
import BluetoothStatusBanner from './components/BluetoothStatusBanner';
import ScanControls from './components/ScanControls';
import DeviceListItem from './components/DeviceListItem';
import DeviceDetails from './components/DeviceDetails';

export default function App() {
  // Initialize OmiConnection
  const omiConnection = useRef(new OmiConnection()).current;

  // Filter state
  const [showOnlyOmi, setShowOnlyOmi] = useState(false);

  // State for remembering the last connected device
  const [lastKnownDeviceId, setLastKnownDeviceId] = useState<string | null>(null);
  const [isAttemptingAutoReconnect, setIsAttemptingAutoReconnect] = useState(false);
  const [triedAutoReconnectForCurrentId, setTriedAutoReconnectForCurrentId] = useState(false);

  // State for WebSocket URL for custom audio streaming
  const [webSocketUrl, setWebSocketUrl] = useState<string>('');
  
  // State for User ID
  const [userId, setUserId] = useState<string>('');
  
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
    omiConnection,
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
    const deviceIdToSave = omiConnection.connectedDeviceId; // Corrected: Use property from OmiConnection instance

    if (deviceIdToSave) {
      console.log('[App.tsx] Saving connected device ID to storage:', deviceIdToSave);
      await saveLastConnectedDeviceId(deviceIdToSave);
      setLastKnownDeviceId(deviceIdToSave); // Update state for consistency
      setTriedAutoReconnectForCurrentId(false); // Reset if a new device connects successfully
    } else {
      console.warn('[App.tsx] onDeviceConnect: Could not determine connected device ID to save. omiConnection.connectedDeviceId was null/undefined.');
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
    omiConnection,
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

      const storedWsUrl = await getWebSocketUrl();
      if (storedWsUrl) {
        console.log('[App.tsx] Loaded WebSocket URL from storage:', storedWsUrl);
        setWebSocketUrl(storedWsUrl);
      }

      const storedUserId = await getUserId();
      if (storedUserId) {
        console.log('[App.tsx] Loaded User ID from storage:', storedUserId);
        setUserId(storedUserId);
      }
    };
    loadSettings();
  }, []);

  // Now that deviceConnection is available, we can define the more specific isAudioReadyToListen
  // for useAudioListener if we were to re-initialize it or if useAudioListener needs it reactively.
  // However, useAudioListener is already initialized. The key is that callbacks passed to hooks are stable.
  // The previous tempIsAudioReadyCb is likely sufficient if it broadly gates on omiConnection.
  // For a more reactive isAudioReadyToListen (if needed by useAudioListener internally beyond init):
  const refinedIsAudioReadyToListen = useCallback(() => {
    return omiConnection.isConnected() && !!deviceConnection.connectedDeviceId;
  }, [omiConnection, deviceConnection.connectedDeviceId]);
  // If useAudioListener needed to react to deviceId changes for its readiness, its internal structure would need to accommodate that.
  // For its initial call, tempIsAudioReadyCb was used. This is a common pattern to break dependency cycles.

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
    if (!webSocketUrl || webSocketUrl.trim() === '') {
      Alert.alert('WebSocket URL Required', 'Please enter the WebSocket URL for streaming.');
      return;
    }
    if (!omiConnection.isConnected() || !deviceConnection.connectedDeviceId) {
      Alert.alert('Device Not Connected', 'Please connect to an OMI device first.');
      return;
    }

    try {
      // Modify WebSocket URL to include user_id if provided
      let finalWebSocketUrl = webSocketUrl;
      if (userId && userId.trim() !== '') {
        const separator = webSocketUrl.includes('?') ? '&' : '?';
        finalWebSocketUrl = `${webSocketUrl}${separator}user_id=${encodeURIComponent(userId.trim())}`;
        console.log('[App.tsx] Using WebSocket URL with user_id:', finalWebSocketUrl);
      } else {
        console.log('[App.tsx] Using WebSocket URL without user_id:', finalWebSocketUrl);
      }

      // Start custom WebSocket streaming first
      await audioStreamer.startStreaming(finalWebSocketUrl);

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
  }, [originalStartAudioListener, audioStreamer, webSocketUrl, userId, omiConnection, deviceConnection.connectedDeviceId]);

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
      if (omiConnection.isConnected()) {
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

  const handleSetAndSaveWebSocketUrl = useCallback(async (url: string) => {
    setWebSocketUrl(url);
    await saveWebSocketUrl(url);
  }, []);

  const handleSetAndSaveUserId = useCallback(async (id: string) => {
    setUserId(id);
    await saveUserId(id || null);
  }, []);

  const fetchUsers = useCallback(async (): Promise<string[]> => {
    if (!webSocketUrl) {
      throw new Error('WebSocket URL not set');
    }
    
    // Convert WebSocket URL to HTTP URL for API call
    const baseUrl = webSocketUrl.replace('ws://', 'http://').replace('wss://', 'https://').split('/ws')[0];
    
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
  }, [webSocketUrl]);

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
              webSocketUrl={webSocketUrl}
              onSetWebSocketUrl={handleSetAndSaveWebSocketUrl}
              isAudioStreaming={audioStreamer.isStreaming}
              isConnectingAudioStreamer={audioStreamer.isConnecting}
              audioStreamerError={audioStreamer.error}
              userId={userId}
              onSetUserId={handleSetAndSaveUserId}
              onFetchUsers={fetchUsers}
            />
          )}
        </ScrollView>
      </KeyboardAvoidingView>
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
