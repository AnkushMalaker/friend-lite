import AsyncStorage from '@react-native-async-storage/async-storage';

const LAST_CONNECTED_DEVICE_ID_KEY = 'LAST_CONNECTED_DEVICE_ID';
const WEBSOCKET_URL_KEY = 'WEBSOCKET_URL_KEY';
const DEEPGRAM_API_KEY_KEY = 'DEEPGRAM_API_KEY_KEY';
const USER_ID_KEY = 'USER_ID_KEY';

export const saveLastConnectedDeviceId = async (deviceId: string | null): Promise<void> => {
  try {
    if (deviceId) {
      await AsyncStorage.setItem(LAST_CONNECTED_DEVICE_ID_KEY, deviceId);
      console.log('[Storage] Last connected device ID saved:', deviceId);
    } else {
      await AsyncStorage.removeItem(LAST_CONNECTED_DEVICE_ID_KEY);
      console.log('[Storage] Last connected device ID removed.');
    }
  } catch (error) {
    console.error('[Storage] Error saving last connected device ID:', error);
  }
};

export const getLastConnectedDeviceId = async (): Promise<string | null> => {
  try {
    const deviceId = await AsyncStorage.getItem(LAST_CONNECTED_DEVICE_ID_KEY);
    console.log('[Storage] Raw value from AsyncStorage.getItem for device ID:', deviceId === null ? "null" : `"${deviceId}"`);
    return deviceId;
  } catch (error) {
    console.error('[Storage] Error retrieving last connected device ID:', error);
    return null;
  }
};

// WebSocket URL
export const saveWebSocketUrl = async (url: string | null): Promise<void> => {
  try {
    if (url) {
      await AsyncStorage.setItem(WEBSOCKET_URL_KEY, url);
      console.log('[Storage] WebSocket URL saved:', url);
    } else {
      await AsyncStorage.removeItem(WEBSOCKET_URL_KEY);
      console.log('[Storage] WebSocket URL removed.');
    }
  } catch (error) {
    console.error('[Storage] Error saving WebSocket URL:', error);
  }
};

export const getWebSocketUrl = async (): Promise<string | null> => {
  try {
    const url = await AsyncStorage.getItem(WEBSOCKET_URL_KEY);
    console.log('[Storage] Retrieved WebSocket URL:', url);
    return url;
  } catch (error) {
    console.error('[Storage] Error retrieving WebSocket URL:', error);
    return null;
  }
};

// Deepgram API Key
export const saveDeepgramApiKey = async (apiKey: string | null): Promise<void> => {
  try {
    if (apiKey) {
      await AsyncStorage.setItem(DEEPGRAM_API_KEY_KEY, apiKey);
      console.log('[Storage] Deepgram API Key saved.'); // Don't log the key itself for security
    } else {
      await AsyncStorage.removeItem(DEEPGRAM_API_KEY_KEY);
      console.log('[Storage] Deepgram API Key removed.');
    }
  } catch (error) {
    console.error('[Storage] Error saving Deepgram API Key:', error);
  }
};

export const getDeepgramApiKey = async (): Promise<string | null> => {
  try {
    const apiKey = await AsyncStorage.getItem(DEEPGRAM_API_KEY_KEY);
    if (apiKey) {
        console.log('[Storage] Retrieved Deepgram API Key.');
    }
    return apiKey;
  } catch (error) {
    console.error('[Storage] Error retrieving Deepgram API Key:', error);
    return null;
  }
};

// User ID
export const saveUserId = async (userId: string | null): Promise<void> => {
  try {
    if (userId) {
      await AsyncStorage.setItem(USER_ID_KEY, userId);
      console.log('[Storage] User ID saved:', userId);
    } else {
      await AsyncStorage.removeItem(USER_ID_KEY);
      console.log('[Storage] User ID removed.');
    }
  } catch (error) {
    console.error('[Storage] Error saving User ID:', error);
  }
};

export const getUserId = async (): Promise<string | null> => {
  try {
    const userId = await AsyncStorage.getItem(USER_ID_KEY);
    console.log('[Storage] Retrieved User ID:', userId);
    return userId;
  } catch (error) {
    console.error('[Storage] Error retrieving User ID:', error);
    return null;
  }
}; 