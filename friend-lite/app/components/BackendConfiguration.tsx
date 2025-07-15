import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  Alert,
} from 'react-native';

interface BackendConfigurationProps {
  backendUrl: string;
  onSetBackendUrl: (url: string) => void;
  authUsername: string;
  authPassword: string;
  authToken: string;
  onSetAuthUsername: (username: string) => void;
  onSetAuthPassword: (password: string) => void;
  onSetAuthToken: (token: string) => void;
  onClearAuthData: () => void;
  onTestAuth: () => Promise<boolean>;
  onShowConversations: () => void;
}

export const BackendConfiguration: React.FC<BackendConfigurationProps> = ({
  backendUrl,
  onSetBackendUrl,
  authUsername,
  authPassword,
  authToken,
  onSetAuthUsername,
  onSetAuthPassword,
  onSetAuthToken,
  onClearAuthData,
  onTestAuth,
  onShowConversations,
}) => {
  const [isTestingAuth, setIsTestingAuth] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [urlEditMessage, setUrlEditMessage] = useState<string>('');

  const handleTestAuth = async () => {
    setIsTestingAuth(true);
    try {
      const success = await onTestAuth();
      if (success) {
        Alert.alert('Authentication Test', 'Authentication successful! Token saved.');
      } else {
        Alert.alert('Authentication Test', 'Authentication failed. Please check your credentials.');
      }
    } catch (error) {
      console.error('[BackendConfiguration] Error testing auth:', error);
      Alert.alert('Authentication Test', 'Authentication test failed. Please check your backend connection.');
    } finally {
      setIsTestingAuth(false);
    }
  };

  const handleClearAuth = () => {
    Alert.alert(
      'Clear Authentication',
      'Are you sure you want to clear all authentication data?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: onClearAuthData
        }
      ]
    );
  };

  const handleUrlChange = (url: string) => {
    // Check if WebSocket paths are being cleaned up
    const originalUrl = url;
    let cleanedUrl = url;
    let message = '';

    // Simulate the same URL cleaning logic as in the main app
    if (url.includes('/ws_pcm') || url.includes('/ws_omi')) {
      cleanedUrl = url.replace(/\/(ws_pcm|ws_omi)($|\?.*$)/, '');
      message = '📝 WebSocket paths automatically cleaned from URL';
    }

    onSetBackendUrl(url);
    
    if (message) {
      setUrlEditMessage(message);
      // Clear message after 3 seconds
      setTimeout(() => setUrlEditMessage(''), 3000);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Backend Configuration</Text>
      
      {/* Backend URL */}
      <Text style={styles.inputLabel}>Backend URL:</Text>
      <TextInput
        style={styles.textInput}
        value={backendUrl}
        onChangeText={handleUrlChange}
        placeholder="http://your-server:8000 or https://your-server.com"
        autoCapitalize="none"
        keyboardType="url"
        returnKeyType="done"
        autoCorrect={false}
      />
      {urlEditMessage ? (
        <Text style={styles.urlEditMessage}>{urlEditMessage}</Text>
      ) : null}
      
      {/* Authentication Section */}
      <View style={styles.authSection}>
        <Text style={styles.subSectionTitle}>Authentication</Text>
        
        <Text style={styles.inputLabel}>Username (email):</Text>
        <TextInput
          style={styles.textInput}
          value={authUsername}
          onChangeText={onSetAuthUsername}
          placeholder="your-email@example.com"
          autoCapitalize="none"
          keyboardType="email-address"
          returnKeyType="next"
          autoCorrect={false}
        />
        
        <Text style={styles.inputLabel}>Password:</Text>
        <View style={styles.passwordContainer}>
          <TextInput
            style={[styles.textInput, styles.passwordInput]}
            value={authPassword}
            onChangeText={onSetAuthPassword}
            placeholder="Your password"
            secureTextEntry={!showPassword}
            returnKeyType="done"
            autoCorrect={false}
          />
          <TouchableOpacity
            style={styles.passwordToggle}
            onPress={() => setShowPassword(!showPassword)}
          >
            <Text style={styles.passwordToggleText}>{showPassword ? '🙈' : '👁️'}</Text>
          </TouchableOpacity>
        </View>
        
        <Text style={styles.inputLabel}>JWT Token (optional):</Text>
        <TextInput
          style={[styles.textInput, styles.tokenInput]}
          value={authToken}
          onChangeText={onSetAuthToken}
          placeholder="Paste JWT token here"
          autoCapitalize="none"
          returnKeyType="done"
          autoCorrect={false}
          multiline={true}
          numberOfLines={2}
        />
        
        <View style={styles.authButtonsContainer}>
          <TouchableOpacity
            style={[styles.button, styles.buttonAuth, { flex: 1, marginRight: 10 }]}
            onPress={handleTestAuth}
            disabled={isTestingAuth || ((!authUsername || !authPassword) && !authToken)}
          >
            <Text style={styles.buttonText}>
              {isTestingAuth ? "Testing..." : "Test Auth"}
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.button, styles.buttonDanger, { flex: 1 }]}
            onPress={handleClearAuth}
          >
            <Text style={styles.buttonText}>Clear</Text>
          </TouchableOpacity>
        </View>
        
        {authToken && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>✅ Authenticated</Text>
            <TouchableOpacity
              style={[styles.button, styles.buttonConversations, { marginTop: 10 }]}
              onPress={onShowConversations}
            >
              <Text style={styles.buttonText}>View Conversations</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    margin: 15,
    marginBottom: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#333',
  },
  subSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    color: '#444',
  },
  inputLabel: {
    fontSize: 14,
    color: '#333',
    marginBottom: 5,
    fontWeight: '500',
  },
  textInput: {
    backgroundColor: '#f0f0f0',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 10,
    fontSize: 14,
    width: '100%',
    marginBottom: 10,
    color: '#000',
  },
  authSection: {
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  passwordContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
  },
  passwordInput: {
    flex: 1,
    marginBottom: 0,
  },
  passwordToggle: {
    position: 'absolute',
    right: 10,
    padding: 5,
  },
  passwordToggleText: {
    fontSize: 16,
  },
  tokenInput: {
    minHeight: 60,
    textAlignVertical: 'top',
  },
  authButtonsContainer: {
    flexDirection: 'row',
    marginTop: 15,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    elevation: 2,
  },
  buttonAuth: {
    backgroundColor: '#34C759',
  },
  buttonDanger: {
    backgroundColor: '#FF3B30',
  },
  buttonConversations: {
    backgroundColor: '#5856D6',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  infoContainer: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    alignItems: 'center',
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: '500',
    color: '#34C759',
  },
  urlEditMessage: {
    fontSize: 12,
    color: '#666',
    marginTop: -8,
    marginBottom: 8,
    fontStyle: 'italic',
  },
});

export default BackendConfiguration;