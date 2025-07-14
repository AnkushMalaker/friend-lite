import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  RefreshControl,
  TextInput,
} from 'react-native';
import { ConversationsApi, Conversation, UserInfo, ConversationsResponse } from '../utils/conversationsApi';

interface ConversationsViewProps {
  webSocketUrl: string;
  authToken: string;
  onClose: () => void;
}

export const ConversationsView: React.FC<ConversationsViewProps> = ({
  webSocketUrl,
  authToken,
  onClose,
}) => {
  const [conversations, setConversations] = useState<ConversationsResponse | null>(null);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [allUsers, setAllUsers] = useState<UserInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string>('all');
  const [searchText, setSearchText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const api = new ConversationsApi(webSocketUrl, authToken);

  const loadData = async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);

      // Load user info and conversations in parallel
      const [userInfoResult, conversationsResult] = await Promise.all([
        api.getUserInfo(),
        api.getConversations(),
      ]);

      setUserInfo(userInfoResult);
      setConversations(conversationsResult);

      // If user is admin, load all users for filtering
      if (userInfoResult.is_superuser) {
        try {
          const usersResult = await api.getAllUsers();
          setAllUsers(usersResult);
        } catch (usersError) {
          console.warn('[ConversationsView] Could not load all users:', usersError);
          // Non-critical error, continue without user filtering
        }
      }
    } catch (err) {
      console.error('[ConversationsView] Error loading data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load conversations');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return 'Unknown';
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes}m ${seconds}s`;
  };

  const getTranscriptPreview = (transcript: any[]) => {
    if (!transcript || transcript.length === 0) return 'No transcript available';
    
    const preview = transcript
      .slice(0, 3)
      .map(segment => segment.text || '')
      .join(' ')
      .trim();
    
    return preview.length > 100 ? preview.substring(0, 100) + '...' : preview;
  };

  const handleCloseConversation = async (clientId: string) => {
    Alert.alert(
      'Close Conversation',
      `Are you sure you want to close the active conversation for client ${clientId}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Close',
          style: 'destructive',
          onPress: async () => {
            try {
              await api.closeConversation(clientId);
              Alert.alert('Success', 'Conversation closed successfully');
              loadData(true); // Refresh data
            } catch (err) {
              console.error('[ConversationsView] Error closing conversation:', err);
              Alert.alert('Error', 'Failed to close conversation');
            }
          },
        },
      ]
    );
  };

  const filterConversationsByUser = (conversations: ConversationsResponse) => {
    if (!userInfo?.is_superuser || selectedUser === 'all') {
      return conversations;
    }

    // Filter conversations to show only those from the selected user
    const filteredConversations: { [client_id: string]: Conversation[] } = {};
    
    Object.entries(conversations.conversations).forEach(([clientId, convs]) => {
      // Extract user ID from client ID (format: user_id_suffix-device_name)
      const userIdSuffix = clientId.split('-')[0];
      const selectedUserSuffix = selectedUser.split('-').pop()?.substring(-6);
      
      if (userIdSuffix === selectedUserSuffix) {
        filteredConversations[clientId] = convs;
      }
    });

    return { conversations: filteredConversations };
  };

  const filterConversationsBySearch = (conversations: ConversationsResponse) => {
    if (!searchText.trim()) {
      return conversations;
    }

    const searchLower = searchText.toLowerCase();
    const filteredConversations: { [client_id: string]: Conversation[] } = {};

    Object.entries(conversations.conversations).forEach(([clientId, convs]) => {
      const filteredConvs = convs.filter(conv => {
        const transcriptText = conv.transcript
          .map(segment => segment.text || '')
          .join(' ')
          .toLowerCase();
        
        return (
          clientId.toLowerCase().includes(searchLower) ||
          transcriptText.includes(searchLower) ||
          conv.audio_uuid.toLowerCase().includes(searchLower)
        );
      });

      if (filteredConvs.length > 0) {
        filteredConversations[clientId] = filteredConvs;
      }
    });

    return { conversations: filteredConversations };
  };

  const getDisplayConversations = () => {
    if (!conversations) return null;
    
    let filtered = filterConversationsByUser(conversations);
    filtered = filterConversationsBySearch(filtered);
    
    return filtered;
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Conversations</Text>
          <TouchableOpacity style={styles.closeButton} onPress={onClose}>
            <Text style={styles.closeButtonText}>✕</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading conversations...</Text>
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Conversations</Text>
          <TouchableOpacity style={styles.closeButton} onPress={onClose}>
            <Text style={styles.closeButtonText}>✕</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.centerContent}>
          <Text style={styles.errorText}>Error: {error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={() => loadData()}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const displayConversations = getDisplayConversations();
  const totalConversations = displayConversations 
    ? Object.values(displayConversations.conversations).reduce((sum, convs) => sum + convs.length, 0)
    : 0;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Conversations</Text>
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>✕</Text>
        </TouchableOpacity>
      </View>

      {userInfo && (
        <View style={styles.userInfo}>
          <Text style={styles.userInfoText}>
            Logged in as: {userInfo.email} {userInfo.is_superuser && '(Admin)'}
          </Text>
          <Text style={styles.conversationCount}>
            {totalConversations} conversation{totalConversations !== 1 ? 's' : ''}
          </Text>
        </View>
      )}

      {/* Search */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search conversations..."
          value={searchText}
          onChangeText={setSearchText}
          autoCapitalize="none"
          autoCorrect={false}
        />
      </View>

      {/* Admin User Filter */}
      {userInfo?.is_superuser && allUsers.length > 0 && (
        <View style={styles.filterContainer}>
          <Text style={styles.filterLabel}>Filter by user:</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.userFilterScroll}>
            <TouchableOpacity
              style={[styles.userFilterButton, selectedUser === 'all' && styles.userFilterButtonActive]}
              onPress={() => setSelectedUser('all')}
            >
              <Text style={[styles.userFilterText, selectedUser === 'all' && styles.userFilterTextActive]}>
                All Users
              </Text>
            </TouchableOpacity>
            {allUsers.map(user => (
              <TouchableOpacity
                key={user.id}
                style={[styles.userFilterButton, selectedUser === user.id && styles.userFilterButtonActive]}
                onPress={() => setSelectedUser(user.id)}
              >
                <Text style={[styles.userFilterText, selectedUser === user.id && styles.userFilterTextActive]}>
                  {user.display_name || user.email}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}

      <ScrollView
        style={styles.conversationsList}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => loadData(true)} />
        }
      >
        {displayConversations && Object.keys(displayConversations.conversations).length > 0 ? (
          Object.entries(displayConversations.conversations).map(([clientId, clientConversations]) => (
            <View key={clientId} style={styles.clientSection}>
              <View style={styles.clientHeader}>
                <Text style={styles.clientId}>Client: {clientId}</Text>
                <TouchableOpacity
                  style={styles.closeConversationButton}
                  onPress={() => handleCloseConversation(clientId)}
                >
                  <Text style={styles.closeConversationButtonText}>Close Active</Text>
                </TouchableOpacity>
              </View>

              {clientConversations.map(conversation => (
                <View key={conversation.audio_uuid} style={styles.conversationItem}>
                  <View style={styles.conversationHeader}>
                    <Text style={styles.conversationTime}>
                      {formatTimestamp(conversation.timestamp)}
                    </Text>
                    <Text style={styles.conversationDuration}>
                      {formatDuration(conversation.cropped_duration)}
                    </Text>
                  </View>

                  <Text style={styles.conversationPreview}>
                    {getTranscriptPreview(conversation.transcript)}
                  </Text>

                  <View style={styles.conversationFooter}>
                    <Text style={styles.speakersText}>
                      Speakers: {conversation.speakers_identified.length || 'Unknown'}
                    </Text>
                    <Text style={styles.memoryText}>
                      {conversation.has_memory ? '🧠 Memory created' : '⏳ Processing...'}
                    </Text>
                  </View>

                  <Text style={styles.audioUuid}>ID: {conversation.audio_uuid}</Text>
                </View>
              ))}
            </View>
          ))
        ) : (
          <View style={styles.centerContent}>
            <Text style={styles.noConversationsText}>
              {searchText.trim() || selectedUser !== 'all' 
                ? 'No conversations match your filter'
                : 'No conversations found'}
            </Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: 16,
    color: '#666',
  },
  userInfo: {
    padding: 15,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  userInfoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  conversationCount: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
  },
  searchContainer: {
    padding: 15,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  searchInput: {
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
  },
  filterContainer: {
    padding: 15,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  filterLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  userFilterScroll: {
    flexDirection: 'row',
  },
  userFilterButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#f0f0f0',
    marginRight: 8,
  },
  userFilterButtonActive: {
    backgroundColor: '#007AFF',
  },
  userFilterText: {
    fontSize: 12,
    color: '#666',
  },
  userFilterTextActive: {
    color: 'white',
    fontWeight: '600',
  },
  conversationsList: {
    flex: 1,
  },
  clientSection: {
    marginBottom: 20,
    backgroundColor: 'white',
    borderRadius: 10,
    margin: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  clientHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  clientId: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  closeConversationButton: {
    backgroundColor: '#FF9500',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  closeConversationButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  conversationItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  conversationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  conversationTime: {
    fontSize: 12,
    color: '#666',
  },
  conversationDuration: {
    fontSize: 12,
    color: '#007AFF',
    fontWeight: '600',
  },
  conversationPreview: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 8,
  },
  conversationFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  speakersText: {
    fontSize: 12,
    color: '#666',
  },
  memoryText: {
    fontSize: 12,
    color: '#666',
  },
  audioUuid: {
    fontSize: 10,
    color: '#999',
    fontFamily: 'monospace',
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    fontSize: 16,
    color: '#FF3B30',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  noConversationsText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
});

export default ConversationsView;