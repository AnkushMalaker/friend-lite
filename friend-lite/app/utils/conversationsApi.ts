export interface ConversationSegment {
  text: string;
  speaker: string;
  start: number;
  end: number;
}

export interface Memory {
  memory_id: string;
  created_at: string;
  status: 'created' | 'processing' | 'completed';
  updated_at: string;
}

export interface Conversation {
  audio_uuid: string;
  audio_path: string;
  timestamp: number;
  transcript: ConversationSegment[];
  speakers_identified: string[];
  cropped_audio_path?: string;
  speech_segments?: Array<{ start: number; end: number }>;
  cropped_duration?: number;
  memories: Memory[];
  has_memory: boolean;
}

export interface ConversationsResponse {
  conversations: {
    [client_id: string]: Conversation[];
  };
}

export interface UserInfo {
  id: string;
  email: string;
  is_superuser: boolean;
  is_active: boolean;
  display_name?: string;
}

export class ConversationsApi {
  private baseUrl: string;
  private token: string;

  constructor(baseUrl: string, token: string) {
    // Convert WebSocket URL to HTTP URL
    this.baseUrl = baseUrl.replace('ws://', 'http://').replace('wss://', 'https://').split('/ws')[0];
    this.token = token;
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    return response.json();
  }

  async getConversations(): Promise<ConversationsResponse> {
    return this.makeRequest<ConversationsResponse>('/api/conversations');
  }

  async getUserInfo(): Promise<UserInfo> {
    return this.makeRequest<UserInfo>('/api/users/me');
  }

  async getAllUsers(): Promise<UserInfo[]> {
    return this.makeRequest<UserInfo[]>('/api/users');
  }

  async closeConversation(clientId: string): Promise<{ message: string; timestamp: number }> {
    return this.makeRequest(`/api/conversations/${clientId}/close`, {
      method: 'POST',
    });
  }

  async getCroppedAudioInfo(audioUuid: string): Promise<any> {
    return this.makeRequest(`/api/conversations/${audioUuid}/cropped`);
  }

  async reprocessAudio(audioUuid: string): Promise<any> {
    return this.makeRequest(`/api/conversations/${audioUuid}/reprocess`, {
      method: 'POST',
    });
  }

  async addSpeaker(audioUuid: string, speakerId: string): Promise<any> {
    return this.makeRequest(`/api/conversations/${audioUuid}/speakers`, {
      method: 'POST',
      body: JSON.stringify({ speaker_id: speakerId }),
    });
  }

  async updateTranscriptSegment(
    audioUuid: string,
    segmentIndex: number,
    updates: {
      speaker_id?: string;
      start_time?: number;
      end_time?: number;
    }
  ): Promise<any> {
    return this.makeRequest(`/api/conversations/${audioUuid}/transcript/${segmentIndex}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }
}