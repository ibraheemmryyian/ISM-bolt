import { supabase } from './supabase';
import { toast } from 'react-toastify';

export interface Conversation {
  id: string;
  participant1: string;
  participant2: string;
  participant1_name: string;
  participant2_name: string;
  last_message?: string;
  last_message_at?: string;
  unread_count: number;
  error?: string;
}

export interface Message {
  id: string;
  content: string;
  sender_id: string;
  receiver_id: string;
  conversation_id: string;
  created_at: string;
  read: boolean;
  error?: string;
}

export interface Connection {
  following_id: string;
  following: {
    id: string;
    name: string;
    company_profiles?: {
      location?: string;
      organization_type?: string;
    }[];
  };
  error?: string;
}

export interface Favorite {
  id: string;
  user_id: string;
  material_id: string;
  material: {
    material_name: string;
    description: string;
    companies: {
      name: string;
    };
  };
  created_at: string;
  error?: string;
}

class MessagingAPIClient {
  private retryAttempts = 3;
  private retryDelay = 1000;

  async request<T>(
    endpoint: string, 
    options: RequestInit = {}, 
    retryCount = 0
  ): Promise<{ success: boolean; data?: T; error?: string }> {
    const url = `/api${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      return {
        success: true,
        data: data.data || data
      };

    } catch (error: any) {
      // Retry logic for network errors
      if (retryCount < this.retryAttempts && this.isRetryableError(error)) {
        await this.delay(this.retryDelay * Math.pow(2, retryCount));
        return this.request<T>(endpoint, options, retryCount + 1);
      }

      return {
        success: false,
        error: this.getUserFriendlyError(error)
      };
    }
  }

  private isRetryableError(error: any): boolean {
    return error.message.includes('Network') || 
           error.message.includes('fetch') ||
           error.message.includes('500') ||
           error.message.includes('502') ||
           error.message.includes('503');
  }

  private getUserFriendlyError(error: any): string {
    if (error.message.includes('Network')) {
      return 'Network connection failed. Please check your internet connection.';
    }
    if (error.message.includes('500')) {
      return 'Server error. Our team has been notified.';
    }
    if (error.message.includes('404')) {
      return 'Service not found. Please contact support.';
    }
    return 'An unexpected error occurred. Please try again.';
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class MessagingService {
  private apiClient = new MessagingAPIClient();

  // Conversation methods
  async getConversations(userId: string): Promise<Conversation[]> {
    try {
      const response = await this.apiClient.request<Conversation[]>(`/conversations/${userId}`);
      
      if (!response.success) {
        toast.error(response.error || 'Failed to load conversations');
        return [];
      }
      
      return response.data || [];
    } catch (error: any) {
      console.error('Error getting conversations:', error);
      toast.error('Failed to load conversations. Please try again.');
      return [];
    }
  }

  async getMessages(conversationId: string): Promise<Message[]> {
    try {
      const response = await this.apiClient.request<Message[]>(`/conversations/${conversationId}/messages`);
      
      if (!response.success) {
        toast.error(response.error || 'Failed to load messages');
        return [];
      }
      
      return response.data || [];
    } catch (error: any) {
      console.error('Error getting messages:', error);
      toast.error('Failed to load messages. Please try again.');
      return [];
    }
  }

  async sendMessage(conversationId: string, senderId: string, content: string): Promise<Message | null> {
    try {
      const response = await this.apiClient.request<Message>('/messages', {
        method: 'POST',
        body: JSON.stringify({
          conversationId,
          senderId,
          content,
        }),
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to send message');
        return null;
      }
      
      toast.success('Message sent successfully');
      return response.data || null;
    } catch (error: any) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
      return null;
    }
  }

  async createConversation(participantIds: string[]): Promise<{ id: string } | null> {
    try {
      const response = await this.apiClient.request<{ id: string }>('/conversations', {
        method: 'POST',
        body: JSON.stringify({
          participantIds,
        }),
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to create conversation');
        return null;
      }
      
      toast.success('Conversation created successfully');
      return response.data || null;
    } catch (error: any) {
      console.error('Error creating conversation:', error);
      toast.error('Failed to create conversation. Please try again.');
      return null;
    }
  }

  // Connection methods
  async connectWithCompany(followerId: string, followingId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request('/connections', {
        method: 'POST',
        body: JSON.stringify({
          followerId,
          followingId,
        }),
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to connect with company');
        return false;
      }
      
      toast.success('Successfully connected with company');
      return true;
    } catch (error: any) {
      console.error('Error connecting with company:', error);
      toast.error('Failed to connect with company. Please try again.');
      return false;
    }
  }

  async disconnectFromCompany(followerId: string, followingId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request(`/connections/${followerId}/${followingId}`, {
        method: 'DELETE',
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to disconnect from company');
        return false;
      }
      
      toast.success('Successfully disconnected from company');
      return true;
    } catch (error: any) {
      console.error('Error disconnecting from company:', error);
      toast.error('Failed to disconnect from company. Please try again.');
      return false;
    }
  }

  async getFollowing(userId: string): Promise<Connection[]> {
    try {
      const response = await this.apiClient.request<Connection[]>(`/connections/following/${userId}`);
      
      if (!response.success) {
        console.warn('Failed to get following list:', response.error);
        return [];
      }
      
      return response.data || [];
    } catch (error: any) {
      console.error('Error getting following:', error);
      return [];
    }
  }

  async getFollowers(userId: string): Promise<Connection[]> {
    try {
      const response = await this.apiClient.request<Connection[]>(`/connections/followers/${userId}`);
      
      if (!response.success) {
        console.warn('Failed to get followers list:', response.error);
        return [];
      }
      
      return response.data || [];
    } catch (error: any) {
      console.error('Error getting followers:', error);
      return [];
    }
  }

  async checkConnection(followerId: string, followingId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request<{ isConnected: boolean }>(`/connections/check/${followerId}/${followingId}`);
      
      if (!response.success) {
        console.warn('Failed to check connection status:', response.error);
        return false;
      }
      
      return response.data?.isConnected || false;
    } catch (error: any) {
      console.error('Error checking connection:', error);
      return false;
    }
  }

  // Favorite methods
  async addToFavorites(userId: string, materialId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request('/favorites', {
        method: 'POST',
        body: JSON.stringify({
          userId,
          materialId,
        }),
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to add to favorites');
        return false;
      }
      
      toast.success('Added to favorites');
      return true;
    } catch (error: any) {
      console.error('Error adding to favorites:', error);
      toast.error('Failed to add to favorites. Please try again.');
      return false;
    }
  }

  async removeFromFavorites(userId: string, materialId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request(`/favorites/${userId}/${materialId}`, {
        method: 'DELETE',
      });
      
      if (!response.success) {
        toast.error(response.error || 'Failed to remove from favorites');
        return false;
      }
      
      toast.success('Removed from favorites');
      return true;
    } catch (error: any) {
      console.error('Error removing from favorites:', error);
      toast.error('Failed to remove from favorites. Please try again.');
      return false;
    }
  }

  async getFavorites(userId: string): Promise<Favorite[]> {
    try {
      const response = await this.apiClient.request<Favorite[]>(`/favorites/${userId}`);
      
      if (!response.success) {
        console.warn('Failed to get favorites:', response.error);
        return [];
      }
      
      return response.data || [];
    } catch (error: any) {
      console.error('Error getting favorites:', error);
      return [];
    }
  }

  async checkFavorite(userId: string, materialId: string): Promise<boolean> {
    try {
      const response = await this.apiClient.request<{ isFavorited: boolean }>(`/favorites/check/${userId}/${materialId}`);
      
      if (!response.success) {
        console.warn('Failed to check favorite status:', response.error);
        return false;
      }
      
      return response.data?.isFavorited || false;
    } catch (error: any) {
      console.error('Error checking favorite:', error);
      return false;
    }
  }

  // Real-time messaging with WebSocket fallback
  async subscribeToMessages(conversationId: string, callback: (message: Message) => void): Promise<() => void> {
    try {
      // Try WebSocket first
      const ws = new WebSocket(`ws://localhost:5000/ws/conversations/${conversationId}`);
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          callback(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.warn('WebSocket error, falling back to polling:', error);
        this.startPolling(conversationId, callback);
      };

      return () => {
        ws.close();
      };
    } catch (error) {
      console.warn('WebSocket not available, using polling:', error);
      return this.startPolling(conversationId, callback);
    }
  }

  private startPolling(conversationId: string, callback: (message: Message) => void): () => void {
    let lastMessageId = '';
    const interval = setInterval(async () => {
      try {
        const messages = await this.getMessages(conversationId);
        const newMessages = messages.filter(msg => msg.id > lastMessageId);
        
        if (newMessages.length > 0) {
          lastMessageId = newMessages[newMessages.length - 1].id;
          newMessages.forEach(callback);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }
}

// Export singleton instance
export const messagingService = new MessagingService(); 