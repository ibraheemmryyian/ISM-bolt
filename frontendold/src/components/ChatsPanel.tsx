import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { MessageSquare, Send, ArrowLeft, Users } from 'lucide-react';

interface Message {
  id: string;
  content: string;
  sender_id: string;
  receiver_id: string;
  created_at: string;
}

interface Conversation {
  id: string;
  participant1: string;
  participant2: string;
  created_at: string;
  participant1_name?: string;
  participant2_name?: string;
}

export function ChatsPanel() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    loadUserAndConversations();
  }, []);

  useEffect(() => {
    const conversationId = searchParams.get('conversation');
    if (conversationId) {
      loadConversation(conversationId);
    }
  }, [searchParams]);

  async function loadUserAndConversations() {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUserId(user.id);
        await loadConversations(user.id);
      }
    } catch (error) {
      console.error('Error loading user:', error);
    } finally {
      setLoading(false);
    }
  }

  async function loadConversations(userId: string) {
    try {
      const { data: conversationsData, error } = await supabase
        .from('conversations')
        .select(`
          *,
          p1:companies!conversations_participant1_fkey(name),
          p2:companies!conversations_participant2_fkey(name)
        `)
        .or(`participant1.eq.${userId},participant2.eq.${userId}`)
        .order('created_at', { ascending: false });

      if (error) throw error;

      const formattedConversations = conversationsData?.map(conv => ({
        ...conv,
        participant1_name: conv.p1?.name || 'Unknown Company',
        participant2_name: conv.p2?.name || 'Unknown Company'
      })) || [];

      setConversations(formattedConversations);
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  }

  async function loadConversation(conversationId: string) {
    try {
      // Get conversation details
      const { data: conversation, error: convError } = await supabase
        .from('conversations')
        .select(`
          *,
          p1:companies!conversations_participant1_fkey(name),
          p2:companies!conversations_participant2_fkey(name)
        `)
        .eq('id', conversationId)
        .single();

      if (convError) throw convError;

      const formattedConversation = {
        ...conversation,
        participant1_name: conversation.p1?.name || 'Unknown Company',
        participant2_name: conversation.p2?.name || 'Unknown Company'
      };

      setCurrentConversation(formattedConversation);

      // Load messages for this conversation
      const { data: messagesData, error: msgError } = await supabase
        .from('messages')
        .select('*')
        .or(`sender_id.eq.${conversation.participant1},sender_id.eq.${conversation.participant2}`)
        .or(`receiver_id.eq.${conversation.participant1},receiver_id.eq.${conversation.participant2}`)
        .order('created_at', { ascending: true });

      if (msgError) throw msgError;
      setMessages(messagesData || []);
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  }

  async function sendMessage() {
    if (!newMessage.trim() || !currentConversation || !currentUserId) return;

    try {
      const otherParticipant = currentConversation.participant1 === currentUserId 
        ? currentConversation.participant2 
        : currentConversation.participant1;

      const { data: message, error } = await supabase
        .from('messages')
        .insert({
          sender_id: currentUserId,
          receiver_id: otherParticipant,
          content: newMessage.trim(),
          conversation_id: currentConversation.id
        })
        .select()
        .single();

      if (error) throw error;

      setMessages(prev => [...prev, message]);
      setNewMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
  }

  function getOtherParticipantName(conversation: Conversation): string {
    if (!currentUserId) return 'Unknown';
    return conversation.participant1 === currentUserId 
      ? conversation.participant2_name || 'Unknown Company'
      : conversation.participant1_name || 'Unknown Company';
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading chats...</p>
        </div>
      </div>
    );
  }

  if (currentConversation) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-4xl mx-auto bg-white shadow-sm">
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => {
                  setCurrentConversation(null);
                  navigate('/chats');
                }}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <ArrowLeft className="h-5 w-5" />
              </button>
              <div>
                <h2 className="text-lg font-semibold">
                  {getOtherParticipantName(currentConversation)}
                </h2>
                <p className="text-sm text-gray-500">Active now</p>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="h-96 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-8">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No messages yet. Start the conversation!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender_id === currentUserId ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.sender_id === currentUserId
                        ? 'bg-emerald-500 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <p className="text-sm">{message.content}</p>
                    <p className={`text-xs mt-1 ${
                      message.sender_id === currentUserId ? 'text-emerald-100' : 'text-gray-500'
                    }`}>
                      {new Date(message.created_at).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Message Input */}
          <div className="p-4 border-t">
            <div className="flex space-x-2">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Type your message..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              />
              <button
                onClick={sendMessage}
                disabled={!newMessage.trim()}
                className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Chats</h1>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
          >
            ← Back to Dashboard
          </button>
        </div>

        {conversations.length === 0 ? (
          <div className="bg-white rounded-xl shadow-sm p-8 text-center">
            <MessageSquare className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">No conversations yet</h2>
            <p className="text-gray-600 mb-6">Start connecting with companies to begin chatting</p>
            <button
              onClick={() => navigate('/marketplace')}
              className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600"
            >
              Browse Marketplace
            </button>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-sm">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                onClick={() => navigate(`/chats?conversation=${conversation.id}`)}
                className="p-4 border-b last:border-b-0 hover:bg-gray-50 cursor-pointer transition"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-emerald-100 rounded-full">
                      <Users className="h-5 w-5 text-emerald-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">
                        {getOtherParticipantName(conversation)}
                      </h3>
                      <p className="text-sm text-gray-500">
                        {new Date(conversation.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-sm text-gray-400">
                    Click to open
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 