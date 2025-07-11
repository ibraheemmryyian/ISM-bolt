import React, { useEffect, useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Bell, CheckCircle, XCircle, Info, AlertTriangle } from 'lucide-react';

interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  data?: any;
  read_at?: string;
  created_at: string;
}

interface NotificationsPanelProps {
  companyId: string;
}

export const NotificationsPanel: React.FC<NotificationsPanelProps> = ({ companyId }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  useEffect(() => {
    if (!companyId) return;
    const eventSource = new EventSource(`/api/v1/companies/${companyId}/notifications/stream`);
    eventSource.onmessage = (event) => {
      const notification = JSON.parse(event.data);
      setNotifications(prev => {
        if (prev.find(n => n.id === notification.id)) return prev;
        return [notification, ...prev].slice(0, 10);
      });
    };
    return () => eventSource.close();
  }, [companyId]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'match_update': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error': return <XCircle className="w-5 h-5 text-red-400" />;
      case 'info': return <Info className="w-5 h-5 text-blue-400" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      default: return <Bell className="w-5 h-5 text-emerald-400" />;
    }
  };

  return (
    <div className="space-y-2">
      {notifications.length === 0 ? (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4 text-center text-gray-400">
            <Bell className="w-8 h-8 mx-auto mb-2" />
            No notifications yet.
          </CardContent>
        </Card>
      ) : (
        notifications.map(n => (
          <Card key={n.id} className="bg-slate-800/50 border-slate-700">
            <CardContent className="flex items-center gap-3 p-4">
              {getIcon(n.type)}
              <div>
                <div className="font-semibold text-white">{n.title}</div>
                <div className="text-gray-300 text-sm">{n.message}</div>
                <div className="text-xs text-gray-500 mt-1">{new Date(n.created_at).toLocaleString()}</div>
              </div>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
}; 