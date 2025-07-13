import React, { useState } from 'react';

interface Notification {
  id: string;
  type: 'match' | 'message' | 'admin' | 'info';
  message: string;
  timestamp: string;
}

const mockNotifications: Notification[] = [
  { id: '1', type: 'match', message: 'New material match found: Spent Grain → Bioenergy Co.', timestamp: '2 hours ago' },
  { id: '2', type: 'message', message: 'You received a message from Green Plastics.', timestamp: '5 hours ago' },
  { id: '3', type: 'admin', message: 'Your subscription has been upgraded to Pro.', timestamp: '1 day ago' },
];

export function NotificationsPanel() {
  const [notifications] = useState<Notification[]>(mockNotifications);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white rounded-xl shadow-sm p-8 w-full max-w-lg">
        <h2 className="text-2xl font-bold mb-6">Notifications</h2>
        <div className="space-y-4">
          {notifications.length === 0 ? (
            <div className="text-gray-500 text-center">No notifications yet.</div>
          ) : (
            notifications.map((n) => (
              <div key={n.id} className="flex items-center space-x-3 border-b pb-3 last:border-b-0">
                <span className={`h-8 w-8 flex items-center justify-center rounded-full ${n.type === 'match' ? 'bg-emerald-100 text-emerald-600' : n.type === 'message' ? 'bg-blue-100 text-blue-600' : n.type === 'admin' ? 'bg-yellow-100 text-yellow-600' : 'bg-gray-100 text-gray-500'}`}>{n.type.charAt(0).toUpperCase()}</span>
                <div className="flex-1">
                  <div className="text-gray-900">{n.message}</div>
                  <div className="text-xs text-gray-400">{n.timestamp}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
} 