import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

interface Connection {
  from_location: string;
  to_location: string;
  type: string;
}

export function GlobalMap() {
  const [connections, setConnections] = useState<Connection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadConnections();
  }, []);

  async function loadConnections() {
    try {
      const { data, error } = await supabase
        .from('companies')
        .select('location')
        .not('location', 'is', null);

      if (error) throw error;

      // Get actual connections from database
      const { data: connectionsData, error: connectionsError } = await supabase
        .from('connections')
        .select('*')
        .eq('status', 'active');

      if (connectionsError) throw connectionsError;

      // Transform connections to map format
      const actualConnections = connectionsData.map(connection => ({
        from_location: connection.requester_location || 'Unknown',
        to_location: connection.recipient_location || 'Unknown',
        type: connection.connection_type || 'partnership'
      }));

      setConnections(actualConnections);
    } catch (error) {
      console.error('Error loading connections:', error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-[600px] flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 min-h-[600px] relative">
      <div className="absolute inset-0 opacity-20">
        {/* World map background - in production, use a proper map visualization library */}
        <img
          src="https://images.unsplash.com/photo-1589519160732-57fc498494f8?auto=format&fit=crop&q=80"
          alt="World Map"
          className="w-full h-full object-cover"
        />
      </div>
      
      <div className="relative z-10 container mx-auto px-6 py-12">
        <h2 className="text-3xl font-bold text-white mb-8">Global Impact Network</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          {connections.map((connection, index) => (
            <div key={index} className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
              <div className="text-emerald-400 font-semibold">
                {connection.type === 'material_exchange' ? 'Material Exchange' : 'Research Collaboration'}
              </div>
              <div className="text-white/80 text-sm mt-2">
                {connection.from_location} → {connection.to_location}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}