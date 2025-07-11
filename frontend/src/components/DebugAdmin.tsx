import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

export function DebugAdmin() {
  const [debugData, setDebugData] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const testTables = async () => {
    setLoading(true);
    setError('');
    
    const results: any = {};
    
    // Test each table
    const tables = ['companies', 'company_applications', 'materials', 'users', 'subscriptions'];
    
    for (const table of tables) {
      try {
        console.log(`Testing table: ${table}`);
        const { data, error } = await supabase
          .from(table)
          .select('*')
          .limit(5);
        
        if (error) {
          results[table] = { error: error.message };
          console.error(`Error with ${table}:`, error);
        } else {
          results[table] = { 
            count: data?.length || 0,
            data: data || [],
            columns: data && data.length > 0 ? Object.keys(data[0]) : []
          };
          console.log(`${table}:`, data?.length || 0, 'records');
        }
      } catch (err: any) {
        results[table] = { error: err.message };
        console.error(`Exception with ${table}:`, err);
      }
    }
    
    setDebugData(results);
    setLoading(false);
  };

  useEffect(() => {
    testTables();
  }, []);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Database Debug Info</h1>
      
      {loading && <p>Loading...</p>}
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      
      <div className="space-y-4">
        {Object.entries(debugData).map(([tableName, info]: [string, any]) => (
          <div key={tableName} className="border rounded p-4">
            <h3 className="font-bold text-lg mb-2">{tableName}</h3>
            
            {info.error ? (
              <div className="text-red-600">
                ❌ Error: {info.error}
              </div>
            ) : (
              <div>
                <p>✅ Records: {info.count}</p>
                {info.columns && info.columns.length > 0 && (
                  <p>Columns: {info.columns.join(', ')}</p>
                )}
                {info.data && info.data.length > 0 && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-blue-600">View Data</summary>
                    <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-auto">
                      {JSON.stringify(info.data, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <button 
        onClick={testTables}
        className="mt-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Refresh Data
      </button>
    </div>
  );
} 