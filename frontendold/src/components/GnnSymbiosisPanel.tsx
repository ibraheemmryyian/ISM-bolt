import React, { useEffect, useState } from 'react';

interface GnnLink {
  0: string;
  1: string;
  2: number;
}

const GnnSymbiosisPanel: React.FC = () => {
  const [links, setLinks] = useState<GnnLink[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchLinks() {
      setLoading(true);
      try {
        const res = await fetch('/api/gnn-symbiosis');
        const data = await res.json();
        setLinks(data.links || []);
      } catch (e) {
        setLinks([]);
      }
      setLoading(false);
    }
    fetchLinks();
  }, []);

  if (loading) return <div>Loading AI-discovered symbiosis opportunities...</div>;
  if (links.length === 0) return <div>No new AI-discovered symbiosis opportunities found.</div>;

  return (
    <div className="gnn-symbiosis-panel">
      <h2>AI-Discovered Symbiosis Opportunities</h2>
      <ul>
        {links.map((link, idx) => (
          <li key={idx} className="gnn-link">
            <strong>{link[0]}</strong> &rarr; <strong>{link[1]}</strong> <span style={{ color: '#888' }}>(Score: {link[2].toFixed(3)})</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default GnnSymbiosisPanel; 