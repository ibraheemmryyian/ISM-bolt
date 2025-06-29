import React, { useEffect, useState } from 'react';

interface Opportunity {
  id: string;
  title: string;
  description: string;
  impact: string;
  actionUrl?: string;
}

const ProactiveOpportunitiesPanel: React.FC = () => {
  const [opportunities, setOpportunities] = useState<Opportunity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Replace with real service call
    async function fetchOpportunities() {
      setLoading(true);
      // Placeholder: simulate fetch
      setTimeout(() => {
        setOpportunities([
          {
            id: 'opp1',
            title: 'New EU Regulation on Plastic Waste',
            description: 'A new regulation creates demand for recycled plastics. Your company can supply to 3 new partners.',
            impact: 'Potential revenue: €120,000/year, Carbon reduction: 200 tons/year',
            actionUrl: '#'
          }
        ]);
        setLoading(false);
      }, 1000);
    }
    fetchOpportunities();
  }, []);

  if (loading) return <div>Loading opportunities...</div>;
  if (opportunities.length === 0) return <div>No new opportunities at this time.</div>;

  return (
    <div className="proactive-opportunities-panel">
      <h2>Proactive Symbiosis Opportunities</h2>
      <ul>
        {opportunities.map(opp => (
          <li key={opp.id} className="opportunity-card">
            <h3>{opp.title}</h3>
            <p>{opp.description}</p>
            <div><strong>Impact:</strong> {opp.impact}</div>
            {opp.actionUrl && <a href={opp.actionUrl} className="btn-act-now">Act Now</a>}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ProactiveOpportunitiesPanel; 