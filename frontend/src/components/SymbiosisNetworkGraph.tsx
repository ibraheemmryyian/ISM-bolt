import React, { useEffect, useRef, useState } from 'react';

interface NetworkNode {
  id: string;
  name?: string;
  industry?: string;
  location?: string;
  annual_waste?: number;
  carbon_footprint?: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  explanation: any;
}

interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  clusters: string[][];
  metrics: {
    total_nodes: number;
    total_edges: number;
    density: number;
    average_clustering: number;
    symbiosis_potential: number;
  };
}

interface SymbiosisNetworkGraphProps {
  networkData: NetworkData;
  onNodeClick?: (nodeId: string) => void;
  onEdgeClick?: (edge: NetworkEdge) => void;
}

export function SymbiosisNetworkGraph({ networkData, onNodeClick, onEdgeClick }: SymbiosisNetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<NetworkEdge | null>(null);

  useEffect(() => {
    if (!svgRef.current || !networkData) return;

    // Clear previous content
    const svg = svgRef.current;
    svg.innerHTML = '';

    // Simple force-directed graph visualization
    // In production, you'd use D3.js or a library like vis.js
    drawSimpleNetwork(svg, networkData);
  }, [networkData]);

  const drawSimpleNetwork = (svg: SVGSVGElement, data: NetworkData) => {
    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Draw edges first (so they appear behind nodes)
    data.edges.forEach((edge, index) => {
      const sourceNode = data.nodes.find(n => n.id === edge.source);
      const targetNode = data.nodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode) {
        const angle = (index * 2 * Math.PI) / data.edges.length;
        const sourceX = centerX + Math.cos(angle) * 150;
        const sourceY = centerY + Math.sin(angle) * 150;
        const targetX = centerX + Math.cos(angle + Math.PI) * 150;
        const targetY = centerY + Math.sin(angle + Math.PI) * 150;

        const strokeWidth = Math.max(1, edge.weight * 5);
        const strokeColor = edge.weight > 0.8 ? '#10b981' : edge.weight > 0.6 ? '#f59e0b' : '#ef4444';

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', sourceX.toString());
        line.setAttribute('y1', sourceY.toString());
        line.setAttribute('x2', targetX.toString());
        line.setAttribute('y2', targetY.toString());
        line.setAttribute('stroke', strokeColor);
        line.setAttribute('stroke-width', strokeWidth.toString());
        line.setAttribute('opacity', '0.6');
        line.style.cursor = 'pointer';
        
        line.addEventListener('click', () => {
          setSelectedEdge(edge);
          onEdgeClick?.(edge);
        });

        svg.appendChild(line);
      }
    });

    // Draw nodes
    data.nodes.forEach((node, index) => {
      const angle = (index * 2 * Math.PI) / data.nodes.length;
      const x = centerX + Math.cos(angle) * 150;
      const y = centerY + Math.sin(angle) * 150;

      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', x.toString());
      circle.setAttribute('cy', y.toString());
      circle.setAttribute('r', '20');
      circle.setAttribute('fill', '#3b82f6');
      circle.setAttribute('stroke', '#1e40af');
      circle.setAttribute('stroke-width', '2');
      circle.style.cursor = 'pointer';
      
      circle.addEventListener('click', () => {
        setSelectedNode(node.id);
        onNodeClick?.(node.id);
      });

      svg.appendChild(circle);

      // Node label
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', x.toString());
      text.setAttribute('y', (y + 5).toString());
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', 'white');
      text.setAttribute('font-size', '12');
      text.setAttribute('font-weight', 'bold');
      text.textContent = node.id.substring(0, 3).toUpperCase();
      
      svg.appendChild(text);
    });
  };

  return (
    <div className="bg-white rounded-lg p-6 shadow-lg">
      <div className="mb-4">
        <h3 className="text-xl font-bold text-gray-900 mb-2">Symbiosis Network</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="text-center">
            <div className="font-bold text-blue-600">{networkData.metrics.total_nodes}</div>
            <div className="text-gray-600">Companies</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-green-600">{networkData.metrics.total_edges}</div>
            <div className="text-gray-600">Connections</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-purple-600">{networkData.metrics.symbiosis_potential.toFixed(2)}</div>
            <div className="text-gray-600">Symbiosis Score</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-orange-600">{networkData.clusters.length}</div>
            <div className="text-gray-600">Clusters</div>
          </div>
        </div>
      </div>

      {/* Network Visualization */}
      <div className="border rounded-lg bg-gray-50 p-4">
        <svg
          ref={svgRef}
          width="800"
          height="600"
          className="w-full h-auto"
          viewBox="0 0 800 600"
        />
      </div>

      {/* Selected Node/Edge Info */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <h4 className="font-semibold text-blue-900">Selected Company: {selectedNode}</h4>
          <p className="text-sm text-blue-700">Click on connections to see match details</p>
        </div>
      )}

      {selectedEdge && (
        <div className="mt-4 p-4 bg-green-50 rounded-lg">
          <h4 className="font-semibold text-green-900">
            Match: {selectedEdge.source} ↔ {selectedEdge.target}
          </h4>
          <p className="text-sm text-green-700">
            Score: {(selectedEdge.weight * 100).toFixed(0)}%
          </p>
          <button
            onClick={() => setSelectedEdge(null)}
            className="mt-2 text-sm text-green-600 hover:text-green-800"
          >
            Clear Selection
          </button>
        </div>
      )}

      {/* Clusters */}
      <div className="mt-6">
        <h4 className="font-semibold text-gray-900 mb-3">Symbiosis Clusters</h4>
        <div className="space-y-3">
          {networkData.clusters.map((cluster, index) => (
            <div key={index} className="p-3 bg-gray-50 rounded-lg">
              <div className="font-medium text-gray-900 mb-1">
                Cluster {index + 1} ({cluster.length} companies)
              </div>
              <div className="text-sm text-gray-600">
                {cluster.join(', ')}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 