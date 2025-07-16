"""
Advanced Multi-Hop Symbiosis Detection Service
AI-Powered Circular Economy Network Analysis
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from flask import Flask, request, jsonify
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import os
import threading
import queue
import time
from collections import defaultdict, deque
import heapq
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Advanced Multi-Hop Configuration
@dataclass
class MultiHopConfig:
    """Multi-Hop Symbiosis Detection Configuration"""
    max_hops: int = 5
    min_symbiosis_score: float = 0.7
    max_path_length: int = 10
    clustering_method: str = "dbscan"  # dbscan, kmeans, hierarchical
    similarity_threshold: float = 0.8
    graph_algorithm: str = "dijkstra"  # dijkstra, bellman_ford, floyd_warshall
    ai_enhancement: bool = True
    pattern_recognition: bool = True
    real_time_updates: bool = True
    cache_results: bool = True
    max_cache_size: int = 1000
    update_frequency: int = 300  # seconds
    parallel_processing: bool = True
    num_workers: int = 4

class SymbiosisGraph:
    """Advanced Graph Representation for Industrial Symbiosis"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_attributes = {}
        self.edge_attributes = {}
        self.symbiosis_patterns = {}
        self.clusters = {}
        
    def add_company(self, company_id: str, attributes: Dict):
        """Add company node to graph"""
        self.graph.add_node(company_id, **attributes)
        self.node_attributes[company_id] = attributes
    
    def add_symbiosis_relationship(self, source: str, target: str, 
                                 relationship_type: str, strength: float):
        """Add symbiosis relationship between companies"""
        self.graph.add_edge(source, target, 
                           type=relationship_type, 
                           strength=strength,
                           timestamp=datetime.now())
        self.edge_attributes[(source, target)] = {
            'type': relationship_type,
            'strength': strength,
            'timestamp': datetime.now()
        }
    
    def get_company_neighbors(self, company_id: str, max_hops: int = 1) -> Set[str]:
        """Get companies within specified hop distance"""
        neighbors = set()
        visited = set()
        queue = deque([(company_id, 0)])
        
        while queue:
            current, hops = queue.popleft()
            
            if current in visited or hops > max_hops:
                continue
            
            visited.add(current)
            if hops > 0:
                neighbors.add(current)
            
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
        
        return neighbors
    
    def calculate_symbiosis_score(self, path: List[str]) -> float:
        """Calculate symbiosis score for a path"""
        if len(path) < 2:
            return 0.0
        
        total_strength = 0.0
        num_edges = 0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if self.graph.has_edge(source, target):
                strength = self.graph[source][target]['strength']
                total_strength += strength
                num_edges += 1
        
        return total_strength / num_edges if num_edges > 0 else 0.0
    
    def find_all_paths(self, source: str, target: str, max_length: int) -> List[List[str]]:
        """Find all paths between two companies"""
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return
            
            if current == target and len(path) > 1:
                paths.append(path[:])
                return
            
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, path + [neighbor], visited)
                    visited.remove(neighbor)
        
        dfs(source, [source], {source})
        return paths

class SymbiosisPatternRecognizer:
    """AI-Powered Pattern Recognition for Symbiosis Networks"""
    
    def __init__(self, config: MultiHopConfig):
        self.config = config
        self.patterns = {}
        self.pattern_embeddings = {}
        self.similarity_matrix = None
        
    def extract_patterns(self, graph: SymbiosisGraph) -> Dict:
        """Extract common patterns from symbiosis network"""
        patterns = {
            'linear_chains': self.find_linear_chains(graph),
            'circular_loops': self.find_circular_loops(graph),
            'hub_spoke': self.find_hub_spoke_patterns(graph),
            'clusters': self.find_clusters(graph),
            'bridges': self.find_bridge_patterns(graph)
        }
        
        self.patterns = patterns
        return patterns
    
    def find_linear_chains(self, graph: SymbiosisGraph) -> List[List[str]]:
        """Find linear supply chains"""
        chains = []
        visited = set()
        
        for node in graph.graph.nodes():
            if node in visited:
                continue
            
            # Find longest chain starting from this node
            chain = self.find_longest_chain(graph, node, visited)
            if len(chain) >= 3:  # Minimum chain length
                chains.append(chain)
        
        return chains
    
    def find_longest_chain(self, graph: SymbiosisGraph, start: str, 
                          visited: Set[str]) -> List[str]:
        """Find longest chain starting from a node"""
        def dfs(current: str, path: List[str]) -> List[str]:
            visited.add(current)
            longest_path = path[:]
            
            for neighbor in graph.graph.successors(current):
                if neighbor not in visited:
                    candidate_path = dfs(neighbor, path + [neighbor])
                    if len(candidate_path) > len(longest_path):
                        longest_path = candidate_path
            
            return longest_path
        
        return dfs(start, [start])
    
    def find_circular_loops(self, graph: SymbiosisGraph) -> List[List[str]]:
        """Find circular symbiosis loops"""
        cycles = []
        
        # Find all simple cycles
        for cycle in nx.simple_cycles(graph.graph):
            if len(cycle) >= 3:  # Minimum cycle length
                cycles.append(cycle)
        
        return cycles
    
    def find_hub_spoke_patterns(self, graph: SymbiosisGraph) -> List[Dict]:
        """Find hub-spoke patterns"""
        hub_patterns = []
        
        for node in graph.graph.nodes():
            in_degree = graph.graph.in_degree(node)
            out_degree = graph.graph.out_degree(node)
            
            # Identify hubs (high degree nodes)
            if in_degree + out_degree >= 5:
                spokes = list(graph.graph.predecessors(node)) + list(graph.graph.successors(node))
                hub_patterns.append({
                    'hub': node,
                    'spokes': spokes,
                    'in_degree': in_degree,
                    'out_degree': out_degree
                })
        
        return hub_patterns
    
    def find_clusters(self, graph: SymbiosisGraph) -> List[List[str]]:
        """Find densely connected clusters"""
        # Convert to undirected graph for clustering
        undirected_graph = graph.graph.to_undirected()
        if undirected_graph.number_of_nodes() == 0:
            return []
        # Use community detection
        communities = nx.community.greedy_modularity_communities(undirected_graph)
        return [list(community) for community in communities if len(community) >= 3]
    
    def find_bridge_patterns(self, graph: SymbiosisGraph) -> List[Dict]:
        """Find bridge patterns connecting different clusters"""
        bridges = []
        
        # Find bridge edges
        bridge_edges = nx.bridges(graph.graph.to_undirected())
        
        for edge in bridge_edges:
            source, target = edge
            bridges.append({
                'source': source,
                'target': target,
                'strength': graph.graph[source][target]['strength']
            })
        
        return bridges
    
    def calculate_pattern_similarity(self, pattern1: List[str], 
                                   pattern2: List[str]) -> float:
        """Calculate similarity between two patterns"""
        # Jaccard similarity
        set1 = set(pattern1)
        set2 = set(pattern2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def cluster_patterns(self, patterns: List[List[str]]) -> List[List[int]]:
        """Cluster similar patterns"""
        if len(patterns) < 2:
            return [[0]] if patterns else []
        
        # Calculate similarity matrix
        n = len(patterns)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.calculate_pattern_similarity(patterns[i], patterns[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=self.config.similarity_threshold, min_samples=2)
        cluster_labels = clustering.fit_predict(similarity_matrix)
        
        # Group patterns by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
        
        return list(clusters.values())

class MultiHopSymbiosisDetector:
    """Advanced Multi-Hop Symbiosis Detection Engine"""
    
    def __init__(self, config: MultiHopConfig):
        self.config = config
        self.graph = SymbiosisGraph()
        self.pattern_recognizer = SymbiosisPatternRecognizer(config)
        self.cache = {}
        self.cache_timestamps = {}
        self.update_queue = queue.Queue()
        
        # Start background update thread
        if self.config.real_time_updates:
            self.update_thread = threading.Thread(target=self._background_updates)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def add_company_data(self, company_id: str, company_data: Dict):
        """Add company data to the symbiosis network"""
        self.graph.add_company(company_id, company_data)
        
        # Add to update queue
        if self.config.real_time_updates:
            self.update_queue.put(('add_company', company_id, company_data))
    
    def add_symbiosis_relationship(self, source: str, target: str, 
                                 relationship_type: str, strength: float):
        """Add symbiosis relationship"""
        self.graph.add_symbiosis_relationship(source, target, relationship_type, strength)
        
        # Add to update queue
        if self.config.real_time_updates:
            self.update_queue.put(('add_relationship', source, target, relationship_type, strength))
    
    def find_multi_hop_symbiosis(self, source: str, target: str, 
                                max_hops: int = None) -> List[Dict]:
        """Find multi-hop symbiosis opportunities"""
        if max_hops is None:
            max_hops = self.config.max_hops
        
        # Check cache first
        cache_key = f"symbiosis_{source}_{target}_{max_hops}"
        if self.config.cache_results and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cache_time = self.cache_timestamps[cache_key]
            
            # Check if cache is still valid
            if (datetime.now() - cache_time).seconds < self.config.update_frequency:
                return cached_result
        
        # Find all paths
        paths = self.graph.find_all_paths(source, target, max_hops)
        
        # Calculate symbiosis scores
        symbiosis_opportunities = []
        for path in paths:
            score = self.graph.calculate_symbiosis_score(path)
            
            if score >= self.config.min_symbiosis_score:
                opportunity = {
                    'path': path,
                    'score': score,
                    'length': len(path) - 1,
                    'companies': path,
                    'estimated_impact': self.calculate_impact(path, score),
                    'feasibility': self.assess_feasibility(path),
                    'sustainability_score': self.calculate_sustainability(path)
                }
                symbiosis_opportunities.append(opportunity)
        
        # Sort by score
        symbiosis_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Cache result
        if self.config.cache_results:
            self.cache[cache_key] = symbiosis_opportunities
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Clean cache if too large
            if len(self.cache) > self.config.max_cache_size:
                self._clean_cache()
        
        return symbiosis_opportunities
    
    def find_optimal_paths(self, source: str, target: str, 
                          max_hops: int = None) -> List[Dict]:
        """Find optimal paths using different algorithms"""
        if max_hops is None:
            max_hops = self.config.max_hops
        
        optimal_paths = []
        
        if self.config.graph_algorithm == "dijkstra":
            optimal_paths = self._dijkstra_shortest_paths(source, target, max_hops)
        elif self.config.graph_algorithm == "bellman_ford":
            optimal_paths = self._bellman_ford_paths(source, target, max_hops)
        elif self.config.graph_algorithm == "floyd_warshall":
            optimal_paths = self._floyd_warshall_paths(source, target, max_hops)
        
        return optimal_paths
    
    def _dijkstra_shortest_paths(self, source: str, target: str, 
                                max_hops: int) -> List[Dict]:
        """Find shortest paths using Dijkstra's algorithm"""
        paths = []
        
        # Use NetworkX implementation
        try:
            # Ensure cutoff is strictly less than 1
            cutoff = max_hops
            if cutoff >= 1:
                cutoff = min(cutoff, 0.99) if cutoff == 1 else cutoff
            shortest_paths = nx.all_simple_paths(self.graph.graph, source, target, cutoff=cutoff)
            
            for path in shortest_paths:
                if len(path) <= max_hops + 1:
                    score = self.graph.calculate_symbiosis_score(path)
                    paths.append({
                        'path': path,
                        'score': score,
                        'length': len(path) - 1,
                        'algorithm': 'dijkstra'
                    })
        except nx.NetworkXNoPath:
            pass
        
        return sorted(paths, key=lambda x: x['score'], reverse=True)
    
    def _bellman_ford_paths(self, source: str, target: str, 
                           max_hops: int) -> List[Dict]:
        """Find paths using Bellman-Ford algorithm"""
        # Simplified implementation
        return self._dijkstra_shortest_paths(source, target, max_hops)
    
    def _floyd_warshall_paths(self, source: str, target: str, 
                             max_hops: int) -> List[Dict]:
        """Find paths using Floyd-Warshall algorithm"""
        # Simplified implementation
        return self._dijkstra_shortest_paths(source, target, max_hops)
    
    def calculate_impact(self, path: List[str], score: float) -> Dict:
        """Calculate environmental and economic impact"""
        # Mock impact calculation
        num_companies = len(path)
        base_impact = score * num_companies
        
        return {
            'environmental_savings': base_impact * 1000,  # kg CO2
            'economic_benefit': base_impact * 50000,  # USD
            'waste_reduction': base_impact * 500,  # kg
            'energy_savings': base_impact * 2000  # kWh
        }
    
    def assess_feasibility(self, path: List[str]) -> Dict:
        """Assess feasibility of symbiosis path"""
        feasibility_score = 0.0
        factors = {}
        
        # Distance factor
        total_distance = self._calculate_total_distance(path)
        distance_factor = max(0, 1 - total_distance / 1000)  # Normalize to 0-1
        factors['distance'] = distance_factor
        
        # Technology compatibility
        tech_compatibility = self._assess_technology_compatibility(path)
        factors['technology'] = tech_compatibility
        
        # Regulatory compliance
        regulatory_score = self._assess_regulatory_compliance(path)
        factors['regulatory'] = regulatory_score
        
        # Economic viability
        economic_score = self._assess_economic_viability(path)
        factors['economic'] = economic_score
        
        # Calculate overall feasibility
        feasibility_score = np.mean(list(factors.values()))
        
        return {
            'overall_score': feasibility_score,
            'factors': factors,
            'recommendations': self._generate_recommendations(factors)
        }
    
    def calculate_sustainability(self, path: List[str]) -> float:
        """Calculate sustainability score"""
        # Mock sustainability calculation
        base_score = 0.8
        
        # Adjust based on path characteristics
        if len(path) > 3:
            base_score += 0.1  # Longer chains are more sustainable
        
        # Add some randomness for demonstration
        return min(1.0, base_score + np.random.normal(0, 0.05))
    
    def _calculate_total_distance(self, path: List[str]) -> float:
        """Calculate total distance between companies in path"""
        # Mock distance calculation
        return len(path) * 50  # km
    
    def _assess_technology_compatibility(self, path: List[str]) -> float:
        """Assess technology compatibility between companies"""
        # Mock assessment
        return np.random.uniform(0.6, 0.95)
    
    def _assess_regulatory_compliance(self, path: List[str]) -> float:
        """Assess regulatory compliance"""
        # Mock assessment
        return np.random.uniform(0.7, 0.98)
    
    def _assess_economic_viability(self, path: List[str]) -> float:
        """Assess economic viability"""
        # Mock assessment
        return np.random.uniform(0.5, 0.9)
    
    def _generate_recommendations(self, factors: Dict) -> List[str]:
        """Generate recommendations based on feasibility factors"""
        recommendations = []
        
        if factors['distance'] < 0.5:
            recommendations.append("Consider logistics optimization to reduce transportation costs")
        
        if factors['technology'] < 0.7:
            recommendations.append("Invest in technology upgrades for better compatibility")
        
        if factors['regulatory'] < 0.8:
            recommendations.append("Ensure regulatory compliance before implementation")
        
        if factors['economic'] < 0.6:
            recommendations.append("Conduct detailed cost-benefit analysis")
        
        return recommendations
    
    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).seconds > self.config.update_frequency * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    def _background_updates(self):
        """Background thread for real-time updates"""
        while True:
            try:
                # Process update queue
                while not self.update_queue.empty():
                    update = self.update_queue.get_nowait()
                    self._process_update(update)
                # Update patterns periodically
                if self.config.pattern_recognition:
                    try:
                        self.pattern_recognizer.extract_patterns(self.graph)
                    except Exception as e:
                        import traceback
                        logging.error(f"Background update error: {e}\n{traceback.format_exc()}")
                time.sleep(1)  # Check every second
            except Exception as e:
                import traceback
                logging.error(f"Background update error: {e}\n{traceback.format_exc()}")
                time.sleep(5)
    
    def _process_update(self, update: Tuple):
        """Process a single update"""
        update_type = update[0]
        
        if update_type == 'add_company':
            # Handle company addition
            pass
        elif update_type == 'add_relationship':
            # Handle relationship addition
            pass

# Flask Application for Multi-Hop Symbiosis Detection
multi_hop_app = Flask(__name__)

# Initialize multi-hop detection service
multi_hop_config = MultiHopConfig()
multi_hop_detector = MultiHopSymbiosisDetector(multi_hop_config)

@multi_hop_app.route('/health', methods=['GET'])
def multi_hop_health_check():
    """Health check for multi-hop symbiosis service"""
    return jsonify({
        'status': 'healthy',
        'service': 'multi_hop_symbiosis',
        'graph_nodes': len(multi_hop_detector.graph.graph.nodes()),
        'graph_edges': len(multi_hop_detector.graph.graph.edges()),
        'cache_size': len(multi_hop_detector.cache)
    })

@multi_hop_app.route('/add_company', methods=['POST'])
def add_company():
    """Add company to symbiosis network"""
    try:
        data = request.get_json()
        company_id = data.get('company_id')
        company_data = data.get('company_data', {})
        
        if not company_id:
            return jsonify({'error': 'Company ID required'}), 400
        
        multi_hop_detector.add_company_data(company_id, company_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Company {company_id} added successfully'
        })
        
    except Exception as e:
        logging.error(f"Add company error: {e}")
        return jsonify({'error': str(e)}), 500

@multi_hop_app.route('/add_relationship', methods=['POST'])
def add_relationship():
    """Add symbiosis relationship"""
    try:
        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        relationship_type = data.get('type', 'material_exchange')
        strength = data.get('strength', 0.5)
        
        if not source or not target:
            return jsonify({'error': 'Source and target required'}), 400
        
        multi_hop_detector.add_symbiosis_relationship(source, target, relationship_type, strength)
        
        return jsonify({
            'status': 'success',
            'message': f'Relationship {source} -> {target} added successfully'
        })
        
    except Exception as e:
        logging.error(f"Add relationship error: {e}")
        return jsonify({'error': str(e)}), 500

@multi_hop_app.route('/find_symbiosis', methods=['POST'])
def find_symbiosis():
    """Find multi-hop symbiosis opportunities"""
    try:
        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        max_hops = data.get('max_hops', multi_hop_config.max_hops)
        
        if not source or not target:
            return jsonify({'error': 'Source and target required'}), 400
        
        opportunities = multi_hop_detector.find_multi_hop_symbiosis(source, target, max_hops)
        
        return jsonify({
            'status': 'success',
            'opportunities': opportunities,
            'count': len(opportunities)
        })
        
    except Exception as e:
        logging.error(f"Find symbiosis error: {e}")
        return jsonify({'error': str(e)}), 500

@multi_hop_app.route('/optimal_paths', methods=['POST'])
def find_optimal_paths():
    """Find optimal paths between companies"""
    try:
        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        max_hops = data.get('max_hops', multi_hop_config.max_hops)
        
        if not source or not target:
            return jsonify({'error': 'Source and target required'}), 400
        
        paths = multi_hop_detector.find_optimal_paths(source, target, max_hops)
        
        return jsonify({
            'status': 'success',
            'paths': paths,
            'count': len(paths)
        })
        
    except Exception as e:
        logging.error(f"Find optimal paths error: {e}")
        return jsonify({'error': str(e)}), 500

@multi_hop_app.route('/patterns', methods=['GET'])
def get_patterns():
    """Get detected symbiosis patterns"""
    try:
        patterns = multi_hop_detector.pattern_recognizer.extract_patterns(multi_hop_detector.graph)
        
        return jsonify({
            'status': 'success',
            'patterns': patterns
        })
        
    except Exception as e:
        logging.error(f"Get patterns error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    multi_hop_app.run(host='0.0.0.0', port=5003, debug=False) 