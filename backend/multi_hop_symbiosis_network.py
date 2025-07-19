import numpy as np
import torch
import torch.nn as nn

class MultiHopSymbiosisNetwork:
    def __init__(self, max_hops=3, embedding_dim=64):
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.symbiosis_model = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def find_multi_hop_matches(self, source_company, material, max_hops=3, top_k=3):
        """Find multi-hop symbiosis matches"""
        try:
            # Generate sample multi-hop matches
            matches = []
            for hop in range(1, max_hops + 1):
                for i in range(top_k):
                    match = {
                        'id': f'hop_{hop}_company_{i+1}',
                        'name': f'Hop {hop} Company {i+1}',
                        'company_id': f'hop_{hop}_company_{i+1}',
                        'score': 0.9 - (hop * 0.2) - (i * 0.1),  # Decreasing scores with hops
                        'hops': hop,
                        'reason': f'Multi-hop symbiosis: {hop} hop(s) away'
                    }
                    matches.append(match)
            return matches
        except Exception as e:
            print(f"Error in find_multi_hop_matches: {e}")
            return []
    
    async def find_symbiosis_matches(self, material):
        """Find symbiosis matches for a given material"""
        try:
            # Generate sample symbiosis matches
            material_name = material.get('material_name', 'Unknown Material')
            
            # Create sample matches
            matches = []
            for i in range(2):  # Generate 2 sample matches
                match = {
                    'company_id': f'symbiosis_company_{i+1}',
                    'company_name': f'Symbiosis Company {i+1}',
                    'material_name': f'Symbiotic {material_name}',
                    'score': 0.7 - (i * 0.1),  # Decreasing scores
                    'type': 'symbiosis',
                    'potential_value': 600 + (i * 200)
                }
                matches.append(match)
            
            return matches
        except Exception as e:
            print(f"Error in find_symbiosis_matches: {e}")
            return []
    
    def run(self):
        return 'MultiHopSymbiosisNetwork running.' 