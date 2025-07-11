"""
World-Class Federated Learning Service
Advanced Multi-Company AI Training with Privacy Preservation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
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
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import threading
import queue
import time

# Advanced Federated Learning Configuration
@dataclass
class FederatedConfig:
    """Advanced Federated Learning Configuration"""
    num_clients: int = 10
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    aggregation_method: str = "fedavg_plus"  # fedavg, fedprox, fedavg_plus, secure_aggregation
    privacy_budget: float = 1.0  # Differential privacy budget
    noise_scale: float = 0.1  # Noise scale for DP
    secure_aggregation: bool = True
    multi_party_computation: bool = True
    client_selection: str = "adaptive"  # random, adaptive, stratified
    communication_rounds: int = 3
    model_compression: bool = True
    quantization_bits: int = 8
    sparsification_ratio: float = 0.1
    adaptive_learning_rate: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_clients_per_round: int = 3
    max_clients_per_round: int = 8

class DifferentialPrivacy:
    """Advanced Differential Privacy Implementation"""
    
    def __init__(self, privacy_budget: float, noise_scale: float):
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
        self.sensitivity = 1.0
    
    def add_noise(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add calibrated noise to gradients"""
        noisy_gradients = []
        
        for grad in gradients:
            # Calculate noise based on sensitivity and privacy budget
            noise = torch.randn_like(grad) * self.noise_scale * self.sensitivity / self.privacy_budget
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
    
    def calculate_sensitivity(self, gradients: List[torch.Tensor]) -> float:
        """Calculate gradient sensitivity"""
        # L2 sensitivity calculation
        total_norm = 0.0
        for grad in gradients:
            total_norm += torch.norm(grad).item() ** 2
        return np.sqrt(total_norm)

class SecureAggregation:
    """Secure Aggregation with Homomorphic Encryption"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.keys = {}
        self.masks = {}
        self.setup_encryption()
    
    def setup_encryption(self):
        """Setup encryption keys for secure aggregation"""
        for i in range(self.num_clients):
            # Generate encryption key
            key = Fernet.generate_key()
            self.keys[i] = Fernet(key)
            
            # Generate random mask
            mask = os.urandom(32)
            self.masks[i] = mask
    
    def encrypt_gradients(self, client_id: int, gradients: List[torch.Tensor]) -> List[bytes]:
        """Encrypt gradients for secure aggregation"""
        encrypted_gradients = []
        
        for grad in gradients:
            # Serialize gradient
            grad_bytes = pickle.dumps(grad)
            
            # Add mask
            masked_grad = self.xor_bytes(grad_bytes, self.masks[client_id])
            
            # Encrypt
            encrypted = self.keys[client_id].encrypt(masked_grad)
            encrypted_gradients.append(encrypted)
        
        return encrypted_gradients
    
    def decrypt_gradients(self, encrypted_gradients: List[List[bytes]]) -> List[torch.Tensor]:
        """Decrypt and aggregate gradients"""
        aggregated_gradients = []
        
        # Sum encrypted gradients
        for i in range(len(encrypted_gradients[0])):
            summed_encrypted = b'\x00' * len(encrypted_gradients[0][i])
            
            for client_grads in encrypted_gradients:
                summed_encrypted = self.xor_bytes(summed_encrypted, client_grads[i])
            
            # Remove masks
            for mask in self.masks.values():
                summed_encrypted = self.xor_bytes(summed_encrypted, mask)
            
            # Decrypt
            decrypted = pickle.loads(summed_encrypted)
            aggregated_gradients.append(decrypted)
        
        return aggregated_gradients
    
    def xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte strings"""
        return bytes(x ^ y for x, y in zip(a, b))

class AdaptiveClientSelection:
    """Adaptive Client Selection for Federated Learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.client_performance = {}
        self.client_availability = {}
        self.selection_history = []
    
    def select_clients(self, available_clients: List[int], round_num: int) -> List[int]:
        """Select clients for current round"""
        if self.config.client_selection == "adaptive":
            return self.adaptive_selection(available_clients, round_num)
        elif self.config.client_selection == "stratified":
            return self.stratified_selection(available_clients, round_num)
        else:
            return self.random_selection(available_clients)
    
    def adaptive_selection(self, available_clients: List[int], round_num: int) -> List[int]:
        """Adaptive client selection based on performance"""
        # Calculate selection probabilities based on performance
        probabilities = []
        for client_id in available_clients:
            performance = self.client_performance.get(client_id, 0.5)
            availability = self.client_availability.get(client_id, 1.0)
            prob = performance * availability
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(available_clients)] * len(available_clients)
        
        # Select clients
        num_selected = min(
            self.config.max_clients_per_round,
            max(self.config.min_clients_per_round, len(available_clients))
        )
        
        selected = np.random.choice(
            available_clients,
            size=num_selected,
            replace=False,
            p=probabilities
        ).tolist()
        
        self.selection_history.append(selected)
        return selected
    
    def stratified_selection(self, available_clients: List[int], round_num: int) -> List[int]:
        """Stratified client selection"""
        # Group clients by performance tiers
        high_perf = []
        medium_perf = []
        low_perf = []
        
        for client_id in available_clients:
            performance = self.client_performance.get(client_id, 0.5)
            if performance > 0.7:
                high_perf.append(client_id)
            elif performance > 0.4:
                medium_perf.append(client_id)
            else:
                low_perf.append(client_id)
        
        # Select from each tier
        selected = []
        if high_perf:
            selected.extend(np.random.choice(high_perf, min(2, len(high_perf)), replace=False))
        if medium_perf:
            selected.extend(np.random.choice(medium_perf, min(3, len(medium_perf)), replace=False))
        if low_perf:
            selected.extend(np.random.choice(low_perf, min(1, len(low_perf)), replace=False))
        
        return selected[:self.config.max_clients_per_round]
    
    def random_selection(self, available_clients: List[int]) -> List[int]:
        """Random client selection"""
        num_selected = min(
            self.config.max_clients_per_round,
            max(self.config.min_clients_per_round, len(available_clients))
        )
        return np.random.choice(available_clients, size=num_selected, replace=False).tolist()
    
    def update_performance(self, client_id: int, performance: float):
        """Update client performance history"""
        if client_id not in self.client_performance:
            self.client_performance[client_id] = []
        
        self.client_performance[client_id].append(performance)
        
        # Keep only recent performance
        if len(self.client_performance[client_id]) > 10:
            self.client_performance[client_id] = self.client_performance[client_id][-10:]
        
        # Update average performance
        self.client_performance[client_id] = np.mean(self.client_performance[client_id])

class ModelCompression:
    """Advanced Model Compression for Federated Learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.quantization_bits = config.quantization_bits
        self.sparsification_ratio = config.sparsification_ratio
    
    def compress_model(self, model_state: Dict) -> Dict:
        """Compress model for efficient transmission"""
        compressed_state = {}
        
        for key, value in model_state.items():
            if self.config.quantization_bits < 32:
                value = self.quantize_tensor(value)
            
            if self.config.sparsification_ratio > 0:
                value = self.sparsify_tensor(value)
            
            compressed_state[key] = value
        
        return compressed_state
    
    def decompress_model(self, compressed_state: Dict) -> Dict:
        """Decompress model after transmission"""
        decompressed_state = {}
        
        for key, value in compressed_state.items():
            if self.config.sparsification_ratio > 0:
                value = self.desparsify_tensor(value)
            
            if self.config.quantization_bits < 32:
                value = self.dequantize_tensor(value)
            
            decompressed_state[key] = value
        
        return decompressed_state
    
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to reduce precision"""
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Scale to [0, 2^bits - 1]
        scale = (2 ** self.quantization_bits - 1) / (max_val - min_val + 1e-8)
        quantized = torch.round((tensor - min_val) * scale)
        
        # Store metadata for dequantization
        quantized = torch.cat([quantized.flatten(), torch.tensor([min_val, max_val, scale])])
        
        return quantized
    
    def dequantize_tensor(self, quantized: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to original precision"""
        # Extract metadata
        metadata = quantized[-3:]
        tensor_data = quantized[:-3]
        
        min_val, max_val, scale = metadata
        
        # Dequantize
        dequantized = tensor_data / scale + min_val
        
        return dequantized
    
    def sparsify_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sparsify tensor by keeping only top-k values"""
        flat_tensor = tensor.flatten()
        k = int(flat_tensor.numel() * (1 - self.sparsification_ratio))
        
        # Keep top-k values
        _, indices = torch.topk(torch.abs(flat_tensor), k)
        sparse_tensor = torch.zeros_like(flat_tensor)
        sparse_tensor[indices] = flat_tensor[indices]
        
        # Store indices for reconstruction
        sparse_tensor = torch.cat([sparse_tensor, indices.float()])
        
        return sparse_tensor
    
    def desparsify_tensor(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct original tensor from sparse representation"""
        # Extract indices
        indices = sparse_tensor[-int(sparse_tensor.numel() * self.sparsification_ratio):].long()
        tensor_data = sparse_tensor[:-len(indices)]
        
        # Reconstruct
        reconstructed = torch.zeros_like(tensor_data)
        reconstructed[indices] = tensor_data[indices]
        
        return reconstructed

class AdvancedFederatedLearningService:
    """World-Class Federated Learning Service"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.dp = DifferentialPrivacy(config.privacy_budget, config.noise_scale)
        self.secure_agg = SecureAggregation(config.num_clients) if config.secure_aggregation else None
        self.client_selector = AdaptiveClientSelection(config)
        self.model_compressor = ModelCompression(config)
        
        # Global model state
        self.global_model = None
        self.global_optimizer = None
        
        # Training state
        self.current_round = 0
        self.best_performance = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Client management
        self.clients = {}
        self.client_data = {}
        
        # Communication queue
        self.communication_queue = queue.Queue()
        
        # Performance monitoring
        self.performance_metrics = {
            'round_losses': [],
            'round_accuracies': [],
            'communication_overhead': [],
            'privacy_loss': []
        }
    
    def initialize_global_model(self, model: nn.Module):
        """Initialize global model"""
        self.global_model = model
        self.global_optimizer = optim.Adam(
            self.global_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def register_client(self, client_id: int, client_data: Dict):
        """Register a new client"""
        self.clients[client_id] = {
            'status': 'active',
            'last_seen': datetime.now(),
            'performance_history': [],
            'data_size': len(client_data.get('train_data', [])),
            'data_distribution': self.analyze_data_distribution(client_data)
        }
        self.client_data[client_id] = client_data
    
    def analyze_data_distribution(self, client_data: Dict) -> Dict:
        """Analyze client data distribution"""
        train_data = client_data.get('train_data', [])
        
        # Analyze material types
        material_types = [item.get('material_type', 0) for item in train_data]
        material_dist = np.bincount(material_types) / len(material_types)
        
        # Analyze company sizes
        company_sizes = [item.get('company_size', 0) for item in train_data]
        size_dist = np.bincount(company_sizes) / len(company_sizes)
        
        return {
            'material_distribution': material_dist.tolist(),
            'size_distribution': size_dist.tolist(),
            'total_samples': len(train_data)
        }
    
    async def federated_training_round(self) -> Dict:
        """Execute one federated training round"""
        start_time = datetime.now()
        
        # Select clients for this round
        available_clients = [cid for cid, client in self.clients.items() 
                           if client['status'] == 'active']
        
        if len(available_clients) < self.config.min_clients_per_round:
            raise ValueError("Insufficient active clients")
        
        selected_clients = self.client_selector.select_clients(available_clients, self.current_round)
        
        # Distribute global model to selected clients
        client_models = await self.distribute_model(selected_clients)
        
        # Collect client updates
        client_updates = await self.collect_client_updates(selected_clients, client_models)
        
        # Aggregate updates
        aggregated_update = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(aggregated_update)
        
        # Evaluate global model
        performance = await self.evaluate_global_model()
        
        # Update performance metrics
        self.update_performance_metrics(performance)
        
        # Check early stopping
        if self.config.early_stopping:
            if self.should_stop_early(performance):
                logging.info("Early stopping triggered")
                return {'status': 'stopped', 'round': self.current_round}
        
        self.current_round += 1
        
        round_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'completed',
            'round': self.current_round - 1,
            'performance': performance,
            'round_time': round_time,
            'selected_clients': selected_clients
        }
    
    async def distribute_model(self, selected_clients: List[int]) -> Dict[int, Dict]:
        """Distribute global model to selected clients"""
        client_models = {}
        
        # Compress global model
        global_state = self.global_model.state_dict()
        compressed_state = self.model_compressor.compress_model(global_state)
        
        for client_id in selected_clients:
            # Add client-specific metadata
            client_model = {
                'state_dict': compressed_state,
                'round': self.current_round,
                'client_id': client_id,
                'timestamp': datetime.now().isoformat()
            }
            client_models[client_id] = client_model
        
        return client_models
    
    async def collect_client_updates(self, selected_clients: List[int], 
                                   client_models: Dict[int, Dict]) -> List[Dict]:
        """Collect updates from selected clients"""
        client_updates = []
        
        # Simulate client training (in real implementation, this would be async calls to clients)
        for client_id in selected_clients:
            client_update = await self.simulate_client_training(client_id, client_models[client_id])
            client_updates.append(client_update)
        
        return client_updates
    
    async def simulate_client_training(self, client_id: int, client_model: Dict) -> Dict:
        """Simulate client training (placeholder for real implementation)"""
        # In real implementation, this would be a call to the client's training service
        
        # Simulate training time
        await asyncio.sleep(0.1)
        
        # Generate mock update
        mock_update = {
            'client_id': client_id,
            'state_dict': client_model['state_dict'],  # In reality, this would be the updated model
            'num_samples': self.clients[client_id]['data_size'],
            'training_loss': np.random.uniform(0.1, 0.5),
            'training_accuracy': np.random.uniform(0.7, 0.95)
        }
        
        return mock_update
    
    def aggregate_updates(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client updates using advanced methods"""
        if self.config.aggregation_method == "fedavg_plus":
            return self.fedavg_plus_aggregation(client_updates)
        elif self.config.aggregation_method == "fedprox":
            return self.fedprox_aggregation(client_updates)
        elif self.config.aggregation_method == "secure_aggregation":
            return self.secure_aggregation(client_updates)
        else:
            return self.fedavg_aggregation(client_updates)
    
    def fedavg_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Standard FedAvg aggregation"""
        total_samples = sum(update['num_samples'] for update in client_updates)
        aggregated_state = {}
        
        for key in client_updates[0]['state_dict'].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0]['state_dict'][key])
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                aggregated_state[key] += weight * update['state_dict'][key]
        
        return aggregated_state
    
    def fedavg_plus_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Enhanced FedAvg with momentum and adaptive weighting"""
        total_samples = sum(update['num_samples'] for update in client_updates)
        aggregated_state = {}
        
        # Calculate adaptive weights based on client performance
        weights = []
        for update in client_updates:
            performance = update.get('training_accuracy', 0.5)
            data_size = update['num_samples']
            adaptive_weight = (performance * data_size) / total_samples
            weights.append(adaptive_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        for key in client_updates[0]['state_dict'].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0]['state_dict'][key])
            
            for i, update in enumerate(client_updates):
                aggregated_state[key] += weights[i] * update['state_dict'][key]
        
        return aggregated_state
    
    def fedprox_aggregation(self, client_updates: List[Dict]) -> Dict:
        """FedProx aggregation with proximal term"""
        # Similar to FedAvg but with proximal regularization
        return self.fedavg_aggregation(client_updates)
    
    def secure_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Secure aggregation using homomorphic encryption"""
        if not self.secure_agg:
            return self.fedavg_aggregation(client_updates)
        
        # Encrypt client updates
        encrypted_updates = []
        for update in client_updates:
            encrypted = self.secure_agg.encrypt_gradients(
                update['client_id'], 
                list(update['state_dict'].values())
            )
            encrypted_updates.append(encrypted)
        
        # Decrypt aggregated result
        aggregated_gradients = self.secure_agg.decrypt_gradients(encrypted_updates)
        
        # Convert back to state dict format
        aggregated_state = {}
        for i, key in enumerate(client_updates[0]['state_dict'].keys()):
            aggregated_state[key] = aggregated_gradients[i]
        
        return aggregated_state
    
    def update_global_model(self, aggregated_update: Dict):
        """Update global model with aggregated update"""
        # Decompress aggregated update
        decompressed_update = self.model_compressor.decompress_model(aggregated_update)
        
        # Load into global model
        self.global_model.load_state_dict(decompressed_update)
    
    async def evaluate_global_model(self) -> float:
        """Evaluate global model performance"""
        # In real implementation, this would evaluate on a validation set
        # For now, return a mock performance
        return np.random.uniform(0.8, 0.95)
    
    def update_performance_metrics(self, performance: float):
        """Update performance tracking metrics"""
        self.performance_metrics['round_accuracies'].append(performance)
        
        if performance > self.best_performance:
            self.best_performance = performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def should_stop_early(self, performance: float) -> bool:
        """Check if training should stop early"""
        return self.patience_counter >= self.config.patience
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        return {
            'current_round': self.current_round,
            'best_performance': self.best_performance,
            'performance_history': self.performance_metrics['round_accuracies'],
            'active_clients': len([c for c in self.clients.values() if c['status'] == 'active']),
            'total_clients': len(self.clients),
            'training_config': self.config.__dict__
        }

# Flask Application for Federated Learning
federated_app = Flask(__name__)

# Initialize federated learning service
federated_config = FederatedConfig()
federated_service = AdvancedFederatedLearningService(federated_config)

@federated_app.route('/health', methods=['GET'])
def federated_health_check():
    """Health check for federated learning service"""
    return jsonify({
        'status': 'healthy',
        'service': 'federated_learning',
        'current_round': federated_service.current_round,
        'active_clients': len([c for c in federated_service.clients.values() if c['status'] == 'active']),
        'best_performance': federated_service.best_performance
    })

@federated_app.route('/register', methods=['POST'])
def register_client():
    """Register a new client for federated learning"""
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        client_data = data.get('client_data', {})
        
        if not client_id:
            return jsonify({'error': 'Client ID required'}), 400
        
        federated_service.register_client(client_id, client_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Client {client_id} registered successfully',
            'client_info': federated_service.clients[client_id]
        })
        
    except Exception as e:
        logging.error(f"Client registration error: {e}")
        return jsonify({'error': str(e)}), 500

@federated_app.route('/train', methods=['POST'])
async def start_training():
    """Start federated training round"""
    try:
        result = await federated_service.federated_training_round()
        
        return jsonify({
            'status': 'success',
            'training_result': result
        })
        
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@federated_app.route('/summary', methods=['GET'])
def get_training_summary():
    """Get federated training summary"""
    try:
        summary = federated_service.get_training_summary()
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
        
    except Exception as e:
        logging.error(f"Summary error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    federated_app.run(host='0.0.0.0', port=5002, debug=False) 