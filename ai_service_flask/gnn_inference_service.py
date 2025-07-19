"""
World-Class GNN Inference Service
Advanced Industrial Symbiosis Graph Neural Network with Multi-Modal Processing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import json
import logging
from flask import Flask, request, jsonify
import torch
from torch_geometric.nn import GCNConv, GATConv
from ml_core.models import ModelFactory
from ml_core.utils import ModelRegistry
from ml_core.monitoring import MLMetricsTracker
from ml_core.data_processing import DataValidator

app = Flask(__name__)

model_registry = ModelRegistry()
metrics_tracker = MLMetricsTracker()
data_validator = DataValidator()

def get_gnn_model(model_id):
    model_info = model_registry.get_model(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in registry")
    model = model_info['model_class'](**model_info['model_params'])
    model.load_state_dict(torch.load(model_info['model_path'], map_location='cpu'))
    return model.eval()

@app.route('/gnn_inference', methods=['POST'])
def gnn_inference():
    try:
        data = request.json
        model_id = data.get('model_id')
        graph_data = data.get('graph_data')
        model = get_gnn_model(model_id)
        validated = data_validator.validate_gnn_input(graph_data)
        # Assume validated is a torch_geometric.data.Data object
        with torch.no_grad():
            output = model(validated)
        metrics_tracker.record_inference_metrics({'model_id': model_id, 'success': True})
        return jsonify({'result': output.cpu().numpy().tolist()})
    except Exception as e:
        metrics_tracker.record_inference_metrics({'model_id': request.json.get('model_id', 'unknown'), 'success': False, 'error': str(e)})
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        health_metrics = metrics_tracker.get_latest_metrics()
        return jsonify({'status': 'healthy', 'metrics': health_metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False) 