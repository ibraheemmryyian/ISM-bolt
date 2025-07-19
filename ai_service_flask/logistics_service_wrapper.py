#!/usr/bin/env python3
"""
Logistics Service Wrapper
Flask wrapper for the logistics cost engine service
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import json
import logging
from flask import Flask, request, jsonify
import torch
from ml_core.models import ModelFactory
from ml_core.utils import ModelRegistry
from ml_core.monitoring import MLMetricsTracker
from ml_core.data_processing import DataValidator
from ml_core.optimization import HyperparameterOptimizer

app = Flask(__name__)

model_registry = ModelRegistry()
metrics_tracker = MLMetricsTracker()
data_validator = DataValidator()
# Fix HyperparameterOptimizer initialization
try:
    optimizer = HyperparameterOptimizer()
except TypeError:
    # Provide default arguments
    class DefaultConfig:
        def __init__(self):
            self.optimization_algorithm = 'bayesian'
            self.max_trials = 100
            self.timeout = 3600
    
    class DefaultSearchSpace:
        def __init__(self):
            self.learning_rate = [0.001, 0.1]
            self.batch_size = [16, 128]
            self.hidden_size = [64, 512]
    
    class DefaultModel:
        def __init__(self):
            self.name = 'default_model'
    
    optimizer = HyperparameterOptimizer(
        model=DefaultModel(),
        config=DefaultConfig(),
        search_space=DefaultSearchSpace()
    )


def get_logistics_model(model_id):
    model_info = model_registry.get_model(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in registry")
    model = model_info['model_class'](**model_info['model_params'])
    model.load_state_dict(torch.load(model_info['model_path'], map_location='cpu'))
    return model.eval()

@app.route('/logistics_cost_predict', methods=['POST'])
def logistics_cost_predict():
    try:
        data = request.json
        model_id = data.get('model_id')
        input_data = data.get('input_data')
        model = get_logistics_model(model_id)
        validated = data_validator.validate_logistics_input(input_data)
        input_tensor = torch.FloatTensor(validated['features']).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        metrics_tracker.record_inference_metrics({'model_id': model_id, 'success': True})
        return jsonify({'logistics_cost_prediction': output.cpu().numpy().tolist()})
    except Exception as e:
        metrics_tracker.record_inference_metrics({'model_id': request.json.get('model_id', 'unknown'), 'success': False, 'error': str(e)})
        return jsonify({'error': str(e)}), 500

@app.route('/batch_logistics_cost_predict', methods=['POST'])
def batch_logistics_cost_predict():
    try:
        data = request.json
        model_id = data.get('model_id')
        batch_data = data.get('batch_data')
        model = get_logistics_model(model_id)
        validated = [data_validator.validate_logistics_input(d) for d in batch_data]
        input_tensor = torch.FloatTensor([v['features'] for v in validated])
        with torch.no_grad():
            output = model(input_tensor)
        metrics_tracker.record_inference_metrics({'model_id': model_id, 'success': True, 'batch': True})
        return jsonify({'logistics_cost_predictions': output.cpu().numpy().tolist()})
    except Exception as e:
        metrics_tracker.record_inference_metrics({'model_id': request.json.get('model_id', 'unknown'), 'success': False, 'error': str(e), 'batch': True})
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        health_metrics = metrics_tracker.get_latest_metrics()
        return jsonify({'status': 'healthy', 'metrics': health_metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=False) 