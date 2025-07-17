import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import shap
from ml_core.models import ModelFactory
from ml_core.utils import ModelRegistry
from ml_core.monitoring import MLMetricsTracker
from ml_core.optimization import HyperparameterOptimizer
from backend.utils.distributed_logger import DistributedLogger
from backend.utils.advanced_data_validator import AdvancedDataValidator

app = Flask(__name__)
api = Api(app, version='1.0', title='AI Matchmaking Service', description='Insanely Advanced Modular ML Matchmaking', doc='/docs')

logger = DistributedLogger('AIMatchmakingService', log_file='logs/ai_matchmaking_service.log')
model_registry = ModelRegistry()
metrics_tracker = MLMetricsTracker()
optimizer = HyperparameterOptimizer()
data_validator = AdvancedDataValidator(logger=logger)

def get_model(model_id):
    model_info = model_registry.get_model(model_id)
    if not model_info:
        logger.error(f"Model {model_id} not found in registry")
        raise ValueError(f"Model {model_id} not found in registry")
    model = model_info['model_class'](**model_info['model_params'])
    model.load_state_dict(torch.load(model_info['model_path'], map_location='cpu'))
    return model.eval()

match_input = api.model('MatchInput', {
    'model_id': fields.String(required=True, description='Model identifier'),
    'input_data': fields.Raw(required=True, description='Input data for matchmaking (graph or features)')
})

@api.route('/match')
class Match(Resource):
    @api.expect(match_input)
    @api.response(200, 'Success')
    @api.response(400, 'Invalid input data')
    @api.response(500, 'Internal error')
    def post(self):
        try:
            data = request.json
            model_id = data.get('model_id')
            input_data = data.get('input_data')
            # Schema-based validation
            schema = {
                'type': 'object',
                'properties': {
                    'features': {'type': 'array'},
                    'graph': {'type': 'object'}
                },
                'anyOf': [
                    {'required': ['features']},
                    {'required': ['graph']}
                ]
            }
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error("Input data failed schema validation.")
                return {'error': 'Invalid input data'}, 400
            model = get_model(model_id)
            # Use GNN if graph, else transformer/MLP
            if 'graph' in input_data:
                # Assume input_data['graph'] is a torch_geometric.data.Data object
                graph_data = input_data['graph']
                with torch.no_grad():
                    output = model(graph_data)
            else:
                features = torch.FloatTensor(input_data['features']).unsqueeze(0)
                with torch.no_grad():
                    output = model(features)
            metrics_tracker.record_inference_metrics({'model_id': model_id, 'success': True})
            logger.info(f"Matchmaking successful for model {model_id}")
            return {'match_result': output.cpu().numpy().tolist()}
        except Exception as e:
            logger.error(f"Matchmaking error: {e}")
            metrics_tracker.record_inference_metrics({'model_id': request.json.get('model_id', 'unknown'), 'success': False, 'error': str(e)})
            return {'error': str(e)}, 500

@api.route('/explain')
class Explain(Resource):
    @api.expect(match_input)
    @api.response(200, 'Success')
    @api.response(400, 'Invalid input data')
    @api.response(500, 'Internal error')
    def post(self):
        try:
            data = request.json
            model_id = data.get('model_id')
            input_data = data.get('input_data')
            schema = {
                'type': 'object',
                'properties': {
                    'features': {'type': 'array'},
                    'graph': {'type': 'object'}
                },
                'anyOf': [
                    {'required': ['features']},
                    {'required': ['graph']}
                ]
            }
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error("Input data failed schema validation.")
                return {'error': 'Invalid input data'}, 400
            model = get_model(model_id)
            if 'features' in input_data:
                features = torch.FloatTensor(input_data['features']).unsqueeze(0)
                explainer = shap.Explainer(model, features)
                shap_values = explainer(features)
                logger.info(f"Explanation generated for model {model_id}")
                return {'shap_values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist()}
            else:
                return {'error': 'Explainability for GNN graphs not yet implemented'}, 400
        except Exception as e:
            logger.error(f"Explainability error: {e}")
            return {'error': str(e)}, 500

@api.route('/train')
class Train(Resource):
    @api.expect(match_input)
    @api.response(200, 'Training started')
    @api.response(500, 'Error')
    def post(self):
        try:
            data = request.json
            model_id = data.get('model_id')
            input_data = data.get('input_data')
            # Assume input_data contains 'features' and 'labels' or 'graph' and 'labels'
            model = get_model(model_id)
            if 'features' in input_data:
                features = torch.FloatTensor(input_data['features'])
                labels = torch.FloatTensor(input_data['labels'])
                dataset = TensorDataset(features, labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)
                optimizer_ = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.BCEWithLogitsLoss()
                for epoch in range(10):
                    for X, y in loader:
                        optimizer_.zero_grad()
                        output = model(X)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer_.step()
                logger.info(f"Training completed for model {model_id}")
            elif 'graph' in input_data:
                # Placeholder for GNN training
                logger.info(f"GNN training for model {model_id} (not implemented in this stub)")
            return {'status': 'training started', 'model_id': model_id}
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'error': str(e)}, 500

@api.route('/optimize')
class Optimize(Resource):
    @api.expect(match_input)
    @api.response(200, 'Optimization started')
    @api.response(500, 'Error')
    def post(self):
        try:
            data = request.json
            model_id = data.get('model_id')
            input_data = data.get('input_data')
            logger.info(f"Optimization requested for model {model_id}")
            model_info = model_registry.get_model(model_id)
            optimizer.optimize_hyperparameters(
                model_type=model_info['model_type'],
                training_data=input_data.get('training_data'),
                validation_data=input_data.get('validation_data'),
                optimization_strategy='bayesian'
            )
            return {'status': 'optimization started', 'model_id': model_id}
            except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'error': str(e)}, 500

@api.route('/health')
class Health(Resource):
    @api.response(200, 'Healthy')
    @api.response(500, 'Error')
    def get(self):
        try:
            return {'status': 'healthy'}
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {'status': 'error', 'error': str(e)}, 500

if __name__ == '__main__':
    logger.info("Starting AI Matchmaking Service...")
    app.run(host='0.0.0.0', port=8020, debug=False) 