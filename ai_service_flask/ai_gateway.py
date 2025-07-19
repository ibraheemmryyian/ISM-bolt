"""
Advanced AI Service Gateway
Orchestrates all AI services with intelligent routing, load balancing, and health monitoring
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import json
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from ml_core.models import ModelFactory
from ml_core.utils import ModelRegistry
from ml_core.monitoring import MLMetricsTracker
# Fix import error
try:
    from backend.utils.distributed_logger import DistributedLogger
except ImportError:
    # Fallback implementation
    import logging
    class DistributedLogger:
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(__name__)
        def info(self, msg):
            self.logger.info(msg)
        def error(self, msg):
            self.logger.error(msg)
        def warning(self, msg):
            self.logger.warning(msg)
from backend.utils.advanced_data_validator import AdvancedDataValidator
from ml_core.optimization import HyperparameterOptimizer
import shap
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='AI Gateway API', description='Revolutionary Modular ML Gateway', doc='/docs')

# Initialize advanced utilities
logger = DistributedLogger('AIGateway', log_file='logs/ai_gateway.log')
model_registry = ModelRegistry()
metrics_tracker = MLMetricsTracker()
data_validator = AdvancedDataValidator(logger=logger)
optimizer = HyperparameterOptimizer()
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.pad_token = tokenizer.eos_token

inference_input = api.model('InferenceInput', {
    'model_id': fields.String(required=True, description='Model identifier'),
    'input_data': fields.Raw(required=True, description='Input data for inference')
})

explain_input = api.model('ExplainInput', {
    'model_id': fields.String(required=True, description='Model identifier'),
    'input_data': fields.Raw(required=True, description='Input data for explanation')
})

def get_model(model_id):
    model_info = model_registry.get_model(model_id)
    if not model_info:
        logger.error(f"Model {model_id} not found in registry")
        raise ValueError(f"Model {model_id} not found in registry")
    model = model_info['model_class'](**model_info['model_params'])
    model.load_state_dict(torch.load(model_info['model_path'], map_location='cpu'))
    return model.eval()

@api.route('/inference')
class Inference(Resource):
    @api.expect(inference_input)
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
                    'text': {'type': 'string'},
                    'features': {'type': 'array'}
                },
                'anyOf': [
                    {'required': ['text']},
                    {'required': ['features']}
                ]
            }
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error("Input data failed schema validation.")
                return {'error': 'Invalid input data'}, 400
            model = get_model(model_id)
            # Tokenize if text
            if 'text' in input_data:
                encoding = tokenizer(input_data['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
                input_tensor = encoding['input_ids']
            else:
                input_tensor = torch.FloatTensor(input_data['features']).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            metrics_tracker.record_inference_metrics({'model_id': model_id, 'success': True})
            logger.info(f"Inference successful for model {model_id}")
            return {'result': output.cpu().numpy().tolist()}
        except Exception as e:
            logger.error(f"Inference error: {e}")
            metrics_tracker.record_inference_metrics({'model_id': request.json.get('model_id', 'unknown'), 'success': False, 'error': str(e)})
            return {'error': str(e)}, 500

@api.route('/explain')
class Explain(Resource):
    @api.expect(explain_input)
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
                    'text': {'type': 'string'},
                    'features': {'type': 'array'}
                },
                'anyOf': [
                    {'required': ['text']},
                    {'required': ['features']}
                ]
            }
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error("Input data failed schema validation.")
                return {'error': 'Invalid input data'}, 400
            model = get_model(model_id)
            # Tokenize if text
            if 'text' in input_data:
                encoding = tokenizer(input_data['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
                input_tensor = encoding['input_ids']
            else:
                input_tensor = torch.FloatTensor(input_data['features']).unsqueeze(0)
            # SHAP explainability
            explainer = shap.Explainer(model, input_tensor)
            shap_values = explainer(input_tensor)
            logger.info(f"Explanation generated for model {model_id}")
            return {'shap_values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist()}
        except Exception as e:
            logger.error(f"Explainability error: {e}")
            return {'error': str(e)}, 500

@api.route('/optimize')
class Optimize(Resource):
    @api.expect(inference_input)
    @api.response(200, 'Success')
    @api.response(500, 'Internal error')
    def post(self):
        try:
            data = request.json
            model_type = data.get('model_type')
            training_data = data.get('training_data')
            validation_data = data.get('validation_data')
            strategy = data.get('strategy', 'bayesian')
            logger.info(f"Starting hyperparameter optimization for {model_type} with strategy {strategy}")
            result = optimizer.optimize_hyperparameters(
                model_type=model_type,
                training_data=training_data,
                validation_data=validation_data,
                optimization_strategy=strategy
            )
            logger.info(f"Optimization completed for {model_type}")
            return {'optimization_result': result}
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'error': str(e)}, 500

@api.route('/health')
class Health(Resource):
    @api.response(200, 'Healthy')
    @api.response(500, 'Error')
    def get(self):
        try:
            health_metrics = metrics_tracker.get_latest_metrics()
            logger.info("Health check successful.")
            return {'status': 'healthy', 'metrics': health_metrics}
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {'status': 'error', 'error': str(e)}, 500

if __name__ == '__main__':
    logger.info("Starting AI Gateway service...")
    app.run(host='0.0.0.0', port=8000, debug=False) 