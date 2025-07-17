#!/usr/bin/env python3
"""
Logistics Service Wrapper
Flask wrapper for the logistics cost engine service
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

# Add parent directory to path to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logistics_service.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global flag to track service status
service_healthy = True
startup_time = datetime.now()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy' if service_healthy else 'unhealthy',
            'service': 'logistics_service',
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - startup_time),
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/logistics/route-planning', methods=['POST'])
def route_planning():
    """Plan optimal logistics routes"""
    try:
        data = request.get_json()
        logger.info(f"Route planning request: {data}")
        
        # Mock route planning response
        response = {
            'optimal_route': {
                'distance_km': 1250,
                'estimated_time_hours': 18,
                'cost_usd': 850,
                'route_points': [
                    {'lat': 25.2048, 'lng': 55.2708, 'name': 'Dubai'},
                    {'lat': 24.7136, 'lng': 46.6753, 'name': 'Riyadh'},
                    {'lat': 21.2703, 'lng': -157.8083, 'name': 'Honolulu'}
                ]
            },
            'alternatives': [
                {
                    'distance_km': 1400,
                    'estimated_time_hours': 22,
                    'cost_usd': 920,
                    'reason': 'Avoiding high traffic areas'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Route planning failed: {e}")
        return jsonify({
            'error': 'Route planning failed',
            'message': str(e)
        }), 500

@app.route('/api/logistics/cost-calculation', methods=['POST'])
def cost_calculation():
    """Calculate logistics costs"""
    try:
        data = request.get_json()
        logger.info(f"Cost calculation request: {data}")
        
        # Mock cost calculation
        response = {
            'total_cost': 1250.0,
            'breakdown': {
                'transportation': 850.0,
                'handling': 150.0,
                'insurance': 75.0,
                'customs': 100.0,
                'storage': 75.0
            },
            'currency': 'USD',
            'valid_until': (datetime.now().replace(hour=23, minute=59, second=59)).isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Cost calculation failed: {e}")
        return jsonify({
            'error': 'Cost calculation failed',
            'message': str(e)
        }), 500

@app.route('/api/logistics/shipping-rates', methods=['POST'])
def shipping_rates():
    """Get shipping rates"""
    try:
        data = request.get_json()
        logger.info(f"Shipping rates request: {data}")
        
        # Mock shipping rates
        response = {
            'rates': [
                {
                    'service': 'Express',
                    'carrier': 'DHL',
                    'cost': 450.0,
                    'delivery_time': '2-3 days',
                    'tracking': True
                },
                {
                    'service': 'Standard',
                    'carrier': 'FedEx',
                    'cost': 320.0,
                    'delivery_time': '5-7 days',
                    'tracking': True
                },
                {
                    'service': 'Economy',
                    'carrier': 'UPS',
                    'cost': 280.0,
                    'delivery_time': '8-12 days',
                    'tracking': True
                }
            ],
            'currency': 'USD',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Shipping rates failed: {e}")
        return jsonify({
            'error': 'Shipping rates failed',
            'message': str(e)
        }), 500

@app.route('/api/logistics/carbon-footprint', methods=['POST'])
def carbon_footprint():
    """Calculate carbon footprint for logistics"""
    try:
        data = request.get_json()
        logger.info(f"Carbon footprint request: {data}")
        
        # Mock carbon footprint calculation
        response = {
            'total_co2_kg': 125.5,
            'breakdown': {
                'transportation': 95.2,
                'packaging': 15.3,
                'warehousing': 10.0,
                'last_mile': 5.0
            },
            'offset_cost_usd': 12.55,
            'recommendations': [
                'Use electric vehicles for last-mile delivery',
                'Optimize route planning to reduce distance',
                'Consider rail transport for long distances'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Carbon footprint calculation failed: {e}")
        return jsonify({
            'error': 'Carbon footprint calculation failed',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def start_service():
    """Start the logistics service"""
    global service_healthy
    
    try:
        logger.info("Starting Logistics Service...")
        logger.info("Service will be available at: http://localhost:5006")
        
        # Set service as healthy
        service_healthy = True
        
        # Start Flask app
        app.run(
            host='0.0.0.0',
            port=5006,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start logistics service: {e}")
        service_healthy = False
        raise

if __name__ == '__main__':
    start_service() 