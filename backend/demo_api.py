#!/usr/bin/env python3
"""
Simple Demo API for SymbioFlows
Flask API wrapper for the demo service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from demo_simple_service import SimpleDemoService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize the demo service
demo_service = SimpleDemoService()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'SymbioFlows Demo API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/api/materials': 'Get all materials',
            '/api/companies': 'Get all companies',
            '/api/matches/<material_id>': 'Get matches for a material',
            '/api/statistics': 'Get match statistics',
            '/api/materials/add': 'Add new material (POST)'
        }
    })

@app.route('/api/materials', methods=['GET'])
def get_materials():
    """Get all materials"""
    try:
        materials = demo_service.get_all_materials()
        return jsonify({
            'success': True,
            'data': materials,
            'count': len(materials)
        })
    except Exception as e:
        logger.error(f"Error getting materials: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/companies', methods=['GET'])
def get_companies():
    """Get all companies"""
    try:
        companies = demo_service.get_all_companies()
        return jsonify({
            'success': True,
            'data': companies,
            'count': len(companies)
        })
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/matches/<material_id>', methods=['GET'])
def get_matches(material_id):
    """Get matches for a specific material"""
    try:
        matches = demo_service.generate_matches(material_id)
        return jsonify({
            'success': True,
            'data': matches,
            'material_id': material_id,
            'count': len(matches)
        })
    except Exception as e:
        logger.error(f"Error getting matches for {material_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get match statistics"""
    try:
        stats = demo_service.get_match_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/materials/add', methods=['POST'])
def add_material():
    """Add a new material"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        result = demo_service.add_material(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error adding material: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/<format>', methods=['GET'])
def export_data(format):
    """Export data in specified format"""
    try:
        if format not in ['json', 'csv']:
            return jsonify({
                'success': False,
                'error': 'Format must be json or csv'
            }), 400
        
        file_path = demo_service.export_data(format)
        return jsonify({
            'success': True,
            'file_path': file_path,
            'format': format
        })
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SymbioFlows Demo API',
        'timestamp': '2025-07-29T01:43:53Z'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting SymbioFlows Demo API...")
    print("📍 API will be available at: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)