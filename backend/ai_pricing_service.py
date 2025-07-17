"""
AI Pricing Service - Flask wrapper for AI pricing integration
Provides REST API endpoints for pricing validation and integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime

# Import the pricing integration modules
try:
    from ai_pricing_orchestrator import AI_PricingOrchestrator
    # Temporarily disable pricing integration manager to avoid circular imports
    # from ai_pricing_integration import PricingIntegrationManager, pricing_integration_manager
except ImportError as e:
    logging.error(f"Failed to import pricing modules: {e}")

# Create dummy classes for fallback
class PricingIntegrationManager:
    def __init__(self):
        self.integrated_modules = {}
        self.integration_status = {}
    
    def get_integration_status(self):
        return {"error": "Pricing integration manager not available due to circular imports"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize pricing services
try:
    pricing_manager = PricingIntegrationManager()
    pricing_orchestrator = AI_PricingOrchestrator()
    logger.info("AI Pricing Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pricing services: {e}")
    pricing_manager = None
    pricing_orchestrator = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ai_pricing_service",
        "timestamp": datetime.utcnow().isoformat(),
        "pricing_manager_available": pricing_manager is not None,
        "pricing_orchestrator_available": pricing_orchestrator is not None
    })

@app.route('/api/pricing/validate', methods=['POST'])
def validate_pricing():
    """Validate pricing for a match"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        material = data.get('material')
        quantity = data.get('quantity', 1.0)
        quality = data.get('quality', 'clean')
        source_location = data.get('source_location', 'unknown')
        destination_location = data.get('destination_location', 'unknown')
        proposed_price = data.get('proposed_price')
        
        if not all([material, proposed_price]):
            return jsonify({"error": "Missing required fields: material, proposed_price"}), 400
        
        if pricing_orchestrator:
            validation = pricing_orchestrator.validate_match_pricing(
                material, quantity, quality, source_location, destination_location, proposed_price
            )
            
            return jsonify({
                "success": True,
                "is_valid": validation.is_valid,
                "reason": validation.reason,
                "required_adjustments": validation.required_adjustments,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Pricing orchestrator not available",
                "is_valid": False
            }), 503
            
    except Exception as e:
        logger.error(f"Error in pricing validation: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "is_valid": False
        }), 500

@app.route('/api/pricing/material/<material>', methods=['GET'])
def get_material_pricing(material):
    """Get pricing data for a material"""
    try:
        if not pricing_orchestrator:
            return jsonify({
                "success": False,
                "error": "Pricing orchestrator not available"
            }), 503
        
        price_data = pricing_orchestrator.get_material_price(material)
        
        if price_data:
            return jsonify({
                "success": True,
                "material": material,
                "price_data": {
                    "price": price_data.price,
                    "currency": price_data.currency,
                    "source": price_data.source.value,
                    "timestamp": price_data.timestamp.isoformat(),
                    "confidence": price_data.confidence
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": f"No pricing data available for {material}"
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting material pricing: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/pricing/integration/status', methods=['GET'])
def get_integration_status():
    """Get pricing integration status"""
    try:
        if pricing_manager:
            status = pricing_manager.get_integration_status()
            return jsonify({
                "success": True,
                "status": status,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Pricing manager not available",
                "status": {"error": "Service not available"}
            }), 503
            
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/pricing/calculate', methods=['POST'])
def calculate_pricing():
    """Calculate pricing for a match"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        material = data.get('material')
        quantity = data.get('quantity', 1.0)
        quality = data.get('quality', 'clean')
        source_location = data.get('source_location', 'unknown')
        destination_location = data.get('destination_location', 'unknown')
        
        if not material:
            return jsonify({"error": "Missing required field: material"}), 400
        
        if pricing_orchestrator:
            pricing_result = pricing_orchestrator.calculate_match_pricing(
                material, quantity, quality, source_location, destination_location
            )
            
            return jsonify({
                "success": True,
                "pricing_result": {
                    "material": pricing_result.material,
                    "virgin_price": pricing_result.virgin_price,
                    "recycled_price": pricing_result.recycled_price,
                    "savings_percentage": pricing_result.savings_percentage,
                    "profit_margin": pricing_result.profit_margin,
                    "shipping_cost": pricing_result.shipping_cost,
                    "refining_cost": pricing_result.refining_cost,
                    "total_cost": pricing_result.total_cost,
                    "confidence": pricing_result.confidence,
                    "risk_level": pricing_result.risk_level,
                    "alerts": pricing_result.alerts,
                    "timestamp": pricing_result.timestamp.isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "Pricing orchestrator not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error calculating pricing: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting AI Pricing Service on port 5005")
    app.run(host='0.0.0.0', port=5005, debug=True) 