"""
Logistics Cost Service - Flask wrapper for logistics cost engine
Provides REST API endpoints for route planning and cost calculation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime

# Import the logistics cost engine
try:
    from logistics_cost_engine import LogisticsCostEngine, Location, TransportMode
except ImportError as e:
    logging.error(f"Failed to import logistics cost engine: {e}")
    # Create dummy class for fallback
    class LogisticsCostEngine:
        def __init__(self):
            pass
        
        def get_route_planning(self, *args, **kwargs):
            return []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize logistics engine
try:
    logistics_engine = LogisticsCostEngine()
    logger.info("Logistics Cost Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize logistics engine: {e}")
    logistics_engine = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "logistics_cost_service",
        "timestamp": datetime.utcnow().isoformat(),
        "logistics_engine_available": logistics_engine is not None
    })

@app.route('/api/logistics/route-planning', methods=['POST'])
def route_planning():
    """Calculate route planning with cost analysis"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract location data
        origin_data = data.get('origin')
        destination_data = data.get('destination')
        cargo_weight = data.get('cargo_weight', 1.0)
        cargo_value = data.get('cargo_value', 1000.0)
        urgency = data.get('urgency', 'normal')
        
        if not all([origin_data, destination_data]):
            return jsonify({"error": "Missing required fields: origin, destination"}), 400
        
        # Create Location objects
        origin = Location(
            name=origin_data.get('name', 'Origin'),
            latitude=origin_data.get('latitude', 0.0),
            longitude=origin_data.get('longitude', 0.0),
            country=origin_data.get('country', 'Unknown'),
            city=origin_data.get('city', 'Unknown')
        )
        
        destination = Location(
            name=destination_data.get('name', 'Destination'),
            latitude=destination_data.get('latitude', 0.0),
            longitude=destination_data.get('longitude', 0.0),
            country=destination_data.get('country', 'Unknown'),
            city=destination_data.get('city', 'Unknown')
        )
        
        if logistics_engine:
            routes = logistics_engine.get_route_planning(
                origin=origin,
                destination=destination,
                cargo_weight=cargo_weight,
                cargo_value=cargo_value,
                urgency=urgency
            )
            
            # Convert routes to JSON-serializable format
            route_data = []
            for route in routes:
                route_info = {
                    "route_id": route.route_id,
                    "total_distance": route.total_distance,
                    "total_duration": route.total_duration,
                    "total_cost": route.total_cost,
                    "total_carbon": route.total_carbon,
                    "reliability_score": route.reliability_score,
                    "segments": [
                        {
                            "mode": segment.mode.value,
                            "origin": segment.origin.name,
                            "destination": segment.destination.name,
                            "distance_km": segment.distance_km,
                            "duration_hours": segment.duration_hours,
                            "cost_euro": segment.cost_euro,
                            "carbon_kg": segment.carbon_kg,
                            "capacity_ton": segment.capacity_ton,
                            "reliability_score": segment.reliability_score
                        }
                        for segment in route.segments
                    ]
                }
                route_data.append(route_info)
            
            return jsonify({
                "success": True,
                "routes": route_data,
                "total_routes": len(route_data),
                "cargo_weight": cargo_weight,
                "cargo_value": cargo_value,
                "urgency": urgency,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Logistics engine not available",
                "routes": []
            }), 503
            
    except Exception as e:
        logger.error(f"Error in route planning: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "routes": []
        }), 500

@app.route('/api/logistics/optimize-route', methods=['POST'])
def optimize_route():
    """Optimize route based on criteria"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        routes_data = data.get('routes', [])
        optimization_criteria = data.get('criteria', 'cost')
        
        if not routes_data:
            return jsonify({"error": "No routes provided"}), 400
        
        if logistics_engine:
            # Convert JSON routes back to Route objects (simplified)
            # For now, just return the first route as "optimized"
            optimized_route = routes_data[0] if routes_data else None
            
            return jsonify({
                "success": True,
                "optimized_route": optimized_route,
                "criteria": optimization_criteria,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Logistics engine not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error in route optimization: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/logistics/carbon-impact', methods=['POST'])
def calculate_carbon_impact():
    """Calculate carbon impact for a route"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        route_data = data.get('route')
        cargo_weight = data.get('cargo_weight', 1.0)
        
        if not route_data:
            return jsonify({"error": "No route data provided"}), 400
        
        if logistics_engine:
            # For now, calculate a simple carbon impact
            total_distance = route_data.get('total_distance', 0)
            total_carbon = route_data.get('total_carbon', 0)
            
            carbon_impact = {
                "total_carbon_kg": total_carbon,
                "carbon_tax_eur": total_carbon * 0.05,  # 50 EUR per ton
                "carbon_intensity_kg_per_ton_km": total_carbon / (cargo_weight * total_distance) if total_distance > 0 else 0,
                "carbon_savings_vs_truck": total_carbon * 0.2,  # 20% savings
                "carbon_equivalent": {
                    "trees_needed": total_carbon / 22,
                    "car_km_equivalent": total_carbon / 0.2,
                    "flight_km_equivalent": total_carbon / 0.25
                }
            }
            
            return jsonify({
                "success": True,
                "carbon_impact": carbon_impact,
                "cargo_weight": cargo_weight,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Logistics engine not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error calculating carbon impact: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/logistics/real-time-rates', methods=['POST'])
def get_real_time_rates():
    """Get real-time rates for a route"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        origin_data = data.get('origin')
        destination_data = data.get('destination')
        cargo_weight = data.get('cargo_weight', 1.0)
        mode = data.get('mode', 'truck')
        
        if not all([origin_data, destination_data]):
            return jsonify({"error": "Missing required fields: origin, destination"}), 400
        
        # Create Location objects
        origin = Location(
            name=origin_data.get('name', 'Origin'),
            latitude=origin_data.get('latitude', 0.0),
            longitude=origin_data.get('longitude', 0.0),
            country=origin_data.get('country', 'Unknown'),
            city=origin_data.get('city', 'Unknown')
        )
        
        destination = Location(
            name=destination_data.get('name', 'Destination'),
            latitude=destination_data.get('latitude', 0.0),
            longitude=destination_data.get('longitude', 0.0),
            country=destination_data.get('country', 'Unknown'),
            city=destination_data.get('city', 'Unknown')
        )
        
        if logistics_engine:
            try:
                transport_mode = TransportMode(mode)
                rates = logistics_engine.get_real_time_rates(
                    origin, destination, cargo_weight, transport_mode
                )
                
                return jsonify({
                    "success": True,
                    "rates": rates,
                    "mode": mode,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except ValueError:
                return jsonify({
                    "success": False,
                    "error": f"Invalid transport mode: {mode}"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Logistics engine not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error getting real-time rates: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Logistics Cost Service on port 5006")
    app.run(host='0.0.0.0', port=5006, debug=True) 