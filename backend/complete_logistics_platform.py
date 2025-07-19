#!/usr/bin/env python3
"""
Complete Logistics Platform for SymbioFlows
Unified interface for all logistics operations: matching, pricing, booking, tracking
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import requests
from flask import Flask, request, jsonify, render_template_string
import redis
import jwt
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LogisticsDeal:
    """Complete logistics deal with all details"""
    deal_id: str
    match_id: str
    buyer_company_id: str
    seller_company_id: str
    material_id: str
    quantity: float
    unit_price: Decimal
    total_material_cost: Decimal
    shipping_cost: Decimal
    customs_cost: Decimal
    insurance_cost: Decimal
    handling_cost: Decimal
    platform_fee: Decimal
    total_cost_to_buyer: Decimal
    net_revenue_to_seller: Decimal
    freightos_booking_id: Optional[str] = None
    tracking_number: Optional[str] = None
    status: str = 'pending'
    buyer_accepted: bool = False
    seller_accepted: bool = False
    created_at: datetime = None
    expires_at: datetime = None
    shipment_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None

@dataclass
class MaterialProfile:
    """Material profile with logistics properties"""
    material_id: str
    name: str
    density: float  # kg/m³
    hazardous: bool
    temperature_sensitive: bool
    customs_classification: str
    packaging_requirements: Dict[str, Any]

class CompleteLogisticsPlatform:
    """Complete logistics platform with all features"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.deals: Dict[str, LogisticsDeal] = {}
        self.materials: Dict[str, MaterialProfile] = {}
        self.platform_fee_percentage = Decimal('0.05')  # 5%
        
        # Load material profiles
        self._load_material_profiles()
        
        # Initialize external service connections
        self._init_service_connections()
    
    def _load_material_profiles(self):
        """Load material profiles"""
        self.materials = {
            "mat_001": MaterialProfile(
                material_id="mat_001",
                name="Recycled Aluminum",
                density=2700.0,  # kg/m³
                hazardous=False,
                temperature_sensitive=False,
                customs_classification="7602.00.00",
                packaging_requirements={
                    "container_type": "standard",
                    "palletization": True,
                    "weather_protection": False
                }
            ),
            "mat_002": MaterialProfile(
                material_id="mat_002",
                name="Recycled Plastic",
                density=950.0,  # kg/m³
                hazardous=False,
                temperature_sensitive=False,
                customs_classification="3915.90.00",
                packaging_requirements={
                    "container_type": "standard",
                    "palletization": True,
                    "weather_protection": True
                }
            )
        }
    
    def _init_service_connections(self):
        """Initialize connections to other services"""
        self.ai_matchmaking_url = "http://localhost:8020"
        self.ai_pricing_url = "http://localhost:5005"
        self.freightos_url = "http://localhost:5025"
        self.monitoring_url = "http://localhost:5011"
    
    async def create_ai_match(self, buyer_requirements: Dict[str, Any], 
                            seller_capabilities: Dict[str, Any]) -> Optional[str]:
        """Create AI-powered match using matchmaking service"""
        try:
            payload = {
                "buyer_requirements": buyer_requirements,
                "seller_capabilities": seller_capabilities,
                "match_type": "logistics_optimized"
            }
            
            response = requests.post(
                f"{self.ai_matchmaking_url}/create_match",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('match_id')
            else:
                logger.error(f"AI matchmaking failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"AI matchmaking error: {e}")
            return None
    
    async def get_ai_pricing(self, material_id: str, quantity: float, 
                           origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """Get AI-powered pricing from pricing service"""
        try:
            payload = {
                "material_id": material_id,
                "quantity": quantity,
                "origin": origin,
                "destination": destination,
                "include_logistics": True
            }
            
            response = requests.post(
                f"{self.ai_pricing_url}/calculate_price",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"AI pricing failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"AI pricing error: {e}")
            return None
    
    async def create_complete_deal(self, buyer_id: str, seller_id: str, 
                                 material_id: str, quantity: float) -> Optional[LogisticsDeal]:
        """Create complete logistics deal with AI pricing and matching"""
        
        deal_id = f"deal_{uuid.uuid4().hex[:8]}"
        
        # Get material profile
        material = self.materials.get(material_id)
        if not material:
            logger.error(f"Material {material_id} not found")
            return None
        
        # Get AI pricing
        pricing_data = await self.get_ai_pricing(
            material_id=material_id,
            quantity=quantity,
            origin="seller_location",  # Would come from seller profile
            destination="buyer_location"  # Would come from buyer profile
        )
        
        if not pricing_data:
            logger.error("Failed to get AI pricing")
            return None
        
        # Calculate costs
        unit_price = Decimal(str(pricing_data.get('unit_price', 0)))
        total_material_cost = unit_price * Decimal(str(quantity))
        shipping_cost = Decimal(str(pricing_data.get('shipping_cost', 0)))
        customs_cost = Decimal(str(pricing_data.get('customs_cost', 0)))
        insurance_cost = Decimal(str(pricing_data.get('insurance_cost', 0)))
        handling_cost = Decimal(str(pricing_data.get('handling_cost', 0)))
        platform_fee = total_material_cost * self.platform_fee_percentage
        
        # Calculate totals
        total_cost_to_buyer = (
            total_material_cost + 
            shipping_cost + 
            customs_cost + 
            insurance_cost + 
            handling_cost + 
            platform_fee
        )
        
        net_revenue_to_seller = total_material_cost - platform_fee
        
        # Create deal
        deal = LogisticsDeal(
            deal_id=deal_id,
            match_id=pricing_data.get('match_id', ''),
            buyer_company_id=buyer_id,
            seller_company_id=seller_id,
            material_id=material_id,
            quantity=quantity,
            unit_price=unit_price,
            total_material_cost=total_material_cost,
            shipping_cost=shipping_cost,
            customs_cost=customs_cost,
            insurance_cost=insurance_cost,
            handling_cost=handling_cost,
            platform_fee=platform_fee,
            total_cost_to_buyer=total_cost_to_buyer,
            net_revenue_to_seller=net_revenue_to_seller
        )
        
        # Store deal
        self.deals[deal_id] = deal
        self.redis_client.setex(f"logistics_deal:{deal_id}", 86400, json.dumps(asdict(deal), default=str))
        
        # Send to monitoring
        await self._send_to_monitoring(deal)
        
        logger.info(f"Created complete deal {deal_id}: Buyer pays ${total_cost_to_buyer}, Seller gets ${net_revenue_to_seller}")
        
        return deal
    
    async def _send_to_monitoring(self, deal: LogisticsDeal):
        """Send deal to monitoring service"""
        try:
            payload = {
                "deal_id": deal.deal_id,
                "type": "logistics_deal",
                "value": float(deal.total_cost_to_buyer),
                "status": deal.status,
                "timestamp": datetime.now().isoformat()
            }
            
            requests.post(
                f"{self.monitoring_url}/log_event",
                json=payload,
                timeout=10
            )
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    async def get_buyer_dashboard(self, buyer_id: str) -> Dict[str, Any]:
        """Get buyer dashboard with all their deals"""
        buyer_deals = [
            deal for deal in self.deals.values() 
            if deal.buyer_company_id == buyer_id
        ]
        
        return {
            "buyer_id": buyer_id,
            "total_deals": len(buyer_deals),
            "active_deals": len([d for d in buyer_deals if d.status in ['pending', 'accepted']]),
            "total_spent": sum(float(d.total_cost_to_buyer) for d in buyer_deals if d.status == 'completed'),
            "pending_deals": [
                {
                    "deal_id": d.deal_id,
                    "material": self.materials.get(d.material_id, {}).get('name', 'Unknown'),
                    "quantity": d.quantity,
                    "total_cost": float(d.total_cost_to_buyer),
                    "status": d.status,
                    "expires_at": d.expires_at.isoformat() if d.expires_at else None
                }
                for d in buyer_deals if d.status in ['pending', 'buyer_accepted']
            ]
        }
    
    async def get_seller_dashboard(self, seller_id: str) -> Dict[str, Any]:
        """Get seller dashboard with all their deals"""
        seller_deals = [
            deal for deal in self.deals.values() 
            if deal.seller_company_id == seller_id
        ]
        
        return {
            "seller_id": seller_id,
            "total_deals": len(seller_deals),
            "active_deals": len([d for d in seller_deals if d.status in ['pending', 'accepted']]),
            "total_revenue": sum(float(d.net_revenue_to_seller) for d in seller_deals if d.status == 'completed'),
            "pending_deals": [
                {
                    "deal_id": d.deal_id,
                    "material": self.materials.get(d.material_id, {}).get('name', 'Unknown'),
                    "quantity": d.quantity,
                    "net_revenue": float(d.net_revenue_to_seller),
                    "status": d.status,
                    "expires_at": d.expires_at.isoformat() if d.expires_at else None
                }
                for d in seller_deals if d.status in ['pending', 'seller_accepted']
            ]
        }
    
    async def accept_deal(self, deal_id: str, company_id: str) -> bool:
        """Accept a logistics deal"""
        deal = self.deals.get(deal_id)
        
        if not deal:
            return False
        
        if company_id not in [deal.buyer_company_id, deal.seller_company_id]:
            return False
        
        # Update acceptance status
        if company_id == deal.buyer_company_id:
            deal.buyer_accepted = True
        elif company_id == deal.seller_company_id:
            deal.seller_accepted = True
        
        # Check if both accepted
        if deal.buyer_accepted and deal.seller_accepted:
            deal.status = 'accepted'
            await self._finalize_deal(deal)
        elif deal.buyer_accepted:
            deal.status = 'buyer_accepted'
        elif deal.seller_accepted:
            deal.status = 'seller_accepted'
        
        # Update stored deal
        self.redis_client.setex(f"logistics_deal:{deal_id}", 86400, json.dumps(asdict(deal), default=str))
        
        # Send to monitoring
        await self._send_to_monitoring(deal)
        
        logger.info(f"Deal {deal_id} accepted by {company_id}, status: {deal.status}")
        return True
    
    async def _finalize_deal(self, deal: LogisticsDeal):
        """Finalize the deal and book logistics"""
        try:
            # Book with Freightos
            freightos_payload = {
                "match_id": deal.match_id,
                "buyer_id": deal.buyer_company_id,
                "seller_id": deal.seller_company_id,
                "material_id": deal.material_id,
                "quantity": deal.quantity,
                "unit_price": float(deal.unit_price)
            }
            
            response = requests.post(
                f"{self.freightos_url}/matches/create",
                json=freightos_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                freightos_data = response.json()
                deal.freightos_booking_id = freightos_data.get('booking_id')
                deal.status = 'logistics_booked'
                deal.shipment_date = datetime.now() + timedelta(days=2)  # Estimated
                deal.delivery_date = deal.shipment_date + timedelta(days=7)  # Estimated
                
                logger.info(f"Logistics booked for deal {deal.deal_id}")
            else:
                deal.status = 'booking_failed'
                logger.error(f"Failed to book logistics for deal {deal.deal_id}")
            
        except Exception as e:
            logger.error(f"Error finalizing deal {deal.deal_id}: {e}")
            deal.status = 'error'

# Initialize platform
logistics_platform = CompleteLogisticsPlatform()

# Flask app
app = Flask(__name__)

def require_auth(f):
    """Authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Complete Logistics Platform',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/deals/create', methods=['POST'])
@require_auth
async def create_deal():
    """Create a new complete logistics deal"""
    try:
        data = request.get_json()
        
        deal = await logistics_platform.create_complete_deal(
            buyer_id=data['buyer_id'],
            seller_id=data['seller_id'],
            material_id=data['material_id'],
            quantity=float(data['quantity'])
        )
        
        if deal:
            return jsonify({
                'status': 'success',
                'deal_id': deal.deal_id,
                'total_cost_to_buyer': float(deal.total_cost_to_buyer),
                'net_revenue_to_seller': float(deal.net_revenue_to_seller),
                'message': 'Complete logistics deal created successfully'
            })
        else:
            return jsonify({'error': 'Failed to create deal'}), 500
        
    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deals/<deal_id>/buyer', methods=['GET'])
@require_auth
async def get_deal_buyer(deal_id):
    """Get deal details for buyer"""
    try:
        buyer_id = request.args.get('buyer_id')
        if not buyer_id:
            return jsonify({'error': 'buyer_id required'}), 400
        
        deal = logistics_platform.deals.get(deal_id)
        
        if not deal or deal.buyer_company_id != buyer_id:
            return jsonify({'error': 'Deal not found or access denied'}), 404
        
        material = logistics_platform.materials.get(deal.material_id, {})
        
        return jsonify({
            "deal_id": deal.deal_id,
            "material": material.get('name', 'Unknown'),
            "quantity": deal.quantity,
            "unit_price": float(deal.unit_price),
            "total_material_cost": float(deal.total_material_cost),
            "shipping_cost": float(deal.shipping_cost),
            "customs_cost": float(deal.customs_cost),
            "insurance_cost": float(deal.insurance_cost),
            "handling_cost": float(deal.handling_cost),
            "platform_fee": float(deal.platform_fee),
            "total_cost_to_buyer": float(deal.total_cost_to_buyer),
            "status": deal.status,
            "expires_at": deal.expires_at.isoformat() if deal.expires_at else None,
            "shipment_date": deal.shipment_date.isoformat() if deal.shipment_date else None,
            "delivery_date": deal.delivery_date.isoformat() if deal.delivery_date else None
        })
        
    except Exception as e:
        logger.error(f"Error getting buyer deal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deals/<deal_id>/seller', methods=['GET'])
@require_auth
async def get_deal_seller(deal_id):
    """Get deal details for seller"""
    try:
        seller_id = request.args.get('seller_id')
        if not seller_id:
            return jsonify({'error': 'seller_id required'}), 400
        
        deal = logistics_platform.deals.get(deal_id)
        
        if not deal or deal.seller_company_id != seller_id:
            return jsonify({'error': 'Deal not found or access denied'}), 404
        
        material = logistics_platform.materials.get(deal.material_id, {})
        
        return jsonify({
            "deal_id": deal.deal_id,
            "material": material.get('name', 'Unknown'),
            "quantity": deal.quantity,
            "unit_price": float(deal.unit_price),
            "total_material_cost": float(deal.total_material_cost),
            "platform_fee": float(deal.platform_fee),
            "net_revenue_to_seller": float(deal.net_revenue_to_seller),
            "status": deal.status,
            "expires_at": deal.expires_at.isoformat() if deal.expires_at else None
        })
        
    except Exception as e:
        logger.error(f"Error getting seller deal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deals/<deal_id>/accept', methods=['POST'])
@require_auth
async def accept_deal(deal_id):
    """Accept a logistics deal"""
    try:
        data = request.get_json()
        company_id = data.get('company_id')
        
        if not company_id:
            return jsonify({'error': 'company_id required'}), 400
        
        success = await logistics_platform.accept_deal(deal_id, company_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Deal accepted successfully'
            })
        else:
            return jsonify({'error': 'Failed to accept deal'}), 400
        
    except Exception as e:
        logger.error(f"Error accepting deal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/buyer/<buyer_id>', methods=['GET'])
@require_auth
async def get_buyer_dashboard(buyer_id):
    """Get buyer dashboard"""
    try:
        dashboard = await logistics_platform.get_buyer_dashboard(buyer_id)
        return jsonify(dashboard)
        
    except Exception as e:
        logger.error(f"Error getting buyer dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/seller/<seller_id>', methods=['GET'])
@require_auth
async def get_seller_dashboard(seller_id):
    """Get seller dashboard"""
    try:
        dashboard = await logistics_platform.get_seller_dashboard(seller_id)
        return jsonify(dashboard)
        
    except Exception as e:
        logger.error(f"Error getting seller dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/admin', methods=['GET'])
@require_auth
async def get_admin_dashboard():
    """Get admin dashboard with system overview"""
    try:
        total_deals = len(logistics_platform.deals)
        active_deals = len([d for d in logistics_platform.deals.values() if d.status in ['pending', 'accepted']])
        completed_deals = len([d for d in logistics_platform.deals.values() if d.status == 'completed'])
        total_volume = sum(d.quantity for d in logistics_platform.deals.values() if d.status == 'completed')
        total_revenue = sum(float(d.platform_fee) for d in logistics_platform.deals.values() if d.status == 'completed')
        
        return jsonify({
            "total_deals": total_deals,
            "active_deals": active_deals,
            "completed_deals": completed_deals,
            "total_volume": total_volume,
            "total_revenue": total_revenue,
            "platform_fee_percentage": float(logistics_platform.platform_fee_percentage),
            "system_status": "operational"
        })
        
    except Exception as e:
        logger.error(f"Error getting admin dashboard: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚚 Starting Complete Logistics Platform on port 5026...")
    app.run(host='0.0.0.0', port=5026, debug=False) 