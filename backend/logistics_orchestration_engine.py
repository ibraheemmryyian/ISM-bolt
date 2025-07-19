#!/usr/bin/env python3
"""
Logistics Orchestration Engine for SymbioFlows
Handles complete logistics flow: AI matching → Freightos integration → Cost calculation → Deal orchestration
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
from flask import Flask, request, jsonify
import redis
import jwt
from functools import wraps
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LogisticsMatch:
    """Complete logistics match with all costs and details"""
    match_id: str
    buyer_company_id: str
    seller_company_id: str
    material_id: str
    quantity: float
    unit_price: Decimal
    total_material_cost: Decimal
    freightos_quote_id: Optional[str] = None
    shipping_cost: Decimal = Decimal('0')
    customs_cost: Decimal = Decimal('0')
    insurance_cost: Decimal = Decimal('0')
    handling_cost: Decimal = Decimal('0')
    platform_fee: Decimal = Decimal('0')
    total_cost_to_buyer: Decimal = Decimal('0')
    net_revenue_to_seller: Decimal = Decimal('0')
    status: str = 'pending'
    created_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=24)

@dataclass
class FreightosQuote:
    """Freightos shipping quote"""
    quote_id: str
    origin: str
    destination: str
    weight: float
    volume: float
    shipping_cost: Decimal
    transit_time_days: int
    service_level: str
    carrier: str
    valid_until: datetime

@dataclass
class CompanyProfile:
    """Company profile with logistics preferences"""
    company_id: str
    name: str
    location: str
    shipping_preferences: Dict[str, Any]
    payment_terms: str
    credit_rating: str
    logistics_requirements: Dict[str, Any]

class FreightosIntegration:
    """Freightos API integration for shipping quotes"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.freightos.com/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_shipping_quote(self, origin: str, destination: str, weight: float, 
                                volume: float, cargo_type: str = "general") -> Optional[FreightosQuote]:
        """Get shipping quote from Freightos"""
        try:
            payload = {
                "origin": origin,
                "destination": destination,
                "weight": weight,
                "volume": volume,
                "cargo_type": cargo_type,
                "service_level": "standard"
            }
            
            response = requests.post(
                f"{self.base_url}/quotes",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return FreightosQuote(
                    quote_id=data.get('id'),
                    origin=origin,
                    destination=destination,
                    weight=weight,
                    volume=volume,
                    shipping_cost=Decimal(str(data.get('total_cost', 0))),
                    transit_time_days=data.get('transit_time_days', 7),
                    service_level=data.get('service_level', 'standard'),
                    carrier=data.get('carrier', 'unknown'),
                    valid_until=datetime.now() + timedelta(hours=24)
                )
            else:
                logger.error(f"Freightos API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Freightos integration error: {e}")
            return None
    
    async def book_shipment(self, quote_id: str, match_id: str) -> bool:
        """Book shipment with Freightos"""
        try:
            payload = {
                "quote_id": quote_id,
                "booking_reference": match_id,
                "shipper_details": {
                    "name": "SymbioFlows Logistics",
                    "email": "logistics@symbioflows.com"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/bookings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Freightos booking error: {e}")
            return False

class LogisticsOrchestrationEngine:
    """Complete logistics orchestration engine"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.freightos = FreightosIntegration(os.getenv('FREIGHTOS_API_KEY', 'demo_key'))
        self.matches: Dict[str, LogisticsMatch] = {}
        self.companies: Dict[str, CompanyProfile] = {}
        self.platform_fee_percentage = Decimal('0.05')  # 5% platform fee
        
        # Load company profiles
        self._load_company_profiles()
    
    def _load_company_profiles(self):
        """Load company profiles from database"""
        # This would normally load from your database
        # For now, creating sample profiles
        self.companies = {
            "buyer_001": CompanyProfile(
                company_id="buyer_001",
                name="Tech Manufacturing Co",
                location="San Francisco, CA",
                shipping_preferences={
                    "preferred_carriers": ["FedEx", "UPS"],
                    "max_transit_time": 7,
                    "insurance_required": True
                },
                payment_terms="Net 30",
                credit_rating="A+",
                logistics_requirements={
                    "temperature_controlled": False,
                    "hazardous_materials": False
                }
            ),
            "seller_001": CompanyProfile(
                company_id="seller_001",
                name="Green Materials Inc",
                location="Houston, TX",
                shipping_preferences={
                    "preferred_carriers": ["DHL", "FedEx"],
                    "max_transit_time": 10,
                    "insurance_required": False
                },
                payment_terms="Net 15",
                credit_rating="A",
                logistics_requirements={
                    "temperature_controlled": False,
                    "hazardous_materials": False
                }
            )
        }
    
    async def create_logistics_match(self, buyer_id: str, seller_id: str, material_id: str,
                                   quantity: float, unit_price: Decimal) -> LogisticsMatch:
        """Create a new logistics match"""
        
        match_id = f"match_{uuid.uuid4().hex[:8]}"
        
        # Calculate material costs
        total_material_cost = unit_price * Decimal(str(quantity))
        
        # Create match
        match = LogisticsMatch(
            match_id=match_id,
            buyer_company_id=buyer_id,
            seller_company_id=seller_id,
            material_id=material_id,
            quantity=quantity,
            unit_price=unit_price,
            total_material_cost=total_material_cost
        )
        
        # Get shipping quote
        buyer_profile = self.companies.get(buyer_id)
        seller_profile = self.companies.get(seller_id)
        
        if buyer_profile and seller_profile:
            # Estimate weight and volume (this would come from material properties)
            estimated_weight = quantity * 2.5  # kg per unit
            estimated_volume = quantity * 0.1   # m³ per unit
            
            freightos_quote = await self.freightos.get_shipping_quote(
                origin=seller_profile.location,
                destination=buyer_profile.location,
                weight=estimated_weight,
                volume=estimated_volume
            )
            
            if freightos_quote:
                match.freightos_quote_id = freightos_quote.quote_id
                match.shipping_cost = freightos_quote.shipping_cost
                
                # Calculate additional costs
                match.customs_cost = self._calculate_customs_cost(total_material_cost)
                match.insurance_cost = self._calculate_insurance_cost(total_material_cost)
                match.handling_cost = self._calculate_handling_cost(estimated_weight)
                match.platform_fee = total_material_cost * self.platform_fee_percentage
                
                # Calculate totals
                match.total_cost_to_buyer = (
                    total_material_cost + 
                    match.shipping_cost + 
                    match.customs_cost + 
                    match.insurance_cost + 
                    match.handling_cost + 
                    match.platform_fee
                )
                
                match.net_revenue_to_seller = (
                    total_material_cost - 
                    match.platform_fee
                )
        
        # Store match
        self.matches[match_id] = match
        self.redis_client.setex(f"logistics_match:{match_id}", 86400, json.dumps(asdict(match), default=str))
        
        logger.info(f"Created logistics match {match_id}: Buyer pays ${match.total_cost_to_buyer}, Seller gets ${match.net_revenue_to_seller}")
        
        return match
    
    def _calculate_customs_cost(self, material_cost: Decimal) -> Decimal:
        """Calculate customs duties and taxes"""
        # Simplified calculation - would be more complex in reality
        customs_rate = Decimal('0.05')  # 5% customs duty
        return material_cost * customs_rate
    
    def _calculate_insurance_cost(self, material_cost: Decimal) -> Decimal:
        """Calculate insurance cost"""
        insurance_rate = Decimal('0.02')  # 2% insurance
        return material_cost * insurance_rate
    
    def _calculate_handling_cost(self, weight: float) -> Decimal:
        """Calculate handling and processing costs"""
        # $10 per 100kg
        return Decimal(str(weight * 0.1))
    
    async def get_match_for_buyer(self, match_id: str, buyer_id: str) -> Optional[Dict[str, Any]]:
        """Get match details for buyer (shows total cost)"""
        match = self.matches.get(match_id)
        
        if not match or match.buyer_company_id != buyer_id:
            return None
        
        return {
            "match_id": match.match_id,
            "seller_company": self.companies.get(match.seller_company_id, {}).get('name', 'Unknown'),
            "material_id": match.material_id,
            "quantity": match.quantity,
            "unit_price": float(match.unit_price),
            "total_material_cost": float(match.total_material_cost),
            "shipping_cost": float(match.shipping_cost),
            "customs_cost": float(match.customs_cost),
            "insurance_cost": float(match.insurance_cost),
            "handling_cost": float(match.handling_cost),
            "platform_fee": float(match.platform_fee),
            "total_cost_to_buyer": float(match.total_cost_to_buyer),
            "status": match.status,
            "expires_at": match.expires_at.isoformat(),
            "freightos_quote_id": match.freightos_quote_id
        }
    
    async def get_match_for_seller(self, match_id: str, seller_id: str) -> Optional[Dict[str, Any]]:
        """Get match details for seller (shows net revenue)"""
        match = self.matches.get(match_id)
        
        if not match or match.seller_company_id != seller_id:
            return None
        
        return {
            "match_id": match.match_id,
            "buyer_company": self.companies.get(match.buyer_company_id, {}).get('name', 'Unknown'),
            "material_id": match.material_id,
            "quantity": match.quantity,
            "unit_price": float(match.unit_price),
            "total_material_cost": float(match.total_material_cost),
            "platform_fee": float(match.platform_fee),
            "net_revenue_to_seller": float(match.net_revenue_to_seller),
            "status": match.status,
            "expires_at": match.expires_at.isoformat()
        }
    
    async def accept_match(self, match_id: str, company_id: str) -> bool:
        """Accept a logistics match"""
        match = self.matches.get(match_id)
        
        if not match:
            return False
        
        if company_id not in [match.buyer_company_id, match.seller_company_id]:
            return False
        
        # Update status based on who accepted
        if company_id == match.buyer_company_id:
            if match.status == 'pending':
                match.status = 'buyer_accepted'
            elif match.status == 'seller_accepted':
                match.status = 'both_accepted'
        elif company_id == match.seller_company_id:
            if match.status == 'pending':
                match.status = 'seller_accepted'
            elif match.status == 'buyer_accepted':
                match.status = 'both_accepted'
        
        # If both accepted, book shipment
        if match.status == 'both_accepted':
            await self._finalize_deal(match)
        
        # Update stored match
        self.redis_client.setex(f"logistics_match:{match_id}", 86400, json.dumps(asdict(match), default=str))
        
        logger.info(f"Match {match_id} accepted by {company_id}, status: {match.status}")
        return True
    
    async def _finalize_deal(self, match: LogisticsMatch):
        """Finalize the deal and book shipment"""
        try:
            # Book shipment with Freightos
            if match.freightos_quote_id:
                booking_success = await self.freightos.book_shipment(
                    match.freightos_quote_id, 
                    match.match_id
                )
                
                if booking_success:
                    match.status = 'shipment_booked'
                    logger.info(f"Shipment booked for match {match.match_id}")
                else:
                    match.status = 'booking_failed'
                    logger.error(f"Failed to book shipment for match {match.match_id}")
            
            # Here you would also:
            # 1. Create purchase order
            # 2. Set up payment processing
            # 3. Notify both parties
            # 4. Track shipment
            # 5. Handle delivery confirmation
            
        except Exception as e:
            logger.error(f"Error finalizing deal {match.match_id}: {e}")
            match.status = 'error'

# Initialize the engine
logistics_engine = LogisticsOrchestrationEngine()

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
        'service': 'Logistics Orchestration Engine',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/matches/create', methods=['POST'])
@require_auth
async def create_match():
    """Create a new logistics match"""
    try:
        data = request.get_json()
        
        match = await logistics_engine.create_logistics_match(
            buyer_id=data['buyer_id'],
            seller_id=data['seller_id'],
            material_id=data['material_id'],
            quantity=float(data['quantity']),
            unit_price=Decimal(str(data['unit_price']))
        )
        
        return jsonify({
            'status': 'success',
            'match_id': match.match_id,
            'message': 'Logistics match created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating match: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/matches/<match_id>/buyer', methods=['GET'])
@require_auth
async def get_match_buyer(match_id):
    """Get match details for buyer"""
    try:
        buyer_id = request.args.get('buyer_id')
        if not buyer_id:
            return jsonify({'error': 'buyer_id required'}), 400
        
        match_data = await logistics_engine.get_match_for_buyer(match_id, buyer_id)
        
        if not match_data:
            return jsonify({'error': 'Match not found or access denied'}), 404
        
        return jsonify(match_data)
        
    except Exception as e:
        logger.error(f"Error getting buyer match: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/matches/<match_id>/seller', methods=['GET'])
@require_auth
async def get_match_seller(match_id):
    """Get match details for seller"""
    try:
        seller_id = request.args.get('seller_id')
        if not seller_id:
            return jsonify({'error': 'seller_id required'}), 400
        
        match_data = await logistics_engine.get_match_for_seller(match_id, seller_id)
        
        if not match_data:
            return jsonify({'error': 'Match not found or access denied'}), 404
        
        return jsonify(match_data)
        
    except Exception as e:
        logger.error(f"Error getting seller match: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/matches/<match_id>/accept', methods=['POST'])
@require_auth
async def accept_match(match_id):
    """Accept a logistics match"""
    try:
        data = request.get_json()
        company_id = data.get('company_id')
        
        if not company_id:
            return jsonify({'error': 'company_id required'}), 400
        
        success = await logistics_engine.accept_match(match_id, company_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Match accepted successfully'
            })
        else:
            return jsonify({'error': 'Failed to accept match'}), 400
        
    except Exception as e:
        logger.error(f"Error accepting match: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/matches/<match_id>/status', methods=['GET'])
@require_auth
async def get_match_status(match_id):
    """Get match status"""
    try:
        match = logistics_engine.matches.get(match_id)
        
        if not match:
            return jsonify({'error': 'Match not found'}), 404
        
        return jsonify({
            'match_id': match_id,
            'status': match.status,
            'created_at': match.created_at.isoformat(),
            'expires_at': match.expires_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting match status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚚 Starting Logistics Orchestration Engine on port 5025...")
    app.run(host='0.0.0.0', port=5025, debug=False) 