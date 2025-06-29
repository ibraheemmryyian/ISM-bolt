"""
Advanced Logistics & Cost Engine for Industrial Symbiosis
Handles multi-modal transportation, cost optimization, and carbon impact analysis
"""

import os
import json
import math
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransportMode(Enum):
    TRUCK = "truck"
    TRAIN = "train"
    SHIP = "ship"
    AIR = "air"
    MULTIMODAL = "multimodal"

@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    country: str
    city: str
    postal_code: str = ""

@dataclass
class RouteSegment:
    mode: TransportMode
    origin: Location
    destination: Location
    distance_km: float
    duration_hours: float
    cost_euro: float
    carbon_kg: float
    capacity_ton: float
    reliability_score: float

@dataclass
class Route:
    segments: List[RouteSegment]
    total_distance: float
    total_duration: float
    total_cost: float
    total_carbon: float
    reliability_score: float
    route_id: str

@dataclass
class CostBreakdown:
    transport: float
    customs: float
    insurance: float
    storage: float
    carbon_tax: float
    handling: float
    total: float

class LogisticsCostEngine:
    def __init__(self):
        self.api_keys = {
            'google_maps': os.getenv('GOOGLE_MAPS_API_KEY', ''),
            'openweather': os.getenv('OPENWEATHER_API_KEY', ''),
            'freightos': os.getenv('FREIGHTOS_API_KEY', ''),
            'carbon_interface': os.getenv('CARBON_INTERFACE_API_KEY', '')
        }
        
        # Cost parameters (EUR)
        self.cost_rates = {
            'truck': {
                'per_km': 0.15,
                'per_ton_km': 0.08,
                'base_cost': 50,
                'carbon_per_km': 0.15  # kg CO2 per km
            },
            'train': {
                'per_km': 0.08,
                'per_ton_km': 0.04,
                'base_cost': 200,
                'carbon_per_km': 0.04
            },
            'ship': {
                'per_km': 0.02,
                'per_ton_km': 0.01,
                'base_cost': 500,
                'carbon_per_km': 0.03
            },
            'air': {
                'per_km': 0.50,
                'per_ton_km': 0.25,
                'base_cost': 1000,
                'carbon_per_km': 0.50
            }
        }
        
        # Customs and insurance rates
        self.customs_rate = 0.05  # 5% of cargo value
        self.insurance_rate = 0.02  # 2% of cargo value
        self.storage_rate = 50  # EUR per day per ton
        self.carbon_tax_rate = 50  # EUR per ton CO2
        
    def calculate_distance(self, origin: Location, destination: Location) -> float:
        """Calculate distance between two locations using Haversine formula"""
        lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
        lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def get_route_planning(self, origin: Location, destination: Location, 
                          cargo_weight: float, cargo_value: float,
                          urgency: str = 'normal') -> List[Route]:
        """
        Generate multi-modal route options with cost and carbon analysis
        """
        routes = []
        
        # Calculate base distance
        direct_distance = self.calculate_distance(origin, destination)
        
        # Generate different route options
        route_options = self._generate_route_options(origin, destination, direct_distance)
        
        for i, option in enumerate(route_options):
            route = self._calculate_route_details(option, cargo_weight, cargo_value, urgency)
            if route:
                routes.append(route)
        
        # Sort by total cost
        routes.sort(key=lambda x: x.total_cost)
        
        return routes
    
    def _generate_route_options(self, origin: Location, destination: Location, 
                               direct_distance: float) -> List[List[Dict]]:
        """Generate different route combinations"""
        options = []
        
        # Option 1: Direct truck
        if direct_distance <= 800:  # Suitable for truck
            options.append([{
                'mode': TransportMode.TRUCK,
                'origin': origin,
                'destination': destination,
                'distance': direct_distance
            }])
        
        # Option 2: Train (if distance > 300km)
        if direct_distance > 300:
            options.append([{
                'mode': TransportMode.TRAIN,
                'origin': origin,
                'destination': destination,
                'distance': direct_distance
            }])
        
        # Option 3: Ship (if international or coastal)
        if self._is_international(origin, destination) or self._is_coastal(origin, destination):
            options.append([{
                'mode': TransportMode.SHIP,
                'origin': origin,
                'destination': destination,
                'distance': direct_distance * 1.2  # Sea routes are longer
            }])
        
        # Option 4: Air (for urgent shipments)
        options.append([{
            'mode': TransportMode.AIR,
            'origin': origin,
            'destination': destination,
            'distance': direct_distance
        }])
        
        # Option 5: Multimodal (truck + train + truck)
        if direct_distance > 500:
            # Find intermediate points for multimodal
            intermediate = self._find_intermediate_point(origin, destination)
            if intermediate:
                options.append([
                    {
                        'mode': TransportMode.TRUCK,
                        'origin': origin,
                        'destination': intermediate,
                        'distance': self.calculate_distance(origin, intermediate)
                    },
                    {
                        'mode': TransportMode.TRAIN,
                        'origin': intermediate,
                        'destination': intermediate,  # Same point for transfer
                        'distance': 0  # Transfer distance
                    },
                    {
                        'mode': TransportMode.TRUCK,
                        'origin': intermediate,
                        'destination': destination,
                        'distance': self.calculate_distance(intermediate, destination)
                    }
                ])
        
        return options
    
    def _calculate_route_details(self, route_option: List[Dict], cargo_weight: float,
                                cargo_value: float, urgency: str) -> Optional[Route]:
        """Calculate detailed costs and carbon impact for a route"""
        segments = []
        total_distance = 0
        total_duration = 0
        total_cost = 0
        total_carbon = 0
        reliability_score = 1.0
        
        for i, segment in enumerate(route_option):
            # Calculate segment details
            segment_details = self._calculate_segment_cost(
                segment['mode'], segment['distance'], cargo_weight, cargo_value
            )
            
            # Create route segment
            route_segment = RouteSegment(
                mode=segment['mode'],
                origin=segment['origin'],
                destination=segment['destination'],
                distance_km=segment['distance'],
                duration_hours=segment_details['duration'],
                cost_euro=segment_details['cost'],
                carbon_kg=segment_details['carbon'],
                capacity_ton=segment_details['capacity'],
                reliability_score=segment_details['reliability']
            )
            
            segments.append(route_segment)
            total_distance += segment['distance']
            total_duration += segment_details['duration']
            total_cost += segment_details['cost']
            total_carbon += segment_details['carbon']
            reliability_score *= segment_details['reliability']
        
        # Add additional costs
        additional_costs = self._calculate_additional_costs(cargo_weight, cargo_value, total_distance)
        total_cost += additional_costs['total']
        
        # Apply urgency multiplier
        if urgency == 'urgent':
            total_cost *= 1.5
            total_duration *= 0.7
        elif urgency == 'express':
            total_cost *= 2.0
            total_duration *= 0.5
        
        return Route(
            segments=segments,
            total_distance=total_distance,
            total_duration=total_duration,
            total_cost=total_cost,
            total_carbon=total_carbon,
            reliability_score=reliability_score,
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(segments)}"
        )
    
    def _calculate_segment_cost(self, mode: TransportMode, distance: float,
                               cargo_weight: float, cargo_value: float) -> Dict:
        """Calculate cost breakdown for a single segment"""
        rates = self.cost_rates[mode.value]
        
        # Base transport cost
        transport_cost = rates['base_cost'] + (distance * rates['per_km']) + (distance * cargo_weight * rates['per_ton_km'])
        
        # Carbon impact
        carbon_kg = distance * rates['carbon_per_km'] * cargo_weight
        
        # Duration estimation
        speed_kmh = {
            'truck': 60,
            'train': 80,
            'ship': 25,
            'air': 800
        }
        duration = distance / speed_kmh[mode.value]
        
        # Capacity and reliability
        capacity = {
            'truck': 25,
            'train': 1000,
            'ship': 5000,
            'air': 100
        }
        
        reliability = {
            'truck': 0.95,
            'train': 0.98,
            'ship': 0.90,
            'air': 0.99
        }
        
        return {
            'cost': transport_cost,
            'carbon': carbon_kg,
            'duration': duration,
            'capacity': capacity[mode.value],
            'reliability': reliability[mode.value]
        }
    
    def _calculate_additional_costs(self, cargo_weight: float, cargo_value: float, 
                                   distance: float) -> CostBreakdown:
        """Calculate additional costs (customs, insurance, storage, etc.)"""
        customs = cargo_value * self.customs_rate
        insurance = cargo_value * self.insurance_rate
        storage = cargo_weight * self.storage_rate * (distance / 500)  # Days based on distance
        carbon_tax = 0  # Will be calculated based on actual carbon
        handling = cargo_weight * 10  # EUR per ton
        
        total = customs + insurance + storage + carbon_tax + handling
        
        return CostBreakdown(
            transport=0,  # Already calculated
            customs=customs,
            insurance=insurance,
            storage=storage,
            carbon_tax=carbon_tax,
            handling=handling,
            total=total
        )
    
    def _is_international(self, origin: Location, destination: Location) -> bool:
        """Check if route is international"""
        return origin.country != destination.country
    
    def _is_coastal(self, location: Location) -> bool:
        """Check if location is coastal (simplified)"""
        coastal_cities = ['rotterdam', 'hamburg', 'antwerp', 'barcelona', 'genoa']
        return location.city.lower() in coastal_cities
    
    def _find_intermediate_point(self, origin: Location, destination: Location) -> Optional[Location]:
        """Find intermediate point for multimodal routes"""
        # Major logistics hubs
        hubs = [
            Location("Rotterdam", 51.9225, 4.4792, "Netherlands", "Rotterdam"),
            Location("Hamburg", 53.5511, 9.9937, "Germany", "Hamburg"),
            Location("Antwerp", 51.2194, 4.4025, "Belgium", "Antwerp"),
            Location("Barcelona", 41.3851, 2.1734, "Spain", "Barcelona"),
            Location("Milan", 45.4642, 9.1900, "Italy", "Milan")
        ]
        
        # Find closest hub to midpoint
        mid_lat = (origin.latitude + destination.latitude) / 2
        mid_lon = (origin.longitude + destination.longitude) / 2
        
        closest_hub = None
        min_distance = float('inf')
        
        for hub in hubs:
            distance = self.calculate_distance(
                Location("mid", mid_lat, mid_lon, "", ""),
                hub
            )
            if distance < min_distance:
                min_distance = distance
                closest_hub = hub
        
        return closest_hub
    
    def get_real_time_rates(self, origin: Location, destination: Location,
                           cargo_weight: float, mode: TransportMode) -> Dict:
        """Get real-time rates from external APIs (placeholder)"""
        # This would integrate with real logistics APIs
        # For now, return estimated rates
        base_rate = self.cost_rates[mode.value]['per_km']
        
        # Simulate real-time pricing variations
        import random
        variation = random.uniform(0.8, 1.2)
        
        return {
            'rate_per_km': base_rate * variation,
            'availability': True,
            'transit_time_days': self._estimate_transit_time(mode, origin, destination),
            'last_updated': datetime.now().isoformat()
        }
    
    def _estimate_transit_time(self, mode: TransportMode, origin: Location, 
                              destination: Location) -> int:
        """Estimate transit time in days"""
        distance = self.calculate_distance(origin, destination)
        
        speeds = {
            'truck': 60,  # km/h
            'train': 80,
            'ship': 25,
            'air': 800
        }
        
        hours = distance / speeds[mode.value]
        days = math.ceil(hours / 24)
        
        # Add buffer days
        buffer_days = {
            'truck': 1,
            'train': 2,
            'ship': 3,
            'air': 1
        }
        
        return days + buffer_days[mode.value]
    
    def calculate_carbon_impact(self, route: Route, cargo_weight: float) -> Dict:
        """Calculate detailed carbon impact"""
        total_carbon = route.total_carbon
        
        # Carbon tax calculation
        carbon_tax = total_carbon * (self.carbon_tax_rate / 1000)  # Convert to EUR
        
        # Carbon intensity per ton-km
        carbon_intensity = total_carbon / (cargo_weight * route.total_distance)
        
        return {
            'total_carbon_kg': total_carbon,
            'carbon_tax_eur': carbon_tax,
            'carbon_intensity_kg_per_ton_km': carbon_intensity,
            'carbon_savings_vs_truck': self._calculate_carbon_savings(route),
            'carbon_equivalent': {
                'trees_needed': total_carbon / 22,  # kg CO2 per tree per year
                'car_km_equivalent': total_carbon / 0.2,  # kg CO2 per km
                'flight_km_equivalent': total_carbon / 0.25  # kg CO2 per km
            }
        }
    
    def _calculate_carbon_savings(self, route: Route) -> float:
        """Calculate carbon savings compared to truck-only route"""
        truck_carbon = route.total_distance * self.cost_rates['truck']['carbon_per_km']
        return truck_carbon - route.total_carbon
    
    def optimize_route(self, routes: List[Route], optimization_criteria: str = 'cost') -> Route:
        """Optimize route based on different criteria"""
        if optimization_criteria == 'cost':
            return min(routes, key=lambda x: x.total_cost)
        elif optimization_criteria == 'carbon':
            return min(routes, key=lambda x: x.total_carbon)
        elif optimization_criteria == 'time':
            return min(routes, key=lambda x: x.total_duration)
        elif optimization_criteria == 'reliability':
            return max(routes, key=lambda x: x.reliability_score)
        else:
            # Multi-criteria optimization
            return self._multi_criteria_optimization(routes)
    
    def _multi_criteria_optimization(self, routes: List[Route]) -> Route:
        """Multi-criteria optimization using weighted scoring"""
        best_route = None
        best_score = -1
        
        for route in routes:
            # Normalize scores (0-1)
            cost_score = 1 - (route.total_cost / max(r.total_cost for r in routes))
            carbon_score = 1 - (route.total_carbon / max(r.total_carbon for r in routes))
            time_score = 1 - (route.total_duration / max(r.total_duration for r in routes))
            reliability_score = route.reliability_score
            
            # Weighted combination
            total_score = (
                cost_score * 0.4 +
                carbon_score * 0.3 +
                time_score * 0.2 +
                reliability_score * 0.1
            )
            
            if total_score > best_score:
                best_score = total_score
                best_route = route
        
        return best_route
    
    def generate_route_report(self, route: Route, cargo_weight: float, 
                             cargo_value: float) -> Dict:
        """Generate comprehensive route report"""
        carbon_impact = self.calculate_carbon_impact(route, cargo_weight)
        
        return {
            'route_id': route.route_id,
            'summary': {
                'total_distance_km': route.total_distance,
                'total_duration_hours': route.total_duration,
                'total_cost_eur': route.total_cost,
                'total_carbon_kg': route.total_carbon,
                'reliability_score': route.reliability_score
            },
            'segments': [
                {
                    'mode': segment.mode.value,
                    'origin': segment.origin.name,
                    'destination': segment.destination.name,
                    'distance_km': segment.distance_km,
                    'duration_hours': segment.duration_hours,
                    'cost_eur': segment.cost_euro,
                    'carbon_kg': segment.carbon_kg
                }
                for segment in route.segments
            ],
            'carbon_impact': carbon_impact,
            'cost_breakdown': {
                'transport': sum(s.cost_euro for s in route.segments),
                'carbon_tax': carbon_impact['carbon_tax_eur'],
                'total': route.total_cost
            },
            'recommendations': self._generate_recommendations(route, carbon_impact),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, route: Route, carbon_impact: Dict) -> List[str]:
        """Generate recommendations for route optimization"""
        recommendations = []
        
        if carbon_impact['carbon_savings_vs_truck'] > 0:
            recommendations.append(
                f"Carbon savings: {carbon_impact['carbon_savings_vs_truck']:.1f} kg CO2 vs truck-only route"
            )
        
        if route.reliability_score < 0.95:
            recommendations.append("Consider alternative routes for better reliability")
        
        if route.total_duration > 48:
            recommendations.append("Route may be suitable for non-urgent shipments")
        
        if carbon_impact['carbon_intensity_kg_per_ton_km'] > 0.1:
            recommendations.append("Consider more carbon-efficient transport modes")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = LogisticsCostEngine()
    
    # Example locations
    origin = Location("Factory A", 52.3676, 4.9041, "Netherlands", "Amsterdam")
    destination = Location("Factory B", 48.8566, 2.3522, "France", "Paris")
    
    # Calculate routes
    routes = engine.get_route_planning(
        origin=origin,
        destination=destination,
        cargo_weight=10.0,  # 10 tons
        cargo_value=50000,  # 50,000 EUR
        urgency='normal'
    )
    
    # Generate report for best route
    if routes:
        best_route = engine.optimize_route(routes, 'cost')
        report = engine.generate_route_report(best_route, 10.0, 50000)
        
        print("=== Logistics Route Report ===")
        print(json.dumps(report, indent=2, default=str)) 