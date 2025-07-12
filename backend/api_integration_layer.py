"""
Advanced API Integration Layer for Industrial Symbiosis
Integrates external APIs for chemical analysis, logistics, emissions, and market intelligence
"""

import os
import requests
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class ChemicalStructure:
    """Chemical structure data from Next Gen Materials Project"""
    formula: str
    molecular_weight: float
    density: float
    melting_point: float
    boiling_point: float
    solubility: Dict[str, float]
    toxicity_data: Dict[str, Any]
    environmental_impact: Dict[str, Any]
    recycling_potential: float
    market_value: float

@dataclass
class ShippingQuote:
    """Shipping quote from Freightos API"""
    origin: str
    destination: str
    weight: float
    volume: float
    cost: float
    transit_time: int
    carbon_emissions: float
    service_level: str
    carrier: str
    quote_id: str

@dataclass
class CarbonFootprint:
    """Carbon footprint data from Freightos Carbon API"""
    total_emissions: float
    transport_emissions: float
    packaging_emissions: float
    handling_emissions: float
    carbon_intensity: float
    offset_cost: float
    sustainability_score: float

@dataclass
class MarketIntelligence:
    """Market intelligence from NewsAPI"""
    articles: List[Dict[str, Any]]
    sentiment_score: float
    market_trends: List[str]
    regulatory_updates: List[Dict[str, Any]]
    price_forecasts: Dict[str, float]
    risk_factors: List[str]

class APIIntegrationLayer:
    """
    Advanced API Integration Layer for Industrial Symbiosis
    Provides seamless access to external APIs for enhanced matching and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API Integration Layer"""
        self.config = config or {}
        
        # API Keys (should be loaded from environment variables)
        self.next_gen_materials_api_key = os.getenv('NEXT_GEN_MATERIALS_API_KEY')
        self.freightos_api_key = os.getenv('FREIGHTOS_API_KEY')
        self.freightos_carbon_api_key = os.getenv('FREIGHTOS_CARBON_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_R1_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        
        # API Base URLs
        self.next_gen_materials_base_url = "https://api.nextgenmaterials.org/v1"
        self.freightos_base_url = "https://api.freightos.com/v2"
        self.freightos_carbon_base_url = "https://api.freightos.com/carbon/v1"
        self.deepseek_base_url = "https://api.deepseek.com/v1"
        self.newsapi_base_url = "https://newsapi.org/v2"
        
        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Rate limiting
        self.rate_limits = {
            'next_gen_materials': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 3600},
            'freightos': {'calls': 0, 'limit': 500, 'reset_time': time.time() + 3600},
            'freightos_carbon': {'calls': 0, 'limit': 500, 'reset_time': time.time() + 3600},
            'deepseek': {'calls': 0, 'limit': 100, 'reset_time': time.time() + 3600},
            'newsapi': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 3600}
        }
        
        logger.info("API Integration Layer initialized")
    
    async def analyze_chemical_structure(self, material_name: str, 
                                       waste_stream: Optional[str] = None) -> Optional[ChemicalStructure]:
        """
        Analyze chemical structure using Next Gen Materials Project API
        
        Args:
            material_name: Name of the material/waste
            waste_stream: Optional waste stream description
            
        Returns:
            Chemical structure data or None if analysis fails
        """
        if not self.next_gen_materials_api_key:
            logger.warning("Next Gen Materials API key not configured")
            return None
        
        cache_key = f"chemical_{material_name}_{waste_stream}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Check rate limits
            if not self._check_rate_limit('next_gen_materials'):
                logger.warning("Rate limit exceeded for Next Gen Materials API")
                return None
            
            url = f"{self.next_gen_materials_base_url}/analyze"
            payload = {
                "material": material_name,
                "waste_stream": waste_stream,
                "include_toxicity": True,
                "include_environmental": True,
                "include_market": True
            }
            
            headers = {
                "Authorization": f"Bearer {self.next_gen_materials_api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        chemical_structure = ChemicalStructure(
                            formula=data.get('formula', ''),
                            molecular_weight=data.get('molecular_weight', 0.0),
                            density=data.get('density', 0.0),
                            melting_point=data.get('melting_point', 0.0),
                            boiling_point=data.get('boiling_point', 0.0),
                            solubility=data.get('solubility', {}),
                            toxicity_data=data.get('toxicity', {}),
                            environmental_impact=data.get('environmental_impact', {}),
                            recycling_potential=data.get('recycling_potential', 0.0),
                            market_value=data.get('market_value', 0.0)
                        )
                        
                        # Cache the result
                        self.cache[cache_key] = chemical_structure
                        self._increment_rate_limit('next_gen_materials')
                        
                        logger.info(f"Chemical analysis completed for {material_name}")
                        return chemical_structure
                    else:
                        logger.error(f"Chemical analysis failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error analyzing chemical structure: {e}")
            return None
    
    async def get_shipping_quote(self, origin: str, destination: str, 
                               weight: float, volume: float, 
                               material_type: str) -> Optional[ShippingQuote]:
        """
        Get shipping quote from Freightos API
        
        Args:
            origin: Origin location
            destination: Destination location
            weight: Weight in kg
            volume: Volume in cubic meters
            material_type: Type of material being shipped
            
        Returns:
            Shipping quote or None if quote fails
        """
        if not self.freightos_api_key:
            logger.warning("Freightos API key not configured")
            return None
        
        cache_key = f"shipping_{origin}_{destination}_{weight}_{volume}_{material_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Check rate limits
            if not self._check_rate_limit('freightos'):
                logger.warning("Rate limit exceeded for Freightos API")
                return None
            
            url = f"{self.freightos_base_url}/quote"
            payload = {
                "origin": origin,
                "destination": destination,
                "weight": weight,
                "volume": volume,
                "material_type": material_type,
                "service_level": "standard",
                "include_carbon": True
            }
            
            headers = {
                "Authorization": f"Bearer {self.freightos_api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        shipping_quote = ShippingQuote(
                            origin=origin,
                            destination=destination,
                            weight=weight,
                            volume=volume,
                            cost=data.get('cost', 0.0),
                            transit_time=data.get('transit_time', 0),
                            carbon_emissions=data.get('carbon_emissions', 0.0),
                            service_level=data.get('service_level', 'standard'),
                            carrier=data.get('carrier', ''),
                            quote_id=data.get('quote_id', '')
                        )
                        
                        # Cache the result
                        self.cache[cache_key] = shipping_quote
                        self._increment_rate_limit('freightos')
                        
                        logger.info(f"Shipping quote obtained: ${shipping_quote.cost}")
                        return shipping_quote
                    else:
                        logger.error(f"Shipping quote failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting shipping quote: {e}")
            return None
    
    async def calculate_carbon_footprint(self, origin: str, destination: str,
                                       weight: float, volume: float,
                                       transport_mode: str = "truck") -> Optional[CarbonFootprint]:
        """
        Calculate carbon footprint using Freightos Carbon API
        
        Args:
            origin: Origin location
            destination: Destination location
            weight: Weight in kg
            volume: Volume in cubic meters
            transport_mode: Mode of transport (truck, rail, ship, air)
            
        Returns:
            Carbon footprint data or None if calculation fails
        """
        if not self.freightos_carbon_api_key:
            logger.warning("Freightos Carbon API key not configured")
            return None
        
        cache_key = f"carbon_{origin}_{destination}_{weight}_{volume}_{transport_mode}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Check rate limits
            if not self._check_rate_limit('freightos_carbon'):
                logger.warning("Rate limit exceeded for Freightos Carbon API")
                return None
            
            url = f"{self.freightos_carbon_base_url}/calculate"
            payload = {
                "origin": origin,
                "destination": destination,
                "weight": weight,
                "volume": volume,
                "transport_mode": transport_mode,
                "include_packaging": True,
                "include_handling": True
            }
            
            headers = {
                "Authorization": f"Bearer {self.freightos_carbon_api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        carbon_footprint = CarbonFootprint(
                            total_emissions=data.get('total_emissions', 0.0),
                            transport_emissions=data.get('transport_emissions', 0.0),
                            packaging_emissions=data.get('packaging_emissions', 0.0),
                            handling_emissions=data.get('handling_emissions', 0.0),
                            carbon_intensity=data.get('carbon_intensity', 0.0),
                            offset_cost=data.get('offset_cost', 0.0),
                            sustainability_score=data.get('sustainability_score', 0.0)
                        )
                        
                        # Cache the result
                        self.cache[cache_key] = carbon_footprint
                        self._increment_rate_limit('freightos_carbon')
                        
                        logger.info(f"Carbon footprint calculated: {carbon_footprint.total_emissions} kg CO2")
                        return carbon_footprint
                    else:
                        logger.error(f"Carbon calculation failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error calculating carbon footprint: {e}")
            return None
    
    async def get_market_intelligence(self, industry: str, 
                                    material: Optional[str] = None,
                                    location: Optional[str] = None) -> Optional[MarketIntelligence]:
        """
        Get market intelligence from NewsAPI
        
        Args:
            industry: Industry sector
            material: Specific material (optional)
            location: Geographic location (optional)
            
        Returns:
            Market intelligence data or None if retrieval fails
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return None
        
        cache_key = f"market_{industry}_{material}_{location}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Check rate limits
            if not self._check_rate_limit('newsapi'):
                logger.warning("Rate limit exceeded for NewsAPI")
                return None
            
            # Build query
            query_parts = [industry]
            if material:
                query_parts.append(material)
            if location:
                query_parts.append(location)
            
            query = " AND ".join(query_parts)
            
            url = f"{self.newsapi_base_url}/everything"
            params = {
                "q": query,
                "apiKey": self.newsapi_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = data.get('articles', [])
                        
                        # Analyze sentiment and extract insights
                        sentiment_score = self._analyze_sentiment(articles)
                        market_trends = self._extract_market_trends(articles)
                        regulatory_updates = self._extract_regulatory_updates(articles)
                        price_forecasts = self._extract_price_forecasts(articles)
                        risk_factors = self._extract_risk_factors(articles)
                        
                        market_intelligence = MarketIntelligence(
                            articles=articles,
                            sentiment_score=sentiment_score,
                            market_trends=market_trends,
                            regulatory_updates=regulatory_updates,
                            price_forecasts=price_forecasts,
                            risk_factors=risk_factors
                        )
                        
                        # Cache the result
                        self.cache[cache_key] = market_intelligence
                        self._increment_rate_limit('newsapi')
                        
                        logger.info(f"Market intelligence retrieved: {len(articles)} articles")
                        return market_intelligence
                    else:
                        logger.error(f"Market intelligence failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting market intelligence: {e}")
            return None
    
    async def advanced_llm_analysis(self, prompt: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform advanced LLM analysis using DeepSeek R1 API
        
        Args:
            prompt: Analysis prompt
            context: Context data for analysis
            
        Returns:
            LLM analysis results or None if analysis fails
        """
        if not self.deepseek_api_key:
            logger.warning("DeepSeek R1 API key not configured")
            return None
        
        try:
            # Check rate limits
            if not self._check_rate_limit('deepseek'):
                logger.warning("Rate limit exceeded for DeepSeek R1 API")
                return None
            
            url = f"{self.deepseek_base_url}/chat/completions"
            payload = {
                "model": "deepseek-r1",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert industrial symbiosis analyst. Provide detailed, actionable insights."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContext: {json.dumps(context, indent=2)}"
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        analysis_result = {
                            "analysis": data['choices'][0]['message']['content'],
                            "model": "deepseek-r1",
                            "timestamp": datetime.now().isoformat(),
                            "context": context
                        }
                        
                        self._increment_rate_limit('deepseek')
                        
                        logger.info("Advanced LLM analysis completed")
                        return analysis_result
                    else:
                        logger.error(f"LLM analysis failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error performing LLM analysis: {e}")
            return None
    
    async def comprehensive_symbiosis_analysis(self, buyer_data: Dict[str, Any], 
                                             seller_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive symbiosis analysis using all available APIs
        
        Args:
            buyer_data: Buyer company data
            seller_data: Seller company data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            results = {
                "chemical_analysis": None,
                "shipping_analysis": None,
                "carbon_analysis": None,
                "market_intelligence": None,
                "llm_insights": None,
                "overall_assessment": {}
            }
            
            # Chemical structure analysis
            if seller_data.get('waste_materials'):
                for material in seller_data['waste_materials']:
                    chemical_structure = await self.analyze_chemical_structure(
                        material['name'], material.get('description')
                    )
                    if chemical_structure:
                        results["chemical_analysis"] = chemical_structure
                        break
            
            # Shipping analysis
            if buyer_data.get('location') and seller_data.get('location'):
                shipping_quote = await self.get_shipping_quote(
                    seller_data['location'],
                    buyer_data['location'],
                    seller_data.get('waste_volume', 1000),
                    seller_data.get('waste_volume', 10),
                    seller_data.get('waste_type', 'general')
                )
                results["shipping_analysis"] = shipping_quote
            
            # Carbon footprint analysis
            if results["shipping_analysis"]:
                carbon_footprint = await self.calculate_carbon_footprint(
                    seller_data['location'],
                    buyer_data['location'],
                    seller_data.get('waste_volume', 1000),
                    seller_data.get('waste_volume', 10)
                )
                results["carbon_analysis"] = carbon_footprint
            
            # Market intelligence
            market_intelligence = await self.get_market_intelligence(
                buyer_data.get('industry', 'manufacturing'),
                seller_data.get('waste_type'),
                buyer_data.get('location')
            )
            results["market_intelligence"] = market_intelligence
            
            # Advanced LLM analysis
            context = {
                "buyer": buyer_data,
                "seller": seller_data,
                "chemical_analysis": results["chemical_analysis"],
                "shipping_analysis": results["shipping_analysis"],
                "carbon_analysis": results["carbon_analysis"],
                "market_intelligence": results["market_intelligence"]
            }
            
            llm_prompt = """
            Analyze this industrial symbiosis opportunity comprehensively:
            1. Evaluate the chemical compatibility and safety
            2. Assess the economic viability considering shipping costs
            3. Calculate the environmental impact and carbon savings
            4. Consider market conditions and regulatory factors
            5. Provide specific recommendations for implementation
            6. Identify potential risks and mitigation strategies
            """
            
            llm_insights = await self.advanced_llm_analysis(llm_prompt, context)
            results["llm_insights"] = llm_insights
            
            # Overall assessment
            results["overall_assessment"] = self._calculate_overall_assessment(results)
            
            logger.info("Comprehensive symbiosis analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows another call"""
        limit_info = self.rate_limits[api_name]
        
        # Reset counter if time has passed
        if time.time() > limit_info['reset_time']:
            limit_info['calls'] = 0
            limit_info['reset_time'] = time.time() + 3600
        
        return limit_info['calls'] < limit_info['limit']
    
    def _increment_rate_limit(self, api_name: str):
        """Increment API call counter"""
        self.rate_limits[api_name]['calls'] += 1
    
    def _analyze_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Analyze sentiment of news articles"""
        # Simple sentiment analysis based on keywords
        positive_words = ['growth', 'profit', 'increase', 'positive', 'opportunity']
        negative_words = ['decline', 'loss', 'decrease', 'negative', 'risk']
        
        total_sentiment = 0
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            sentiment = (positive_count - negative_count) / max(len(text.split()), 1)
            total_sentiment += sentiment
        
        return total_sentiment / len(articles) if articles else 0.0
    
    def _extract_market_trends(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract market trends from articles"""
        trends = []
        trend_keywords = ['trend', 'growth', 'decline', 'market', 'demand', 'supply']
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if any(keyword in text.lower() for keyword in trend_keywords):
                trends.append(article.get('title', ''))
        
        return trends[:5]  # Return top 5 trends
    
    def _extract_regulatory_updates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract regulatory updates from articles"""
        updates = []
        regulatory_keywords = ['regulation', 'law', 'policy', 'compliance', 'government']
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if any(keyword in text.lower() for keyword in regulatory_keywords):
                updates.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', '')
                })
        
        return updates[:5]  # Return top 5 updates
    
    def _extract_price_forecasts(self, articles: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract price forecasts from articles"""
        forecasts = {}
        # This would require more sophisticated NLP to extract actual price predictions
        # For now, return a simple estimate based on sentiment
        sentiment = self._analyze_sentiment(articles)
        forecasts['price_trend'] = 1.0 + (sentiment * 0.1)  # ±10% based on sentiment
        return forecasts
    
    def _extract_risk_factors(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract risk factors from articles"""
        risks = []
        risk_keywords = ['risk', 'threat', 'challenge', 'uncertainty', 'volatility']
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if any(keyword in text.lower() for keyword in risk_keywords):
                risks.append(article.get('title', ''))
        
        return risks[:5]  # Return top 5 risks
    
    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall assessment score"""
        scores = {
            'chemical_compatibility': 0.0,
            'economic_viability': 0.0,
            'environmental_impact': 0.0,
            'market_conditions': 0.0,
            'overall_score': 0.0
        }
        
        # Chemical compatibility score
        if results.get('chemical_analysis'):
            chemical = results['chemical_analysis']
            scores['chemical_compatibility'] = chemical.recycling_potential
        
        # Economic viability score
        if results.get('shipping_analysis'):
            shipping = results['shipping_analysis']
            # Simple economic calculation (would be more sophisticated in practice)
            scores['economic_viability'] = min(1.0, 1000 / max(shipping.cost, 1))
        
        # Environmental impact score
        if results.get('carbon_analysis'):
            carbon = results['carbon_analysis']
            scores['environmental_impact'] = carbon.sustainability_score
        
        # Market conditions score
        if results.get('market_intelligence'):
            market = results['market_intelligence']
            scores['market_conditions'] = (market.sentiment_score + 1) / 2  # Normalize to 0-1
        
        # Overall score (weighted average)
        weights = {
            'chemical_compatibility': 0.3,
            'economic_viability': 0.25,
            'environmental_impact': 0.25,
            'market_conditions': 0.2
        }
        
        scores['overall_score'] = sum(
            scores[key] * weights[key] 
            for key in weights.keys()
        )
        
        return scores
    
    def clear_cache(self):
        """Clear API response cache"""
        self.cache.clear()
        logger.info("API cache cleared")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all APIs"""
        return {
            'next_gen_materials': {
                'configured': bool(self.next_gen_materials_api_key),
                'rate_limit': self.rate_limits['next_gen_materials']
            },
            'freightos': {
                'configured': bool(self.freightos_api_key),
                'rate_limit': self.rate_limits['freightos']
            },
            'freightos_carbon': {
                'configured': bool(self.freightos_carbon_api_key),
                'rate_limit': self.rate_limits['freightos_carbon']
            },
            'deepseek': {
                'configured': bool(self.deepseek_api_key),
                'rate_limit': self.rate_limits['deepseek']
            },
            'newsapi': {
                'configured': bool(self.newsapi_key),
                'rate_limit': self.rate_limits['newsapi']
            },
            'cache_size': len(self.cache)
        }