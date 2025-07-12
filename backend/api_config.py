"""
API Configuration for Industrial Symbiosis System
Manages API keys and settings for external services
"""

import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class APIConfig:
    """Configuration manager for external APIs"""
    
    def __init__(self):
        """Initialize API configuration"""
        self.api_keys = self._load_api_keys()
        self.api_settings = self._load_api_settings()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables or use provided keys"""
        return {
            'next_gen_materials': os.getenv('NEXT_GEN_MATERIALS_API_KEY', 'zSFjfpRg6m020aK84yOjM7oLIhjDNPjE'),
            'freightos': os.getenv('FREIGHTOS_API_KEY', 'V2C6teoe9xSKKpTxL8j4xxuOFGHxQWhx'),
            'freightos_secret': os.getenv('FREIGHTOS_SECRET_KEY', 'k6hEyfd3b6ao8rKQ'),
            'freightos_carbon': os.getenv('FREIGHTOS_CARBON_API_KEY', 'V2C6teoe9xSKKpTxL8j4xxuOFGHxQWhx'),  # Same as freightos
            'deepseek': os.getenv('DEEPSEEK_R1_API_KEY', 'sk-7ce79f30332d45d5b3acb8968b052132'),
            'newsapi': os.getenv('NEWSAPI_KEY', ''),  # Will need to be set by user
        }
    
    def _load_api_settings(self) -> Dict[str, Any]:
        """Load API settings and configurations"""
        return {
            'next_gen_materials': {
                'base_url': 'https://api.nextgenmaterials.com',
                'rate_limit': 1000,
                'timeout': 30
            },
            'freightos': {
                'base_url': 'https://api.freightos.com',
                'rate_limit': 500,
                'timeout': 30
            },
            'freightos_carbon': {
                'base_url': 'https://api.freightos.com',
                'rate_limit': 500,
                'timeout': 30
            },
            'deepseek': {
                'base_url': 'https://api.deepseek.com/v1',
                'rate_limit': 100,
                'timeout': 60
            },
            'newsapi': {
                'base_url': 'https://newsapi.org/v2',
                'rate_limit': 1000,
                'timeout': 30
            }
        }
    
    def get_api_key(self, api_name: str) -> str:
        """Get API key for specified service"""
        return self.api_keys.get(api_name, '')
    
    def is_api_configured(self, api_name: str) -> bool:
        """Check if API is configured with valid key"""
        return bool(self.get_api_key(api_name))
    
    def get_api_settings(self, api_name: str) -> Dict[str, Any]:
        """Get API settings for specified service"""
        return self.api_settings.get(api_name, {})
    
    def get_all_configured_apis(self) -> Dict[str, bool]:
        """Get status of all APIs"""
        return {
            api_name: self.is_api_configured(api_name)
            for api_name in self.api_keys.keys()
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate API configuration and return status"""
        status = {
            'configured_apis': [],
            'missing_apis': [],
            'total_configured': 0,
            'total_apis': len(self.api_keys)
        }
        
        for api_name, key in self.api_keys.items():
            if key:
                status['configured_apis'].append(api_name)
                status['total_configured'] += 1
            else:
                status['missing_apis'].append(api_name)
        
        return status

# Global API configuration instance
api_config = APIConfig()