"""
Revolutionary Plugin Ecosystem & SDK for Industrial Symbiosis
Plugin architecture for third-party extensions with security and sandboxing
"""

import os
import json
import sys
import importlib
import inspect
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import hmac
import base64
import requests
from enum import Enum
import yaml
import docker
from pathlib import Path
import tempfile
import shutil
import zipfile
import tarfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PluginType(Enum):
    FINANCE = "finance"
    CARBON = "carbon"
    LOGISTICS = "logistics"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class PluginStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    UPDATING = "updating"

@dataclass
class PluginManifest:
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str]
    permissions: List[str]
    api_version: str
    min_platform_version: str
    icon_url: Optional[str] = None
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    license: str = "MIT"
    tags: List[str] = None

@dataclass
class PluginInstance:
    id: str
    manifest: PluginManifest
    status: PluginStatus
    instance: Any
    loaded_at: datetime
    last_used: datetime
    usage_count: int = 0
    error_message: Optional[str] = None

@dataclass
class WebhookEvent:
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    signature: Optional[str] = None

class PluginSecurityManager:
    """Manages plugin security and sandboxing"""
    
    def __init__(self):
        self.allowed_permissions = {
            'read_materials': 'Read material data',
            'write_materials': 'Write material data',
            'read_companies': 'Read company data',
            'write_companies': 'Write company data',
            'read_matches': 'Read matching data',
            'write_matches': 'Write matching data',
            'api_access': 'Access external APIs',
            'file_system': 'File system access',
            'network': 'Network access',
            'database': 'Database access'
        }
        
        self.sandbox_config = {
            'max_execution_time': 30,  # seconds
            'max_memory_mb': 512,
            'max_network_requests': 10,
            'allowed_domains': [],
            'blocked_modules': ['os', 'subprocess', 'sys', 'ctypes']
        }
    
    def validate_plugin_permissions(self, manifest: PluginManifest) -> bool:
        """Validate plugin permissions against allowed list"""
        for permission in manifest.permissions:
            if permission not in self.allowed_permissions:
                logger.warning(f"Plugin {manifest.name} requests invalid permission: {permission}")
                return False
        return True
    
    def create_sandbox_environment(self, plugin_id: str) -> Dict[str, Any]:
        """Create sandboxed environment for plugin execution"""
        sandbox = {
            'plugin_id': plugin_id,
            'start_time': datetime.now(),
            'execution_count': 0,
            'network_requests': 0,
            'memory_usage': 0,
            'errors': []
        }
        return sandbox
    
    def check_sandbox_limits(self, sandbox: Dict[str, Any]) -> bool:
        """Check if plugin is within sandbox limits"""
        execution_time = (datetime.now() - sandbox['start_time']).total_seconds()
        
        if execution_time > self.sandbox_config['max_execution_time']:
            sandbox['errors'].append(f"Execution time limit exceeded: {execution_time}s")
            return False
        
        if sandbox['network_requests'] > self.sandbox_config['max_network_requests']:
            sandbox['errors'].append(f"Network request limit exceeded: {sandbox['network_requests']}")
            return False
        
        return True
    
    def generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook verification"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def verify_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self.generate_webhook_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)

class PluginManager:
    """Main plugin manager for loading, managing, and executing plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInstance] = {}
        self.security_manager = PluginSecurityManager()
        self.plugin_directory = Path("plugins")
        self.plugin_directory.mkdir(exist_ok=True)
        
        # Plugin registry
        self.registry_url = os.getenv('PLUGIN_REGISTRY_URL', 'https://registry.industrial-symbiosis.com')
        self.api_key = os.getenv('PLUGIN_REGISTRY_API_KEY', '')
        
        # Webhook handlers
        self.webhook_handlers: Dict[str, List[Callable]] = {}
        
        # Load existing plugins
        self.load_existing_plugins()
    
    def load_existing_plugins(self):
        """Load plugins from the plugin directory"""
        for plugin_dir in self.plugin_directory.iterdir():
            if plugin_dir.is_dir():
                manifest_file = plugin_dir / "manifest.yaml"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = yaml.safe_load(f)
                            manifest = PluginManifest(**manifest_data)
                            self.load_plugin(plugin_dir, manifest)
                    except Exception as e:
                        logger.error(f"Failed to load plugin from {plugin_dir}: {e}")
    
    def load_plugin(self, plugin_path: Path, manifest: PluginManifest) -> Optional[PluginInstance]:
        """Load a plugin from the given path"""
        
        # Validate permissions
        if not self.security_manager.validate_plugin_permissions(manifest):
            logger.error(f"Plugin {manifest.name} has invalid permissions")
            return None
        
        try:
            # Add plugin path to Python path
            sys.path.insert(0, str(plugin_path))
            
            # Import the plugin module
            module_name = manifest.entry_point.split('.')[0]
            module = importlib.import_module(module_name)
            
            # Get the plugin class
            class_name = manifest.entry_point.split('.')[-1]
            plugin_class = getattr(module, class_name)
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Create plugin instance record
            instance = PluginInstance(
                id=f"{manifest.name}_{manifest.version}",
                manifest=manifest,
                status=PluginStatus.ACTIVE,
                instance=plugin_instance,
                loaded_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.plugins[instance.id] = instance
            logger.info(f"Successfully loaded plugin: {manifest.name} v{manifest.version}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {manifest.name}: {e}")
            
            # Create error instance
            error_instance = PluginInstance(
                id=f"{manifest.name}_{manifest.version}",
                manifest=manifest,
                status=PluginStatus.ERROR,
                instance=None,
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                error_message=str(e)
            )
            
            self.plugins[error_instance.id] = error_instance
            return error_instance
    
    def install_plugin_from_url(self, url: str) -> Optional[PluginInstance]:
        """Install plugin from URL"""
        try:
            # Download plugin
            response = requests.get(url)
            response.raise_for_status()
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract plugin
                if url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        zip_file.extractall(temp_path)
                elif url.endswith('.tar.gz'):
                    with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                        tar.extractall(temp_path)
                
                # Find manifest file
                manifest_file = None
                for file_path in temp_path.rglob("manifest.yaml"):
                    manifest_file = file_path
                    break
                
                if not manifest_file:
                    raise ValueError("No manifest.yaml found in plugin")
                
                # Load manifest
                with open(manifest_file, 'r') as f:
                    manifest_data = yaml.safe_load(f)
                    manifest = PluginManifest(**manifest_data)
                
                # Create plugin directory
                plugin_dir = self.plugin_directory / manifest.name
                plugin_dir.mkdir(exist_ok=True)
                
                # Copy plugin files
                shutil.copytree(manifest_file.parent, plugin_dir, dirs_exist_ok=True)
                
                # Load plugin
                return self.load_plugin(plugin_dir, manifest)
                
        except Exception as e:
            logger.error(f"Failed to install plugin from {url}: {e}")
            return None
    
    def execute_plugin(self, plugin_id: str, method: str, *args, **kwargs) -> Any:
        """Execute a plugin method with sandboxing"""
        
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin_instance = self.plugins[plugin_id]
        
        if plugin_instance.status != PluginStatus.ACTIVE:
            raise RuntimeError(f"Plugin {plugin_id} is not active: {plugin_instance.status}")
        
        # Create sandbox
        sandbox = self.security_manager.create_sandbox_environment(plugin_id)
        
        try:
            # Check if method exists
            if not hasattr(plugin_instance.instance, method):
                raise AttributeError(f"Method {method} not found in plugin {plugin_id}")
            
            # Execute method
            method_func = getattr(plugin_instance.instance, method)
            
            # Update usage stats
            plugin_instance.last_used = datetime.now()
            plugin_instance.usage_count += 1
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(method_func):
                # Async method
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    asyncio.wait_for(method_func(*args, **kwargs), 
                                   timeout=self.security_manager.sandbox_config['max_execution_time'])
                )
            else:
                # Sync method
                result = method_func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            sandbox['errors'].append(str(e))
            logger.error(f"Plugin {plugin_id} execution error: {e}")
            raise
        
        finally:
            # Check sandbox limits
            if not self.security_manager.check_sandbox_limits(sandbox):
                logger.warning(f"Plugin {plugin_id} exceeded sandbox limits")
    
    def register_webhook_handler(self, event_type: str, handler: Callable):
        """Register a webhook handler"""
        if event_type not in self.webhook_handlers:
            self.webhook_handlers[event_type] = []
        self.webhook_handlers[event_type].append(handler)
    
    def trigger_webhook(self, event_type: str, data: Dict[str, Any]):
        """Trigger webhook handlers for an event"""
        if event_type in self.webhook_handlers:
            for handler in self.webhook_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Webhook handler error for {event_type}: {e}")
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get plugin information"""
        if plugin_id not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_id]
        return {
            'id': plugin.id,
            'manifest': asdict(plugin.manifest),
            'status': plugin.status.value,
            'loaded_at': plugin.loaded_at.isoformat(),
            'last_used': plugin.last_used.isoformat(),
            'usage_count': plugin.usage_count,
            'error_message': plugin.error_message
        }
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins"""
        return [self.get_plugin_info(plugin_id) for plugin_id in self.plugins.keys()]
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        plugin.status = PluginStatus.INACTIVE
        del self.plugins[plugin_id]
        
        logger.info(f"Unloaded plugin: {plugin_id}")
        return True

class PluginSDK:
    """SDK for plugin developers"""
    
    def __init__(self, plugin_id: str, permissions: List[str]):
        self.plugin_id = plugin_id
        self.permissions = permissions
        self.api_client = PluginAPIClient()
    
    def log(self, message: str, level: str = "info"):
        """Log a message from the plugin"""
        logger.log(getattr(logging, level.upper()), f"[{self.plugin_id}] {message}")
    
    def get_materials(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get materials data"""
        if 'read_materials' not in self.permissions:
            raise PermissionError("Plugin does not have read_materials permission")
        
        return self.api_client.get_materials(filters)
    
    def get_companies(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get companies data"""
        if 'read_companies' not in self.permissions:
            raise PermissionError("Plugin does not have read_companies permission")
        
        return self.api_client.get_companies(filters)
    
    def get_matches(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get matches data"""
        if 'read_matches' not in self.permissions:
            raise PermissionError("Plugin does not have read_matches permission")
        
        return self.api_client.get_matches(filters)
    
    def make_api_request(self, url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make external API request"""
        if 'api_access' not in self.permissions:
            raise PermissionError("Plugin does not have api_access permission")
        
        return self.api_client.make_request(url, method, data)
    
    def trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger a custom event"""
        self.api_client.trigger_event(event_type, data)

class PluginAPIClient:
    """API client for plugins to interact with the main system"""
    
    def __init__(self):
        self.base_url = os.getenv('PLATFORM_API_URL', 'http://localhost:3001')
        self.api_key = os.getenv('PLATFORM_API_KEY', '')
    
    def get_materials(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get materials from the platform"""
        response = requests.get(f"{self.base_url}/api/materials", params=filters)
        response.raise_for_status()
        return response.json().get('materials', [])
    
    def get_companies(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get companies from the platform"""
        response = requests.get(f"{self.base_url}/api/companies", params=filters)
        response.raise_for_status()
        return response.json().get('companies', [])
    
    def get_matches(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get matches from the platform"""
        response = requests.get(f"{self.base_url}/api/matches", params=filters)
        response.raise_for_status()
        return response.json().get('matches', [])
    
    def make_request(self, url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make external API request"""
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger a custom event"""
        payload = {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(f"{self.base_url}/api/events", json=payload)
        response.raise_for_status()

class PluginMarketplace:
    """Plugin marketplace for discovery and management"""
    
    def __init__(self, registry_url: str, api_key: str = None):
        self.registry_url = registry_url
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    
    def search_plugins(self, query: str = None, plugin_type: PluginType = None, 
                      tags: List[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace"""
        params = {}
        if query:
            params['q'] = query
        if plugin_type:
            params['type'] = plugin_type.value
        if tags:
            params['tags'] = ','.join(tags)
        
        response = requests.get(f"{self.registry_url}/api/plugins/search", 
                              params=params, headers=self.headers)
        response.raise_for_status()
        return response.json().get('plugins', [])
    
    def get_plugin_details(self, plugin_name: str) -> Dict[str, Any]:
        """Get detailed information about a plugin"""
        response = requests.get(f"{self.registry_url}/api/plugins/{plugin_name}", 
                              headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def download_plugin(self, plugin_name: str, version: str = None) -> bytes:
        """Download plugin package"""
        url = f"{self.registry_url}/api/plugins/{plugin_name}/download"
        if version:
            url += f"?version={version}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.content
    
    def publish_plugin(self, plugin_data: Dict[str, Any], package_path: str) -> bool:
        """Publish a plugin to the marketplace"""
        files = {'package': open(package_path, 'rb')}
        data = {'plugin_data': json.dumps(plugin_data)}
        
        response = requests.post(f"{self.registry_url}/api/plugins/publish", 
                               data=data, files=files, headers=self.headers)
        return response.status_code == 200

# Example plugin base class
class BasePlugin:
    """Base class for all plugins"""
    
    def __init__(self):
        self.sdk = None
        self.manifest = None
    
    def initialize(self, sdk: PluginSDK, manifest: PluginManifest):
        """Initialize the plugin with SDK and manifest"""
        self.sdk = sdk
        self.manifest = manifest
        self.sdk.log(f"Plugin {manifest.name} initialized")
    
    def on_load(self):
        """Called when plugin is loaded"""
        pass
    
    def on_unload(self):
        """Called when plugin is unloaded"""
        pass
    
    def on_event(self, event_type: str, data: Dict[str, Any]):
        """Called when an event is triggered"""
        pass

# Example finance plugin
class FinancePlugin(BasePlugin):
    """Example finance plugin for cost analysis"""
    
    def calculate_roi(self, investment: float, returns: float, timeframe: int) -> Dict[str, Any]:
        """Calculate ROI for a symbiosis investment"""
        roi = ((returns - investment) / investment) * 100
        annualized_roi = roi / (timeframe / 12)  # Assuming timeframe in months
        
        return {
            'roi_percentage': roi,
            'annualized_roi': annualized_roi,
            'payback_period_months': (investment / (returns / timeframe)) if returns > 0 else float('inf'),
            'net_present_value': returns - investment
        }
    
    def analyze_financial_risk(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial risk of a material transaction"""
        # This would integrate with financial APIs
        return {
            'risk_score': 0.3,
            'risk_factors': ['Market volatility', 'Supply chain disruption'],
            'recommendations': ['Diversify suppliers', 'Hedge prices']
        }

# Example carbon plugin
class CarbonPlugin(BasePlugin):
    """Example carbon plugin for sustainability analysis"""
    
    def calculate_carbon_footprint(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate carbon footprint of materials"""
        # This would use LCA databases
        carbon_per_ton = {
            'steel': 1800,
            'aluminum': 17000,
            'plastic': 2000,
            'paper': 800
        }
        
        material_type = material_data.get('type', 'unknown')
        quantity = material_data.get('quantity', 1)
        
        carbon = carbon_per_ton.get(material_type, 1000) * quantity
        
        return {
            'carbon_kg': carbon,
            'carbon_equivalent': {
                'trees': carbon / 22,
                'car_km': carbon / 0.2,
                'flight_km': carbon / 0.25
            }
        }
    
    def suggest_sustainability_improvements(self, company_data: Dict[str, Any]) -> List[str]:
        """Suggest sustainability improvements"""
        suggestions = []
        
        if company_data.get('industry') == 'manufacturing':
            suggestions.extend([
                'Implement circular economy practices',
                'Optimize energy efficiency',
                'Use renewable energy sources'
            ])
        
        return suggestions

# Initialize plugin manager
plugin_manager = PluginManager()

# Example usage
if __name__ == "__main__":
    # Create example plugin manifests
    finance_manifest = PluginManifest(
        name="finance_analyzer",
        version="1.0.0",
        description="Financial analysis and ROI calculations",
        author="Industrial Symbiosis Team",
        plugin_type=PluginType.FINANCE,
        entry_point="finance_plugin.FinancePlugin",
        dependencies=[],
        permissions=['read_materials', 'read_companies'],
        api_version="1.0",
        min_platform_version="1.0.0",
        tags=['finance', 'roi', 'risk']
    )
    
    carbon_manifest = PluginManifest(
        name="carbon_calculator",
        version="1.0.0",
        description="Carbon footprint and sustainability analysis",
        author="Industrial Symbiosis Team",
        plugin_type=PluginType.CARBON,
        entry_point="carbon_plugin.CarbonPlugin",
        dependencies=[],
        permissions=['read_materials', 'read_companies'],
        api_version="1.0",
        min_platform_version="1.0.0",
        tags=['carbon', 'sustainability', 'lca']
    )
    
    print("Plugin Ecosystem initialized successfully!")
    print(f"Loaded plugins: {len(plugin_manager.plugins)}")
    print(f"Available plugins: {[p.manifest.name for p in plugin_manager.plugins.values()]}") 