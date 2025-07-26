"""
Simple Service Registry for SymbioFlows
Handles service registration and discovery without external dependencies
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import threading
import time

class ServiceRegistry:
    def __init__(self):
        self.services = {}
        self.health_checks = {}
        self.logger = logging.getLogger(__name__)
        
    def register_service(self, service_name: str, service_url: str, 
                        service_type: str = "python", port: int = None) -> bool:
        """Register a service with the registry"""
        try:
            service_info = {
                'name': service_name,
                'url': service_url,
                'type': service_type,
                'port': port,
                'registered_at': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.services[service_name] = service_info
            self.logger.info(f"✅ Service registered: {service_name} at {service_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register service {service_name}: {e}")
            return False
    
    def deregister_service(self, service_name: str) -> bool:
        """Deregister a service"""
        try:
            if service_name in self.services:
                del self.services[service_name]
                self.logger.info(f"✅ Service deregistered: {service_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to deregister service {service_name}: {e}")
            return False
    
    def get_service(self, service_name: str) -> Optional[Dict]:
        """Get service information"""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, Dict]:
        """Get all registered services"""
        return self.services.copy()
    
    def update_heartbeat(self, service_name: str) -> bool:
        """Update service heartbeat"""
        try:
            if service_name in self.services:
                self.services[service_name]['last_heartbeat'] = datetime.now().isoformat()
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to update heartbeat for {service_name}: {e}")
            return False
    
    def cleanup_stale_services(self, timeout_minutes: int = 5):
        """Remove services that haven't sent heartbeat in timeout_minutes"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
            stale_services = []
            
            for service_name, service_info in self.services.items():
                last_heartbeat = datetime.fromisoformat(service_info['last_heartbeat'])
                if last_heartbeat < cutoff_time:
                    stale_services.append(service_name)
            
            for service_name in stale_services:
                self.deregister_service(service_name)
                self.logger.warning(f"⚠️ Removed stale service: {service_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up stale services: {e}")

# Global registry instance
registry = ServiceRegistry()

# Flask app for service registry API
app = Flask(__name__)

@app.route('/v1/agent/service/register', methods=['POST', 'PUT'])
def register_service():
    """Register a service"""
    try:
        data = request.get_json()
        service_name = data.get('name')
        service_url = data.get('url')
        service_type = data.get('type', 'python')
        port = data.get('port')
        
        if not service_name or not service_url:
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = registry.register_service(service_name, service_url, service_type, port)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Service {service_name} registered'})
        else:
            return jsonify({'error': 'Failed to register service'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/agent/service/deregister', methods=['POST', 'PUT'])
def deregister_service():
    """Deregister a service"""
    try:
        data = request.get_json()
        service_name = data.get('name')
        
        if not service_name:
            return jsonify({'error': 'Missing service name'}), 400
        
        success = registry.deregister_service(service_name)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Service {service_name} deregistered'})
        else:
            return jsonify({'error': 'Service not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/agent/service/heartbeat', methods=['POST', 'PUT'])
def update_heartbeat():
    """Update service heartbeat"""
    try:
        data = request.get_json()
        service_name = data.get('name')
        
        if not service_name:
            return jsonify({'error': 'Missing service name'}), 400
        
        success = registry.update_heartbeat(service_name)
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Service not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/agent/service/list', methods=['GET'])
def list_services():
    """List all registered services"""
    try:
        services = registry.get_all_services()
        return jsonify({'services': services})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/agent/service/<service_name>', methods=['GET'])
def get_service(service_name):
    """Get specific service information"""
    try:
        service = registry.get_service(service_name)
        if service:
            return jsonify({'service': service})
        else:
            return jsonify({'error': 'Service not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service_count': len(registry.services)
    })

def start_registry_server(host='0.0.0.0', port=8500):
    """Start the service registry server"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Start cleanup thread
    def cleanup_loop():
        while True:
            try:
                registry.cleanup_stale_services()
                time.sleep(60)  # Clean up every minute
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    
    logger.info(f"🚀 Starting Service Registry on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    start_registry_server() 