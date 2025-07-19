#!/usr/bin/env python3
"""
Complete System Integration for SymbioFlows
Connects ALL services together with proper configuration and startup
"""

import asyncio
import json
import logging
import time
import uuid
import os
import subprocess
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
import requests
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    name: str
    host: str
    port: int
    health_endpoint: str
    startup_timeout: int
    dependencies: List[str]
    environment_vars: Dict[str, str]
    startup_command: str
    working_directory: str

class SystemIntegrator:
    """Complete system integration and orchestration"""
    
    def __init__(self):
        self.redis_client = None
        self.services = {}
        self.service_processes = {}
        self.service_status = {}
        self.config = self._load_configuration()
        
        # Initialize Redis
        self._init_redis()
        
        # Initialize services
        self._init_services()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        return {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            'jaeger_host': os.getenv('JAEGER_HOST', 'localhost'),
            'jaeger_port': int(os.getenv('JAEGER_PORT', 6831)),
            'base_url': os.getenv('BASE_URL', 'http://localhost'),
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            # Create in-memory fallback
            self.redis_client = None
    
    def _init_services(self):
        """Initialize all service configurations"""
        self.services = {
            # Core Infrastructure
            'redis': ServiceConfig(
                name="Redis",
                host="localhost",
                port=6379,
                health_endpoint="/",
                startup_timeout=10,
                dependencies=[],
                environment_vars={},
                startup_command="redis-server",
                working_directory="."
            ),
            
            # Orchestration Services
            'advanced_orchestration_engine': ServiceConfig(
                name="Advanced Orchestration Engine",
                host="localhost",
                port=5018,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379',
                    'JAEGER_HOST': 'localhost',
                    'JAEGER_PORT': '6831'
                },
                startup_command="python advanced_orchestration_engine.py",
                working_directory="backend"
            ),
            
            'service_mesh_proxy': ServiceConfig(
                name="Service Mesh Proxy",
                host="localhost",
                port=5019,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python service_mesh_proxy.py",
                working_directory="backend"
            ),
            
            'real_service_communication': ServiceConfig(
                name="Real Service Communication",
                host="localhost",
                port=5020,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python real_service_communication.py",
                working_directory="backend"
            ),
            
            'workflow_orchestrator': ServiceConfig(
                name="Workflow Orchestrator",
                host="localhost",
                port=5021,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python workflow_orchestrator.py",
                working_directory="backend"
            ),
            
            'distributed_tracing': ServiceConfig(
                name="Distributed Tracing",
                host="localhost",
                port=5022,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'JAEGER_HOST': 'localhost',
                    'JAEGER_PORT': '6831'
                },
                startup_command="python distributed_tracing.py",
                working_directory="backend"
            ),
            
            'event_driven_architecture': ServiceConfig(
                name="Event-Driven Architecture",
                host="localhost",
                port=5023,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python event_driven_architecture.py",
                working_directory="backend"
            ),
            
            # Backend Services
            'adaptive_onboarding_server': ServiceConfig(
                name="Adaptive Onboarding Server",
                host="localhost",
                port=5003,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python adaptive_onboarding_server.py",
                working_directory="backend"
            ),
            
            'ai_pricing_service': ServiceConfig(
                name="AI Pricing Service",
                host="localhost",
                port=5005,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_pricing_service.py",
                working_directory="backend"
            ),
            
            'ai_pricing_orchestrator': ServiceConfig(
                name="AI Pricing Orchestrator",
                host="localhost",
                port=8030,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_pricing_orchestrator.py",
                working_directory="backend"
            ),
            
            'meta_learning_orchestrator': ServiceConfig(
                name="Meta-Learning Orchestrator",
                host="localhost",
                port=8010,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python meta_learning_orchestrator.py",
                working_directory="backend"
            ),
            
            'ai_matchmaking_service': ServiceConfig(
                name="AI Matchmaking Service",
                host="localhost",
                port=8020,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_matchmaking_service.py",
                working_directory="backend"
            ),
            
            'materials_bert_service_simple': ServiceConfig(
                name="MaterialsBERT Simple Service",
                host="localhost",
                port=5002,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python materials_bert_service_simple.py",
                working_directory="backend"
            ),
            
            'ai_listings_generator': ServiceConfig(
                name="AI Listings Generator",
                host="localhost",
                port=5010,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_listings_generator.py",
                working_directory="backend"
            ),
            
            'ai_monitoring_dashboard': ServiceConfig(
                name="AI Monitoring Dashboard",
                host="localhost",
                port=5011,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_monitoring_dashboard.py",
                working_directory="backend"
            ),
            
            'ultra_ai_listings_generator': ServiceConfig(
                name="Ultra AI Listings Generator",
                host="localhost",
                port=5012,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ultra_ai_listings_generator.py",
                working_directory="backend"
            ),
            
            'regulatory_compliance': ServiceConfig(
                name="Regulatory Compliance Service",
                host="localhost",
                port=5013,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python regulatory_compliance.py",
                working_directory="backend"
            ),
            
            'proactive_opportunity_engine': ServiceConfig(
                name="Proactive Opportunity Engine",
                host="localhost",
                port=5014,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python proactive_opportunity_engine.py",
                working_directory="backend"
            ),
            
            'ai_feedback_orchestrator': ServiceConfig(
                name="AI Feedback Orchestrator",
                host="localhost",
                port=5015,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_feedback_orchestrator.py",
                working_directory="backend"
            ),
            
            'value_function_arbiter': ServiceConfig(
                name="Value Function Arbiter",
                host="localhost",
                port=5016,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python value_function_arbiter.py",
                working_directory="backend"
            ),
            
            'financial_analysis_engine': ServiceConfig(
                name="Financial Analysis Engine",
                host="localhost",
                port=5017,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python financial_analysis_engine.py",
                working_directory="."
            ),
            
            'logistics_cost_service': ServiceConfig(
                name="Logistics Cost Service",
                host="localhost",
                port=5006,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python logistics_cost_service.py",
                working_directory="."
            ),
            
            # AI Service Flask
            'ai_gateway': ServiceConfig(
                name="AI Gateway",
                host="localhost",
                port=8000,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_gateway.py",
                working_directory="ai_service_flask"
            ),
            
            'advanced_analytics_service': ServiceConfig(
                name="Advanced Analytics Service",
                host="localhost",
                port=5004,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python advanced_analytics_service.py",
                working_directory="ai_service_flask"
            ),
            
            'ai_pricing_service_wrapper': ServiceConfig(
                name="AI Pricing Service Wrapper",
                host="localhost",
                port=8002,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python ai_pricing_service_wrapper.py",
                working_directory="ai_service_flask"
            ),
            
            'gnn_inference_service': ServiceConfig(
                name="GNN Inference Service",
                host="localhost",
                port=8001,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python gnn_inference_service.py",
                working_directory="ai_service_flask"
            ),
            
            'logistics_service_wrapper': ServiceConfig(
                name="Logistics Service Wrapper",
                host="localhost",
                port=8003,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python logistics_service_wrapper.py",
                working_directory="ai_service_flask"
            ),
            
            'multi_hop_symbiosis_service': ServiceConfig(
                name="Multi-Hop Symbiosis Service",
                host="localhost",
                port=5003,
                health_endpoint="/health",
                startup_timeout=30,
                dependencies=['redis'],
                environment_vars={
                    'REDIS_HOST': 'localhost',
                    'REDIS_PORT': '6379'
                },
                startup_command="python multi_hop_symbiosis_service.py",
                working_directory="ai_service_flask"
            )
        }
    
    async def start_redis(self):
        """Start Redis server"""
        try:
            # Check if Redis is already running
            test_client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            test_client.ping()
            logger.info("✅ Redis is already running")
            return True
        except:
            logger.info("🔄 Starting Redis server...")
            try:
                # Try to start Redis (this might fail on Windows)
                process = subprocess.Popen(
                    ['redis-server'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await asyncio.sleep(3)
                
                # Test connection
                test_client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=5)
                test_client.ping()
                logger.info("✅ Redis started successfully")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Could not start Redis: {e}")
                logger.info("💡 Please start Redis manually: redis-server")
                return False
    
    async def start_service(self, service_name: str) -> bool:
        """Start a single service"""
        if service_name not in self.services:
            logger.error(f"❌ Service {service_name} not found")
            return False
        
        service = self.services[service_name]
        
        # Check dependencies
        for dep in service.dependencies:
            if dep not in self.service_status or self.service_status[dep] != 'healthy':
                logger.warning(f"⚠️ Dependency {dep} not ready for {service_name}")
                return False
        
        logger.info(f"🚀 Starting {service.name}...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env.update(service.environment_vars)
            
            # Start service process
            process = subprocess.Popen(
                service.startup_command.split(),
                cwd=service.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.service_processes[service_name] = process
            
            # Wait for service to start
            await asyncio.sleep(5)
            
            # Check health
            health_url = f"http://{service.host}:{service.port}{service.health_endpoint}"
            
            for attempt in range(service.startup_timeout // 5):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=5) as response:
                            if response.status == 200:
                                self.service_status[service_name] = 'healthy'
                                logger.info(f"✅ {service.name} started successfully")
                                return True
                except:
                    pass
                
                await asyncio.sleep(5)
            
            logger.error(f"❌ {service.name} failed to start")
            self.service_status[service_name] = 'failed'
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting {service.name}: {e}")
            self.service_status[service_name] = 'failed'
            return False
    
    async def start_all_services(self):
        """Start all services in dependency order"""
        logger.info("🚀 Starting SymbioFlows Production System...")
        
        # Start Redis first
        redis_ready = await self.start_redis()
        if redis_ready:
            self.service_status['redis'] = 'healthy'
        
        # Start services in dependency order
        started_services = set()
        max_attempts = 3
        
        for attempt in range(max_attempts):
            for service_name, service in self.services.items():
                if service_name in started_services:
                    continue
                
                # Check if all dependencies are ready
                dependencies_ready = all(
                    dep in self.service_status and self.service_status[dep] == 'healthy'
                    for dep in service.dependencies
                )
                
                if dependencies_ready:
                    success = await self.start_service(service_name)
                    if success:
                        started_services.add(service_name)
            
            # Check if all services are started
            if len(started_services) == len(self.services):
                break
            
            await asyncio.sleep(10)
        
        # Final status report
        logger.info("📊 Service Status Report:")
        for service_name, status in self.service_status.items():
            status_icon = "✅" if status == 'healthy' else "❌"
            logger.info(f"{status_icon} {service_name}: {status}")
        
        healthy_count = sum(1 for status in self.service_status.values() if status == 'healthy')
        total_count = len(self.services)
        
        logger.info(f"🎯 {healthy_count}/{total_count} services healthy")
        
        if healthy_count == total_count:
            logger.info("🎉 ALL SERVICES STARTED SUCCESSFULLY!")
            return True
        else:
            logger.warning("⚠️ Some services failed to start")
            return False
    
    async def health_check_all(self):
        """Check health of all services"""
        logger.info("🔍 Performing health check...")
        
        async with aiohttp.ClientSession() as session:
            for service_name, service in self.services.items():
                try:
                    health_url = f"http://{service.host}:{service.port}{service.health_endpoint}"
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            self.service_status[service_name] = 'healthy'
                        else:
                            self.service_status[service_name] = 'unhealthy'
                except:
                    self.service_status[service_name] = 'unhealthy'
        
        healthy_count = sum(1 for status in self.service_status.values() if status == 'healthy')
        total_count = len(self.services)
        
        logger.info(f"📊 Health Check: {healthy_count}/{total_count} services healthy")
        return healthy_count == total_count
    
    def stop_all_services(self):
        """Stop all services"""
        logger.info("🛑 Stopping all services...")
        
        for service_name, process in self.service_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped {service_name}")
            except:
                try:
                    process.kill()
                    logger.info(f"🔪 Force killed {service_name}")
                except:
                    pass

# Initialize system integrator
system_integrator = SystemIntegrator()

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Complete System Integration',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/start', methods=['POST'])
async def start_system():
    """Start the complete system"""
    try:
        success = await system_integrator.start_all_services()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'All services started successfully',
                'services': system_integrator.service_status
            })
        else:
            return jsonify({
                'status': 'partial',
                'message': 'Some services failed to start',
                'services': system_integrator.service_status
            }), 500
        
    except Exception as e:
        logger.error(f"Start system error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    return jsonify({
        'services': system_integrator.service_status,
        'total_services': len(system_integrator.services),
        'healthy_services': sum(1 for status in system_integrator.service_status.values() if status == 'healthy')
    })

@app.route('/stop', methods=['POST'])
def stop_system():
    """Stop the complete system"""
    try:
        system_integrator.stop_all_services()
        return jsonify({
            'status': 'success',
            'message': 'All services stopped'
        })
    except Exception as e:
        logger.error(f"Stop system error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Complete System Integration on port 5024...")
    app.run(host='0.0.0.0', port=5024, debug=False) 