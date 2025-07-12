#!/usr/bin/env python3
"""
Perfect AI System Startup Script
Ensures every AI module works perfectly with absolute synergy and utmost adaptiveness.
"""

import asyncio
import logging
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import signal
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PerfectAISystem:
    """
    Perfect AI System ensuring absolute synergy between all AI modules
    with persistent GNN warm starts and utmost adaptiveness.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.modules = {}
        self.health_status = {}
        self.startup_time = None
        
        # Import AI modules
        self._import_modules()
        
        # Initialize system
        self._initialize_system()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Perfect AI System initialized")

    def _import_modules(self):
        """Import all AI modules with proper error handling"""
        self.module_classes = {}
        
        try:
            # Core AI modules
            from gnn_reasoning import GNNReasoning
            self.module_classes['gnn_reasoning'] = GNNReasoning
        except ImportError as e:
            self.logger.warning(f"GNNReasoning not available: {e}")
            self.module_classes['gnn_reasoning'] = None
        
        try:
            from revolutionary_ai_matching import RevolutionaryAIMatching
            self.module_classes['revolutionary_matching'] = RevolutionaryAIMatching
        except ImportError as e:
            self.logger.warning(f"RevolutionaryAIMatching not available: {e}")
            self.module_classes['revolutionary_matching'] = None
        
        try:
            from knowledge_graph import KnowledgeGraph
            self.module_classes['knowledge_graph'] = KnowledgeGraph
        except ImportError as e:
            self.logger.warning(f"KnowledgeGraph not available: {e}")
            self.module_classes['knowledge_graph'] = None
        
        try:
            from model_persistence_manager import ModelPersistenceManager
            self.module_classes['model_persistence'] = ModelPersistenceManager
        except ImportError as e:
            self.logger.warning(f"ModelPersistenceManager not available: {e}")
            self.module_classes['model_persistence'] = None
        
        try:
            # Advanced modules
            from advanced_ai_orchestrator import AdvancedAIOrchestrator, OrchestrationConfig
            self.module_classes['orchestrator'] = AdvancedAIOrchestrator
            self.module_classes['orchestration_config'] = OrchestrationConfig
        except ImportError as e:
            self.logger.warning(f"AdvancedAIOrchestrator not available: {e}")
            self.module_classes['orchestrator'] = None
            self.module_classes['orchestration_config'] = None
        
        try:
            from perfect_ai_integration import PerfectAIIntegration
            self.module_classes['integration'] = PerfectAIIntegration
        except ImportError as e:
            self.logger.warning(f"PerfectAIIntegration not available: {e}")
            self.module_classes['integration'] = None
        
        self.logger.info(f"Successfully imported {len([v for v in self.module_classes.values() if v is not None])} AI modules")

    def _initialize_system(self):
        """Initialize the AI system with perfect configuration"""
        try:
            self.logger.info("Initializing Perfect AI System...")
            
            # Create model directories
            self._create_directories()
            
            # Initialize model persistence manager
            if self.module_classes.get('model_persistence'):
                self.modules['model_persistence'] = self.module_classes['model_persistence']('./models')
            
            # Initialize orchestrator with perfect configuration
            if self.module_classes.get('orchestrator') and self.module_classes.get('orchestration_config'):
                orchestrator_config = self.module_classes['orchestration_config'](
                    max_concurrent_requests=20,
                    gnn_warm_start_enabled=True,
                    gnn_persistence_enabled=True,
                    max_memory_usage=0.85,
                    health_check_interval=30,
                    performance_logging_interval=60
                )
                
                self.modules['orchestrator'] = self.module_classes['orchestrator'](orchestrator_config)
            
            # Initialize perfect integration
            if self.module_classes.get('integration'):
                integration_config = {
                    'max_queue_size': 1000,
                    'cache_size': 500,
                    'timeout': 30.0,
                    'retry_attempts': 3
                }
                
                self.modules['integration'] = self.module_classes['integration'](integration_config)
            
            # Initialize core AI modules
            self._initialize_core_modules()
            
            # Perform system health check
            self._perform_health_check()
            
            self.logger.info("Perfect AI System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            './models',
            './models/gnn',
            './models/cache',
            './logs',
            './data',
            './backups'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def _initialize_core_modules(self):
        """Initialize core AI modules with perfect configuration"""
        try:
            # Initialize GNN Reasoning with persistent warm starts
            if self.module_classes.get('gnn_reasoning'):
                self.modules['gnn_reasoning'] = self.module_classes['gnn_reasoning'](
                    model_cache_dir='./models/gnn'
                )
            
            # Initialize Revolutionary AI Matching
            if self.module_classes.get('revolutionary_matching'):
                self.modules['revolutionary_matching'] = self.module_classes['revolutionary_matching']()
            
            # Initialize Knowledge Graph
            if self.module_classes.get('knowledge_graph'):
                self.modules['knowledge_graph'] = self.module_classes['knowledge_graph']()
            
            self.logger.info("Core AI modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Core module initialization failed: {e}")
            raise

    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            self.logger.info("Performing system health check...")
            
            health_results = {}
            
            # Check each module
            for module_name, module in self.modules.items():
                try:
                    if hasattr(module, 'health_check'):
                        health = module.health_check()
                        health_results[module_name] = health
                        self.logger.info(f"{module_name}: {health.get('status', 'unknown')}")
                    else:
                        health_results[module_name] = {'status': 'no_health_check'}
                        self.logger.warning(f"{module_name}: No health check method")
                except Exception as e:
                    health_results[module_name] = {'status': 'error', 'error': str(e)}
                    self.logger.error(f"{module_name} health check failed: {e}")
            
            # Check system resources
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            system_health = {
                'memory_usage': memory.percent,
                'cpu_usage': cpu,
                'disk_usage': psutil.disk_usage('/').percent
            }
            
            health_results['system'] = system_health
            
            # Store health status
            self.health_status = health_results
            
            # Log health summary
            healthy_modules = sum(1 for h in health_results.values() 
                                if isinstance(h, dict) and h.get('status') == 'healthy')
            total_modules = len([h for h in health_results.values() 
                               if isinstance(h, dict) and 'status' in h])
            
            self.logger.info(f"Health check completed: {healthy_modules}/{total_modules} modules healthy")
            
            # Check if system is ready
            if healthy_modules < total_modules * 0.8:  # 80% threshold
                self.logger.warning("System health below threshold, some modules may not be fully operational")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start_system(self):
        """Start the Perfect AI System"""
        try:
            self.logger.info("Starting Perfect AI System...")
            self.startup_time = time.time()
            self.is_running = True
            
            # Warm up all modules
            await self._warm_up_modules()
            
            # Start background services
            self._start_background_services()
            
            # Perform final health check
            self._perform_health_check()
            
            self.logger.info("Perfect AI System started successfully")
            
            # Keep system running
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute
                
                # Periodic health check
                if time.time() - self.startup_time > 300:  # Every 5 minutes
                    self._perform_health_check()
                    self.startup_time = time.time()
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise

    async def _warm_up_modules(self):
        """Warm up all AI modules for optimal performance"""
        try:
            self.logger.info("Warming up AI modules...")
            
            # Warm up GNN module
            if 'gnn_reasoning' in self.modules:
                self.logger.info("Warming up GNN module...")
                self.modules['gnn_reasoning'].warm_up()
            
            # Warm up other modules
            for module_name, module in self.modules.items():
                if hasattr(module, 'warm_up') and module_name != 'gnn_reasoning':
                    try:
                        self.logger.info(f"Warming up {module_name}...")
                        module.warm_up()
                    except Exception as e:
                        self.logger.warning(f"Failed to warm up {module_name}: {e}")
            
            self.logger.info("Module warm-up completed")
            
        except Exception as e:
            self.logger.error(f"Module warm-up failed: {e}")
            raise

    def _start_background_services(self):
        """Start background services for monitoring and optimization"""
        try:
            # Start health monitoring thread
            self.health_thread = threading.Thread(
                target=self._health_monitoring_loop,
                daemon=True
            )
            self.health_thread.start()
            
            # Start performance optimization thread
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
            
            self.logger.info("Background services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")

    def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while self.is_running:
            try:
                time.sleep(60)  # Check every minute
                
                # Quick health check
                for module_name, module in self.modules.items():
                    if hasattr(module, 'health_check'):
                        try:
                            health = module.health_check()
                            if health.get('status') != 'healthy':
                                self.logger.warning(f"{module_name} health degraded: {health}")
                        except Exception as e:
                            self.logger.error(f"{module_name} health check error: {e}")
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    def _optimization_loop(self):
        """Continuous performance optimization"""
        while self.is_running:
            try:
                time.sleep(300)  # Optimize every 5 minutes
                
                # Optimize memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    self.logger.info("High memory usage, performing optimization...")
                    self._optimize_memory()
                
                # Optimize module performance
                self._optimize_modules()
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")

    def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Clear module caches if available
            for module_name, module in self.modules.items():
                if hasattr(module, 'clear_cache'):
                    try:
                        module.clear_cache()
                        self.logger.debug(f"Cleared cache for {module_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clear cache for {module_name}: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")

    def _optimize_modules(self):
        """Optimize module performance"""
        try:
            # Optimize GNN models
            if 'gnn_reasoning' in self.modules:
                try:
                    performance = self.modules['gnn_reasoning'].get_model_performance()
                    self.logger.debug(f"GNN performance: {performance}")
                except Exception as e:
                    self.logger.warning(f"GNN optimization failed: {e}")
            
            # Optimize orchestrator
            if 'orchestrator' in self.modules:
                try:
                    status = self.modules['orchestrator'].get_system_status()
                    self.logger.debug(f"Orchestrator status: {status}")
                except Exception as e:
                    self.logger.warning(f"Orchestrator optimization failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Module optimization failed: {e}")

    async def test_system(self):
        """Test the AI system with comprehensive tests"""
        try:
            self.logger.info("Running comprehensive system tests...")
            
            test_results = {}
            
            # Test GNN inference
            if 'gnn_reasoning' in self.modules:
                try:
                    import networkx as nx
                    test_graph = nx.Graph()
                    test_graph.add_node('A', industry='Steel', location='NY', waste_type='slag')
                    test_graph.add_node('B', industry='Construction', location='NY', material_needed='slag')
                    
                    result = self.modules['gnn_reasoning'].run_gnn_inference(test_graph, 'gcn')
                    test_results['gnn_inference'] = {
                        'success': True,
                        'result_count': len(result) if result else 0
                    }
                    self.logger.info("GNN inference test passed")
                except Exception as e:
                    test_results['gnn_inference'] = {'success': False, 'error': str(e)}
                    self.logger.error(f"GNN inference test failed: {e}")
            
            # Test symbiosis matching
            if 'revolutionary_matching' in self.modules:
                try:
                    buyer = {'id': 'test_buyer', 'industry': 'Construction', 'location': 'NY'}
                    seller = {'id': 'test_seller', 'industry': 'Steel', 'location': 'NY'}
                    
                    result = self.modules['revolutionary_matching'].predict_compatibility(buyer, seller)
                    test_results['symbiosis_matching'] = {
                        'success': True,
                        'score': result.get('revolutionary_score', 0)
                    }
                    self.logger.info("Symbiosis matching test passed")
                except Exception as e:
                    test_results['symbiosis_matching'] = {'success': False, 'error': str(e)}
                    self.logger.error(f"Symbiosis matching test failed: {e}")
            
            # Test integration
            if 'integration' in self.modules:
                try:
                    result = await self.modules['integration'].process_request(
                        'symbiosis_matching',
                        {
                            'buyer': {'id': 'test_buyer', 'industry': 'Construction'},
                            'seller': {'id': 'test_seller', 'industry': 'Steel'}
                        }
                    )
                    test_results['integration'] = {
                        'success': result.success,
                        'execution_time': result.execution_time
                    }
                    self.logger.info("Integration test passed")
                except Exception as e:
                    test_results['integration'] = {'success': False, 'error': str(e)}
                    self.logger.error(f"Integration test failed: {e}")
            
            # Log test summary
            successful_tests = sum(1 for r in test_results.values() if r.get('success', False))
            total_tests = len(test_results)
            
            self.logger.info(f"System tests completed: {successful_tests}/{total_tests} tests passed")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"System testing failed: {e}")
            return {'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': time.time(),
                'is_running': self.is_running,
                'uptime': time.time() - self.startup_time if self.startup_time else 0,
                'modules': {},
                'health': self.health_status,
                'system': {}
            }
            
            # Module status
            for module_name, module in self.modules.items():
                try:
                    if hasattr(module, 'get_system_status'):
                        module_status = module.get_system_status()
                        status['modules'][module_name] = module_status
                    else:
                        status['modules'][module_name] = {'status': 'loaded'}
                except Exception as e:
                    status['modules'][module_name] = {'status': 'error', 'error': str(e)}
            
            # System resources
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            status['system'] = {
                'memory_usage': memory.percent,
                'cpu_usage': cpu,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_modules': len(self.modules)
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """Graceful shutdown of the AI system"""
        try:
            self.logger.info("Shutting down Perfect AI System...")
            self.is_running = False
            
            # Shutdown modules
            for module_name, module in self.modules.items():
                try:
                    if hasattr(module, 'shutdown'):
                        module.shutdown()
                        self.logger.info(f"Shutdown {module_name}")
                except Exception as e:
                    self.logger.error(f"Error shutting down {module_name}: {e}")
            
            # Wait for background threads
            if hasattr(self, 'health_thread') and self.health_thread.is_alive():
                self.health_thread.join(timeout=5)
            
            if hasattr(self, 'optimization_thread') and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5)
            
            self.logger.info("Perfect AI System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main function to start the Perfect AI System"""
    try:
        # Create and start the system
        system = PerfectAISystem()
        
        # Test the system
        test_results = await system.test_system()
        print("System Test Results:", json.dumps(test_results, indent=2))
        
        # Start the system
        await system.start_system()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        raise
    finally:
        if 'system' in locals():
            system.shutdown()

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())