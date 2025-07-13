#!/usr/bin/env python3
"""
Enhanced Materials System Startup
Launches the complete enhanced materials system with Next Gen Materials API and MaterialsBERT
for maximum potential utilization in industrial symbiosis.
"""

import os
import sys
import json
import logging
import subprocess
import time
import requests
import threading
from typing import Dict, List, Any
from datetime import datetime
import signal
import atexit

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_materials_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMaterialsSystem:
    """
    Complete enhanced materials system with Next Gen Materials API and MaterialsBERT integration.
    """
    
    def __init__(self):
        self.services = {}
        self.processes = []
        self.running = False
        self.config = self.load_configuration()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open('enhanced_materials_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Configuration file not found, using defaults")
            return {
                'services': {
                    'materialsbert': {'port': 8001, 'enabled': True},
                    'enhanced_materials': {'port': 3000, 'enabled': True},
                    'ai_prompts': {'port': 3001, 'enabled': True}
                },
                'api_keys': {
                    'next_gen_materials': os.environ.get('NEXT_GEN_MATERIALS_API_KEY'),
                    'deepseek': os.environ.get('DEEPSEEK_API_KEY')
                }
            }
    
    def start_complete_system(self):
        """Start the complete enhanced materials system"""
        logger.info("STARTING ENHANCED MATERIALS SYSTEM")
        logger.info("=" * 60)
        logger.info("Maximum Potential Utilization Mode")
        logger.info("=" * 60)
        
        try:
            # Step 1: Validate environment
            self.validate_environment()
            
            # Step 2: Start MaterialsBERT service
            self.start_materialsbert_service()
            
            # Step 3: Start enhanced materials service
            self.start_enhanced_materials_service()
            
            # Step 4: Start AI prompts service
            self.start_ai_prompts_service()
            
            # Step 5: Initialize monitoring
            self.initialize_monitoring()
            
            # Step 6: Run system tests
            self.run_system_tests()
            
            # Step 7: Start performance optimization
            self.start_performance_optimization()
            
            # Step 8: Display system status
            self.display_system_status()
            
            # Step 9: Start interactive mode
            self.start_interactive_mode()
            
        except Exception as e:
            logger.error(f"System startup failed: {str(e)}")
            self.cleanup()
            sys.exit(1)
    
    def validate_environment(self):
        """Validate environment and dependencies"""
        logger.info("VALIDATING ENVIRONMENT")
        logger.info("-" * 40)
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            raise RuntimeError("Python 3.8+ required")
        logger.info(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = [
            'torch', 'transformers', 'flask', 'requests', 'numpy', 'scipy', 'scikit-learn'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"[OK] {package}")
            except ImportError:
                logger.warning(f"[WARN] {package} not found - some features may be limited")
        
        # Check environment variables
        env_vars = [
            'NEXT_GEN_MATERIALS_API_KEY',
            'DEEPSEEK_API_KEY',
            'MATERIALSBERT_ENABLED',
            'MATERIALSBERT_ENDPOINT'
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                logger.info(f"[OK] {var}: Configured")
            else:
                logger.warning(f"[WARN] {var}: Not configured")
        
        # Check Docker availability
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            logger.info("[OK] Docker: Available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("[WARN] Docker: Not available - MaterialsBERT will run in Python mode")
    
    def start_materialsbert_service(self):
        """Start MaterialsBERT service"""
        logger.info("STARTING MATERIALSBERT SERVICE")
        logger.info("-" * 40)
        
        if not self.config.get('services', {}).get('materialsbert', {}).get('enabled', True):
            logger.info("[SKIP] MaterialsBERT service disabled")
            return
        
        try:
            # Try to start with Docker first
            if self.start_materialsbert_docker():
                logger.info("[OK] MaterialsBERT service started with Docker")
                return
            
            # Fallback to Python mode
            if self.start_materialsbert_python():
                logger.info("[OK] MaterialsBERT service started in Python mode")
                return
            
            logger.error("[ERROR] Failed to start MaterialsBERT service")
            
        except Exception as e:
            logger.error(f"[ERROR] Error starting MaterialsBERT service: {str(e)}")
    
    def start_materialsbert_docker(self) -> bool:
        """Start MaterialsBERT service with Docker"""
        try:
            # Check if Docker Compose file exists
            if not os.path.exists('docker-compose.materialsbert.yml'):
                logger.info("[INFO] Creating Docker Compose configuration...")
                self.create_materialsbert_docker_compose()
            
            # Start service
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.materialsbert.yml', 'up', '-d'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Wait for service to be ready
                for i in range(30):
                    try:
                        response = requests.get('http://localhost:8001/health', timeout=5)
                        if response.status_code == 200:
                            return True
                    except requests.exceptions.ConnectionError:
                        pass
                    time.sleep(1)
            
            return False
            
        except Exception as e:
            logger.warning(f"Docker start failed: {str(e)}")
            return False
    
    def start_materialsbert_python(self) -> bool:
        """Start MaterialsBERT service in Python mode"""
        try:
            # Start MaterialsBERT service in a separate process
            process = subprocess.Popen([
                sys.executable, 'materials_bert_service.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(('materialsbert', process))
            
            # Wait for service to be ready
            for i in range(30):
                try:
                    response = requests.get('http://localhost:8001/health', timeout=5)
                    if response.status_code == 200:
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
            
            return False
            
        except Exception as e:
            logger.error(f"Python start failed: {str(e)}")
            return False
    
    def create_materialsbert_docker_compose(self):
        """Create Docker Compose configuration for MaterialsBERT"""
        docker_compose = {
            'version': '3.8',
            'services': {
                'materialsbert': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'materials_bert_dockerfile'
                    },
                    'ports': ['8001:8001'],
                    'environment': [
                        'MATERIALSBERT_ENABLED=true',
                        'MATERIALSBERT_CACHE_SIZE=1000'
                    ],
                    'volumes': [
                        './model_storage:/app/model_storage'
                    ],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8001/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': '3'
                    },
                    'restart': 'unless-stopped'
                }
            }
        }
        
        import yaml
        with open('docker-compose.materialsbert.yml', 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
    
    def start_enhanced_materials_service(self):
        """Start enhanced materials service"""
        logger.info("STARTING ENHANCED MATERIALS SERVICE")
        logger.info("-" * 40)
        
        if not self.config.get('services', {}).get('enhanced_materials', {}).get('enabled', True):
            logger.info("[SKIP] Enhanced materials service disabled")
            return
        
        try:
            # Import and initialize enhanced materials service
            from enhancedMaterialsService import EnhancedMaterialsService
            
            # Create service instance
            service = EnhancedMaterialsService()
            
            # Test service
            test_result = service.getComprehensiveMaterialAnalysis(
                'polyethylene_terephthalate',
                {'industry': 'packaging', 'location': 'Global'}
            )
            
            if test_result:
                self.services['enhanced_materials'] = service
                logger.info("[OK] Enhanced materials service initialized")
                logger.info(f"   [INFO] Next Gen API: {'Available' if test_result.get('material') else 'Not available'}")
                logger.info(f"   [INFO] MaterialsBERT: {'Available' if test_result.get('materials_bert_insights') else 'Not available'}")
            else:
                logger.warning("[WARN] Enhanced materials service test failed")
            
        except Exception as e:
            logger.error(f"[ERROR] Error starting enhanced materials service: {str(e)}")
    
    def start_ai_prompts_service(self):
        """Start AI prompts service"""
        logger.info("STARTING AI PROMPTS SERVICE")
        logger.info("-" * 40)
        
        if not self.config.get('services', {}).get('ai_prompts', {}).get('enabled', True):
            logger.info("[SKIP] AI prompts service disabled")
            return
        
        try:
            # Import and initialize AI prompts service
            from advanced_ai_prompts_service import AdvancedAIPromptsService
            
            # Create service instance
            service = AdvancedAIPromptsService()
            
            # Test service
            test_profile = {
                'name': 'Test Company',
                'industry': 'manufacturing',
                'main_materials': 'steel, aluminum, polymers'
            }
            
            test_result = service.strategic_material_analysis(test_profile)
            
            if test_result and 'executive_summary' in test_result:
                self.services['ai_prompts'] = service
                logger.info("[OK] AI prompts service initialized")
                logger.info(f"   [INFO] MaterialsBERT Integration: {'Enabled' if service.materialsbert_enabled else 'Disabled'}")
                logger.info(f"   [INFO] Analysis Generated: {len(test_result.get('predicted_outputs', []))} outputs")
            else:
                logger.warning("[WARN] AI prompts service test failed")
            
        except Exception as e:
            logger.error(f"[ERROR] Error starting AI prompts service: {str(e)}")
    
    def initialize_monitoring(self):
        """Initialize system monitoring"""
        logger.info("INITIALIZING MONITORING")
        logger.info("-" * 40)
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("[OK] Monitoring initialized")
    
    def monitoring_loop(self):
        """Monitoring loop for system health"""
        while self.running:
            try:
                # Check MaterialsBERT service
                try:
                    response = requests.get('http://localhost:8001/health', timeout=5)
                    if response.status_code != 200:
                        logger.warning("[WARN] MaterialsBERT service health check failed")
                except requests.exceptions.ConnectionError:
                    logger.warning("[WARN] MaterialsBERT service not responding")
                
                # Check other services
                for service_name, service in self.services.items():
                    if hasattr(service, 'health_check'):
                        try:
                            health = service.health_check()
                            if not health:
                                logger.warning(f"[WARN] {service_name} health check failed")
                        except Exception as e:
                            logger.warning(f"[WARN] {service_name} health check error: {str(e)}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)
    
    def run_system_tests(self):
        """Run comprehensive system tests"""
        logger.info("RUNNING SYSTEM TESTS")
        logger.info("-" * 40)
        
        # Test 1: MaterialsBERT analysis
        logger.info("Test 1: MaterialsBERT Analysis")
        try:
            response = requests.post(
                'http://localhost:8001/analyze',
                json={
                    'text': 'Material: graphene. Single layer of carbon atoms.',
                    'material_name': 'graphene',
                    'context': {'industry': 'electronics'}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"   [OK] Classification: {result.get('material_classification', {}).get('predicted_category', 'N/A')}")
                logger.info(f"   [OK] Confidence: {result.get('material_classification', {}).get('confidence', 'N/A')}")
            else:
                logger.warning(f"   [WARN] Status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"   [ERROR] Error: {str(e)}")
        
        # Test 2: Enhanced materials service
        logger.info("Test 2: Enhanced Materials Service")
        try:
            if 'enhanced_materials' in self.services:
                service = self.services['enhanced_materials']
                result = service.getComprehensiveMaterialAnalysis(
                    'carbon_fiber',
                    {'industry': 'aerospace', 'location': 'Global'}
                )
                
                if result:
                    logger.info(f"   [OK] Next Gen Score: {result.get('material', {}).get('next_gen_score', {}).get('overall_score', 'N/A')}")
                    logger.info(f"   [OK] AI Enhanced: {'Yes' if result.get('ai_enhanced_insights') else 'No'}")
                else:
                    logger.warning("   [WARN] No result returned")
            else:
                logger.warning("   [WARN] Service not available")
                
        except Exception as e:
            logger.error(f"   [ERROR] Error: {str(e)}")
        
        # Test 3: AI prompts service
        logger.info("Test 3: AI Prompts Service")
        try:
            if 'ai_prompts' in self.services:
                service = self.services['ai_prompts']
                test_profile = {
                    'name': 'Advanced Materials Corp',
                    'industry': 'chemical',
                    'main_materials': 'polymers, composites, nanomaterials'
                }
                
                result = service.strategic_material_analysis(test_profile)
                
                if result and 'executive_summary' in result:
                    logger.info(f"   [OK] Analysis Generated: {len(result.get('predicted_outputs', []))} outputs")
                    logger.info(f"   [OK] AI Enhanced: {'Yes' if result.get('ai_enhanced_analysis') else 'No'}")
                else:
                    logger.warning("   [WARN] No analysis generated")
            else:
                logger.warning("   [WARN] Service not available")
                
        except Exception as e:
            logger.error(f"   [ERROR] Error: {str(e)}")
    
    def start_performance_optimization(self):
        """Start performance optimization"""
        logger.info("STARTING PERFORMANCE OPTIMIZATION")
        logger.info("-" * 40)
        
        # Start performance monitoring thread
        perf_thread = threading.Thread(target=self.performance_optimization_loop, daemon=True)
        perf_thread.start()
        
        logger.info("[OK] Performance optimization started")
    
    def performance_optimization_loop(self):
        """Performance optimization loop"""
        while self.running:
            try:
                # Monitor system performance
                import psutil
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 80:
                    logger.warning(f"[WARN] High CPU usage: {cpu_percent}%")
                
                # Memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    logger.warning(f"[WARN] High memory usage: {memory.percent}%")
                
                # Optimize cache if needed
                for service_name, service in self.services.items():
                    if hasattr(service, 'optimize_cache'):
                        try:
                            service.optimize_cache()
                        except Exception as e:
                            logger.debug(f"Cache optimization error for {service_name}: {str(e)}")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance optimization error: {str(e)}")
                time.sleep(300)
    
    def display_system_status(self):
        """Display system status"""
        logger.info("SYSTEM STATUS")
        logger.info("=" * 60)
        
        # Service status
        logger.info("Services:")
        for service_name, service in self.services.items():
            logger.info(f"   [OK] {service_name}: Running")
        
        # API status
        logger.info("API Status:")
        try:
            response = requests.get('http://localhost:8001/health', timeout=5)
            if response.status_code == 200:
                logger.info("   [OK] MaterialsBERT API: Online")
            else:
                logger.warning("   [WARN] MaterialsBERT API: Unhealthy")
        except:
            logger.error("   [ERROR] MaterialsBERT API: Offline")
        
        # Configuration
        logger.info("Configuration:")
        logger.info(f"   [INFO] MaterialsBERT: {'Enabled' if os.environ.get('MATERIALSBERT_ENABLED') == 'true' else 'Disabled'}")
        logger.info(f"   [INFO] Next Gen API: {'Configured' if os.environ.get('NEXT_GEN_MATERIALS_API_KEY') else 'Not configured'}")
        logger.info(f"   [INFO] DeepSeek API: {'Configured' if os.environ.get('DEEPSEEK_API_KEY') else 'Not configured'}")
        
        # Performance
        logger.info("Performance:")
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            logger.info(f"   [INFO] CPU Usage: {cpu_percent}%")
            logger.info(f"   [INFO] Memory Usage: {memory.percent}%")
        except ImportError:
            logger.info("   [INFO] psutil not available for performance monitoring")
        
        logger.info("System is ready for maximum potential utilization!")
        logger.info("=" * 60)
    
    def start_interactive_mode(self):
        """Start interactive mode for user interaction"""
        logger.info("STARTING INTERACTIVE MODE")
        logger.info("-" * 40)
        logger.info("Type 'help' for available commands")
        logger.info("Type 'exit' to stop the system")
        
        self.running = True
        
        while self.running:
            try:
                command = input("\nCommand: ").strip().lower()
                
                if command == 'exit':
                    logger.info("Stopping system...")
                    self.running = False
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'status':
                    self.display_system_status()
                elif command == 'test':
                    self.run_system_tests()
                elif command == 'analyze':
                    self.interactive_analysis()
                elif command == 'optimize':
                    self.interactive_optimization()
                elif command == 'monitor':
                    self.show_monitoring_data()
                else:
                    logger.info("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                logger.info("\nStopping system...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Command error: {str(e)}")
    
    def show_help(self):
        """Show available commands"""
        help_text = """
Available Commands:
  help     - Show this help message
  status   - Display system status
  test     - Run system tests
  analyze  - Interactive material analysis
  optimize - Performance optimization
  monitor  - Show monitoring data
  exit     - Stop the system
        """
        print(help_text)
    
    def interactive_analysis(self):
        """Interactive material analysis"""
        logger.info("INTERACTIVE MATERIAL ANALYSIS")
        logger.info("-" * 40)
        
        material_name = input("Enter material name: ").strip()
        if not material_name:
            logger.info("[ERROR] Material name required")
            return
        
        industry = input("Enter industry (optional): ").strip() or "general"
        location = input("Enter location (optional): ").strip() or "Global"
        
        context = {
            'industry': industry,
            'location': location
        }
        
        logger.info(f"Analyzing {material_name}...")
        
        try:
            if 'enhanced_materials' in self.services:
                service = self.services['enhanced_materials']
                result = service.getComprehensiveMaterialAnalysis(material_name, context)
                
                if result:
                    logger.info("[OK] Analysis completed!")
                    logger.info(f"   [INFO] Next Gen Score: {result.get('material', {}).get('next_gen_score', {}).get('overall_score', 'N/A')}%")
                    logger.info(f"   [INFO] Sustainability Score: {result.get('sustainability_score', {}).get('overall_score', 'N/A')}%")
                    logger.info(f"   [INFO] Business Opportunity: {result.get('business_opportunity_score', {}).get('overall_score', 'N/A')}%")
                    logger.info(f"   [INFO] AI Enhanced: {'Yes' if result.get('ai_enhanced_insights') else 'No'}")
                else:
                    logger.warning("[WARN] Analysis failed")
            else:
                logger.error("[ERROR] Enhanced materials service not available")
                
        except Exception as e:
            logger.error(f"[ERROR] Analysis error: {str(e)}")
    
    def interactive_optimization(self):
        """Interactive performance optimization"""
        logger.info("PERFORMANCE OPTIMIZATION")
        logger.info("-" * 40)
        
        logger.info("Optimizing system performance...")
        
        try:
            # Optimize cache
            for service_name, service in self.services.items():
                if hasattr(service, 'optimize_cache'):
                    service.optimize_cache()
                    logger.info(f"   [OK] {service_name} cache optimized")
            
            # Clear old data
            import gc
            gc.collect()
            logger.info("   [OK] Memory cleaned")
            
            logger.info("[OK] Performance optimization completed")
            
        except Exception as e:
            logger.error(f"[ERROR] Optimization error: {str(e)}")
    
    def show_monitoring_data(self):
        """Show monitoring data"""
        logger.info("MONITORING DATA")
        logger.info("-" * 40)
        
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            logger.info(f"[INFO] CPU Usage: {cpu_percent}%")
            logger.info(f"[INFO] Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
            logger.info(f"[INFO] Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
            
            # Service metrics
            logger.info(f"[INFO] Active Services: {len(self.services)}")
            for service_name in self.services.keys():
                logger.info(f"   [OK] {service_name}")
        except ImportError:
            logger.info("[INFO] psutil not available for monitoring")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, stopping system...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up system resources...")
        
        # Stop all processes
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"   [OK] Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"   [WARN] Force killed {name}")
            except Exception as e:
                logger.error(f"   [ERROR] Error stopping {name}: {str(e)}")
        
        # Stop Docker containers
        try:
            subprocess.run(['docker-compose', '-f', 'docker-compose.materialsbert.yml', 'down'], 
                         capture_output=True)
            logger.info("   [OK] Stopped Docker containers")
        except Exception as e:
            logger.debug(f"Docker cleanup error: {str(e)}")
        
        logger.info("[OK] Cleanup completed")

def main():
    """Main function to start the enhanced materials system"""
    system = EnhancedMaterialsSystem()
    system.start_complete_system()

if __name__ == "__main__":
    main() 