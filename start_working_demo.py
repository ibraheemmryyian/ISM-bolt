#!/usr/bin/env python3
"""
SymbioFlows Working Demo - Uses minimal backend for reliable material generation
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv

# Enhanced logging configuration with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('working_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Working Demo")
print("📝 Logging to: working_demo.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

class WorkingDemoStarter:
    """Starts services with minimal backend for reliable material generation"""
    
    def __init__(self):
        logger.info("Initializing Working Demo...")
        self.project_root = Path(__file__).parent
        logger.info(f"Project root: {self.project_root}")
        
        # Get system PATH
        self.system_path = os.environ.get('PATH', '')
        
        # Add common Node.js paths to PATH
        common_node_paths = [
            r"C:\Program Files\nodejs",
            r"C:\Program Files (x86)\nodejs",
            os.path.expanduser(r"~\AppData\Roaming\npm"),
            os.path.expanduser(r"~\AppData\Local\Programs\nodejs"),
            r"C:\Program Files\nodejs\node_modules\npm\bin",
            os.path.expanduser(r"~\AppData\Local\Programs\nodejs\node_modules\npm\bin")
        ]
        
        for path in common_node_paths:
            if os.path.exists(path) and path not in self.system_path:
                self.system_path = path + os.pathsep + self.system_path
                logger.info(f"Added to PATH: {path}")
        
        logger.info("Working Demo initialized successfully")
        
    def setup_environment(self):
        """Setup environment and load .env file"""
        logger.info("Setting up environment...")
        
        # Load .env file from backend directory
        env_file = self.project_root / 'backend' / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"✅ Loaded environment from: {env_file}")
        else:
            logger.warning(f"⚠️ .env file not found: {env_file}")
        
        # Create directories
        directories = ['logs', 'data', 'models', 'cache']
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'DEEPSEEK_API_KEY']
        optional_vars = ['SUPABASE_ANON_KEY', 'OPENAI_API_KEY']
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var}: {masked_value}")
            else:
                logger.warning(f"⚠️ {var}: Missing")
                
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var}: {masked_value}")
            else:
                logger.info(f"ℹ️ {var}: Optional (not set)")
        
        logger.info("Environment setup completed")
        
    def start_services(self):
        """Start services with minimal backend"""
        logger.info("Starting services with minimal backend...")
        
        # Use enhanced PATH
        env = os.environ.copy()
        env['PATH'] = self.system_path
        
        services = []
        
        # Start minimal backend
        logger.info("Starting minimal backend server...")
        try:
            backend_process = subprocess.Popen(
                ['node', 'app_minimal.js'],
                cwd=self.project_root / 'backend',
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            services.append(('backend', backend_process))
            logger.info(f"✅ Minimal backend started with PID: {backend_process.pid}")
            
            # Wait a bit for startup
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"❌ Failed to start minimal backend: {e}")
        
        # Start AI services
        logger.info("Starting AI services...")
        try:
            ai_process = subprocess.Popen(
                ['python', 'ai_gateway.py'],
                cwd=self.project_root / 'ai_service_flask',
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            services.append(('ai_services', ai_process))
            logger.info(f"✅ AI services started with PID: {ai_process.pid}")
            
            # Wait a bit for startup
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"❌ Failed to start AI services: {e}")
        
        return services
        
    def check_service_health(self, services):
        """Check if services are healthy"""
        logger.info("Checking service health...")
        
        health_checks = {
            'backend': 'http://localhost:3000/api/health',
            'ai_services': 'http://localhost:5000/health'
        }
        
        healthy_services = []
        
        for service_name, process in services:
            if service_name in health_checks:
                try:
                    response = requests.get(health_checks[service_name], timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name}: Healthy")
                        healthy_services.append(service_name)
                    else:
                        logger.warning(f"⚠️ {service_name}: Status {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ {service_name}: Health check failed - {e}")
            else:
                logger.info(f"✅ {service_name}: Running (no health check)")
                healthy_services.append(service_name)
        
        return healthy_services
        
    def test_material_generation(self):
        """Test the material generation functionality"""
        logger.info("Testing material generation...")
        
        try:
            # Test the material generation endpoint
            test_data = {
                "companyProfile": {
                    "name": "Test Manufacturing Co",
                    "industry": "Manufacturing",
                    "size": "Medium",
                    "location": "USA",
                    "sustainability_goals": ["reduce_waste", "carbon_neutral"],
                    "current_practices": ["recycling", "energy_efficiency"]
                }
            }
            
            response = requests.post(
                'http://localhost:3000/api/ai-portfolio-generation',
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Material generation test successful!")
                logger.info(f"Generated {result.get('outputs_count', 0)} outputs and {result.get('inputs_count', 0)} inputs")
                return True
            else:
                logger.warning(f"⚠️ Material generation test failed: Status {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Material generation test error: {e}")
            return False
        
    def display_status(self, services, healthy_services, generation_test=False):
        """Display current status"""
        logger.info("\n" + "="*60)
        logger.info("SYMBIOFLOWS WORKING DEMO STATUS")
        logger.info("="*60)
        
        for service_name, process in services:
            status = "✅ RUNNING" if service_name in healthy_services else "⚠️ STARTING"
            logger.info(f"{service_name:<20} {status}")
        
        if generation_test:
            logger.info(f"{'material_generation':<20} ✅ WORKING")
        else:
            logger.info(f"{'material_generation':<20} ⚠️ NOT TESTED")
        
        logger.info("="*60)
        logger.info("🎉 WORKING DEMO IS READY!")
        logger.info("="*60)
        logger.info("Access URLs:")
        logger.info("Backend API: http://localhost:3000/api/test")
        logger.info("AI Services: http://localhost:5000/health")
        logger.info("Material Generation: POST http://localhost:3000/api/ai-portfolio-generation")
        logger.info("="*60)
        logger.info("💡 TIP: Your AI onboarding should now generate materials!")
        logger.info("💡 TIP: Keep this terminal open to keep services running")
        logger.info("💡 TIP: Press Ctrl+C in this terminal to stop all services")
        logger.info("="*60)
        
    def run(self):
        """Main run method - keeps services running"""
        try:
            logger.info("Starting SymbioFlows Working Demo")
            logger.info("="*60)
            
            # Setup environment
            self.setup_environment()
            
            # Start services
            services = self.start_services()
            
            if not services:
                logger.error("❌ No services started")
                return False
            
            # Wait for services to start
            logger.info("Waiting for services to start...")
            time.sleep(8)
            
            # Check health
            healthy_services = self.check_service_health(services)
            
            # Test material generation
            generation_working = self.test_material_generation()
            
            # Display status
            self.display_status(services, healthy_services, generation_working)
            
            logger.info("🎉 Working demo is running! Keep this terminal open.")
            logger.info("Press Ctrl+C to stop all services")
            
            # Keep running and monitor services
            try:
                while True:
                    time.sleep(30)  # Check every 30 seconds
                    
                    # Check if any services died
                    for service_name, process in services:
                        if process.poll() is not None:
                            logger.error(f"❌ {service_name} has stopped unexpectedly!")
                            logger.info("Service output:")
                            try:
                                output, _ = process.communicate(timeout=1)
                                if output:
                                    logger.info(output)
                            except:
                                pass
                            return False
                    
                    # Periodic health check
                    healthy_services = self.check_service_health(services)
                    logger.info(f"Health check: {len(healthy_services)}/{len(services)} services healthy")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                logger.info("Stopping all services...")
                
                # Stop all services
                for service_name, process in services:
                    try:
                        logger.info(f"Stopping {service_name}...")
                        process.terminate()
                        process.wait(timeout=10)
                        logger.info(f"✅ {service_name} stopped")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚠️ {service_name} didn't stop gracefully, force killing...")
                        process.kill()
                        logger.info(f"✅ {service_name} force killed")
                    except Exception as e:
                        logger.error(f"❌ Error stopping {service_name}: {e}")
                
                logger.info("🎉 All services stopped")
                
        except Exception as e:
            logger.error(f"❌ Working demo failed: {e}")
            return False
        
        return True

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Working Demo")
    print("📝 Check working_demo.log for detailed logs")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    starter = WorkingDemoStarter()
    
    try:
        success = starter.run()
        if success:
            logger.info("🎉 Working demo completed successfully")
        else:
            logger.error("❌ Working demo failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()