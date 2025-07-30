#!/usr/bin/env python3
"""
SymbioFlows Production Demo System
Complete orchestration for user signup, AI onboarding, material listings, and matches generation
"""

import os
import sys
import time
import subprocess
import threading
import signal
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
import psutil
import traceback

# Enhanced logging configuration with immediate output and UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Ensure output goes to console
    ],
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Production Demo System Starting...")
print("📝 Logging to: production_demo.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

class ProductionDemoOrchestrator:
    """Production-ready demo orchestrator for complete user flow"""
    
    def __init__(self):
        logger.info("🔧 Initializing Production Demo Orchestrator...")
        self.project_root = Path(__file__).parent
        logger.info(f"📁 Project root: {self.project_root}")
        
        self.processes: Dict[str, subprocess.Popen] = {}
        self.services_status: Dict[str, bool] = {}
        self.service_outputs: Dict[str, List[str]] = {}
        self.required_services = [
            'supabase',
            'backend',
            'ai_services',
            'frontend'
        ]
        
        # Service configurations
        self.service_configs = {
            'supabase': {
                'name': 'Supabase Database',
                'health_url': 'http://localhost:54321/rest/v1/',
                'startup_time': 10,
                'required': True
            },
            'backend': {
                'name': 'Backend API Server',
                'health_url': 'http://localhost:3000/api/health',
                'startup_time': 15,
                'required': True,
                'command': ['node', 'app.js'],
                'cwd': self.project_root / 'backend',
                'env': {
                    'NODE_ENV': 'production',
                    'PORT': '3000'
                }
            },
            'ai_services': {
                'name': 'AI Microservices Gateway',
                'health_url': 'http://localhost:5000/health',
                'startup_time': 20,
                'required': True,
                'command': ['python', 'ai_gateway.py'],
                'cwd': self.project_root / 'ai_service_flask',
                'env': {
                    'FLASK_ENV': 'production',
                    'FLASK_APP': 'ai_gateway.py'
                }
            },
            'frontend': {
                'name': 'Frontend Application',
                'health_url': 'http://localhost:5173',
                'startup_time': 10,
                'required': True,
                'command': ['npm', 'run', 'dev'],
                'cwd': self.project_root / 'frontend',
                'env': {
                    'VITE_API_URL': 'http://localhost:3000',
                    'VITE_AI_SERVICES_URL': 'http://localhost:5000'
                }
            }
        }
        
        # Demo data for testing
        self.demo_company_profile = {
            "name": "EcoTech Manufacturing",
            "industry": "Electronics Manufacturing",
            "size": "Medium (100-500 employees)",
            "location": "Austin, TX",
            "waste_streams": "Electronic waste, plastic components, metal scraps, packaging materials",
            "sustainability_goals": "Zero waste by 2030, 50% recycled materials usage, carbon neutral operations",
            "production_processes": "PCB assembly, plastic injection molding, metal fabrication, quality testing",
            "current_waste_management": "Basic recycling, some landfill disposal, limited material recovery",
            "material_requirements": "Recycled plastics, reclaimed metals, sustainable packaging, bio-based materials"
        }
        
        logger.info("✅ Orchestrator initialized successfully")
        
    def log_system_info(self):
        """Log system information for debugging"""
        logger.info("🔍 System Information:")
        logger.info(f"  Python version: {sys.version}")
        logger.info(f"  Platform: {sys.platform}")
        logger.info(f"  Working directory: {os.getcwd()}")
        logger.info(f"  Project root: {self.project_root}")
        
        # Check for required tools
        tools = ['node', 'npm', 'python', 'pip']
        for tool in tools:
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    logger.info(f"  {tool}: {version}")
                else:
                    logger.warning(f"  {tool}: Not available")
            except Exception as e:
                logger.warning(f"  {tool}: Error checking - {e}")
        
    def setup_environment(self):
        """Setup production environment"""
        logger.info("🔧 Setting up production environment...")
        
        # Create necessary directories
        directories = [
            'logs',
            'data',
            'models',
            'cache',
            'backups'
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            logger.info(f"  📁 Created directory: {dir_path}")
            
        # Check environment variables
        required_env_vars = [
            'SUPABASE_URL',
            'SUPABASE_ANON_KEY',
            'DEEPSEEK_API_KEY',
            'OPENAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                # Log partial value for debugging (hide sensitive parts)
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"  ✅ {var}: {masked_value}")
                
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            logger.info("Please ensure all required environment variables are set")
            logger.info("You can copy .env.example to .env and fill in the values")
        else:
            logger.info("✅ All required environment variables are set")
            
        logger.info("✅ Environment setup completed")
        
    def install_dependencies(self):
        """Install dependencies for all services"""
        logger.info("📦 Installing dependencies...")
        
        # Backend dependencies
        logger.info("  🔧 Installing backend dependencies...")
        try:
            result = subprocess.run(
                ['npm', 'install'], 
                cwd=self.project_root / 'backend', 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            if result.returncode == 0:
                logger.info("  ✅ Backend dependencies installed successfully")
                logger.debug(f"  📄 npm output: {result.stdout[:500]}...")
            else:
                logger.error(f"  ❌ Backend dependencies failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("  ❌ Backend dependencies installation timed out")
            return False
        except Exception as e:
            logger.error(f"  ❌ Backend dependencies error: {e}")
            return False
            
        # Frontend dependencies
        logger.info("  🎨 Installing frontend dependencies...")
        try:
            result = subprocess.run(
                ['npm', 'install'], 
                cwd=self.project_root / 'frontend', 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            if result.returncode == 0:
                logger.info("  ✅ Frontend dependencies installed successfully")
                logger.debug(f"  📄 npm output: {result.stdout[:500]}...")
            else:
                logger.error(f"  ❌ Frontend dependencies failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("  ❌ Frontend dependencies installation timed out")
            return False
        except Exception as e:
            logger.error(f"  ❌ Frontend dependencies error: {e}")
            return False
            
        # AI services dependencies
        logger.info("  🤖 Installing AI services dependencies...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                cwd=self.project_root / 'ai_service_flask', 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            if result.returncode == 0:
                logger.info("  ✅ AI services dependencies installed successfully")
                logger.debug(f"  📄 pip output: {result.stdout[:500]}...")
            else:
                logger.error(f"  ❌ AI services dependencies failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("  ❌ AI services dependencies installation timed out")
            return False
        except Exception as e:
            logger.error(f"  ❌ AI services dependencies error: {e}")
            return False
            
        logger.info("✅ All dependencies installed successfully")
        return True
        
    def capture_service_output(self, service_name: str, process: subprocess.Popen):
        """Capture and log service output in real-time"""
        def capture_output(stream, prefix):
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            logger.info(f"[{service_name}] {prefix}: {line}")
                            if service_name not in self.service_outputs:
                                self.service_outputs[service_name] = []
                            self.service_outputs[service_name].append(f"{prefix}: {line}")
            except Exception as e:
                logger.error(f"Error capturing {service_name} {prefix} output: {e}")
        
        # Start output capture threads
        if process.stdout:
            threading.Thread(target=capture_output, args=(process.stdout, "STDOUT"), daemon=True).start()
        if process.stderr:
            threading.Thread(target=capture_output, args=(process.stderr, "STDERR"), daemon=True).start()
        
    def start_service(self, service_name: str) -> bool:
        """Start a specific service with enhanced logging"""
        config = self.service_configs.get(service_name)
        if not config:
            logger.error(f"❌ Unknown service: {service_name}")
            return False
            
        logger.info(f"🚀 Starting {config['name']}...")
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if 'env' in config:
                env.update(config['env'])
                logger.debug(f"  Environment variables for {service_name}: {list(config['env'].keys())}")
                
            # Start process
            if 'command' in config:
                logger.info(f"  📋 Command: {' '.join(config['command'])}")
                logger.info(f"  📁 Working directory: {config.get('cwd')}")
                
                process = subprocess.Popen(
                    config['command'],
                    cwd=config.get('cwd'),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                self.processes[service_name] = process
                logger.info(f"  🔄 Process started with PID: {process.pid}")
                
                # Start output capture
                self.capture_service_output(service_name, process)
                
                # Wait for startup with progress indicator
                startup_time = config['startup_time']
                logger.info(f"  ⏳ Waiting {startup_time} seconds for startup...")
                
                for i in range(startup_time):
                    time.sleep(1)
                    if i % 5 == 0:  # Show progress every 5 seconds
                        logger.info(f"  ⏳ Startup progress: {i+1}/{startup_time} seconds")
                    
                    # Check if process is still running
                    if process.poll() is not None:
                        logger.error(f"  ❌ Process terminated during startup (exit code: {process.returncode})")
                        return False
                
                # Check health
                logger.info(f"  🔍 Performing health check...")
                if self.check_service_health(service_name):
                    logger.info(f"✅ {config['name']} started successfully")
                    self.services_status[service_name] = True
                    return True
                else:
                    logger.error(f"❌ {config['name']} failed health check")
                    logger.error(f"  📄 Recent output: {self.service_outputs.get(service_name, [])[-5:]}")
                    self.services_status[service_name] = False
                    return False
            else:
                # External service (like Supabase)
                logger.info(f"ℹ️ {config['name']} should be running externally")
                self.services_status[service_name] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to start {config['name']}: {e}")
            logger.error(f"  📄 Traceback: {traceback.format_exc()}")
            self.services_status[service_name] = False
            return False
            
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy with detailed logging"""
        config = self.service_configs.get(service_name)
        if not config or 'health_url' not in config:
            return True
            
        try:
            logger.debug(f"  🔍 Checking health at: {config['health_url']}")
            response = requests.get(config['health_url'], timeout=10)
            logger.debug(f"  📊 Health check response: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"  ✅ Health check passed")
                return True
            else:
                logger.warning(f"  ⚠️ Health check returned status: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            logger.warning(f"  ⚠️ Health check timeout for {service_name}")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning(f"  ⚠️ Health check connection error for {service_name}")
            return False
        except Exception as e:
            logger.debug(f"  ❌ Health check failed for {service_name}: {e}")
            return False
            
    def start_all_services(self) -> bool:
        """Start all required services with enhanced error handling"""
        logger.info("🚀 Starting all production services...")
        
        success = True
        failed_services = []
        
        for service_name in self.required_services:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting service: {service_name}")
            logger.info(f"{'='*50}")
            
            if not self.start_service(service_name):
                if self.service_configs[service_name].get('required', False):
                    success = False
                    failed_services.append(service_name)
                    logger.error(f"❌ Required service {service_name} failed to start")
                else:
                    logger.warning(f"⚠️ Optional service {service_name} failed to start")
        
        if failed_services:
            logger.error(f"❌ Failed to start required services: {failed_services}")
        else:
            logger.info("✅ All required services started successfully")
            
        return success
        
    def run_demo_flow(self):
        """Run the complete demo user flow with enhanced logging"""
        logger.info("\n🎭 Starting production demo flow...")
        logger.info("="*60)
        
        try:
            # Step 1: User Signup
            logger.info("👤 Step 1: User Signup")
            logger.info("-" * 30)
            signup_result = self.demo_user_signup()
            if not signup_result:
                logger.error("❌ User signup failed")
                return False
                
            # Step 2: AI Onboarding
            logger.info("\n🤖 Step 2: AI Onboarding")
            logger.info("-" * 30)
            onboarding_result = self.demo_ai_onboarding(signup_result['user_id'])
            if not onboarding_result:
                logger.error("❌ AI onboarding failed")
                return False
                
            # Step 3: Material Listings Generation
            logger.info("\n📦 Step 3: Material Listings Generation")
            logger.info("-" * 30)
            listings_result = self.demo_generate_listings(signup_result['user_id'])
            if not listings_result:
                logger.error("❌ Material listings generation failed")
                return False
                
            # Step 4: Matches Generation
            logger.info("\n🔗 Step 4: Matches Generation")
            logger.info("-" * 30)
            matches_result = self.demo_generate_matches(signup_result['user_id'])
            if not matches_result:
                logger.error("❌ Matches generation failed")
                return False
                
            logger.info("\n✅ Demo flow completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo flow failed: {e}")
            logger.error(f"📄 Traceback: {traceback.format_exc()}")
            return False
            
    def demo_user_signup(self) -> Optional[Dict]:
        """Demo user signup process with detailed logging"""
        try:
            logger.info("  📝 Creating demo user account...")
            
            # Create demo user
            signup_data = {
                "email": "demo@ecotech.com",
                "password": "DemoPassword123!",
                "company_name": self.demo_company_profile["name"],
                "industry": self.demo_company_profile["industry"],
                "waste_streams": self.demo_company_profile["waste_streams"],
                "sustainability_goals": self.demo_company_profile["sustainability_goals"]
            }
            
            logger.info(f"  📤 Sending signup request to: http://localhost:3000/api/auth/signup")
            logger.debug(f"  📄 Signup data: {json.dumps(signup_data, indent=2)}")
            
            response = requests.post(
                "http://localhost:3000/api/auth/signup",
                json=signup_data,
                timeout=30
            )
            
            logger.info(f"  📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"  ✅ User signup successful: {result.get('user_id')}")
                logger.debug(f"  📄 Response: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"  ❌ User signup failed: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("  ❌ User signup timeout")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("  ❌ User signup connection error - backend may not be running")
            return None
        except Exception as e:
            logger.error(f"  ❌ User signup error: {e}")
            logger.error(f"  📄 Traceback: {traceback.format_exc()}")
            return None
            
    def demo_ai_onboarding(self, user_id: str) -> bool:
        """Demo AI onboarding process with detailed logging"""
        try:
            logger.info("  🤖 Starting AI onboarding process...")
            
            onboarding_data = {
                "session_id": f"demo_session_{user_id}",
                "answers": {
                    "company_size": self.demo_company_profile["size"],
                    "location": self.demo_company_profile["location"],
                    "production_processes": self.demo_company_profile["production_processes"],
                    "current_waste_management": self.demo_company_profile["current_waste_management"],
                    "material_requirements": self.demo_company_profile["material_requirements"]
                }
            }
            
            logger.info(f"  📤 Sending onboarding request to: http://localhost:3000/api/adaptive-onboarding/complete")
            logger.debug(f"  📄 Onboarding data: {json.dumps(onboarding_data, indent=2)}")
            
            response = requests.post(
                "http://localhost:3000/api/adaptive-onboarding/complete",
                json=onboarding_data,
                timeout=60
            )
            
            logger.info(f"  📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                listings_count = len(result.get('generated_listings', []))
                logger.info(f"  ✅ AI onboarding completed: {listings_count} listings generated")
                logger.debug(f"  📄 Response: {json.dumps(result, indent=2)}")
                return True
            else:
                logger.error(f"  ❌ AI onboarding failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("  ❌ AI onboarding timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("  ❌ AI onboarding connection error - backend may not be running")
            return False
        except Exception as e:
            logger.error(f"  ❌ AI onboarding error: {e}")
            logger.error(f"  📄 Traceback: {traceback.format_exc()}")
            return False
            
    def demo_generate_listings(self, user_id: str) -> bool:
        """Demo material listings generation with detailed logging"""
        try:
            logger.info("  📦 Generating material listings...")
            
            url = f"http://localhost:3000/api/v1/companies/{user_id}/generate-listings"
            logger.info(f"  📤 Sending request to: {url}")
            
            response = requests.post(url, timeout=120)
            
            logger.info(f"  📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                materials_count = len(result.get('materials', []))
                logger.info(f"  ✅ Material listings generated: {materials_count} materials")
                logger.debug(f"  📄 Response: {json.dumps(result, indent=2)}")
                return True
            else:
                logger.error(f"  ❌ Material listings generation failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("  ❌ Material listings generation timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("  ❌ Material listings generation connection error")
            return False
        except Exception as e:
            logger.error(f"  ❌ Material listings generation error: {e}")
            logger.error(f"  📄 Traceback: {traceback.format_exc()}")
            return False
            
    def demo_generate_matches(self, user_id: str) -> bool:
        """Demo matches generation with detailed logging"""
        try:
            logger.info("  🔗 Generating matches...")
            
            url = "http://localhost:3000/api/ai-pipeline"
            data = {"id": user_id}
            logger.info(f"  📤 Sending request to: {url}")
            logger.debug(f"  📄 Request data: {json.dumps(data, indent=2)}")
            
            response = requests.post(url, json=data, timeout=180)
            
            logger.info(f"  📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                matches_count = len(result.get('matches', []))
                logger.info(f"  ✅ Matches generated: {matches_count} matches")
                logger.debug(f"  📄 Response: {json.dumps(result, indent=2)}")
                return True
            else:
                logger.error(f"  ❌ Matches generation failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("  ❌ Matches generation timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("  ❌ Matches generation connection error")
            return False
        except Exception as e:
            logger.error(f"  ❌ Matches generation error: {e}")
            logger.error(f"  📄 Traceback: {traceback.format_exc()}")
            return False
            
    def display_demo_status(self):
        """Display current demo status with enhanced formatting"""
        logger.info("\n" + "="*60)
        logger.info("🎭 SYMBIOFLOWS PRODUCTION DEMO STATUS")
        logger.info("="*60)
        
        for service_name, config in self.service_configs.items():
            status = "✅ RUNNING" if self.services_status.get(service_name, False) else "❌ STOPPED"
            logger.info(f"{config['name']:<30} {status}")
            
        logger.info("="*60)
        logger.info("🌐 Access URLs:")
        logger.info("Frontend: http://localhost:5173")
        logger.info("Backend API: http://localhost:3000")
        logger.info("AI Services: http://localhost:5000")
        logger.info("API Documentation: http://localhost:3000/api-docs")
        logger.info("="*60)
        
    def cleanup(self):
        """Cleanup all processes with enhanced logging"""
        logger.info("🧹 Cleaning up processes...")
        
        for service_name, process in self.processes.items():
            try:
                logger.info(f"  🛑 Terminating {service_name} (PID: {process.pid})...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"  ✅ {service_name} terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"  ⚠️ {service_name} didn't terminate gracefully, force killing...")
                process.kill()
                logger.info(f"  ✅ {service_name} force killed")
            except Exception as e:
                logger.error(f"  ❌ Error terminating {service_name}: {e}")
                
        logger.info("✅ Cleanup completed")
        
    def run(self):
        """Main run method with enhanced error handling and logging"""
        try:
            logger.info("🚀 Starting SymbioFlows Production Demo System")
            logger.info("="*60)
            
            # Log system information
            self.log_system_info()
            
            # Setup environment
            self.setup_environment()
            
            # Install dependencies
            if not self.install_dependencies():
                logger.error("❌ Dependency installation failed")
                return False
                
            # Start all services
            if not self.start_all_services():
                logger.error("❌ Failed to start all required services")
                return False
                
            # Display status
            self.display_demo_status()
            
            # Run demo flow
            if self.run_demo_flow():
                logger.info("🎉 Production demo completed successfully!")
                logger.info("Press Ctrl+C to stop all services")
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("🛑 Received shutdown signal")
                    
            else:
                logger.error("❌ Demo flow failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Production demo failed: {e}")
            logger.error(f"📄 Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.cleanup()
            
def main():
    """Main entry point with enhanced error handling"""
    print("🚀 SymbioFlows Production Demo System")
    print("📝 Check production_demo.log for detailed logs")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    orchestrator = ProductionDemoOrchestrator()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Shutdown signal received")
        orchestrator.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the orchestrator
    try:
        success = orchestrator.run()
        if success:
            logger.info("🎉 System completed successfully")
        else:
            logger.error("❌ System failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(f"📄 Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
if __name__ == "__main__":
    main() 