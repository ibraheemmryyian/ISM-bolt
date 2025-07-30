#!/usr/bin/env python3
"""
Persistent SymbioFlows Demo - Starts services and keeps them running
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv
import signal

# Enhanced logging configuration with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('persistent_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Persistent Demo")
print("📝 Logging to: persistent_demo.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

class PersistentDemoOrchestrator:
    """Persistent demo orchestrator - starts services in background"""
    
    def __init__(self):
        logger.info("Initializing Persistent Demo Orchestrator...")
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
        
        self.running_services = []
        logger.info("Orchestrator initialized successfully")
        
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
        required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var}: {masked_value}")
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        else:
            logger.info("All required environment variables are set")
        
        logger.info("Environment setup completed")
        
    def check_tools(self):
        """Check if required tools are available"""
        logger.info("Checking required tools...")
        
        tools = {
            'python': ['python', '--version'],
            'node': ['node', '--version'],
            'npm': ['npm', '--version']
        }
        
        # Also try to find npm in common locations
        npm_locations = [
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd",
            os.path.expanduser(r"~\AppData\Roaming\npm\npm.cmd"),
            os.path.expanduser(r"~\AppData\Local\Programs\nodejs\npm.cmd")
        ]
        
        available_tools = {}
        
        for tool_name, command in tools.items():
            try:
                # Use the enhanced PATH
                env = os.environ.copy()
                env['PATH'] = self.system_path
                
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env
                )
                
                if result.returncode == 0:
                    version = result.stdout.strip()
                    logger.info(f"✅ {tool_name}: {version}")
                    available_tools[tool_name] = version
                else:
                    # For npm, try direct paths
                    if tool_name == 'npm':
                        for npm_path in npm_locations:
                            if os.path.exists(npm_path):
                                try:
                                    result = subprocess.run(
                                        [npm_path, '--version'],
                                        capture_output=True,
                                        text=True,
                                        timeout=10
                                    )
                                    if result.returncode == 0:
                                        version = result.stdout.strip()
                                        logger.info(f"✅ {tool_name}: {version} (found at {npm_path})")
                                        available_tools[tool_name] = version
                                        break
                                except Exception:
                                    continue
                        else:
                            logger.warning(f"⚠️ {tool_name}: Not available")
                    else:
                        logger.warning(f"⚠️ {tool_name}: Not available")
                    
            except Exception as e:
                logger.warning(f"⚠️ {tool_name}: Error checking - {e}")
        
        return available_tools
        
    def start_services(self, available_tools):
        """Start services in background (persistent)"""
        logger.info("Starting services in background...")
        
        # Use enhanced PATH
        env = os.environ.copy()
        env['PATH'] = self.system_path
        
        services_started = []
        
        # Start backend
        if 'node' in available_tools:
            logger.info("Starting backend server in background...")
            try:
                # Create new command window for backend
                backend_process = subprocess.Popen(
                    ['cmd', '/c', 'start', 'cmd', '/k', 'node app.js'],
                    cwd=self.project_root / 'backend',
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                services_started.append(('backend', 'http://localhost:3000'))
                logger.info(f"✅ Backend started in new window")
                
            except Exception as e:
                logger.error(f"❌ Failed to start backend: {e}")
        else:
            logger.warning("⚠️ Node.js not available, skipping backend")
        
        # Start AI services
        logger.info("Starting AI services in background...")
        try:
            # Create new command window for AI services
            ai_process = subprocess.Popen(
                ['cmd', '/c', 'start', 'cmd', '/k', 'python ai_gateway.py'],
                cwd=self.project_root / 'ai_service_flask',
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            services_started.append(('ai_services', 'http://localhost:5000'))
            logger.info(f"✅ AI services started in new window")
            
        except Exception as e:
            logger.error(f"❌ Failed to start AI services: {e}")
        
        # Start frontend if npm is available
        if 'npm' in available_tools:
            logger.info("Starting frontend in background...")
            try:
                # Create new command window for frontend
                frontend_process = subprocess.Popen(
                    ['cmd', '/c', 'start', 'cmd', '/k', 'npm run dev'],
                    cwd=self.project_root / 'frontend',
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                services_started.append(('frontend', 'http://localhost:5173'))
                logger.info(f"✅ Frontend started in new window")
                
            except Exception as e:
                logger.error(f"❌ Failed to start frontend: {e}")
        else:
            logger.warning("⚠️ npm not available, skipping frontend")
        
        return services_started
        
    def check_service_health(self, services, wait_time=30):
        """Check if services are healthy with retries"""
        logger.info(f"Waiting {wait_time} seconds for services to start...")
        time.sleep(wait_time)
        
        logger.info("Checking service health...")
        
        healthy_services = []
        
        for service_name, url in services:
            for attempt in range(3):  # Try 3 times
                try:
                    response = requests.get(f"{url}/health" if service_name != 'frontend' else url, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name}: Healthy at {url}")
                        healthy_services.append((service_name, url))
                        break
                    else:
                        logger.warning(f"⚠️ {service_name}: Status {response.status_code} (attempt {attempt + 1})")
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.warning(f"⚠️ {service_name}: Health check failed - {e}")
                    else:
                        logger.info(f"⚠️ {service_name}: Attempt {attempt + 1} failed, retrying...")
                        time.sleep(5)
        
        return healthy_services
        
    def display_status(self, services, healthy_services):
        """Display current status"""
        logger.info("\n" + "="*60)
        logger.info("SYMBIOFLOWS PERSISTENT DEMO STATUS")
        logger.info("="*60)
        
        for service_name, url in services:
            is_healthy = any(s[0] == service_name for s in healthy_services)
            status = "✅ RUNNING" if is_healthy else "⚠️ STARTING"
            logger.info(f"{service_name:<20} {status:<12} {url}")
        
        logger.info("="*60)
        logger.info("🎉 Services are running in separate windows!")
        logger.info("📝 Check the opened command windows for service logs")
        logger.info("🛑 Close the command windows to stop the services")
        logger.info("="*60)
        
    def run(self):
        """Main run method"""
        try:
            logger.info("Starting SymbioFlows Persistent Demo System")
            logger.info("="*60)
            
            # Setup environment
            self.setup_environment()
            
            # Check tools
            available_tools = self.check_tools()
            
            # Start services in background
            services = self.start_services(available_tools)
            
            if not services:
                logger.error("❌ No services started")
                return False
            
            # Check health
            healthy_services = self.check_service_health(services)
            
            # Display status
            self.display_status(services, healthy_services)
            
            logger.info("🎉 Persistent demo completed successfully!")
            logger.info("Services are now running in background windows")
            logger.info("This script will exit, but services will continue running")
            
        except Exception as e:
            logger.error(f"❌ Persistent demo failed: {e}")
            return False
        
        return True

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Persistent Demo")
    print("📝 Check persistent_demo.log for detailed logs")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    orchestrator = PersistentDemoOrchestrator()
    
    try:
        success = orchestrator.run()
        if success:
            logger.info("🎉 System completed successfully")
            print("\n🎉 Services are now running!")
            print("Check the opened command windows for each service")
            print("To stop services, close the command windows")
        else:
            logger.error("❌ System failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()