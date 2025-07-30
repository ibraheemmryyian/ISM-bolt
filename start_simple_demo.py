#!/usr/bin/env python3
"""
Simplified SymbioFlows Production Demo System
Handles PATH issues and provides better error handling
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
        logging.FileHandler('simple_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Simple Production Demo")
print("📝 Logging to: simple_demo.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

class SimpleDemoOrchestrator:
    """Simplified demo orchestrator with better PATH handling"""
    
    def __init__(self):
        logger.info("Initializing Simple Demo Orchestrator...")
        self.project_root = Path(__file__).parent
        logger.info(f"Project root: {self.project_root}")
        
        # Get system PATH
        self.system_path = os.environ.get('PATH', '')
        logger.info(f"System PATH: {self.system_path[:100]}...")
        
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
        
        logger.info("Orchestrator initialized successfully")
        
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
        
    def setup_environment(self):
        """Setup environment"""
        logger.info("Setting up environment...")
        
        # Load .env file
        env_file = self.project_root / '.env'
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
            logger.info("Please check your .env file contains these variables")
        else:
            logger.info("All required environment variables are set")
        
        logger.info("Environment setup completed")
        
    def install_dependencies(self, available_tools):
        """Install dependencies with better PATH handling"""
        logger.info("Installing dependencies...")
        
        # Use enhanced PATH
        env = os.environ.copy()
        env['PATH'] = self.system_path
        
        # Backend dependencies
        if 'npm' in available_tools:
            logger.info("Installing backend dependencies...")
            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=self.project_root / 'backend',
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env
                )
                
                if result.returncode == 0:
                    logger.info("✅ Backend dependencies installed successfully")
                else:
                    logger.error(f"❌ Backend dependencies failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Backend dependencies error: {e}")
                return False
        else:
            logger.warning("⚠️ npm not available, skipping backend dependencies")
        
        # Frontend dependencies
        if 'npm' in available_tools:
            logger.info("Installing frontend dependencies...")
            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=self.project_root / 'frontend',
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env
                )
                
                if result.returncode == 0:
                    logger.info("✅ Frontend dependencies installed successfully")
                else:
                    logger.error(f"❌ Frontend dependencies failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Frontend dependencies error: {e}")
                return False
        else:
            logger.warning("⚠️ npm not available, skipping frontend dependencies")
        
        # AI services dependencies
        logger.info("Installing AI services dependencies...")
        try:
            # Try the fixed requirements file first, fallback to original
            requirements_file = 'requirements_fixed.txt'
            if not (self.project_root / 'ai_service_flask' / requirements_file).exists():
                requirements_file = 'requirements.txt'
                logger.info(f"Using fallback requirements file: {requirements_file}")
            
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
                cwd=self.project_root / 'ai_service_flask',
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("✅ AI services dependencies installed successfully")
            else:
                logger.error(f"❌ AI services dependencies failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ AI services dependencies error: {e}")
            return False
        
        logger.info("All dependencies installed successfully")
        return True
        
    def start_services(self, available_tools):
        """Start services with better PATH handling"""
        logger.info("Starting services...")
        
        # Use enhanced PATH
        env = os.environ.copy()
        env['PATH'] = self.system_path
        
        services = []
        
        # Start backend
        if 'node' in available_tools:
            logger.info("Starting backend server...")
            try:
                backend_process = subprocess.Popen(
                    ['node', 'app.js'],
                    cwd=self.project_root / 'backend',
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                services.append(('backend', backend_process))
                logger.info(f"✅ Backend started with PID: {backend_process.pid}")
                
                # Wait a bit for startup
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Failed to start backend: {e}")
        else:
            logger.warning("⚠️ Node.js not available, skipping backend")
        
        # Start AI services
        logger.info("Starting AI services...")
        try:
            ai_process = subprocess.Popen(
                ['python', 'ai_gateway.py'],
                cwd=self.project_root / 'ai_service_flask',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            services.append(('ai_services', ai_process))
            logger.info(f"✅ AI services started with PID: {ai_process.pid}")
            
            # Wait a bit for startup
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Failed to start AI services: {e}")
        
        # Start frontend
        if 'npm' in available_tools:
            logger.info("Starting frontend...")
            try:
                frontend_process = subprocess.Popen(
                    ['npm', 'run', 'dev'],
                    cwd=self.project_root / 'frontend',
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                services.append(('frontend', frontend_process))
                logger.info(f"✅ Frontend started with PID: {frontend_process.pid}")
                
            except Exception as e:
                logger.error(f"❌ Failed to start frontend: {e}")
        else:
            logger.warning("⚠️ npm not available, skipping frontend")
        
        return services
        
    def check_service_health(self, services):
        """Check if services are healthy"""
        logger.info("Checking service health...")
        
        health_checks = {
            'backend': 'http://localhost:3000/api/health',
            'ai_services': 'http://localhost:5000/health',
            'frontend': 'http://localhost:5173'
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
        
    def display_status(self, services, healthy_services):
        """Display current status"""
        logger.info("\n" + "="*60)
        logger.info("SYMBIOFLOWS SIMPLE DEMO STATUS")
        logger.info("="*60)
        
        for service_name, process in services:
            status = "✅ RUNNING" if service_name in healthy_services else "⚠️ STARTING"
            logger.info(f"{service_name:<20} {status}")
        
        logger.info("="*60)
        logger.info("Access URLs:")
        logger.info("Frontend: http://localhost:5173")
        logger.info("Backend API: http://localhost:3000")
        logger.info("AI Services: http://localhost:5000")
        logger.info("="*60)
        
    def cleanup(self, services):
        """Cleanup all processes"""
        logger.info("Cleaning up processes...")
        
        for service_name, process in services:
            try:
                logger.info(f"Terminating {service_name} (PID: {process.pid})...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service_name} terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {service_name} didn't terminate gracefully, force killing...")
                process.kill()
                logger.info(f"✅ {service_name} force killed")
            except Exception as e:
                logger.error(f"❌ Error terminating {service_name}: {e}")
        
        logger.info("Cleanup completed")
        
    def run(self):
        """Main run method"""
        try:
            logger.info("Starting SymbioFlows Simple Demo System")
            logger.info("="*60)
            
            # Check tools
            available_tools = self.check_tools()
            
            # Setup environment
            self.setup_environment()
            
            # Install dependencies
            if not self.install_dependencies(available_tools):
                logger.error("❌ Dependency installation failed")
                return False
            
            # Start services
            services = self.start_services(available_tools)
            
            if not services:
                logger.error("❌ No services started")
                return False
            
            # Wait for services to start
            logger.info("Waiting for services to start...")
            time.sleep(10)
            
            # Check health
            healthy_services = self.check_service_health(services)
            
            # Display status
            self.display_status(services, healthy_services)
            
            logger.info("🎉 Simple demo completed successfully!")
            logger.info("Press Ctrl+C to stop all services")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                
        except Exception as e:
            logger.error(f"❌ Simple demo failed: {e}")
            return False
        finally:
            if 'services' in locals():
                self.cleanup(services)
        
        return True

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Simple Production Demo")
    print("📝 Check simple_demo.log for detailed logs")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    orchestrator = SimpleDemoOrchestrator()
    
    try:
        success = orchestrator.run()
        if success:
            logger.info("🎉 System completed successfully")
        else:
            logger.error("❌ System failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 