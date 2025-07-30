#!/usr/bin/env python3
"""
SymbioFlows Demo - Keeps running in the foreground
"""

import os
import sys
import time
import subprocess
import logging
import signal
from pathlib import Path
import requests
from dotenv import load_dotenv

# Enhanced logging configuration with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_running.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Demo - Running in Foreground")
print("📝 Logging to: demo_running.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

# Global variables for signal handling
running_services = []

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    print("\n🛑 Shutdown signal received. Cleaning up...")
    cleanup_services()
    sys.exit(0)

def cleanup_services():
    """Clean up all running services"""
    global running_services
    
    logger.info("🧹 Cleaning up processes...")
    
    for service_name, process in running_services:
        try:
            logger.info(f"  🛑 Terminating {service_name} (PID: {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info(f"  ✅ {service_name} terminated gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.info(f"  ⚠️ {service_name} force killed")
        except Exception as e:
            logger.error(f"  ❌ Error terminating {service_name}: {e}")
    
    running_services = []
    logger.info("✅ Cleanup completed")

def setup_environment():
    """Setup environment and load .env file"""
    logger.info("🔧 Setting up environment...")
    
    project_root = Path(__file__).parent
    
    # Load .env file from backend directory
    env_file = project_root / 'backend' / '.env'
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
    
    # Check environment variables
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
    else:
        logger.info("✅ All required environment variables are set")
    
    logger.info("✅ Environment setup completed")
    return project_root

def start_backend(project_root):
    """Start the backend server"""
    logger.info("🚀 Starting backend server...")
    
    try:
        backend_process = subprocess.Popen(
            ['node', 'app.js'],
            cwd=project_root / 'backend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        running_services.append(('backend', backend_process))
        logger.info(f"✅ Backend started with PID: {backend_process.pid}")
        
        # Start output capture
        start_output_capture(backend_process, 'backend')
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        return False

def start_ai_services(project_root):
    """Start the AI services"""
    logger.info("🤖 Starting AI services...")
    
    try:
        ai_process = subprocess.Popen(
            ['python', 'ai_gateway.py'],
            cwd=project_root / 'ai_service_flask',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        running_services.append(('ai_services', ai_process))
        logger.info(f"✅ AI services started with PID: {ai_process.pid}")
        
        # Start output capture
        start_output_capture(ai_process, 'ai_services')
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start AI services: {e}")
        return False

def start_output_capture(process, service_name):
    """Capture and log service output in real-time"""
    import threading
    
    def capture_output(stream, prefix):
        try:
            for line in iter(stream.readline, ''):
                if line:
                    line = line.strip()
                    if line:
                        logger.info(f"[{service_name}] {prefix}: {line}")
        except Exception as e:
            logger.error(f"Error capturing {service_name} {prefix} output: {e}")
    
    # Start output capture threads
    if process.stdout:
        threading.Thread(target=capture_output, args=(process.stdout, "OUT"), daemon=True).start()
    if process.stderr:
        threading.Thread(target=capture_output, args=(process.stderr, "ERR"), daemon=True).start()

def check_service_health():
    """Check if services are healthy"""
    logger.info("🔍 Checking service health...")
    
    health_checks = {
        'backend': 'http://localhost:3000/api/health',
        'ai_services': 'http://localhost:5000/health',
        'frontend': 'http://localhost:5173'
    }
    
    healthy_services = []
    
    for service_name, url in health_checks.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service_name}: Healthy")
                healthy_services.append(service_name)
            else:
                logger.warning(f"⚠️ {service_name}: Status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ {service_name}: Health check failed - {e}")
    
    return healthy_services

def display_status(healthy_services):
    """Display current status"""
    print("\n" + "="*60)
    print("🔄 SYMBIOFLOWS DEMO STATUS")
    print("="*60)
    
    for service_name, _ in running_services:
        status = "✅ RUNNING" if service_name in healthy_services else "⚠️ STARTING"
        print(f"{service_name:<20} {status}")
    
    print("="*60)
    print("🌐 Access URLs:")
    print("Backend API: http://localhost:3000")
    print("AI Services: http://localhost:5000")
    print("="*60)
    print("Press Ctrl+C to stop all services")
    print("="*60)

def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Setup environment
        project_root = setup_environment()
        
        # Start services
        backend_started = start_backend(project_root)
        ai_started = start_ai_services(project_root)
        
        if not backend_started and not ai_started:
            logger.error("❌ No services started")
            return False
        
        # Wait for services to start
        logger.info("⏳ Waiting for services to start...")
        time.sleep(10)
        
        # Initial health check
        healthy_services = check_service_health()
        display_status(healthy_services)
        
        # Keep running and periodically check health
        logger.info("🔄 Services are running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(30)  # Check health every 30 seconds
                healthy_services = check_service_health()
                display_status(healthy_services)
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False
    finally:
        cleanup_services()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)