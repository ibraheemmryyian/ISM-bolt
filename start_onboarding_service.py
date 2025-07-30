#!/usr/bin/env python3
"""
SymbioFlows Onboarding Service Starter
Starts the adaptive onboarding service that generates material listings
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
        logging.FileHandler('onboarding_service.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add immediate startup message
print("🚀 SymbioFlows Onboarding Service Starter")
print("📝 Logging to: onboarding_service.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

class OnboardingServiceStarter:
    """Starts the onboarding service that generates material listings"""
    
    def __init__(self):
        logger.info("Initializing Onboarding Service Starter...")
        self.project_root = Path(__file__).parent
        logger.info(f"Project root: {self.project_root}")
        
        # Get system PATH
        self.system_path = os.environ.get('PATH', '')
        logger.info("Onboarding Service Starter initialized successfully")
        
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
        
        logger.info("Environment setup completed")
        
    def start_onboarding_service(self):
        """Start the onboarding service"""
        logger.info("Starting Adaptive Onboarding Service...")
        
        # Use enhanced PATH
        env = os.environ.copy()
        env['PATH'] = self.system_path
        
        try:
            # Start the onboarding server
            onboarding_process = subprocess.Popen(
                ['python', 'adaptive_onboarding_server.py'],
                cwd=self.project_root / 'backend',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"✅ Onboarding service started with PID: {onboarding_process.pid}")
            
            # Wait a bit for startup
            time.sleep(5)
            
            return onboarding_process
            
        except Exception as e:
            logger.error(f"❌ Failed to start onboarding service: {e}")
            return None
        
    def check_service_health(self, process):
        """Check if the service is healthy"""
        logger.info("Checking onboarding service health...")
        
        try:
            # Try to connect to the service
            response = requests.get('http://localhost:5003/health', timeout=5)
            if response.status_code == 200:
                logger.info("✅ Onboarding service is healthy")
                return True
            else:
                logger.warning(f"⚠️ Onboarding service returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Onboarding service health check failed: {e}")
            return False
        
    def run(self):
        """Main run method - keeps service running"""
        try:
            logger.info("Starting SymbioFlows Onboarding Service")
            logger.info("="*60)
            
            # Setup environment
            self.setup_environment()
            
            # Start onboarding service
            process = self.start_onboarding_service()
            
            if not process:
                logger.error("❌ Onboarding service failed to start")
                return False
            
            # Wait for service to start
            logger.info("Waiting for onboarding service to start...")
            time.sleep(10)
            
            # Check health
            is_healthy = self.check_service_health(process)
            
            # Display status
            logger.info("\n" + "="*60)
            logger.info("ONBOARDING SERVICE STATUS")
            logger.info("="*60)
            logger.info(f"Status: {'✅ RUNNING' if is_healthy else '⚠️ STARTING'}")
            logger.info("="*60)
            logger.info("🎉 ONBOARDING SERVICE IS NOW RUNNING!")
            logger.info("="*60)
            logger.info("Access URL: http://localhost:5003")
            logger.info("="*60)
            logger.info("💡 TIP: Keep this terminal open to keep the service running")
            logger.info("💡 TIP: Open a new terminal to run other commands")
            logger.info("💡 TIP: Press Ctrl+C in this terminal to stop the service")
            logger.info("="*60)
            
            logger.info("🎉 Onboarding service is running! Keep this terminal open.")
            logger.info("Press Ctrl+C to stop the service")
            
            # Keep running and monitor service
            try:
                while True:
                    time.sleep(30)  # Check every 30 seconds
                    
                    # Check if service died
                    if process.poll() is not None:
                        logger.error("❌ Onboarding service has stopped unexpectedly!")
                        logger.info("Restarting service...")
                        return False
                    
                    # Periodic health check
                    is_healthy = self.check_service_health(process)
                    logger.info(f"Health check: {'✅ Healthy' if is_healthy else '⚠️ Not responding'}")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                logger.info("Stopping onboarding service...")
                
                # Stop the service
                try:
                    logger.info(f"Terminating onboarding service (PID: {process.pid})...")
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info("✅ Onboarding service stopped")
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Onboarding service didn't stop gracefully, force killing...")
                    process.kill()
                    logger.info("✅ Onboarding service force killed")
                except Exception as e:
                    logger.error(f"❌ Error stopping onboarding service: {e}")
                
                logger.info("🎉 Onboarding service stopped")
                
        except Exception as e:
            logger.error(f"❌ Onboarding service failed: {e}")
            return False
        
        return True

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Onboarding Service Starter")
    print("📝 Check onboarding_service.log for detailed logs")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    starter = OnboardingServiceStarter()
    
    try:
        success = starter.run()
        if success:
            logger.info("🎉 Onboarding service completed successfully")
        else:
            logger.error("❌ Onboarding service failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()