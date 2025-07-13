#!/usr/bin/env python3
"""
System Startup with API Validation
Ensures all APIs are working before starting the system
"""

import os
import sys
import subprocess
import time
import logging
from validate_system_apis import SystemAPIValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate environment variables are set"""
    logger.info("🔧 Validating environment configuration...")
    
    required_vars = [
        'DEEPSEEK_API_KEY',
        'FREIGHTOS_API_KEY', 
        'FREIGHTOS_SECRET_KEY',
        'NEXT_GEN_MATERIALS_API_KEY',
        'NEWSAPI_KEY',
        'SUPABASE_URL',
        'SUPABASE_SERVICE_ROLE_KEY'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_') or value == 'required':
            missing.append(var)
    
    if missing:
        logger.error("❌ Missing required environment variables:")
        for var in missing:
            logger.error(f"   - {var}")
        logger.error("❌ Please set all required API keys in your .env file")
        return False
    
    logger.info("✅ All environment variables are configured")
    return True

def validate_apis():
    """Validate all APIs are working"""
    logger.info("🧪 Validating all API connections...")
    
    try:
        validator = SystemAPIValidator()
        results = validator.run_all_tests()
        
        if results['status'] == 'success':
            logger.info("✅ All APIs are working correctly")
            return True
        else:
            logger.error("❌ Some APIs are not working:")
            for api in results['critical_failures']:
                logger.error(f"   - {api}: {results['results'][api]['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ API validation failed: {str(e)}")
        return False

def start_backend():
    """Start the backend server"""
    logger.info("🚀 Starting backend server...")
    
    try:
        # Change to backend directory
        os.chdir('backend')
        
        # Start the backend server
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Backend server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Backend server failed to start:")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {str(e)}")
        return None

def start_frontend():
    """Start the frontend server"""
    logger.info("🚀 Starting frontend server...")
    
    try:
        # Change to frontend directory
        os.chdir('../frontend')
        
        # Start the frontend server
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Frontend server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Frontend server failed to start:")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start frontend: {str(e)}")
        return None

def main():
    """Main startup function"""
    logger.info("🚀 ISM AI Platform - Starting with Full API Validation")
    logger.info("=" * 60)
    
    # Step 1: Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)
    
    # Step 2: Validate APIs
    if not validate_apis():
        logger.error("❌ API validation failed")
        logger.error("❌ System cannot start without all APIs working")
        sys.exit(1)
    
    logger.info("✅ All validations passed - Starting system...")
    logger.info("=" * 60)
    
    # Step 3: Start backend
    backend_process = start_backend()
    if not backend_process:
        logger.error("❌ Backend startup failed")
        sys.exit(1)
    
    # Step 4: Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        logger.error("❌ Frontend startup failed")
        backend_process.terminate()
        sys.exit(1)
    
    logger.info("🎉 System started successfully!")
    logger.info("📱 Frontend: http://localhost:5173")
    logger.info("🔧 Backend: http://localhost:3000")
    logger.info("🏥 Health Check: http://localhost:3000/api/health")
    logger.info("")
    logger.info("Press Ctrl+C to stop all services")
    
    try:
        # Keep the processes running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                logger.error("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                logger.error("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down system...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            logger.info("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            logger.info("✅ Frontend stopped")
        
        logger.info("👋 System shutdown complete")

if __name__ == "__main__":
    main()