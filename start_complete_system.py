#!/usr/bin/env python3
"""
Complete System Startup Script
Starts all AI services and components with proper error handling
"""

import sys
import os
import time
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CompleteSystemManager:
    """Manages the complete AI system startup"""
    
    def __init__(self):
        self.services = {}
        self.processes = {}
        self.startup_time = time.time()
        
    def start_system(self):
        """Start the complete system"""
        logger.info("Starting Complete AI System")
        
        try:
            # Step 1: Check dependencies
            self.check_dependencies()
            
            # Step 2: Initialize AI components
            self.initialize_ai_components()
            
            # Step 3: Start backend services
            self.start_backend_services()
            
            # Step 4: Start frontend
            self.start_frontend()
            
            # Step 5: Verify system health
            self.verify_system_health()
            
            # Step 6: Run tests
            self.run_system_tests()
            
            logger.info("✅ Complete system started successfully!")
            self.print_system_status()
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.cleanup_on_failure()
            sys.exit(1)
    
    def check_dependencies(self):
        """Check all system dependencies"""
        logger.info("Checking system dependencies...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher required")
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'networkx', 'sklearn', 
            'flask', 'requests', 'supabase'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"⚠️ Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            self.install_packages(missing_packages)
        
        logger.info("Dependencies check completed")
    
    def install_packages(self, packages):
        """Install missing packages"""
        try:
            for package in packages:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            raise
    
    def initialize_ai_components(self):
        """Initialize all AI components"""
        logger.info("Initializing AI components...")
        
        try:
            # Import and initialize AI components
            from backend.knowledge_graph import knowledge_graph
            from backend.federated_meta_learning import federated_learner
            from backend.gnn_reasoning_engine import gnn_reasoning_engine
            from backend.model_persistence_manager import model_persistence_manager
            from backend.regulatory_compliance import regulatory_compliance_engine
            from backend.impact_forecasting import impact_forecasting_engine
            from backend.ai_service_integration import AIServiceIntegration
            from backend.revolutionary_ai_matching import RevolutionaryAIMatching
            
            # Store component references
            self.services['knowledge_graph'] = knowledge_graph
            self.services['federated_learner'] = federated_learner
            self.services['gnn_reasoning'] = gnn_reasoning_engine
            self.services['model_persistence'] = model_persistence_manager
            self.services['regulatory_compliance'] = regulatory_compliance_engine
            self.services['impact_forecasting'] = impact_forecasting_engine
            self.services['ai_integration'] = AIServiceIntegration()
            self.services['matching_engine'] = RevolutionaryAIMatching()
            
            logger.info("✅ AI components initialized")
            
        except Exception as e:
            logger.error(f"AI component initialization failed: {e}")
            raise
    
    def start_backend_services(self):
        """Start backend services"""
        logger.info("🔧 Starting backend services...")
        
        try:
            # Start Flask backend
            backend_process = subprocess.Popen([
                sys.executable, 'backend/app.js'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['backend'] = backend_process
            
            # Wait for backend to start
            time.sleep(3)
            
            # Check if backend is running
            if backend_process.poll() is None:
                logger.info("✅ Backend service started")
            else:
                raise RuntimeError("Backend service failed to start")
                
        except Exception as e:
            logger.error(f"❌ Backend service startup failed: {e}")
            raise
    
    def start_frontend(self):
        """Start frontend application"""
        logger.info("🎨 Starting frontend application...")
        
        try:
            # Check if frontend directory exists
            frontend_dir = Path('frontend')
            if not frontend_dir.exists():
                logger.warning("⚠️ Frontend directory not found, skipping frontend startup")
                return
            
            # Install frontend dependencies
            if (frontend_dir / 'package.json').exists():
                logger.info("Installing frontend dependencies...")
                subprocess.check_call(['npm', 'install'], cwd=frontend_dir)
            
            # Start frontend development server
            frontend_process = subprocess.Popen([
                'npm', 'run', 'dev'
            ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['frontend'] = frontend_process
            
            # Wait for frontend to start
            time.sleep(5)
            
            # Check if frontend is running
            if frontend_process.poll() is None:
                logger.info("✅ Frontend application started")
            else:
                logger.warning("⚠️ Frontend service may have failed to start")
                
        except Exception as e:
            logger.error(f"❌ Frontend startup failed: {e}")
            # Don't fail the entire system if frontend fails
    
    def verify_system_health(self):
        """Verify system health"""
        logger.info("🏥 Verifying system health...")
        
        try:
            # Check AI service health
            ai_integration = self.services.get('ai_integration')
            if ai_integration:
                health_status = ai_integration.check_service_health()
                if health_status['overall_status'] == 'healthy':
                    logger.info("✅ AI services healthy")
                else:
                    logger.warning("⚠️ Some AI services may have issues")
            
            # Check backend health
            try:
                import requests
                response = requests.get('http://localhost:3000/health', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Backend API healthy")
                else:
                    logger.warning("⚠️ Backend API may have issues")
            except Exception as e:
                logger.warning(f"⚠️ Backend health check failed: {e}")
            
            # Check frontend health
            try:
                response = requests.get('http://localhost:5173', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Frontend healthy")
                else:
                    logger.warning("⚠️ Frontend may have issues")
            except Exception as e:
                logger.warning(f"⚠️ Frontend health check failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Health verification failed: {e}")
    
    def run_system_tests(self):
        """Run system tests"""
        logger.info("🧪 Running system tests...")
        
        try:
            # Run the comprehensive test suite
            test_script = Path('test_complete_system.py')
            if test_script.exists():
                result = subprocess.run([
                    sys.executable, 'test_complete_system.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ System tests passed")
                else:
                    logger.warning("⚠️ Some system tests failed")
                    logger.warning(f"Test output: {result.stdout}")
            else:
                logger.warning("⚠️ Test script not found, skipping tests")
                
        except Exception as e:
            logger.error(f"❌ System tests failed: {e}")
    
    def print_system_status(self):
        """Print system status"""
        startup_time = time.time() - self.startup_time
        
        print("\n" + "="*80)
        print("🎉 COMPLETE AI SYSTEM STARTED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 System Status:")
        print(f"   Startup Time: {startup_time:.2f} seconds")
        print(f"   AI Components: {len(self.services)} initialized")
        print(f"   Running Processes: {len(self.processes)}")
        
        print(f"\n🔧 Services Running:")
        for service_name in self.services.keys():
            print(f"   ✅ {service_name}")
        
        print(f"\n🌐 Access Points:")
        print(f"   Frontend: http://localhost:5173")
        print(f"   Backend API: http://localhost:3000")
        print(f"   Health Check: http://localhost:3000/health")
        
        print(f"\n📝 Logs:")
        print(f"   System Log: system_startup.log")
        print(f"   Test Report: test_report_*.json")
        
        print(f"\n🛑 To stop the system:")
        print(f"   Press Ctrl+C or run: python stop_system.py")
        
        print("\n" + "="*80)
    
    def cleanup_on_failure(self):
        """Cleanup on startup failure"""
        logger.info("Cleaning up on startup failure...")
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped {name} process")
            except Exception as e:
                logger.error(f"Failed to stop {name} process: {e}")
    
    def stop_system(self):
        """Stop the complete system"""
        logger.info("🛑 Stopping complete system...")
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped {name} process")
            except Exception as e:
                logger.error(f"Failed to stop {name} process: {e}")
        
        logger.info("✅ System stopped")

def main():
    """Main function"""
    manager = CompleteSystemManager()
    
    try:
        # Start the system
        manager.start_system()
        
        # Keep the system running
        logger.info("🔄 System is running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received stop signal")
        manager.stop_system()
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        manager.cleanup_on_failure()
        sys.exit(1)

if __name__ == "__main__":
    main() 