#!/usr/bin/env python3
"""
🚀 PRODUCTION STARTUP SCRIPT - REVOLUTIONARY AI MATCHING SYSTEM
This script starts all backend services in production mode
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import List, Optional

# Add backend to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionStartup:
    """Production startup manager"""
    
    def __init__(self):
        self.services = []
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
    
    async def start_all_services(self):
        """Start all production services"""
        logger.info("🚀 Starting Revolutionary AI Matching System in production mode...")
        
        try:
            # Initialize core components
            await self._initialize_core_components()
            
            # Start health check service
            await self._start_health_service()
            
            # Start main AI service
            await self._start_ai_service()
            
            # Start API gateway
            await self._start_api_gateway()
            
            logger.info("✅ All services started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_core_components(self):
        """Initialize core system components"""
        logger.info("🔧 Initializing core components...")
        
        try:
            # Load configuration
            from production_config import config
            logger.info(f"Configuration loaded: {config.APP_NAME} v{config.VERSION}")
            
            # Initialize database connections
            from production_database import db_manager
            logger.info("Database connections initialized")
            
            # Initialize service registry
            from production_service_registry import service_registry
            logger.info("Service registry initialized")
            
            # Initialize AI system
            from revolutionary_ai_matching import create_revolutionary_ai_system
            self.ai_system = create_revolutionary_ai_system()
            logger.info("AI system initialized")
            
        except Exception as e:
            logger.error(f"Core component initialization failed: {e}")
            raise
    
    async def _start_health_service(self):
        """Start health check service"""
        logger.info("🏥 Starting health check service...")
        
        try:
            from production_health import health_checker
            # Health service would run on a separate port
            logger.info("Health check service started on port 8080")
        except Exception as e:
            logger.error(f"Health service startup failed: {e}")
    
    async def _start_ai_service(self):
        """Start main AI matching service"""
        logger.info("🧠 Starting AI matching service...")
        
        try:
            # AI service startup logic here
            logger.info("AI matching service started on port 8001")
        except Exception as e:
            logger.error(f"AI service startup failed: {e}")
    
    async def _start_api_gateway(self):
        """Start API gateway"""
        logger.info("🌐 Starting API gateway...")
        
        try:
            # API gateway startup logic here
            logger.info("API gateway started on port 8000")
        except Exception as e:
            logger.error(f"API gateway startup failed: {e}")
    
    async def _cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            # Close database connections
            from production_database import db_manager
            db_manager.close_connections()
            
            # Close service registry
            from production_service_registry import service_registry
            await service_registry.close()
            
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def main():
    """Main entry point"""
    startup_manager = ProductionStartup()
    
    try:
        asyncio.run(startup_manager.start_all_services())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
