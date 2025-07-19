#!/usr/bin/env python3
"""
Production AI System Startup Script
Launches the complete production AI system with all components
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import signal
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import production components
from ai_production_orchestrator import AIProductionOrchestrator, ProductionConfig
from ai_feedback_orchestrator import AIFeedbackOrchestrator
from ai_fusion_layer import AIFusionLayer
from ai_hyperparameter_optimizer import AIHyperparameterOptimizer
from ai_retraining_pipeline import AIRetrainingPipeline
from ai_monitoring_dashboard import AIMonitoringDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_ai_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Patch StreamHandler to use UTF-8 encoding if possible
try:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream.reconfigure(encoding='utf-8')
except Exception:
    pass
logger = logging.getLogger(__name__)

class ProductionAISystem:
    """Production AI System Manager"""
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()
        self.orchestrator = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Production AI System Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def start(self):
        """Start the production AI system"""
        try:
            logger.info("🚀 Starting Production AI System...")
            
            # Create and start orchestrator
            self.orchestrator = AIProductionOrchestrator(self.config)
            
            # Start the orchestrator
            await self.orchestrator.start()
            
            logger.info("✅ Production AI System started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("🛑 Shutdown signal received, stopping system...")
            
        except Exception as e:
            logger.error(f"❌ Error starting Production AI System: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the production AI system"""
        try:
            if self.orchestrator:
                self.orchestrator.stop()
                logger.info("✅ Production AI System stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping Production AI System: {e}")
    
    def get_system_status(self) -> dict:
        """Get system status"""
        try:
            if self.orchestrator:
                return self.orchestrator.get_system_status()
            else:
                return {'status': 'not_started'}
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'error': str(e)}

def create_production_config(args) -> ProductionConfig:
    """Create production configuration from command line arguments"""
    return ProductionConfig(
        feedback_enabled=not args.disable_feedback,
        fusion_enabled=not args.disable_fusion,
        optimization_enabled=not args.disable_optimization,
        retraining_enabled=not args.disable_retraining,
        monitoring_enabled=not args.disable_monitoring,
        auto_deploy=not args.disable_auto_deploy,
        health_check_interval=args.health_check_interval,
        backup_interval=args.backup_interval,
        log_level=args.log_level
    )

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Production AI System')
    
    # Component flags
    parser.add_argument('--disable-feedback', action='store_true', 
                       help='Disable feedback orchestrator')
    parser.add_argument('--disable-fusion', action='store_true', 
                       help='Disable fusion layer')
    parser.add_argument('--disable-optimization', action='store_true', 
                       help='Disable hyperparameter optimization')
    parser.add_argument('--disable-retraining', action='store_true', 
                       help='Disable retraining pipeline')
    parser.add_argument('--disable-monitoring', action='store_true', 
                       help='Disable monitoring dashboard')
    parser.add_argument('--disable-auto-deploy', action='store_true', 
                       help='Disable auto-deployment of improved models')
    
    # Configuration
    parser.add_argument('--health-check-interval', type=int, default=60,
                       help='Health check interval in seconds')
    parser.add_argument('--backup-interval', type=int, default=3600,
                       help='Backup interval in seconds')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    return parser

async def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Create production configuration
        config = create_production_config(args)
        
        # Log configuration
        logger.info("Production AI System Configuration:")
        logger.info(f"  Feedback Orchestrator: {'Enabled' if config.feedback_enabled else 'Disabled'}")
        logger.info(f"  Fusion Layer: {'Enabled' if config.fusion_enabled else 'Disabled'}")
        logger.info(f"  Hyperparameter Optimization: {'Enabled' if config.optimization_enabled else 'Disabled'}")
        logger.info(f"  Retraining Pipeline: {'Enabled' if config.retraining_enabled else 'Disabled'}")
        logger.info(f"  Monitoring Dashboard: {'Enabled' if config.monitoring_enabled else 'Disabled'}")
        logger.info(f"  Auto Deploy: {'Enabled' if config.auto_deploy else 'Disabled'}")
        logger.info(f"  Health Check Interval: {config.health_check_interval}s")
        logger.info(f"  Backup Interval: {config.backup_interval}s")
        logger.info(f"  Log Level: {config.log_level}")
        
        # Create and start production system
        system = ProductionAISystem(config)
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 