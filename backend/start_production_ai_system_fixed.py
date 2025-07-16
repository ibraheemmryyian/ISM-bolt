#!/usr/bin/env python3
"""
Production AI System Startup Script - Fixed Version
Launches the complete production AI system with graceful dependency handling
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import signal
import argparse
import subprocess

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class DependencyManager:
    """Manages dependencies and provides fallbacks"""
    
    @staticmethod
    def check_and_install_dependencies():
        """Check and install missing dependencies"""
        logger.info("Checking dependencies...")
        
        missing_packages = []
        
        # Check critical packages
        critical_packages = [
            'numpy', 'pandas', 'flask', 'requests'
        ]
        
        for package in critical_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} missing")
        
        # Install missing packages
        if missing_packages:
            logger.info(f"Installing missing packages: {missing_packages}")
            try:
                for package in missing_packages:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', '--user', package
                    ])
                logger.info("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install dependencies: {e}")
                return False
        
        return True
    
    @staticmethod
    def safe_import(module_name, fallback=None):
        """Safely import a module with fallback"""
        try:
            return __import__(module_name)
        except ImportError:
            logger.warning(f"⚠️ {module_name} not available, using fallback")
            return fallback

# Import production components with fallbacks
try:
    from ai_production_orchestrator import AIProductionOrchestrator, ProductionConfig
    PRODUCTION_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Production Orchestrator not available: {e}")
    PRODUCTION_ORCHESTRATOR_AVAILABLE = False
    AIProductionOrchestrator = None
    ProductionConfig = None

try:
    from ai_feedback_orchestrator import AIFeedbackOrchestrator
    FEEDBACK_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Feedback Orchestrator not available: {e}")
    FEEDBACK_ORCHESTRATOR_AVAILABLE = False
    AIFeedbackOrchestrator = None

try:
    from ai_fusion_layer import AIFusionLayer
    FUSION_LAYER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Fusion Layer not available: {e}")
    FUSION_LAYER_AVAILABLE = False
    AIFusionLayer = None

try:
    from ai_hyperparameter_optimizer import AIHyperparameterOptimizer
    HYPERPARAMETER_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Hyperparameter Optimizer not available: {e}")
    HYPERPARAMETER_OPTIMIZER_AVAILABLE = False
    AIHyperparameterOptimizer = None

try:
    from ai_retraining_pipeline import AIRetrainingPipeline
    RETRAINING_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Retraining Pipeline not available: {e}")
    RETRAINING_PIPELINE_AVAILABLE = False
    AIRetrainingPipeline = None

try:
    from ai_monitoring_dashboard import AIMonitoringDashboard
    MONITORING_DASHBOARD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ AI Monitoring Dashboard not available: {e}")
    MONITORING_DASHBOARD_AVAILABLE = False
    AIMonitoringDashboard = None

class FallbackProductionConfig:
    """Fallback configuration when ProductionConfig is not available"""
    
    def __init__(self, **kwargs):
        self.feedback_enabled = kwargs.get('feedback_enabled', False)
        self.fusion_enabled = kwargs.get('fusion_enabled', False)
        self.optimization_enabled = kwargs.get('optimization_enabled', False)
        self.retraining_enabled = kwargs.get('retraining_enabled', False)
        self.monitoring_enabled = kwargs.get('monitoring_enabled', False)
        self.auto_deploy = kwargs.get('auto_deploy', False)
        self.health_check_interval = kwargs.get('health_check_interval', 60)
        self.backup_interval = kwargs.get('backup_interval', 3600)
        self.log_level = kwargs.get('log_level', 'INFO')

class FallbackProductionOrchestrator:
    """Fallback orchestrator when AIProductionOrchestrator is not available"""
    
    def __init__(self, config):
        self.config = config
        self.status = 'initialized'
        logger.info("Fallback Production Orchestrator initialized")
    
    async def start(self):
        """Start the fallback orchestrator"""
        logger.info("🚀 Starting Fallback Production AI System...")
        self.status = 'running'
        logger.info("✅ Fallback Production AI System started")
    
    def stop(self):
        """Stop the fallback orchestrator"""
        logger.info("🛑 Stopping Fallback Production AI System...")
        self.status = 'stopped'
        logger.info("✅ Fallback Production AI System stopped")
    
    def get_system_status(self):
        """Get system status"""
        return {
            'status': self.status,
            'mode': 'fallback',
            'components': {
                'feedback_orchestrator': 'unavailable',
                'fusion_layer': 'unavailable',
                'hyperparameter_optimizer': 'unavailable',
                'retraining_pipeline': 'unavailable',
                'monitoring_dashboard': 'unavailable'
            }
        }

class ProductionAISystem:
    """Production AI System Manager with fallback support"""
    
    def __init__(self, config=None):
        # Use fallback config if ProductionConfig is not available
        if ProductionConfig is not None:
            self.config = config or ProductionConfig()
        else:
            self.config = config or FallbackProductionConfig()
        
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
            
            # Create orchestrator (use fallback if needed)
            if AIProductionOrchestrator is not None:
                self.orchestrator = AIProductionOrchestrator(self.config)
            else:
                self.orchestrator = FallbackProductionOrchestrator(self.config)
            
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

def create_production_config(args):
    """Create production configuration from command line arguments"""
    if ProductionConfig is not None:
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
    else:
        return FallbackProductionConfig(
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
    parser = argparse.ArgumentParser(description='Production AI System - Fixed Version')
    
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
    
    # Dependency management
    parser.add_argument('--fix-dependencies', action='store_true',
                       help='Automatically fix missing dependencies')
    
    return parser

async def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Fix dependencies if requested
        if args.fix_dependencies:
            logger.info("Fixing dependencies...")
            if not DependencyManager.check_and_install_dependencies():
                logger.error("❌ Failed to fix dependencies")
                sys.exit(1)
        
        # Log component availability
        logger.info("Component Availability:")
        logger.info(f"  Production Orchestrator: {'✅' if PRODUCTION_ORCHESTRATOR_AVAILABLE else '❌'}")
        logger.info(f"  Feedback Orchestrator: {'✅' if FEEDBACK_ORCHESTRATOR_AVAILABLE else '❌'}")
        logger.info(f"  Fusion Layer: {'✅' if FUSION_LAYER_AVAILABLE else '❌'}")
        logger.info(f"  Hyperparameter Optimizer: {'✅' if HYPERPARAMETER_OPTIMIZER_AVAILABLE else '❌'}")
        logger.info(f"  Retraining Pipeline: {'✅' if RETRAINING_PIPELINE_AVAILABLE else '❌'}")
        logger.info(f"  Monitoring Dashboard: {'✅' if MONITORING_DASHBOARD_AVAILABLE else '❌'}")
        
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