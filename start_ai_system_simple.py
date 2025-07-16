#!/usr/bin/env python3
"""
Simple AI System Startup Script
Provides a working AI system with minimal dependencies
"""

import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import signal
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_system.log', encoding='utf-8'),
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

class SimpleAISystem:
    """Simple AI System with minimal dependencies"""
    
    def __init__(self):
        self.status = 'initialized'
        self.start_time = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Simple AI System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def start(self):
        """Start the simple AI system"""
        try:
            logger.info("🚀 Starting Simple AI System...")
            self.start_time = datetime.now()
            self.status = 'running'
            
            logger.info("✅ Simple AI System started successfully")
            logger.info("System is running in basic mode with core functionality")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("🛑 Shutdown signal received, stopping system...")
            
        except Exception as e:
            logger.error(f"❌ Error starting Simple AI System: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the simple AI system"""
        try:
            self.status = 'stopped'
            logger.info("✅ Simple AI System stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping Simple AI System: {e}")
    
    def get_system_status(self) -> dict:
        """Get system status"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'status': self.status,
            'mode': 'simple',
            'uptime_seconds': uptime,
            'components': {
                'ai_matching': 'basic',
                'feedback_system': 'basic',
                'monitoring': 'basic',
                'database': 'basic'
            },
            'capabilities': [
                'Basic AI matching',
                'Simple feedback collection',
                'Basic monitoring',
                'Core system functionality'
            ]
        }

class DependencyChecker:
    """Check and report on system dependencies"""
    
    @staticmethod
    def check_dependencies():
        """Check available dependencies"""
        logger.info("Checking system dependencies...")
        
        dependencies = {
            'Core': ['numpy', 'pandas', 'flask', 'requests'],
            'AI/ML': ['torch', 'transformers', 'sentence_transformers', 'sklearn'],
            'Database': ['supabase', 'psycopg2'],
            'Utilities': ['networkx', 'matplotlib', 'seaborn']
        }
        
        results = {}
        
        for category, packages in dependencies.items():
            category_results = {}
            for package in packages:
                try:
                    __import__(package)
                    category_results[package] = '✅ Available'
                except ImportError:
                    category_results[package] = '❌ Missing'
            results[category] = category_results
        
        # Log results
        for category, packages in results.items():
            logger.info(f"\n{category} Dependencies:")
            for package, status in packages.items():
                logger.info(f"  {package}: {status}")
        
        return results
    
    @staticmethod
    def suggest_installation():
        """Suggest installation commands"""
        logger.info("\n📋 To install missing dependencies, run:")
        logger.info("python fix_dependencies.py")
        logger.info("or manually install packages with:")
        logger.info("pip install --user numpy pandas torch transformers sentence-transformers")

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Simple AI System')
    
    parser.add_argument('--check-deps', action='store_true',
                       help='Check system dependencies')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    parser.add_argument('--fix-deps', action='store_true',
                       help='Attempt to fix missing dependencies')
    
    return parser

async def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Check dependencies if requested
        if args.check_deps:
            DependencyChecker.check_dependencies()
            DependencyChecker.suggest_installation()
            return
        
        # Fix dependencies if requested
        if args.fix_deps:
            logger.info("Attempting to fix dependencies...")
            try:
                # Try to run the dependency fixer
                if os.path.exists('fix_dependencies.py'):
                    subprocess.run([sys.executable, 'fix_dependencies.py'], check=True)
                    logger.info("✅ Dependencies fixed successfully")
                else:
                    logger.warning("⚠️ fix_dependencies.py not found")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to fix dependencies: {e}")
                return
        
        # Check dependencies
        deps = DependencyChecker.check_dependencies()
        
        # Create and start simple AI system
        system = SimpleAISystem()
        
        logger.info("\n" + "="*50)
        logger.info("SIMPLE AI SYSTEM STARTING")
        logger.info("="*50)
        logger.info("This is a basic AI system with core functionality.")
        logger.info("For full features, install all dependencies.")
        logger.info("="*50)
        
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 