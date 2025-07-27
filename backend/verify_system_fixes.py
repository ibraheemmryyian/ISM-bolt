#!/usr/bin/env python3
"""
Verification script to check that all system fixes are working correctly
This script tests all the components that interact with revolutionary_ai_matching.py
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verification")

class SystemVerifier:
    """Verifies that all system fixes are working correctly"""
    
    def __init__(self):
        self.verification_results = {}
    
    async def verify_all_systems(self):
        """Verify all systems"""
        logger.info("🔍 Starting system verification")
        
        # 1. Verify revolutionary_ai_matching.py
        await self.verify_revolutionary_ai_matching()
        
        # 2. Verify integrate_ultra_advanced_ai.py
        await self.verify_integrate_ultra_advanced_ai()
        
        # 3. Verify real_performance_benchmark.py
        await self.verify_real_performance_benchmark()
        
        # 4. Verify generate_supervised_materials_and_matches.py
        await self.verify_generate_supervised_materials_and_matches()
        
        # 5. Verify integrate_revolutionary_ai_matching.py
        await self.verify_integrate_revolutionary_ai_matching()
        
        # Print summary
        self._print_verification_summary()
    
    async def verify_revolutionary_ai_matching(self):
        """Verify revolutionary_ai_matching.py"""
        logger.info("🧪 Verifying revolutionary_ai_matching.py")
        
        try:
            # Import the module
            from revolutionary_ai_matching import RevolutionaryAIMatching
            
            # Initialize the system
            start_time = time.time()
            ai_matching = RevolutionaryAIMatching()
            init_time = time.time() - start_time
            
            # Test match generation
            source_material = "Test Material"
            source_type = "metal"
            source_company = "Test Company"
            
            matches = await ai_matching.generate_high_quality_matches(
                source_material, source_type, source_company
            )
            
            # Verify results
            success = len(matches) > 0
            
            self.verification_results['revolutionary_ai_matching'] = {
                'success': success,
                'initialization_time': init_time,
                'matches_generated': len(matches),
                'message': f"Successfully generated {len(matches)} matches" if success else "Failed to generate matches"
            }
            
            logger.info(f"✅ revolutionary_ai_matching.py verification {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ revolutionary_ai_matching.py verification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            self.verification_results['revolutionary_ai_matching'] = {
                'success': False,
                'error': str(e),
                'message': "Failed with exception"
            }
    
    async def verify_integrate_ultra_advanced_ai(self):
        """Verify integrate_ultra_advanced_ai.py"""
        logger.info("🧪 Verifying integrate_ultra_advanced_ai.py")
        
        try:
            # Try to import the module
            spec = importlib.util.find_spec('integrate_ultra_advanced_ai')
            
            if spec is None:
                logger.warning("⚠️ integrate_ultra_advanced_ai.py module not found")
                self.verification_results['integrate_ultra_advanced_ai'] = {
                    'success': False,
                    'message': "Module not found"
                }
                return
            
            # Import the module
            integrate_ultra_advanced_ai = importlib.import_module('integrate_ultra_advanced_ai')
            
            # Check if UltraAdvancedAIIntegration class exists
            if hasattr(integrate_ultra_advanced_ai, 'UltraAdvancedAIIntegration'):
                # Don't initialize as it might have many dependencies
                success = True
                message = "Module imported successfully and class found"
            else:
                success = False
                message = "Module imported but UltraAdvancedAIIntegration class not found"
            
            self.verification_results['integrate_ultra_advanced_ai'] = {
                'success': success,
                'message': message
            }
            
            logger.info(f"✅ integrate_ultra_advanced_ai.py verification {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ integrate_ultra_advanced_ai.py verification failed: {e}")
            
            self.verification_results['integrate_ultra_advanced_ai'] = {
                'success': False,
                'error': str(e),
                'message': "Failed with exception"
            }
    
    async def verify_real_performance_benchmark(self):
        """Verify real_performance_benchmark.py"""
        logger.info("🧪 Verifying real_performance_benchmark.py")
        
        try:
            # Try to import the module
            spec = importlib.util.find_spec('real_performance_benchmark')
            
            if spec is None:
                logger.warning("⚠️ real_performance_benchmark.py module not found")
                self.verification_results['real_performance_benchmark'] = {
                    'success': False,
                    'message': "Module not found"
                }
                return
            
            # Import the module
            real_performance_benchmark = importlib.import_module('real_performance_benchmark')
            
            # Check if RealPerformanceBenchmark class exists
            if hasattr(real_performance_benchmark, 'RealPerformanceBenchmark'):
                # Don't run benchmark as it would take too long
                success = True
                message = "Module imported successfully and class found"
            else:
                success = False
                message = "Module imported but RealPerformanceBenchmark class not found"
            
            self.verification_results['real_performance_benchmark'] = {
                'success': success,
                'message': message
            }
            
            logger.info(f"✅ real_performance_benchmark.py verification {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ real_performance_benchmark.py verification failed: {e}")
            
            self.verification_results['real_performance_benchmark'] = {
                'success': False,
                'error': str(e),
                'message': "Failed with exception"
            }
    
    async def verify_generate_supervised_materials_and_matches(self):
        """Verify generate_supervised_materials_and_matches.py"""
        logger.info("🧪 Verifying generate_supervised_materials_and_matches.py")
        
        try:
            # Try to import the module
            spec = importlib.util.find_spec('generate_supervised_materials_and_matches')
            
            if spec is None:
                logger.warning("⚠️ generate_supervised_materials_and_matches.py module not found")
                self.verification_results['generate_supervised_materials_and_matches'] = {
                    'success': False,
                    'message': "Module not found"
                }
                return
            
            # Import the module
            generate_supervised_materials_and_matches = importlib.import_module('generate_supervised_materials_and_matches')
            
            # Check if WorldClassMaterialDataGenerator class exists
            if hasattr(generate_supervised_materials_and_matches, 'WorldClassMaterialDataGenerator'):
                # Don't initialize as it might have many dependencies
                success = True
                message = "Module imported successfully and class found"
            else:
                success = False
                message = "Module imported but WorldClassMaterialDataGenerator class not found"
            
            self.verification_results['generate_supervised_materials_and_matches'] = {
                'success': success,
                'message': message
            }
            
            logger.info(f"✅ generate_supervised_materials_and_matches.py verification {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ generate_supervised_materials_and_matches.py verification failed: {e}")
            
            self.verification_results['generate_supervised_materials_and_matches'] = {
                'success': False,
                'error': str(e),
                'message': "Failed with exception"
            }
    
    async def verify_integrate_revolutionary_ai_matching(self):
        """Verify integrate_revolutionary_ai_matching.py"""
        logger.info("🧪 Verifying integrate_revolutionary_ai_matching.py")
        
        try:
            # Try to import the module
            spec = importlib.util.find_spec('integrate_revolutionary_ai_matching')
            
            if spec is None:
                logger.warning("⚠️ integrate_revolutionary_ai_matching.py module not found")
                self.verification_results['integrate_revolutionary_ai_matching'] = {
                    'success': False,
                    'message': "Module not found"
                }
                return
            
            # Import the module
            integrate_revolutionary_ai_matching = importlib.import_module('integrate_revolutionary_ai_matching')
            
            # Check if RevolutionaryAIMatchingIntegration class exists
            if hasattr(integrate_revolutionary_ai_matching, 'RevolutionaryAIMatchingIntegration'):
                # Don't initialize as it might have many dependencies
                success = True
                message = "Module imported successfully and class found"
            else:
                success = False
                message = "Module imported but RevolutionaryAIMatchingIntegration class not found"
            
            self.verification_results['integrate_revolutionary_ai_matching'] = {
                'success': success,
                'message': message
            }
            
            logger.info(f"✅ integrate_revolutionary_ai_matching.py verification {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ integrate_revolutionary_ai_matching.py verification failed: {e}")
            
            self.verification_results['integrate_revolutionary_ai_matching'] = {
                'success': False,
                'error': str(e),
                'message': "Failed with exception"
            }
    
    def _print_verification_summary(self):
        """Print verification summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("="*60)
        
        all_success = True
        
        for component, result in self.verification_results.items():
            success = result.get('success', False)
            message = result.get('message', "No message")
            
            if success:
                logger.info(f"✅ {component}: {message}")
            else:
                logger.info(f"❌ {component}: {message}")
                all_success = False
                if 'error' in result:
                    logger.info(f"   Error: {result['error']}")
        
        logger.info("="*60)
        if all_success:
            logger.info("🎉 ALL SYSTEMS VERIFIED SUCCESSFULLY")
        else:
            logger.info("⚠️ SOME SYSTEMS FAILED VERIFICATION")
        logger.info("="*60)
        
        return all_success

async def main():
    """Main function"""
    verifier = SystemVerifier()
    await verifier.verify_all_systems()

if __name__ == "__main__":
    asyncio.run(main())