"""
Production Integration Script for Revolutionary AI Matching System
This script demonstrates how to integrate the Revolutionary AI Matching system
into a production environment with proper error handling and logging.
"""

import asyncio
import logging
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("revolutionary_ai_matching.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("revolutionary_ai_integration")

# Import the Revolutionary AI Matching system
try:
    from revolutionary_ai_matching import RevolutionaryAIMatching
    logger.info("Successfully imported Revolutionary AI Matching system")
except ImportError as e:
    logger.critical(f"Failed to import Revolutionary AI Matching system: {e}")
    sys.exit(1)

class RevolutionaryAIMatchingIntegration:
    """Integration class for the Revolutionary AI Matching system"""
    
    def __init__(self):
        """Initialize the integration"""
        logger.info("Initializing Revolutionary AI Matching Integration")
        
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        self._check_environment_variables()
        
        # Initialize the Revolutionary AI Matching system
        try:
            self.ai_matching = RevolutionaryAIMatching()
            logger.info("Revolutionary AI Matching system initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize Revolutionary AI Matching system: {e}")
            raise
    
    def _check_environment_variables(self):
        """Check that all required environment variables are set"""
        required_vars = [
            'NEXT_GEN_MATERIALS_API_KEY',
            'MATERIALSBERT_API_KEY',
            'DEEPSEEK_R1_API_KEY',
            'FREIGHTOS_API_KEY',
            'API_NINJA_KEY',
            'SUPABASE_URL',
            'SUPABASE_KEY',
            'NEWSAPI_KEY',
            'CURRENTS_API_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Using dummy values for testing purposes")
    
    async def generate_matches(self, 
                              source_material: str, 
                              source_type: str, 
                              source_company: str,
                              retry_count: int = 3,
                              retry_delay: float = 2.0) -> List[Dict[str, Any]]:
        """
        Generate matches with retry logic
        
        Args:
            source_material: The name of the source material
            source_type: The type of the source material
            source_company: The name of the source company
            retry_count: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of match dictionaries
        """
        logger.info(f"Generating matches for {source_material} ({source_type}) from {source_company}")
        
        for attempt in range(retry_count + 1):
            try:
                start_time = time.time()
                
                matches = await self.ai_matching.generate_high_quality_matches(
                    source_material, source_type, source_company
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated {len(matches)} matches in {elapsed_time:.2f} seconds")
                
                # Save matches to file for reference
                self._save_matches_to_file(matches, source_material, source_type, source_company)
                
                return matches
                
            except Exception as e:
                if attempt < retry_count:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to generate matches after {retry_count + 1} attempts: {e}")
                    raise
    
    def _save_matches_to_file(self, 
                             matches: List[Dict[str, Any]], 
                             source_material: str, 
                             source_type: str, 
                             source_company: str):
        """Save matches to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matches_{source_material.replace(' ', '_')}_{timestamp}.json"
        
        output = {
            "source_material": source_material,
            "source_type": source_type,
            "source_company": source_company,
            "generated_at": datetime.now().isoformat(),
            "match_count": len(matches),
            "matches": matches
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved matches to {filename}")
        except Exception as e:
            logger.error(f"Failed to save matches to file: {e}")
    
    async def process_batch(self, materials: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a batch of materials
        
        Args:
            materials: List of dictionaries containing material information
                Each dictionary should have 'name', 'type', and 'company' keys
                
        Returns:
            Dictionary mapping material names to their matches
        """
        logger.info(f"Processing batch of {len(materials)} materials")
        
        results = {}
        
        for material in materials:
            name = material.get('name')
            type_ = material.get('type')
            company = material.get('company')
            
            if not all([name, type_, company]):
                logger.warning(f"Skipping material with missing information: {material}")
                continue
            
            try:
                matches = await self.generate_matches(name, type_, company)
                results[name] = matches
                logger.info(f"Successfully processed {name}")
            except Exception as e:
                logger.error(f"Failed to process {name}: {e}")
                results[name] = []
        
        logger.info(f"Batch processing complete. Processed {len(results)} materials")
        return results


async def main():
    """Main function demonstrating the integration"""
    try:
        # Initialize the integration
        integration = RevolutionaryAIMatchingIntegration()
        
        # Example materials
        materials = [
            {"name": "Recycled Aluminum", "type": "metal", "company": "EcoMetals Inc."},
            {"name": "PET Plastic", "type": "plastic", "company": "GreenPolymers Ltd."},
            {"name": "Carbon Fiber", "type": "composite", "company": "AdvancedMaterials Co."}
        ]
        
        # Process the materials
        results = await integration.process_batch(materials)
        
        # Display summary
        for material_name, matches in results.items():
            logger.info(f"{material_name}: {len(matches)} matches")
            
            # Display top 3 matches
            for i, match in enumerate(matches[:3]):
                logger.info(f"  {i+1}. {match['target_company_name']} - Score: {match['match_score']:.2f}")
        
        return 0
    
    except Exception as e:
        logger.critical(f"Integration failed: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)