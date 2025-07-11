#!/usr/bin/env python3
"""
Cleanup Bad Materials Script
Removes materials with "Unknown" names and other bad data before running enhanced generator.
"""

import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("CleanupScript")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

def cleanup_bad_materials():
    """Clean up bad materials from the database"""
    logger.info("🧹 Starting cleanup of bad materials...")
    
    try:
        # Get all materials
        logger.info("📋 Fetching all materials...")
        materials_response = supabase.table('materials').select('*').execute()
        materials = materials_response.data
        
        logger.info(f"📦 Found {len(materials)} total materials")
        
        # Identify bad materials
        bad_materials = []
        good_materials = []
        
        for material in materials:
            name = material.get('name', '')
            material_name = material.get('material_name', '')
            
            # Check for bad materials
            is_bad = (
                name == 'Unknown' or 
                material_name == 'Unknown' or
                name == 'Unknown Material' or
                material_name == 'Unknown Material' or
                not name or 
                not material_name or
                name.strip() == '' or
                material_name.strip() == ''
            )
            
            if is_bad:
                bad_materials.append(material)
            else:
                good_materials.append(material)
        
        logger.info(f"✅ Good materials: {len(good_materials)}")
        logger.info(f"❌ Bad materials to remove: {len(bad_materials)}")
        
        if not bad_materials:
            logger.info("🎉 No bad materials found! Database is clean.")
            return
        
        # Show sample of bad materials
        logger.info("📋 Sample bad materials to be removed:")
        for i, material in enumerate(bad_materials[:5]):
            logger.info(f"  {i+1}. Name: '{material.get('name')}', Material Name: '{material.get('material_name')}', Company ID: {material.get('company_id')}")
        
        if len(bad_materials) > 5:
            logger.info(f"  ... and {len(bad_materials) - 5} more")
        
        # Confirm deletion
        logger.info("⚠️ About to delete bad materials. This action cannot be undone.")
        logger.info("💡 To proceed, uncomment the deletion code below.")
        
        # UNCOMMENT THE FOLLOWING LINES TO ACTUALLY DELETE BAD MATERIALS
        # logger.info("🗑️ Deleting bad materials...")
        # 
        # for material in bad_materials:
        #     try:
        #         supabase.table('materials').delete().eq('id', material['id']).execute()
        #         logger.info(f"  ✅ Deleted material: {material.get('name')} (ID: {material['id']})")
        #     except Exception as e:
        #         logger.error(f"  ❌ Failed to delete material {material['id']}: {str(e)}")
        # 
        # logger.info("🎉 Cleanup completed!")
        
        # For now, just show what would be deleted
        logger.info("🔍 DRY RUN MODE: No materials were actually deleted.")
        logger.info("💡 To perform actual cleanup, uncomment the deletion code in this script.")
        
        return len(bad_materials)
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def check_materials_quality():
    """Check the quality of remaining materials"""
    logger.info("🔍 Checking materials quality...")
    
    try:
        materials_response = supabase.table('materials').select('*').execute()
        materials = materials_response.data
        
        # Analyze materials
        total_materials = len(materials)
        materials_with_names = len([m for m in materials if m.get('name') and m.get('name') != 'Unknown'])
        materials_with_descriptions = len([m for m in materials if m.get('description')])
        materials_with_quantities = len([m for m in materials if m.get('quantity')])
        materials_with_company_id = len([m for m in materials if m.get('company_id')])
        
        waste_materials = len([m for m in materials if m.get('type') == 'waste'])
        requirement_materials = len([m for m in materials if m.get('type') == 'requirement'])
        
        logger.info("📊 Materials Quality Report:")
        logger.info(f"  Total materials: {total_materials}")
        logger.info(f"  Materials with proper names: {materials_with_names} ({materials_with_names/total_materials*100:.1f}%)")
        logger.info(f"  Materials with descriptions: {materials_with_descriptions} ({materials_with_descriptions/total_materials*100:.1f}%)")
        logger.info(f"  Materials with quantities: {materials_with_quantities} ({materials_with_quantities/total_materials*100:.1f}%)")
        logger.info(f"  Materials with company_id: {materials_with_company_id} ({materials_with_company_id/total_materials*100:.1f}%)")
        logger.info(f"  Waste materials: {waste_materials}")
        logger.info(f"  Requirement materials: {requirement_materials}")
        
        # Show sample of good materials
        good_materials = [m for m in materials if m.get('name') and m.get('name') != 'Unknown']
        if good_materials:
            logger.info("📋 Sample good materials:")
            for i, material in enumerate(good_materials[:3]):
                logger.info(f"  {i+1}. {material.get('name')} - {material.get('description', 'No description')[:50]}...")
        
    except Exception as e:
        logger.error(f"❌ Error checking materials quality: {str(e)}")

def main():
    """Main cleanup function"""
    logger.info("🚀 Starting Materials Cleanup Process...")
    logger.info("=" * 60)
    
    # Check current state
    check_materials_quality()
    
    print("\n" + "=" * 60)
    
    # Perform cleanup
    bad_count = cleanup_bad_materials()
    
    print("\n" + "=" * 60)
    
    # Check state after cleanup
    if bad_count > 0:
        logger.info("🔄 Checking state after cleanup...")
        check_materials_quality()
    
    logger.info("✅ Cleanup process completed!")
    logger.info("💡 Next step: Run enhanced_ai_generator.py to generate high-quality materials")

if __name__ == "__main__":
    main() 