#!/usr/bin/env python3
"""
Fix Version Conflicts for ISM AI System
Resolves specific import issues with sentence-transformers and sklearn
"""

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_sentence_transformers():
    """Fix sentence-transformers import issue"""
    logger.info("Fixing sentence-transformers import issue...")
    
    try:
        # Uninstall conflicting packages
        logger.info("Uninstalling conflicting packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', '-y', 
            'sentence-transformers', 'transformers', 'huggingface-hub'
        ], check=True)
        
        # Install compatible versions
        logger.info("Installing compatible versions...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'huggingface-hub==0.20.3'
        ], check=True)
        
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'transformers==4.35.0'
        ], check=True)
        
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'sentence-transformers==2.2.2'
        ], check=True)
        
        # Test import
        import sentence_transformers
        logger.info("✅ sentence-transformers fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix sentence-transformers: {e}")
        return False

def fix_sklearn():
    """Fix sklearn import issue"""
    logger.info("Fixing sklearn import issue...")
    
    try:
        # Uninstall conflicting packages
        logger.info("Uninstalling conflicting packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', '-y', 
            'scikit-learn', 'numpy'
        ], check=True)
        
        # Install compatible versions
        logger.info("Installing compatible versions...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user',
            'numpy==1.24.3'
        ], check=True)
        
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user',
            'scikit-learn==1.3.0'
        ], check=True)
        
        # Test import
        import sklearn
        logger.info("✅ sklearn fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix sklearn: {e}")
        return False

def test_imports():
    """Test all critical imports"""
    logger.info("Testing critical imports...")
    
    critical_imports = [
        'numpy', 'pandas', 'torch', 'transformers', 'sentence_transformers',
        'sklearn', 'networkx', 'flask', 'requests', 'supabase'
    ]
    
    failed_imports = []
    for module in critical_imports:
        try:
            __import__(module)
            logger.info(f"✅ {module} imports successfully")
        except ImportError as e:
            logger.error(f"❌ {module} import failed: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"❌ Failed imports: {failed_imports}")
        return False
    else:
        logger.info("✅ All critical imports successful")
        return True

def main():
    """Main fix process"""
    logger.info("🔧 Fixing version conflicts...")
    
    # Fix sentence-transformers
    sentence_transformers_fixed = fix_sentence_transformers()
    
    # Fix sklearn
    sklearn_fixed = fix_sklearn()
    
    # Test all imports
    imports_working = test_imports()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VERSION CONFLICT FIX SUMMARY")
    logger.info("="*50)
    logger.info(f"sentence-transformers: {'✅' if sentence_transformers_fixed else '❌'}")
    logger.info(f"sklearn: {'✅' if sklearn_fixed else '❌'}")
    logger.info(f"All imports: {'✅' if imports_working else '❌'}")
    
    if imports_working:
        logger.info("\n🎉 All version conflicts resolved!")
        logger.info("You can now run your AI system.")
        return True
    else:
        logger.warning("\n⚠️ Some conflicts remain.")
        logger.warning("Try running the fix again or install packages manually.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 