#!/usr/bin/env python3
"""
Fix NumPy Version Conflicts
Resolves the ComplexWarning import issue and version conflicts
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

def fix_numpy_conflicts():
    """Fix NumPy version conflicts"""
    logger.info("Fixing NumPy version conflicts...")
    
    try:
        # Uninstall all conflicting packages
        logger.info("Uninstalling conflicting packages...")
        packages_to_remove = [
            'numpy', 'scipy', 'scikit-learn', 'sentence-transformers', 
            'transformers', 'huggingface-hub', 'thinc'
        ]
        
        for package in packages_to_remove:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'uninstall', '-y', package
                ], check=True)
                logger.info(f"Uninstalled {package}")
            except subprocess.CalledProcessError:
                logger.info(f"{package} not installed or already removed")
        
        # Install compatible versions in order
        logger.info("Installing compatible versions...")
        
        # 1. Install NumPy first (compatible version)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'numpy==1.24.3'
        ], check=True)
        logger.info("✅ NumPy 1.24.3 installed")
        
        # 2. Install SciPy (compatible with NumPy 1.24.3)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'scipy==1.11.1'
        ], check=True)
        logger.info("✅ SciPy 1.11.1 installed")
        
        # 3. Install scikit-learn (compatible with NumPy 1.24.3)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'scikit-learn==1.3.0'
        ], check=True)
        logger.info("✅ scikit-learn 1.3.0 installed")
        
        # 4. Install huggingface-hub (compatible version)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'huggingface-hub==0.20.3'
        ], check=True)
        logger.info("✅ huggingface-hub 0.20.3 installed")
        
        # 5. Install transformers (compatible version)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'transformers==4.35.0'
        ], check=True)
        logger.info("✅ transformers 4.35.0 installed")
        
        # 6. Install sentence-transformers (compatible version)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
            'sentence-transformers==2.2.2'
        ], check=True)
        logger.info("✅ sentence-transformers 2.2.2 installed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix NumPy conflicts: {e}")
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

def check_versions():
    """Check installed versions"""
    logger.info("Checking installed versions...")
    
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
    except ImportError:
        logger.error("NumPy not available")
    
    try:
        import scipy
        logger.info(f"SciPy version: {scipy.__version__}")
    except ImportError:
        logger.error("SciPy not available")
    
    try:
        import sklearn
        logger.info(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        logger.error("scikit-learn not available")
    
    try:
        import sentence_transformers
        logger.info(f"sentence-transformers version: {sentence_transformers.__version__}")
    except ImportError:
        logger.error("sentence-transformers not available")

def main():
    """Main fix process"""
    logger.info("🔧 Fixing NumPy version conflicts...")
    
    # Fix conflicts
    conflicts_fixed = fix_numpy_conflicts()
    
    if conflicts_fixed:
        # Check versions
        check_versions()
        
        # Test imports
        imports_working = test_imports()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("NUMPY CONFLICT FIX SUMMARY")
        logger.info("="*50)
        logger.info(f"Conflicts fixed: {'✅' if conflicts_fixed else '❌'}")
        logger.info(f"All imports: {'✅' if imports_working else '❌'}")
        
        if imports_working:
            logger.info("\n🎉 All NumPy conflicts resolved!")
            logger.info("You can now run your AI system.")
            return True
        else:
            logger.warning("\n⚠️ Some conflicts remain.")
            return False
    else:
        logger.error("❌ Failed to fix NumPy conflicts")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 