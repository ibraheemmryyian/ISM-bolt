#!/usr/bin/env python3
"""
Comprehensive Dependency Fixer for ISM AI System
Resolves all missing packages and import issues
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyFixer:
    """Comprehensive dependency fixer for ISM AI system"""
    
    def __init__(self):
        self.required_packages = {
            # Core scientific computing
            'numpy': '1.24.3',
            'pandas': '2.0.3',
            'scipy': '1.11.1',
            'scikit-learn': '1.3.0',
            
            # Deep learning and AI
            'torch': '2.0.1',
            'torchvision': '0.15.2',
            'transformers': '4.35.0',
            'sentence-transformers': '2.2.2',
            'tokenizers': '0.21.2',
            
            # Graph processing
            'networkx': '3.1',
            'torch-geometric': '2.3.1',
            
            # Web framework
            'flask': '3.0.0',
            'flask-cors': '4.0.0',
            'requests': '2.31.0',
            
            # Database
            'supabase': '1.0.3',
            'psycopg2-binary': '2.9.7',
            
            # Utilities
            'python-dotenv': '1.0.0',
            'tqdm': '4.65.0',
            'click': '8.1.7',
            'rich': '13.5.2',
            
            # Async
            'aiohttp': '3.8.5',
            'asyncio': '3.4.3',
            
            # Monitoring
            'prometheus-client': '0.17.1',
            'structlog': '23.1.0',
            
            # Additional ML
            'xgboost': '1.7.6',
            'lightgbm': '4.0.0',
            'optuna': '3.4.0',
            
            # NLP
            'nltk': '3.8.1',
            'textblob': '0.17.1',
            
            # Visualization
            'matplotlib': '3.7.2',
            'seaborn': '0.12.2',
            'plotly': '5.17.0',
            
            # Security
            'cryptography': '41.0.4',
            'bcrypt': '4.0.1',
            'PyJWT': '2.8.0',
            
            # Testing
            'pytest': '7.4.2',
            'pytest-asyncio': '0.21.1',
            
            # Development
            'black': '23.7.0',
            'flake8': '6.0.0',
            
            # Additional utilities
            'python-multipart': '0.0.6',
            'pydantic': '2.0.0',
            'typing-extensions': '4.7.0',
            'python-dateutil': '2.8.2',
            'openpyxl': '3.1.2',
            'xlrd': '2.0.1',
            
            # Time series
            'statsmodels': '0.14.0',
            'prophet': '1.1.4',
            
            # Financial
            'yfinance': '0.2.18',
            'newsapi-python': '0.2.6',
            
            # Redis
            'redis': '4.5.4',
            
            # Monitoring
            'sentry-sdk': '1.38.0',
            
            # API frameworks
            'fastapi': '0.100.0',
            'uvicorn': '0.23.0',
            
            # Additional ML
            'catboost': '1.2.0',
            'imbalanced-learn': '0.11.0',
            
            # Visualization
            'bokeh': '3.2.0',
            
            # Optimization
            'pulp': '2.7.0',
            'cvxpy': '1.3.2',
            
            # Geospatial
            'geopy': '2.3.0',
            'folium': '0.14.0',
            
            # Additional NLP
            'spacy': '3.7.2',
            'gensim': '4.3.1',
            
            # Financial analysis
            'alpha-vantage': '2.3.1',
            
            # Additional utilities
            'python-decouple': '3.8',
            'python-json-logger': '2.0.7'
        }
        
        self.optional_packages = {
            # Graph processing (optional)
            'igraph': '0.10.6',
            'python-igraph': '0.10.6',
            
            # Time series analysis (optional)
            'arch': '6.2.0',
            'pykalman': '0.9.5',
            
            # Materials science (optional)
            'rdkit-pypi': '2023.3.1',
            'pubchempy': '1.0.4',
            'chemspipy': '2.1.0',
            
            # Experiment tracking (optional)
            'wandb': '0.12.0'
        }
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    
    def upgrade_pip(self):
        """Upgrade pip to latest version"""
        logger.info("Upgrading pip...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'
            ])
            logger.info("✅ Pip upgraded successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to upgrade pip: {e}")
    
    def install_package(self, package_name: str, version: str = None, user_install: bool = True) -> bool:
        """Install a single package with error handling"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install']
            
            if user_install:
                cmd.append('--user')
            
            if version:
                cmd.append(f'{package_name}=={version}')
            else:
                cmd.append(package_name)
            
            # Add additional flags for problematic packages
            if package_name in ['torch', 'torchvision', 'torch-geometric']:
                cmd.extend(['--index-url', 'https://download.pytorch.org/whl/cpu'])
            
            if package_name in ['sentence-transformers', 'transformers']:
                cmd.extend(['--only-binary=all'])
            
            logger.info(f"Installing {package_name}{'==' + version if version else ''}...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ {package_name} installed successfully")
                return True
            else:
                logger.warning(f"⚠️ Failed to install {package_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Timeout installing {package_name}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Error installing {package_name}: {e}")
            return False
    
    def check_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed"""
        try:
            __import__(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False
    
    def install_core_packages(self):
        """Install core packages first"""
        logger.info("Installing core packages...")
        
        core_packages = [
            'numpy', 'pandas', 'scipy', 'scikit-learn',
            'flask', 'requests', 'python-dotenv'
        ]
        
        success_count = 0
        for package in core_packages:
            if not self.check_package_installed(package):
                if self.install_package(package, self.required_packages.get(package)):
                    success_count += 1
            else:
                logger.info(f"✅ {package} already installed")
                success_count += 1
        
        logger.info(f"Core packages: {success_count}/{len(core_packages)} installed")
        return success_count == len(core_packages)
    
    def install_ai_packages(self):
        """Install AI and ML packages"""
        logger.info("Installing AI and ML packages...")
        
        ai_packages = [
            'torch', 'torchvision', 'transformers', 'sentence-transformers',
            'networkx', 'xgboost', 'lightgbm', 'optuna'
        ]
        
        success_count = 0
        for package in ai_packages:
            if not self.check_package_installed(package):
                if self.install_package(package, self.required_packages.get(package)):
                    success_count += 1
            else:
                logger.info(f"✅ {package} already installed")
                success_count += 1
        
        logger.info(f"AI packages: {success_count}/{len(ai_packages)} installed")
        return success_count == len(ai_packages)
    
    def install_database_packages(self):
        """Install database packages"""
        logger.info("Installing database packages...")
        
        db_packages = ['supabase', 'psycopg2-binary']
        
        success_count = 0
        for package in db_packages:
            if not self.check_package_installed(package):
                if self.install_package(package, self.required_packages.get(package)):
                    success_count += 1
            else:
                logger.info(f"✅ {package} already installed")
                success_count += 1
        
        logger.info(f"Database packages: {success_count}/{len(db_packages)} installed")
        return success_count == len(db_packages)
    
    def install_utility_packages(self):
        """Install utility packages"""
        logger.info("Installing utility packages...")
        
        utility_packages = [
            'tqdm', 'click', 'rich', 'aiohttp', 'prometheus-client',
            'structlog', 'nltk', 'textblob', 'matplotlib', 'seaborn',
            'plotly', 'cryptography', 'bcrypt', 'PyJWT', 'pytest',
            'pytest-asyncio', 'black', 'flake8', 'python-multipart',
            'pydantic', 'typing-extensions', 'python-dateutil',
            'openpyxl', 'xlrd', 'statsmodels', 'prophet', 'yfinance',
            'newsapi-python', 'redis', 'sentry-sdk', 'fastapi',
            'uvicorn', 'catboost', 'imbalanced-learn', 'bokeh',
            'pulp', 'cvxpy', 'geopy', 'folium', 'spacy', 'gensim',
            'alpha-vantage', 'python-decouple', 'python-json-logger'
        ]
        
        success_count = 0
        for package in utility_packages:
            if not self.check_package_installed(package):
                if self.install_package(package, self.required_packages.get(package)):
                    success_count += 1
            else:
                logger.info(f"✅ {package} already installed")
                success_count += 1
        
        logger.info(f"Utility packages: {success_count}/{len(utility_packages)} installed")
        return success_count == len(utility_packages)
    
    def install_optional_packages(self):
        """Install optional packages"""
        logger.info("Installing optional packages...")
        
        success_count = 0
        for package, version in self.optional_packages.items():
            if not self.check_package_installed(package):
                if self.install_package(package, version):
                    success_count += 1
            else:
                logger.info(f"✅ {package} already installed")
                success_count += 1
        
        logger.info(f"Optional packages: {success_count}/{len(self.optional_packages)} installed")
        return success_count == len(self.optional_packages)
    
    def create_requirements_file(self):
        """Create updated requirements.txt file"""
        logger.info("Creating updated requirements.txt...")
        
        requirements_content = "# ISM AI System Requirements\n"
        requirements_content += "# Generated by dependency fixer\n\n"
        
        for package, version in self.required_packages.items():
            requirements_content += f"{package}=={version}\n"
        
        requirements_content += "\n# Optional packages\n"
        for package, version in self.optional_packages.items():
            requirements_content += f"# {package}=={version}\n"
        
        with open('requirements_updated.txt', 'w') as f:
            f.write(requirements_content)
        
        logger.info("✅ Created requirements_updated.txt")
    
    def test_imports(self):
        """Test critical imports"""
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
    
    def fix_supabase_import(self):
        """Fix Supabase import issues"""
        logger.info("Fixing Supabase import issues...")
        
        try:
            # Try to install the correct Supabase package
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user', '--upgrade', 'supabase'
            ])
            
            # Test the import
            from supabase import create_client, Client
            logger.info("✅ Supabase import fixed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Supabase import fix failed: {e}")
            return False
    
    def run(self):
        """Run the complete dependency fix process"""
        logger.info("🚀 Starting comprehensive dependency fix...")
        
        try:
            # Check Python version
            self.check_python_version()
            
            # Upgrade pip
            self.upgrade_pip()
            
            # Install packages in order
            core_success = self.install_core_packages()
            ai_success = self.install_ai_packages()
            db_success = self.install_database_packages()
            utility_success = self.install_utility_packages()
            optional_success = self.install_optional_packages()
            
            # Fix specific import issues
            self.fix_supabase_import()
            
            # Create updated requirements file
            self.create_requirements_file()
            
            # Test imports
            import_success = self.test_imports()
            
            # Summary
            logger.info("\n" + "="*50)
            logger.info("DEPENDENCY FIX SUMMARY")
            logger.info("="*50)
            logger.info(f"Core packages: {'✅' if core_success else '❌'}")
            logger.info(f"AI packages: {'✅' if ai_success else '❌'}")
            logger.info(f"Database packages: {'✅' if db_success else '❌'}")
            logger.info(f"Utility packages: {'✅' if utility_success else '❌'}")
            logger.info(f"Optional packages: {'✅' if optional_success else '❌'}")
            logger.info(f"Import test: {'✅' if import_success else '❌'}")
            
            if all([core_success, ai_success, db_success, utility_success, import_success]):
                logger.info("\n🎉 All dependencies installed successfully!")
                logger.info("You can now run the AI production system.")
                return True
            else:
                logger.warning("\n⚠️ Some dependencies failed to install.")
                logger.warning("The system may have limited functionality.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Dependency fix failed: {e}")
            return False

def main():
    """Main entry point"""
    fixer = DependencyFixer()
    success = fixer.run()
    
    if success:
        print("\n✅ Dependency fix completed successfully!")
        print("You can now run: python backend/start_production_ai_system.py")
    else:
        print("\n❌ Dependency fix completed with warnings.")
        print("Some packages may need manual installation.")

if __name__ == "__main__":
    main() 