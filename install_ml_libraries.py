#!/usr/bin/env python3
"""
Comprehensive ML Library Installation Script
Installs all required ML libraries for the Perfect AI System
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLLibraryInstaller:
    """Installer for ML libraries"""
    
    def __init__(self):
        self.install_log = []
        self.failed_installs = []
        
    def run_command(self, command: str, description: str) -> bool:
        """Run a pip install command"""
        logger.info(f"Installing {description}...")
        
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"✓ {description} installed successfully")
            self.install_log.append(f"✓ {description}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {description}: {e}")
            logger.error(f"Error output: {e.stderr}")
            self.failed_installs.append(description)
            self.install_log.append(f"✗ {description} - {e}")
            return False
    
    def install_core_libraries(self):
        """Install core ML libraries"""
        logger.info("Installing core ML libraries...")
        
        core_libraries = [
            ("pip install numpy>=1.24.0", "NumPy"),
            ("pip install pandas>=2.0.0", "Pandas"),
            ("pip install scipy>=1.10.0", "SciPy"),
            ("pip install scikit-learn>=1.3.0", "Scikit-learn"),
            ("pip install joblib>=1.3.0", "Joblib"),
        ]
        
        for command, description in core_libraries:
            self.run_command(command, description)
    
    def install_deep_learning(self):
        """Install deep learning libraries"""
        logger.info("Installing deep learning libraries...")
        
        dl_libraries = [
            ("pip install torch>=2.0.0", "PyTorch"),
            ("pip install torchvision>=0.15.0", "TorchVision"),
            ("pip install torchaudio>=2.0.0", "TorchAudio"),
        ]
        
        for command, description in dl_libraries:
            self.run_command(command, description)
    
    def install_graph_neural_networks(self):
        """Install graph neural network libraries"""
        logger.info("Installing graph neural network libraries...")
        
        gnn_libraries = [
            ("pip install torch-geometric>=2.3.0", "PyTorch Geometric"),
            ("pip install torch-scatter>=2.1.0", "Torch Scatter"),
            ("pip install torch-sparse>=0.6.0", "Torch Sparse"),
            ("pip install torch-cluster>=1.6.0", "Torch Cluster"),
            ("pip install torch-spline-conv>=1.2.0", "Torch Spline Conv"),
            ("pip install networkx>=3.0", "NetworkX"),
        ]
        
        for command, description in gnn_libraries:
            self.run_command(command, description)
    
    def install_boosting_libraries(self):
        """Install boosting libraries"""
        logger.info("Installing boosting libraries...")
        
        boosting_libraries = [
            ("pip install xgboost>=1.7.6", "XGBoost"),
            ("pip install lightgbm>=4.0.0", "LightGBM"),
            ("pip install catboost>=1.2.0", "CatBoost"),
        ]
        
        for command, description in boosting_libraries:
            self.run_command(command, description)
    
    def install_nlp_libraries(self):
        """Install NLP libraries"""
        logger.info("Installing NLP libraries...")
        
        nlp_libraries = [
            ("pip install sentence-transformers>=2.2.0", "Sentence Transformers"),
            ("pip install transformers>=4.30.0", "Transformers"),
            ("pip install tokenizers>=0.13.0", "Tokenizers"),
            ("pip install nltk>=3.8.1", "NLTK"),
            ("pip install spacy>=3.6.0", "SpaCy"),
            ("pip install textblob>=0.17.1", "TextBlob"),
            ("pip install gensim>=4.3.0", "Gensim"),
        ]
        
        for command, description in nlp_libraries:
            self.run_command(command, description)
    
    def install_optimization_libraries(self):
        """Install optimization libraries"""
        logger.info("Installing optimization libraries...")
        
        opt_libraries = [
            ("pip install optuna>=3.2.0", "Optuna"),
            ("pip install hyperopt>=0.2.7", "Hyperopt"),
            ("pip install scikit-optimize>=0.9.0", "Scikit-optimize"),
            ("pip install cvxpy>=1.3.0", "CVXPY"),
            ("pip install pulp>=2.7.0", "PuLP"),
        ]
        
        for command, description in opt_libraries:
            self.run_command(command, description)
    
    def install_clustering_libraries(self):
        """Install clustering libraries"""
        logger.info("Installing clustering libraries...")
        
        clustering_libraries = [
            ("pip install hdbscan>=0.8.29", "HDBSCAN"),
            ("pip install umap-learn>=0.5.3", "UMAP"),
            ("pip install faiss-cpu>=1.7.4", "FAISS"),
        ]
        
        for command, description in clustering_libraries:
            self.run_command(command, description)
    
    def install_interpretability_libraries(self):
        """Install interpretability libraries"""
        logger.info("Installing interpretability libraries...")
        
        interpretability_libraries = [
            ("pip install shap>=0.42.0", "SHAP"),
            ("pip install lime>=0.2.0", "LIME"),
            ("pip install interpret>=0.4.0", "Interpret"),
            ("pip install eli5>=0.13.0", "ELI5"),
            ("pip install alibi>=0.8.0", "Alibi"),
        ]
        
        for command, description in interpretability_libraries:
            self.run_command(command, description)
    
    def install_visualization_libraries(self):
        """Install visualization libraries"""
        logger.info("Installing visualization libraries...")
        
        viz_libraries = [
            ("pip install matplotlib>=3.7.2", "Matplotlib"),
            ("pip install seaborn>=0.12.2", "Seaborn"),
            ("pip install plotly>=5.15.0", "Plotly"),
            ("pip install bokeh>=3.2.0", "Bokeh"),
            ("pip install altair>=5.1.0", "Altair"),
        ]
        
        for command, description in viz_libraries:
            self.run_command(command, description)
    
    def install_development_libraries(self):
        """Install development libraries"""
        logger.info("Installing development libraries...")
        
        dev_libraries = [
            ("pip install pytest>=7.4.0", "Pytest"),
            ("pip install pytest-asyncio>=0.21.0", "Pytest-asyncio"),
            ("pip install pytest-cov>=4.1.0", "Pytest-cov"),
            ("pip install black>=23.7.0", "Black"),
            ("pip install flake8>=6.0.0", "Flake8"),
            ("pip install mypy>=1.5.0", "MyPy"),
            ("pip install isort>=5.12.0", "isort"),
        ]
        
        for command, description in dev_libraries:
            self.run_command(command, description)
    
    def install_all_from_requirements(self):
        """Install all libraries from requirements.txt"""
        logger.info("Installing all libraries from requirements.txt...")
        
        if os.path.exists("requirements.txt"):
            success = self.run_command("pip install -r requirements.txt", "All requirements")
            if success:
                logger.info("✓ All requirements installed successfully")
            else:
                logger.warning("⚠ Some requirements failed to install")
        else:
            logger.error("✗ requirements.txt not found")
    
    def verify_installations(self):
        """Verify that key libraries are installed"""
        logger.info("Verifying installations...")
        
        key_libraries = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("torch", "PyTorch"),
            ("sklearn", "Scikit-learn"),
            ("networkx", "NetworkX"),
            ("xgboost", "XGBoost"),
            ("lightgbm", "LightGBM"),
            ("sentence_transformers", "Sentence Transformers"),
            ("transformers", "Transformers"),
        ]
        
        verification_results = []
        
        for import_name, display_name in key_libraries:
            try:
                __import__(import_name)
                logger.info(f"✓ {display_name} verified")
                verification_results.append(f"✓ {display_name}")
            except ImportError:
                logger.error(f"✗ {display_name} not found")
                verification_results.append(f"✗ {display_name}")
        
        return verification_results
    
    def generate_report(self):
        """Generate installation report"""
        logger.info("\n" + "="*60)
        logger.info("ML LIBRARY INSTALLATION REPORT")
        logger.info("="*60)
        
        logger.info(f"\nInstallation Log:")
        for entry in self.install_log:
            logger.info(f"  {entry}")
        
        if self.failed_installs:
            logger.info(f"\nFailed Installations:")
            for failed in self.failed_installs:
                logger.info(f"  ✗ {failed}")
        
        logger.info("="*60)
    
    def install_all(self):
        """Install all ML libraries"""
        logger.info("Starting comprehensive ML library installation...")
        
        # Install from requirements.txt first
        self.install_all_from_requirements()
        
        # If that fails, install individually
        if self.failed_installs:
            logger.info("Some installations failed, trying individual installations...")
            
            self.install_core_libraries()
            self.install_deep_learning()
            self.install_graph_neural_networks()
            self.install_boosting_libraries()
            self.install_nlp_libraries()
            self.install_optimization_libraries()
            self.install_clustering_libraries()
            self.install_interpretability_libraries()
            self.install_visualization_libraries()
            self.install_development_libraries()
        
        # Verify installations
        verification_results = self.verify_installations()
        
        # Generate report
        self.generate_report()
        
        return len(self.failed_installs) == 0

def main():
    """Main installation function"""
    installer = MLLibraryInstaller()
    success = installer.install_all()
    
    if success:
        logger.info("🎉 All ML libraries installed successfully!")
        return 0
    else:
        logger.warning("⚠ Some libraries failed to install. Check the report above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)