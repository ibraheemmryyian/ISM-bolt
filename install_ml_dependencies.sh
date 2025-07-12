#!/bin/bash

# Perfect AI System - ML Dependencies Installation Script
# This script installs all required ML libraries and dependencies

set -e  # Exit on any error

echo "🚀 Installing Perfect AI System ML Dependencies..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📦 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install numpy pandas scipy

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and dependencies
echo "🔗 Installing PyTorch Geometric..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install scikit-learn and related libraries
echo "🤖 Installing scikit-learn ecosystem..."
pip install scikit-learn

# Install advanced ML libraries
echo "🧠 Installing advanced ML libraries..."
pip install xgboost lightgbm catboost

# Install clustering and dimensionality reduction
echo "🔍 Installing clustering libraries..."
pip install hdbscan umap-learn

# Install optimization libraries
echo "⚡ Installing optimization libraries..."
pip install optuna hyperopt scikit-optimize

# Install NLP libraries
echo "📝 Installing NLP libraries..."
pip install sentence-transformers transformers tokenizers nltk spacy textblob gensim

# Install computer vision libraries
echo "👁️  Installing computer vision libraries..."
pip install opencv-python pillow albumentations

# Install time series libraries
echo "⏰ Installing time series libraries..."
pip install prophet statsmodels arch pyflux

# Install feature engineering libraries
echo "🔧 Installing feature engineering libraries..."
pip install feature-engine boruta

# Install model interpretability libraries
echo "🔍 Installing interpretability libraries..."
pip install shap lime interpret eli5 alibi

# Install federated learning libraries
echo "🌐 Installing federated learning libraries..."
pip install fedml syft flwr

# Install AutoML libraries
echo "🤖 Installing AutoML libraries..."
pip install autosklearn autogluon

# Install database and caching libraries
echo "🗄️  Installing database libraries..."
pip install redis psycopg2-binary sqlalchemy pymongo cassandra-driver

# Install async and concurrency libraries
echo "⚡ Installing async libraries..."
pip install asyncio-mqtt aiohttp websockets aiofiles asyncio-throttle

# Install system monitoring libraries
echo "📊 Installing monitoring libraries..."
pip install psutil prometheus-client grafana-api jaeger-client

# Install configuration and logging libraries
echo "📝 Installing logging libraries..."
pip install python-dotenv structlog colorlog loguru

# Install API and web framework libraries
echo "🌐 Installing web framework libraries..."
pip install fastapi uvicorn pydantic starlette

# Install testing and development libraries
echo "🧪 Installing testing libraries..."
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark

# Install code quality libraries
echo "✨ Installing code quality libraries..."
pip install black flake8 mypy isort bandit

# Install utility libraries
echo "🛠️  Installing utility libraries..."
pip install click rich tqdm typer pydantic-settings

# Install data validation libraries
echo "✅ Installing validation libraries..."
pip install marshmallow cerberus jsonschema

# Install security libraries
echo "🔒 Installing security libraries..."
pip install cryptography bcrypt PyJWT passlib

# Install file processing libraries
echo "📁 Installing file processing libraries..."
pip install pyyaml toml h5py joblib pickle5

# Install visualization libraries
echo "📊 Installing visualization libraries..."
pip install matplotlib seaborn plotly bokeh altair

# Install Jupyter libraries
echo "📓 Installing Jupyter libraries..."
pip install jupyter ipykernel notebook jupyterlab

# Install performance and profiling libraries
echo "⚡ Installing profiling libraries..."
pip install memory-profiler line-profiler py-spy

# Install cloud and deployment libraries
echo "☁️  Installing cloud libraries..."
pip install boto3 google-cloud-storage azure-storage-blob docker kubernetes

# Install monitoring and alerting libraries
echo "🚨 Installing monitoring libraries..."
pip install sentry-sdk datadog newrelic

# Install development tools
echo "🛠️  Installing development tools..."
pip install pre-commit tox coverage

# Install additional ML libraries
echo "🧠 Installing additional ML libraries..."
pip install tsne-cuda faiss-cpu

# Install mathematical optimization libraries
echo "📐 Installing optimization libraries..."
pip install cvxpy pulp

# Install genetic algorithm libraries
echo "🧬 Installing genetic algorithm libraries..."
pip install deap

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install networkx igraph python-igraph dgl spektral

# Verify installations
echo "✅ Verifying installations..."
python3 -c "
import sys
import importlib

required_packages = [
    'torch', 'numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm', 'catboost',
    'sentence_transformers', 'transformers', 'networkx', 'scipy', 'optuna',
    'shap', 'lime', 'umap', 'hdbscan', 'fastapi', 'uvicorn', 'pydantic'
]

print('Verifying package installations...')
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f'✅ {package}')
    except ImportError as e:
        print(f'❌ {package}: {e}')

print('\\n🎉 Installation verification complete!')
"

# Create model directories
echo "📁 Creating model directories..."
mkdir -p models/ml_factory
mkdir -p models/real_ai
mkdir -p models/gnn
mkdir -p models/ensemble
mkdir -p models/clustering
mkdir -p models/anomaly

# Set up environment variables
echo "🔧 Setting up environment variables..."
cat > .env << EOF
# Perfect AI System Environment Variables
AI_MODEL_CACHE_DIR=./models
AI_LOG_LEVEL=INFO
AI_DEVICE=cpu
AI_BATCH_SIZE=32
AI_MAX_WORKERS=4
AI_CACHE_ENABLED=true
AI_MONITORING_ENABLED=true
AI_OPTIMIZATION_ENABLED=true
AI_PERSISTENT_MODELS=true
AI_WARM_START_ENABLED=true
AI_ADAPTIVE_LEARNING=true
AI_CROSS_MODULE_SYNERGY=true
AI_PERFORMANCE_TRACKING=true
AI_HEALTH_MONITORING=true
AI_ERROR_RECOVERY=true
AI_MEMORY_MANAGEMENT=true
AI_RESOURCE_OPTIMIZATION=true
AI_BACKGROUND_SERVICES=true
AI_GNN_WARM_START=true
AI_ORCHESTRATOR_SYNERGY=true
EOF

echo "✅ Environment variables configured"

# Create a test script
echo "🧪 Creating test script..."
cat > test_ml_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify ML installation
"""

import sys
import importlib
from typing import List, Dict

def test_imports() -> Dict[str, bool]:
    """Test all required imports"""
    packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'sentence_transformers': 'Sentence Transformers',
        'transformers': 'Transformers',
        'networkx': 'NetworkX',
        'scipy': 'SciPy',
        'optuna': 'Optuna',
        'shap': 'SHAP',
        'lime': 'LIME',
        'umap': 'UMAP',
        'hdbscan': 'HDBSCAN',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pydantic': 'Pydantic'
    }
    
    results = {}
    for package, name in packages.items():
        try:
            importlib.import_module(package)
            results[name] = True
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            results[name] = False
            print(f"❌ {name} import failed: {e}")
    
    return results

def test_ml_functionality():
    """Test basic ML functionality"""
    print("\n🧪 Testing ML functionality...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make prediction
        prediction = model.predict(X_test[:1])
        
        print(f"✅ ML functionality test passed - Prediction: {prediction[0]:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ ML functionality test failed: {e}")
        return False

def test_deep_learning():
    """Test deep learning functionality"""
    print("\n🧠 Testing deep learning functionality...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create simple neural network
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        
        print(f"✅ Deep learning test passed - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Deep learning test failed: {e}")
        return False

def test_nlp():
    """Test NLP functionality"""
    print("\n📝 Testing NLP functionality...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode sentences
        sentences = ["This is a test sentence.", "This is another test sentence."]
        embeddings = model.encode(sentences)
        
        print(f"✅ NLP test passed - Embeddings shape: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ NLP test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Perfect AI System - ML Installation Test")
    print("=" * 50)
    
    # Test imports
    import_results = test_imports()
    
    # Test functionality
    ml_test = test_ml_functionality()
    dl_test = test_deep_learning()
    nlp_test = test_nlp()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Successful imports: {sum(import_results.values())}/{len(import_results)}")
    print(f"✅ ML functionality: {'PASS' if ml_test else 'FAIL'}")
    print(f"✅ Deep learning: {'PASS' if dl_test else 'FAIL'}")
    print(f"✅ NLP functionality: {'PASS' if nlp_test else 'FAIL'}")
    
    if all(import_results.values()) and ml_test and dl_test and nlp_test:
        print("\n🎉 All tests passed! ML installation is successful.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

echo "✅ Test script created"

# Make scripts executable
chmod +x test_ml_installation.py

echo ""
echo "🎉 Perfect AI System ML Dependencies Installation Complete!"
echo "=========================================================="
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the test script: python test_ml_installation.py"
echo "3. Start the AI system: python startup_script.py"
echo ""
echo "📁 Model directories created:"
echo "   - models/ml_factory/"
echo "   - models/real_ai/"
echo "   - models/gnn/"
echo "   - models/ensemble/"
echo "   - models/clustering/"
echo "   - models/anomaly/"
echo ""
echo "🔧 Environment variables configured in .env"
echo ""
echo "🚀 Ready to run the Perfect AI System!"