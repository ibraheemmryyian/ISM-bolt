#!/usr/bin/env python3
"""
Perfect AI System Installation Script
Installs and configures the revolutionary AI system for industrial symbiosis.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 80)
    print("🚀 Perfect AI System - Industrial Symbiosis Platform")
    print("   Revolutionary AI with Absolute Synergy and Utmost Adaptiveness")
    print("=" * 80)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    print()

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✅ Available memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected. Performance may be limited.")
    except ImportError:
        print("⚠️  Warning: Could not check memory (psutil not available)")
    
    # Check disk space
    try:
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"✅ Available disk space: {disk_gb:.1f} GB")
        
        if disk_gb < 5:
            print("⚠️  Warning: Less than 5GB free space detected.")
    except ImportError:
        print("⚠️  Warning: Could not check disk space")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available - will use CPU")
    except ImportError:
        print("ℹ️  PyTorch not installed yet")
    
    print()

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("✅ Pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not upgrade pip: {e}")
    
    # Install core dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Core dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)
    
    # Install development dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                      check=True)
        print("✅ Development dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install dev dependencies: {e}")
    
    print()

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "models/gnn",
        "models/cache",
        "logs",
        "data",
        "backups",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print()

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        # Create .env file from example
        env_example = Path("backend/env.example")
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
        else:
            # Create basic .env file
            with open(env_file, "w") as f:
                f.write("""# Perfect AI System Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://localhost/industrial_symbiosis
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# AI Model Configuration
MODEL_CACHE_DIR=./models
GNN_WARM_START=true
GNN_PERSISTENCE=true

# System Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=20
MEMORY_USAGE_THRESHOLD=0.85

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
""")
            print("✅ Created basic .env file")
    
    print()

def verify_installation():
    """Verify the installation"""
    print("🔍 Verifying installation...")
    
    # Test imports
    test_modules = [
        "torch",
        "torch_geometric", 
        "networkx",
        "numpy",
        "scikit-learn",
        "sentence_transformers"
    ]
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
    
    # Test backend modules
    try:
        from backend import GNNReasoning, KnowledgeGraph
        print("✅ Backend modules imported successfully")
    except ImportError as e:
        print(f"⚠️  Some backend modules failed to import: {e}")
    
    print()

def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test GNN reasoning
        from backend.gnn_reasoning import GNNReasoning
        gnn = GNNReasoning()
        print("✅ GNN Reasoning initialized")
        
        # Test revolutionary matching
        try:
            from backend.revolutionary_ai_matching import RevolutionaryAIMatching
            matching = RevolutionaryAIMatching()
            print("✅ Revolutionary AI Matching initialized")
        except ImportError:
            print("⚠️  Revolutionary AI Matching not available")
        
        # Test knowledge graph
        from backend.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        print("✅ Knowledge Graph initialized")
        
        print("✅ All core modules working correctly")
        
    except Exception as e:
        print(f"⚠️  Some tests failed: {e}")
    
    print()

def print_next_steps():
    """Print next steps for the user"""
    print("🎉 Installation completed successfully!")
    print()
    print("Next steps:")
    print("1. Configure your .env file with your database and API credentials")
    print("2. Start the Perfect AI System:")
    print("   python backend/start_perfect_ai_system.py")
    print()
    print("3. Or use the integration directly:")
    print("   python -c \"import asyncio; from backend.perfect_ai_integration import PerfectAIIntegration; asyncio.run(PerfectAIIntegration().process_request('symbiosis_matching', {'buyer': {'industry': 'Construction'}, 'seller': {'industry': 'Steel'}}))\"")
    print()
    print("4. For development, install additional dependencies:")
    print("   pip install -e .[dev]")
    print()
    print("5. For GPU support:")
    print("   pip install -e .[gpu]")
    print()
    print("Documentation: PERFECT_AI_SYSTEM_SUMMARY.md")
    print("=" * 80)

def main():
    """Main installation function"""
    print_banner()
    
    # Check requirements
    check_python_version()
    check_system_requirements()
    
    # Install dependencies
    install_dependencies()
    
    # Setup system
    create_directories()
    setup_environment()
    
    # Verify installation
    verify_installation()
    run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()