#!/usr/bin/env python3
"""
Setup script for Perfect AI System - Industrial Symbiosis Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="perfect-ai-system",
    version="1.0.0",
    author="Perfect AI System Team",
    author_email="team@perfectaisystem.com",
    description="Revolutionary AI System for Industrial Symbiosis with Absolute Synergy and Utmost Adaptiveness",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/perfect-ai-system/industrial-symbiosis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",  # CUDA 11.8
            "torch-geometric>=2.3.0",
        ],
        "production": [
            "gunicorn>=21.0.0",
            "uvicorn>=0.23.0",
            "redis>=4.5.0",
            "psycopg2-binary>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "perfect-ai-system=backend.start_perfect_ai_system:main",
            "ai-orchestrator=backend.advanced_ai_orchestrator:main",
            "gnn-reasoning=backend.gnn_reasoning:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "graph neural networks",
        "industrial symbiosis",
        "sustainability",
        "circular economy",
        "ai orchestration",
        "gnn",
        "deep learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/perfect-ai-system/industrial-symbiosis/issues",
        "Source": "https://github.com/perfect-ai-system/industrial-symbiosis",
        "Documentation": "https://perfect-ai-system.readthedocs.io/",
    },
)