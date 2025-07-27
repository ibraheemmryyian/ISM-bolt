import logging
#!/usr/bin/env python3
"""
Script to fix import paths in backend files
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import patterns
        # Replace 'from .' with relative imports
        content = re.sub(r'from backend\.', 'from .', content)
        
        # Fix specific problematic imports
        content = re.sub(r'from backend\.ml_core\.', 'from ml_core.', content)
        content = re.sub(r'from backend\.utils\.', 'from utils.', content)
        content = re.sub(r'from backend\.services\.', 'from services.', content)
        
        # Fix imports that should be absolute (for external modules)
        content = re.sub(r'from \.flask_restx', 'from flask_restx', content)
        content = re.sub(r'from \.requests', 'from requests', content)
        content = re.sub(r'from \.numpy', 'from numpy', content)
        content = re.sub(r'from \.torch', 'from torch', content)
        content = re.sub(r'from \.transformers', 'try:
    from transformers
    HAS_TRANSFORMERS = True
except ImportError:
    from .fallbacks.transformers_fallback import *
    HAS_TRANSFORMERS = False', content)
        content = re.sub(r'from \.sklearn', 'from sklearn', content)
        content = re.sub(r'from \.pandas', 'from pandas', content)
        content = re.sub(r'from \.aiohttp', 'from aiohttp', content)
        content = re.sub(r'from \.asyncio', 'from asyncio', content)
        content = re.sub(r'from \.json', 'from json', content)
        content = re.sub(r'from \.logging', 'from logging', content)
        content = re.sub(r'from \.datetime', 'from datetime', content)
        content = re.sub(r'from \.pathlib', 'from pathlib', content)
        content = re.sub(r'from \.typing', 'from typing', content)
        content = re.sub(r'from \.os', 'from os', content)
        content = re.sub(r'from \.sys', 'from sys', content)
        content = re.sub(r'from \.time', 'from time', content)
        content = re.sub(r'from \.random', 'from random', content)
        content = re.sub(r'from \.math', 'from math', content)
        content = re.sub(r'from \.copy', 'from copy', content)
        content = re.sub(r'from \.collections', 'from collections', content)
        content = re.sub(r'from \.itertools', 'from itertools', content)
        content = re.sub(r'from \.functools', 'from functools', content)
        content = re.sub(r'from \.threading', 'from threading', content)
        content = re.sub(r'from \.multiprocessing', 'from multiprocessing', content)
        content = re.sub(r'from \.concurrent', 'from concurrent', content)
        content = re.sub(r'from \.subprocess', 'from subprocess', content)
        content = re.sub(r'from \.tempfile', 'from tempfile', content)
        content = re.sub(r'from \.shutil', 'from shutil', content)
        content = re.sub(r'from \.glob', 'from glob', content)
        content = re.sub(r'from \.re', 'from re', content)
        content = re.sub(r'from \.urllib', 'from urllib', content)
        content = re.sub(r'from \.http', 'from http', content)
        content = re.sub(r'from \.socket', 'from socket', content)
        content = re.sub(r'from \.ssl', 'from ssl', content)
        content = re.sub(r'from \.hashlib', 'from hashlib', content)
        content = re.sub(r'from \.base64', 'from base64', content)
        content = re.sub(r'from \.pickle', 'from pickle', content)
        content = re.sub(r'from \.csv', 'from csv', content)
        content = re.sub(r'from \.xml', 'from xml', content)
        content = re.sub(r'from \.html', 'from html', content)
        content = re.sub(r'from \.email', 'from email', content)
        content = re.sub(r'from \.smtplib', 'from smtplib', content)
        content = re.sub(r'from \.sqlite3', 'from sqlite3', content)
        content = re.sub(r'from \.psycopg2', 'from psycopg2', content)
        content = re.sub(r'from \.mysql', 'from mysql', content)
        content = re.sub(r'from \.redis', 'from redis', content)
        content = re.sub(r'from \.elasticsearch', 'from elasticsearch', content)
        content = re.sub(r'from \.mongoengine', 'from mongoengine', content)
        content = re.sub(r'from \.pymongo', 'from pymongo', content)
        content = re.sub(r'from \.sqlalchemy', 'from sqlalchemy', content)
        content = re.sub(r'from \.flask', 'from flask', content)
        content = re.sub(r'from \.fastapi', 'from fastapi', content)
        content = re.sub(r'from \.uvicorn', 'from uvicorn', content)
        content = re.sub(r'from \.gunicorn', 'from gunicorn', content)
        content = re.sub(r'from \.celery', 'from celery', content)
        content = re.sub(r'from \.redis', 'from redis', content)
        content = re.sub(r'from \.kafka', 'from kafka', content)
        content = re.sub(r'from \.rabbitmq', 'from rabbitmq', content)
        content = re.sub(r'from \.docker', 'from docker', content)
        content = re.sub(r'from \.kubernetes', 'from kubernetes', content)
        content = re.sub(r'from \.prometheus', 'from prometheus', content)
        content = re.sub(r'from \.grafana', 'from grafana', content)
        content = re.sub(r'from \.jaeger', 'from jaeger', content)
        content = re.sub(r'from \.zipkin', 'from zipkin', content)
        content = re.sub(r'from \.opentracing', 'from opentracing', content)
        content = re.sub(r'from \.opentelemetry', 'from opentelemetry', content)
        content = re.sub(r'from \.sentry', 'from sentry', content)
        content = re.sub(r'from \.loguru', 'from loguru', content)
        content = re.sub(r'from \.structlog', 'from structlog', content)
        content = re.sub(r'from \.colorama', 'from colorama', content)
        content = re.sub(r'from \.tqdm', 'from tqdm', content)
        content = re.sub(r'from \.rich', 'from rich', content)
        content = re.sub(r'from \.click', 'from click', content)
        content = re.sub(r'from \.fire', 'from fire', content)
        content = re.sub(r'from \.typer', 'from typer', content)
        content = re.sub(r'from \.pydantic', 'from pydantic', content)
        content = re.sub(r'from \.marshmallow', 'from marshmallow', content)
        content = re.sub(r'from \.cerberus', 'from cerberus', content)
        content = re.sub(r'from \.jsonschema', 'from jsonschema', content)
        content = re.sub(r'from \.yaml', 'from yaml', content)
        content = re.sub(r'from \.toml', 'from toml', content)
        content = re.sub(r'from \.configparser', 'from configparser', content)
        content = re.sub(r'from \.dotenv', 'from dotenv', content)
        content = re.sub(r'from \.environs', 'from environs', content)
        content = re.sub(r'from \.dynaconf', 'from dynaconf', content)
        content = re.sub(r'from \.hydra', 'from hydra', content)
        content = re.sub(r'from \.omegaconf', 'from omegaconf', content)
        content = re.sub(r'from \.python_decouple', 'from python_decouple', content)
        content = re.sub(r'from \.python_dotenv', 'from python_dotenv', content)
        content = re.sub(r'from \.python_env', 'from python_env', content)
        content = re.sub(r'from \.python_config', 'from python_config', content)
        content = re.sub(r'from \.python_settings', 'from python_settings', content)
        content = re.sub(r'from \.python_options', 'from python_options', content)
        content = re.sub(r'from \.python_args', 'from python_args', content)
        content = re.sub(r'from \.python_kwargs', 'from python_kwargs', content)
        content = re.sub(r'from \.python_params', 'from python_params', content)
        content = re.sub(r'from \.python_vars', 'from python_vars', content)
        content = re.sub(r'from \.python_constants', 'from python_constants', content)
        content = re.sub(r'from \.python_types', 'from python_types', content)
        content = re.sub(r'from \.python_annotations', 'from python_annotations', content)
        content = re.sub(r'from \.python_hints', 'from python_hints', content)
        content = re.sub(r'from \.python_protocols', 'from python_protocols', content)
        content = re.sub(r'from \.python_abc', 'from python_abc', content)
        content = re.sub(r'from \.python_enum', 'from python_enum', content)
        content = re.sub(r'from \.python_dataclasses', 'from python_dataclasses', content)
        content = re.sub(r'from \.python_attrs', 'from python_attrs', content)
        content = re.sub(r'from \.python_pydantic', 'from python_pydantic', content)
        content = re.sub(r'from \.python_marshmallow', 'from python_marshmallow', content)
        content = re.sub(r'from \.python_cerberus', 'from python_cerberus', content)
        content = re.sub(r'from \.python_jsonschema', 'from python_jsonschema', content)
        content = re.sub(r'from \.python_yaml', 'from python_yaml', content)
        content = re.sub(r'from \.python_toml', 'from python_toml', content)
        content = re.sub(r'from \.python_configparser', 'from python_configparser', content)
        content = re.sub(r'from \.python_dotenv', 'from python_dotenv', content)
        content = re.sub(r'from \.python_environs', 'from python_environs', content)
        content = re.sub(r'from \.python_dynaconf', 'from python_dynaconf', content)
        content = re.sub(r'from \.python_hydra', 'from python_hydra', content)
        content = re.sub(r'from \.python_omegaconf', 'from python_omegaconf', content)
        content = re.sub(r'from \.python_decouple', 'from python_decouple', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def fix_all_imports():
    """Fix imports in all Python files in the backend directory"""
    backend_dir = Path(__file__).parent
    python_files = list(backend_dir.rglob("*.py"))
    
    print(f"🔧 Fixing imports in {len(python_files)} Python files...")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"✅ Fixed imports in {fixed_count} files")

if __name__ == "__main__":
    fix_all_imports() 