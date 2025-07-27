"""
Production configuration management
"""
import os
from typing import Any, Optional
from pydantic import BaseSettings, Field
from pathlib import Path

class ProductionConfig(BaseSettings):
    """Production configuration with validation"""
    
    # Application settings
    APP_NAME: str = Field(default="Revolutionary AI Matching System", env="APP_NAME")
    VERSION: str = Field(default="2.0.0", env="VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # Database settings
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # API Keys
    DEEPSEEK_R1_API_KEY: Optional[str] = Field(default=None, env="DEEPSEEK_R1_API_KEY")
    MATERIALS_PROJECT_API_KEY: Optional[str] = Field(default=None, env="MATERIALS_PROJECT_API_KEY")
    FREIGHTOS_API_KEY: Optional[str] = Field(default=None, env="FREIGHTOS_API_KEY")
    
    # Supabase
    SUPABASE_URL: Optional[str] = Field(default=None, env="SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")
    
    # AI Configuration
    AI_MODEL_CACHE_SIZE: int = Field(default=10, env="AI_MODEL_CACHE_SIZE")
    EMBEDDING_DIMENSION: int = Field(default=512, env="EMBEDDING_DIMENSION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

def get_config() -> ProductionConfig:
    """Get production configuration instance"""
    return ProductionConfig()

# Global configuration instance
config = get_config()
