"""
Configuration module untuk API yang mendukung environment-based security.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class APIConfig:
    """Configuration class untuk API settings."""
    
    # Environment
    APP_ENV = os.getenv("APP_ENV", "development")
    
    # Server settings
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    API_KEY = os.getenv("API_KEY")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL")
    GRAPH_DATABASE_URL = os.getenv("GRAPH_DATABASE_URL")
    
    # Cache settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutes default
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))  # seconds
    
    # Content generation
    MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", 500))
    MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", 4000))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 10))
    MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", 3))
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))
    RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", 2.0))
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if APP_ENV == "production" else None,
        "Content-Security-Policy": "default-src 'self'" if APP_ENV == "production" else None
    }
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode."""
        return cls.APP_ENV == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return cls.APP_ENV == "production"
    
    @classmethod
    def get_cors_config(cls) -> Dict[str, Any]:
        """Get CORS configuration based on environment."""
        if cls.is_production():
            return {
                "allow_origins": cls.ALLOWED_ORIGINS,
                "allow_credentials": False,
                "allow_methods": ["GET", "POST"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        else:
            return {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": getattr(logging, cls.LOG_LEVEL.upper()),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if cls.is_production():
            if cls.SECRET_KEY == "dev-secret-key-change-in-production":
                issues.append("SECRET_KEY must be changed in production")
            
            if not cls.API_KEY:
                issues.append("API_KEY must be set in production")
            
            if not cls.DATABASE_URL:
                issues.append("DATABASE_URL must be set")
            
            if "*" in cls.ALLOWED_ORIGINS:
                issues.append("ALLOWED_ORIGINS should not include '*' in production")
        
        return issues
