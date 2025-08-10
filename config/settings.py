"""
Configuration settings for the Nawatech FAQ Chatbot
This module handles all configuration parameters and environment variables
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Configuration settings for the chatbot application"""
    
    def __init__(self):
        """Initialize settings from environment variables"""
        
        # OpenAI Configuration
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # RAG Configuration
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
        self.SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # Memory Configuration
        self.MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "10"))
        
        # Security Configuration
        self.RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
        self.RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))
        self.MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
        
        # Application Configuration
        self.APP_PORT: int = int(os.getenv("PORT", "8501"))
        self.APP_HOST: str = os.getenv("HOST", "0.0.0.0")
        self.DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Data Configuration
        self.FAQ_FILE_PATH: str = os.getenv("FAQ_FILE_PATH", "data/FAQ_Nawa.xlsx")
        self.VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        
        # UI Configuration
        self.APP_TITLE: str = os.getenv("APP_TITLE", "Nawatech FAQ Chatbot")
        self.APP_ICON: str = os.getenv("APP_ICON", "🤖")
        self.COMPANY_NAME: str = os.getenv("COMPANY_NAME", "PT. Nawa Darsana Teknologi")
        
        # Validate critical settings
        self._validate_settings()
        
        logger.info("Settings initialized successfully")
    
    def _validate_settings(self):
        """Validate critical configuration settings"""
        validation_errors = []
        
        # Check OpenAI API Key
        if not self.OPENAI_API_KEY:
            validation_errors.append("OPENAI_API_KEY is required")
        elif not self.OPENAI_API_KEY.startswith("sk-"):
            validation_errors.append("OPENAI_API_KEY format appears invalid")
        
        # Validate numeric ranges
        if not 0.0 <= self.TEMPERATURE <= 2.0:
            validation_errors.append("TEMPERATURE must be between 0.0 and 2.0")
        
        if not 1 <= self.MAX_TOKENS <= 4000:
            validation_errors.append("MAX_TOKENS must be between 1 and 4000")
        
        if not 100 <= self.CHUNK_SIZE <= 2000:
            validation_errors.append("CHUNK_SIZE must be between 100 and 2000")
        
        if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            validation_errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if not 1 <= self.RETRIEVAL_K <= 10:
            validation_errors.append("RETRIEVAL_K must be between 1 and 10")
        
        if not 0.0 <= self.SIMILARITY_THRESHOLD <= 1.0:
            validation_errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Configuration validation error: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
        else:
            logger.info("All configuration settings validated successfully")
    
    def get_openai_config(self) -> dict:
        """Get OpenAI-specific configuration"""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL,
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
            "request_timeout": self.REQUEST_TIMEOUT
        }
    
    def get_rag_config(self) -> dict:
        """Get RAG-specific configuration"""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "retrieval_k": self.RETRIEVAL_K,
            "similarity_threshold": self.SIMILARITY_THRESHOLD
        }
    
    def get_security_config(self) -> dict:
        """Get security-specific configuration"""
        return {
            "rate_limit_per_minute": self.RATE_LIMIT_PER_MINUTE,
            "rate_limit_per_hour": self.RATE_LIMIT_PER_HOUR,
            "max_input_length": self.MAX_INPUT_LENGTH
        }
    
    def get_app_config(self) -> dict:
        """Get application-specific configuration"""
        return {
            "port": self.APP_PORT,
            "host": self.APP_HOST,
            "debug_mode": self.DEBUG_MODE,
            "log_level": self.LOG_LEVEL,
            "title": self.APP_TITLE,
            "icon": self.APP_ICON,
            "company_name": self.COMPANY_NAME
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def __str__(self) -> str:
        """String representation of settings (without sensitive information)"""
        safe_settings = {
            "OPENAI_MODEL": self.OPENAI_MODEL,
            "TEMPERATURE": self.TEMPERATURE,
            "MAX_TOKENS": self.MAX_TOKENS,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "RETRIEVAL_K": self.RETRIEVAL_K,
            "MEMORY_WINDOW": self.MEMORY_WINDOW,
            "APP_PORT": self.APP_PORT,
            "DEBUG_MODE": self.DEBUG_MODE,
            "LOG_LEVEL": self.LOG_LEVEL
        }
        return f"Settings({safe_settings})"