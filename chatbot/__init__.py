# chatbot/__init__.py
"""
Nawatech FAQ Chatbot - Core chatbot modules
This package contains the main components for the FAQ chatbot system.
"""

__version__ = "1.0.0"
__author__ = "Nawatech Development Team"

from .data_processor import DataProcessor
from .llm_handler import LLMHandler
from .security import SecurityManager
from .quality_scorer import QualityScorer

__all__ = [
    "DataProcessor",
    "LLMHandler", 
    "SecurityManager",
    "QualityScorer"
]

# ===================================

# config/__init__.py
"""
Configuration management for Nawatech FAQ Chatbot
This package handles all configuration settings and environment management.
"""

from .settings import Settings

__all__ = ["Settings"]