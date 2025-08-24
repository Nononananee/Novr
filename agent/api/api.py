"""
FastAPI endpoints for the agentic RAG system.
This is now a simple import from the refactored main module.
"""

# Import the main app from the refactored module
from .main import app

# For backward compatibility, expose the app
__all__ = ["app"]