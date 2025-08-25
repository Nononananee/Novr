"""
Compatibility shim so tests can `from agent.api_config import APIConfig`.
"""
from .api.api_config import APIConfig  # re-export

__all__ = ["APIConfig"]
