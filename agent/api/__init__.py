"""API-related modules.

Expose FastAPI app and commonly used utilities for tests that import `agent.api` directly.
"""

# Re-export app so `from agent.api import app` works in tests
try:
    from .main import app
except Exception:  # During import-time errors, provide a placeholder to avoid hard failure in patch-heavy tests
    app = None  # type: ignore

# Expose commonly patched classes for convenience
from .api_utils import (
    ConversationManager,
    AgentExecutor,
    SearchOperations,
    RequestValidator,
    HealthChecker,
)
from .api_cache import cache_manager, cached_operation