"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Generator

# Set test environment
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_novel_rag"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_PASSWORD"] = "test_password"
os.environ["LLM_API_KEY"] = "test_api_key"
os.environ["EMBEDDING_API_KEY"] = "test_embedding_key"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_database():
    """Mock database connection for testing."""
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    
    # Mock connection context manager
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    
    with patch('agent.core.database.connection.db_pool') as mock_db_pool:
        mock_db_pool.pool = mock_pool
        mock_db_pool.acquire.return_value = mock_pool.acquire.return_value
        yield mock_conn


@pytest.fixture
async def mock_graph_client():
    """Mock graph client for testing."""
    mock_client = AsyncMock()
    mock_client._initialized = True
    mock_client.search.return_value = [
        {
            "fact": "Test fact",
            "uuid": "test-uuid-123",
            "valid_at": "2024-01-01T00:00:00Z",
            "invalid_at": None,
            "source_node_uuid": "source-uuid-123"
        }
    ]
    
    with patch('agent.core.graph.operations.graph_client', mock_client):
        yield mock_client


@pytest.fixture
async def mock_llm_client():
    """Mock LLM client for testing."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM.",
                    "role": "assistant"
                }
            }
        ]
    }
    
    with patch('openai.ChatCompletion.acreate', return_value=mock_response):
        yield mock_response


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing."""
    return {
        "message": "Tell me about character development in novels",
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "search_type": "hybrid"
    }


@pytest.fixture
def sample_novel_data():
    """Sample novel data for testing."""
    return {
        "title": "Test Novel",
        "author": "Test Author",
        "genre": "fantasy",
        "summary": "A test novel for unit testing purposes."
    }


@pytest.fixture
def sample_character_data():
    """Sample character data for testing."""
    return {
        "name": "Test Hero",
        "personality_traits": ["brave", "intelligent", "compassionate"],
        "background": "A hero from humble beginnings",
        "role": "protagonist"
    }


@pytest.fixture
async def test_client():
    """Test client for API testing."""
    from fastapi.testclient import TestClient
    from agent.api.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor."""
    mock_monitor = MagicMock()
    mock_monitor.start_monitoring = AsyncMock()
    mock_monitor.stop_monitoring = AsyncMock()
    mock_monitor.get_metrics = MagicMock(return_value={
        "memory_usage_mb": 512.0,
        "cpu_usage_percent": 25.0,
        "active_connections": 5
    })
    
    with patch('agent.monitoring.advanced_system_monitor.system_monitor', mock_monitor):
        yield mock_monitor


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""
    mock_breaker = MagicMock()
    mock_breaker.call = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    mock_breaker.state = "closed"
    mock_breaker.failure_count = 0
    
    return mock_breaker


# Test data cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Cleanup test data after each test."""
    yield
    # Cleanup logic would go here
    # For now, just a placeholder since we're using mocks


# Environment-specific fixtures
@pytest.fixture
def development_env():
    """Set development environment for testing."""
    with patch.dict(os.environ, {"APP_ENV": "development"}):
        yield


@pytest.fixture
def production_env():
    """Set production environment for testing."""
    with patch.dict(os.environ, {"APP_ENV": "production"}):
        yield