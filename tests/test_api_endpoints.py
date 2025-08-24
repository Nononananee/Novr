"""
Comprehensive tests untuk semua API endpoints dengan mocking dan error scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
from datetime import datetime
import json

# Import modules yang akan di-test
from agent.api import app
from agent.api_config import APIConfig
from agent.api_exceptions import (
    ValidationError,
    NotFoundError,
    ServiceUnavailableError,
    AuthenticationError
)
from agent.models import ChatRequest, SearchRequest, HealthStatus


class TestAPIEndpoints:
    """Test suite untuk API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_api_config(self):
        """Mock API configuration untuk testing."""
        with patch.object(APIConfig, 'is_production', return_value=False), \
             patch.object(APIConfig, 'is_development', return_value=True), \
             patch.object(APIConfig, 'ENABLE_CACHING', True):
            yield
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager."""
        with patch('agent.api.cache_manager') as mock:
            mock.get.return_value = None  # Cache miss
            mock.set.return_value = True
            mock.get_stats.return_value = {
                "type": "memory",
                "keys": 10,
                "hits": 50,
                "misses": 10
            }
            yield mock
    
    @pytest.fixture
    def mock_health_checker(self):
        """Mock health checker."""
        with patch('agent.api.HealthChecker') as mock:
            mock.get_comprehensive_health.return_value = {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": datetime.now()
            }
            yield mock
    
    @pytest.fixture
    def mock_conversation_manager(self):
        """Mock conversation manager."""
        with patch('agent.api.ConversationManager') as mock:
            mock.get_or_create_session.return_value = "test-session-123"
            yield mock
    
    @pytest.fixture
    def mock_agent_executor(self):
        """Mock agent executor."""
        with patch('agent.api.AgentExecutor') as mock:
            mock.execute_agent.return_value = (
                "Test response from agent",
                []  # No tools used
            )
            yield mock


class TestHealthEndpoint(TestAPIEndpoints):
    """Tests untuk health endpoint."""
    
    def test_health_check_success(self, client, mock_api_config, mock_health_checker):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] is True
        assert data["graph_database"] is True
        assert data["version"] == "1.0.0"
    
    def test_health_check_degraded(self, client, mock_api_config):
        """Test health check dengan degraded status."""
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.return_value = {
                "status": "degraded",
                "database": True,
                "graph_database": False,
                "llm_connection": True,
                "timestamp": datetime.now()
            }
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "degraded"
    
    def test_health_check_failure(self, client, mock_api_config):
        """Test health check failure."""
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.side_effect = Exception("Database connection failed")
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestChatEndpoint(TestAPIEndpoints):
    """Tests untuk chat endpoints."""
    
    def test_chat_success(self, client, mock_api_config, mock_cache_manager, 
                         mock_conversation_manager, mock_agent_executor):
        """Test successful chat request."""
        with patch('agent.api.RequestValidator') as mock_validator:
            mock_validator.validate_chat_request.return_value = {
                "message": "Hello, how are you?",
                "user_id": "test-user",
                "session_id": None
            }
            
            request_data = {
                "message": "Hello, how are you?",
                "user_id": "test-user"
            }
            
            response = client.post("/chat", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "message" in data
            assert "session_id" in data
            assert data["session_id"] == "test-session-123"
    
    def test_chat_validation_error(self, client, mock_api_config):
        """Test chat dengan validation error."""
        with patch('agent.api.RequestValidator') as mock_validator:
            mock_validator.validate_chat_request.side_effect = ValidationError(
                "Message too short", "message"
            )
            
            request_data = {
                "message": "",
                "user_id": "test-user"
            }
            
            response = client.post("/chat", json=request_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"
    
    def test_chat_with_cache_hit(self, client, mock_api_config, mock_conversation_manager):
        """Test chat dengan cache hit."""
        cached_response = {
            "message": "Cached response",
            "tools_used": [],
            "session_id": "test-session-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        with patch('agent.api.cached_operation') as mock_cached, \
             patch('agent.api.RequestValidator') as mock_validator:
            
            mock_validator.validate_chat_request.return_value = {
                "message": "Hello",
                "user_id": "test-user"
            }
            mock_cached.return_value = cached_response
            
            request_data = {
                "message": "Hello",
                "user_id": "test-user"
            }
            
            response = client.post("/chat", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Cached response"
    
    def test_chat_service_unavailable(self, client, mock_api_config, mock_conversation_manager):
        """Test chat dengan service unavailable."""
        with patch('agent.api.RequestValidator') as mock_validator, \
             patch('agent.api.AgentExecutor') as mock_executor:
            
            mock_validator.validate_chat_request.return_value = {
                "message": "Hello",
                "user_id": "test-user"
            }
            mock_executor.execute_agent.side_effect = Exception("Agent service down")
            
            request_data = {
                "message": "Hello",
                "user_id": "test-user"
            }
            
            response = client.post("/chat", json=request_data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestSearchEndpoints(TestAPIEndpoints):
    """Tests untuk search endpoints."""
    
    def test_vector_search_success(self, client, mock_api_config, mock_cache_manager):
        """Test successful vector search."""
        with patch('agent.api.RequestValidator') as mock_validator, \
             patch('agent.api.SearchOperations') as mock_search:
            
            mock_validator.validate_search_request.return_value = {
                "query": "test query",
                "limit": 10
            }
            mock_search.execute_vector_search.return_value = (
                [{"content": "result 1"}, {"content": "result 2"}],
                150.5  # query time
            )
            
            request_data = {
                "query": "test query",
                "limit": 10
            }
            
            response = client.post("/search/vector", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["search_type"] == "vector"
            assert data["total_results"] == 2
            assert data["query_time_ms"] == 150.5
    
    def test_graph_search_success(self, client, mock_api_config, mock_cache_manager):
        """Test successful graph search."""
        with patch('agent.api.RequestValidator') as mock_validator, \
             patch('agent.api.SearchOperations') as mock_search:
            
            mock_validator.validate_search_request.return_value = {
                "query": "test graph query",
                "limit": 10
            }
            mock_search.execute_graph_search.return_value = (
                [{"node": "entity1", "relationship": "related_to"}],
                200.0
            )
            
            request_data = {
                "query": "test graph query"
            }
            
            response = client.post("/search/graph", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["search_type"] == "graph"
            assert "graph_results" in data
    
    def test_hybrid_search_with_cache(self, client, mock_api_config):
        """Test hybrid search dengan cache hit."""
        cached_result = {
            "results": [{"content": "cached result"}],
            "total_results": 1,
            "search_type": "hybrid",
            "query_time_ms": 100.0
        }
        
        with patch('agent.api.cached_operation') as mock_cached, \
             patch('agent.api.RequestValidator') as mock_validator:
            
            mock_validator.validate_search_request.return_value = {
                "query": "cached query",
                "limit": 5
            }
            mock_cached.return_value = cached_result
            
            request_data = {
                "query": "cached query",
                "limit": 5
            }
            
            response = client.post("/search/hybrid", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["results"][0]["content"] == "cached result"
    
    def test_search_validation_error(self, client, mock_api_config):
        """Test search dengan validation error."""
        with patch('agent.api.RequestValidator') as mock_validator:
            mock_validator.validate_search_request.side_effect = ValidationError(
                "Query too long", "query"
            )
            
            request_data = {
                "query": "x" * 2000,  # Very long query
                "limit": 10
            }
            
            response = client.post("/search/vector", json=request_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestAuthenticationAndSecurity(TestAPIEndpoints):
    """Tests untuk authentication dan security."""
    
    def test_production_requires_api_key(self, client):
        """Test bahwa production requires API key."""
        with patch.object(APIConfig, 'is_production', return_value=True), \
             patch.object(APIConfig, 'API_KEY', 'secret-key'):
            
            # Request tanpa API key
            response = client.get("/health")
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            
            # Request dengan wrong API key
            headers = {"Authorization": "Bearer wrong-key"}
            response = client.get("/health", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            
            # Request dengan correct API key
            headers = {"Authorization": "Bearer secret-key"}
            with patch('agent.api.HealthChecker') as mock_health:
                mock_health.get_comprehensive_health.return_value = {
                    "status": "healthy",
                    "database": True,
                    "graph_database": True,
                    "llm_connection": True,
                    "timestamp": datetime.now()
                }
                
                response = client.get("/health", headers=headers)
                assert response.status_code == status.HTTP_200_OK
    
    def test_development_skips_authentication(self, client, mock_api_config):
        """Test bahwa development mode skip authentication."""
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.return_value = {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": datetime.now()
            }
            
            # Request tanpa headers should work in development
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
    
    def test_security_headers_added(self, client, mock_api_config):
        """Test bahwa security headers ditambahkan."""
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.return_value = {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": datetime.now()
            }
            
            response = client.get("/health")
            
            # Check security headers
            assert "X-Content-Type-Options" in response.headers
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert "X-Frame-Options" in response.headers
            assert response.headers["X-Frame-Options"] == "DENY"


class TestCacheEndpoints(TestAPIEndpoints):
    """Tests untuk cache management endpoints."""
    
    def test_cache_stats(self, client, mock_api_config, mock_cache_manager):
        """Test cache stats endpoint."""
        response = client.get("/cache/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert "cache_stats" in data
        assert data["cache_stats"]["type"] == "memory"
    
    def test_cache_clear(self, client, mock_api_config, mock_cache_manager):
        """Test cache clear endpoint."""
        response = client.post("/cache/clear")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        
        # Test dengan pattern
        response = client.post("/cache/clear?pattern=chat:*")
        assert response.status_code == status.HTTP_200_OK
    
    def test_cache_error_handling(self, client, mock_api_config):
        """Test error handling pada cache endpoints."""
        with patch('agent.api.cache_manager') as mock_cache:
            mock_cache.get_stats.side_effect = Exception("Cache error")
            
            response = client.get("/cache/stats")
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestErrorHandling(TestAPIEndpoints):
    """Tests untuk error handling scenarios."""
    
    def test_api_exception_handler(self, client, mock_api_config):
        """Test custom API exception handling."""
        # Simulasi ValidationError
        with patch('agent.api.RequestValidator') as mock_validator:
            mock_validator.validate_chat_request.side_effect = ValidationError(
                "Invalid input", "message", {"additional": "info"}
            )
            
            response = client.post("/chat", json={"message": ""})
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"
            assert data["error_type"] == "ValidationError"
            assert "request_id" in data
            assert "timestamp" in data
    
    def test_global_exception_handler(self, client, mock_api_config):
        """Test global exception handler."""
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.side_effect = Exception("Unexpected error")
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert data["error_code"] == "INTERNAL_ERROR"
    
    def test_production_error_masking(self, client):
        """Test bahwa production mode mask internal errors."""
        with patch.object(APIConfig, 'is_production', return_value=True), \
             patch('agent.api.HealthChecker') as mock_health:
            
            mock_health.get_comprehensive_health.side_effect = Exception("Internal database error")
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            # Error message should be generic di production
            assert data["error"] == "An internal error occurred"


class TestPerformanceAndResilience(TestAPIEndpoints):
    """Tests untuk performance dan resilience features."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_api_config):
        """Test concurrent request handling."""
        with TestClient(app) as client:
            # Simulasi concurrent requests
            tasks = []
            
            with patch('agent.api.HealthChecker') as mock_health:
                mock_health.get_comprehensive_health.return_value = {
                    "status": "healthy",
                    "database": True,
                    "graph_database": True,
                    "llm_connection": True,
                    "timestamp": datetime.now()
                }
                
                async def make_request():
                    return client.get("/health")
                
                # Create 10 concurrent requests
                for _ in range(10):
                    tasks.append(make_request())
                
                responses = await asyncio.gather(*tasks)
                
                # Semua requests should succeed
                for response in responses:
                    assert response.status_code == status.HTTP_200_OK
    
    def test_retry_mechanism_integration(self, client, mock_api_config):
        """Test retry mechanism dengan failed then success scenario."""
        call_count = 0
        
        def mock_health_with_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": datetime.now()
            }
        
        with patch('agent.api.HealthChecker') as mock_health:
            mock_health.get_comprehensive_health.side_effect = mock_health_with_retry
            
            response = client.get("/health")
            
            # Should succeed after retry
            assert response.status_code == status.HTTP_200_OK
            assert call_count >= 1  # At least one call was made


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
