"""Unit tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from agent.api.main import app


@pytest.mark.unit
@pytest.mark.api
class TestChatEndpoints:
    """Test chat-related endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        with TestClient(app) as client:
            with patch('agent.api.main.HealthChecker.get_comprehensive_health') as mock_health:
                mock_health.return_value = {
                    "status": "healthy",
                    "database": True,
                    "graph_database": True,
                    "llm_connection": True,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
                response = client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["database"] is True
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    @patch('agent.api.endpoints.chat.cached_operation')
    def test_chat_endpoint_success(self, mock_cached_op, mock_execute, mock_session, mock_validate):
        """Test successful chat endpoint."""
        # Setup mocks
        mock_validate.return_value = {
            "message": "Test message",
            "user_id": "test-user"
        }
        mock_session.return_value = "test-session-123"
        mock_execute.return_value = ("Test response", [])
        mock_cached_op.return_value = {
            "message": "Test response",
            "tools_used": [],
            "session_id": "test-session-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": "Test message",
                "search_type": "hybrid"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Test response"
            assert data["session_id"] == "test-session-123"
    
    def test_chat_endpoint_validation_error(self):
        """Test chat endpoint with validation error."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                mock_validate.side_effect = ValueError("Invalid request")
                
                response = client.post("/chat", json={
                    "message": "",  # Empty message should cause validation error
                    "search_type": "hybrid"
                })
                
                # Should handle validation error gracefully
                assert response.status_code in [400, 422, 500]


@pytest.mark.unit
@pytest.mark.api
class TestSearchEndpoints:
    """Test search-related endpoints."""
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_vector_search_endpoint(self, mock_cached_op, mock_search, mock_validate):
        """Test vector search endpoint."""
        # Setup mocks
        mock_validate.return_value = {
            "query": "test query",
            "limit": 10
        }
        mock_search.return_value = ([{"content": "test result"}], 150.5)
        mock_cached_op.return_value = {
            "results": [{"content": "test result"}],
            "total_results": 1,
            "search_type": "vector",
            "query_time_ms": 150.5
        }
        
        with TestClient(app) as client:
            response = client.post("/search/vector", json={
                "query": "test query",
                "limit": 10
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["search_type"] == "vector"
            assert len(data["results"]) == 1
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_graph_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_graph_search_endpoint(self, mock_cached_op, mock_search, mock_validate):
        """Test graph search endpoint."""
        # Setup mocks
        mock_validate.return_value = {
            "query": "character relationships",
            "limit": 5
        }
        mock_search.return_value = ([{"fact": "test fact"}], 200.0)
        mock_cached_op.return_value = {
            "graph_results": [{"fact": "test fact"}],
            "total_results": 1,
            "search_type": "graph",
            "query_time_ms": 200.0
        }
        
        with TestClient(app) as client:
            response = client.post("/search/graph", json={
                "query": "character relationships",
                "limit": 5
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["search_type"] == "graph"
            assert "graph_results" in data


@pytest.mark.unit
@pytest.mark.api
class TestNovelEndpoints:
    """Test novel management endpoints."""
    
    @patch('agent.core.database.novels.create_novel_tables')
    @patch('agent.core.database.novels.create_novel')
    def test_create_novel_endpoint(self, mock_create, mock_tables):
        """Test novel creation endpoint."""
        mock_tables.return_value = None
        mock_create.return_value = "novel-123"
        
        with TestClient(app) as client:
            response = client.post("/novels", params={
                "title": "Test Novel",
                "author": "Test Author",
                "genre": "fantasy"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["novel_id"] == "novel-123"
            assert data["title"] == "Test Novel"
            assert data["status"] == "created"
    
    @patch('agent.core.database.novels.list_novels')
    def test_list_novels_endpoint(self, mock_list):
        """Test novel listing endpoint."""
        mock_novels = [
            {
                "id": "novel-1",
                "title": "Novel 1",
                "author": "Author 1",
                "genre": "fantasy",
                "completion_status": "in_progress",
                "chapter_count": 5,
                "total_word_count": 10000,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ]
        mock_list.return_value = mock_novels
        
        with TestClient(app) as client:
            response = client.get("/novels")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["novels"]) == 1
            assert data["novels"][0]["id"] == "novel-1"
    
    @patch('agent.core.database.novels.create_character')
    def test_create_character_endpoint(self, mock_create):
        """Test character creation endpoint."""
        mock_create.return_value = "character-123"
        
        with TestClient(app) as client:
            response = client.post("/novels/novel-123/characters", params={
                "name": "Test Hero",
                "role": "protagonist",
                "background": "A brave hero"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["character_id"] == "character-123"
            assert data["name"] == "Test Hero"
            assert data["status"] == "created"


@pytest.mark.unit
@pytest.mark.api
class TestSystemEndpoints:
    """Test system monitoring endpoints."""
    
    @patch('agent.monitoring.advanced_system_monitor.get_system_health')
    def test_system_health_endpoint(self, mock_health):
        """Test system health endpoint."""
        mock_health.return_value = {
            "overall_health": "healthy",
            "monitoring_active": True,
            "components": {
                "database": {"status": "healthy"},
                "graph_database": {"status": "healthy"}
            }
        }
        
        with TestClient(app) as client:
            response = client.get("/system/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "health" in data
    
    @patch('agent.api.circuit_breaker.get_all_circuit_status')
    @patch('agent.api.circuit_breaker.get_circuit_breaker_stats')
    def test_circuit_breaker_status_endpoint(self, mock_stats, mock_status):
        """Test circuit breaker status endpoint."""
        mock_status.return_value = {
            "test_circuit": {
                "state": "closed",
                "failure_count": 0,
                "last_failure": None
            }
        }
        mock_stats.return_value = {
            "total_circuits": 1,
            "open_circuits": 0,
            "closed_circuits": 1
        }
        
        with TestClient(app) as client:
            response = client.get("/system/circuit-breakers")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "circuit_breakers" in data