"""Load testing for API endpoints."""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import statistics

from agent.api.main import app


@pytest.mark.performance
class TestAPILoadTesting:
    """Load testing for critical API endpoints."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    @patch('agent.api.endpoints.chat.cached_operation')
    def test_concurrent_chat_requests(self, mock_cache, mock_execute, mock_session, mock_validate):
        """Test concurrent chat requests performance."""
        # Setup mocks for fast responses
        mock_validate.return_value = {"message": "test", "user_id": "test-user"}
        mock_session.return_value = "session-123"
        mock_execute.return_value = ("Test response", [])
        mock_cache.return_value = {
            "message": "Test response",
            "tools_used": [],
            "session_id": "session-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        def make_request(request_id):
            start_time = time.time()
            try:
                with TestClient(app) as client:
                    response = client.post("/chat", json={
                        "message": f"Test message {request_id}",
                        "search_type": "hybrid"
                    })
                    end_time = time.time()
                    return {
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time,
                        "success": response.status_code == 200
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": 500,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Run concurrent requests
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        # Assertions
        success_rate = len(successful_requests) / len(results)
        assert success_rate >= 0.95, f"Success rate too low: {success_rate}"
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
            
            assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time}s"
            assert p95_response_time < 5.0, f"P95 response time too high: {p95_response_time}s"
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_search_endpoint_performance(self, mock_cache, mock_search, mock_validate):
        """Test search endpoint performance under load."""
        # Setup mocks
        mock_validate.return_value = {"query": "test query", "limit": 10}
        mock_search.return_value = ([{"content": "test result"}], 100.0)
        mock_cache.return_value = {
            "results": [{"content": "test result"}],
            "total_results": 1,
            "search_type": "vector",
            "query_time_ms": 100.0
        }
        
        response_times = []
        
        with TestClient(app) as client:
            for i in range(20):
                start_time = time.time()
                response = client.post("/search/vector", json={
                    "query": f"test query {i}",
                    "limit": 10
                })
                end_time = time.time()
                
                assert response.status_code == 200
                response_times.append(end_time - start_time)
        
        # Performance assertions
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        assert avg_time < 1.0, f"Average search time too high: {avg_time}s"
        assert max_time < 3.0, f"Max search time too high: {max_time}s"
    
    def test_health_endpoint_performance(self):
        """Test health endpoint performance."""
        with patch('agent.api.main.HealthChecker.get_comprehensive_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            response_times = []
            
            with TestClient(app) as client:
                for _ in range(100):  # 100 rapid health checks
                    start_time = time.time()
                    response = client.get("/health")
                    end_time = time.time()
                    
                    assert response.status_code == 200
                    response_times.append(end_time - start_time)
            
            # Health checks should be very fast
            avg_time = statistics.mean(response_times)
            p95_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
            
            assert avg_time < 0.1, f"Health check too slow: {avg_time}s"
            assert p95_time < 0.5, f"P95 health check too slow: {p95_time}s"


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage under load."""
    
    def test_memory_usage_during_load(self):
        """Test memory usage doesn't grow excessively during load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
            with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                    with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                        # Setup mocks
                        mock_validate.return_value = {"message": "test", "user_id": "test-user"}
                        mock_session.return_value = "session-123"
                        mock_execute.return_value = ("Test response", [])
                        mock_cache.return_value = {
                            "message": "Test response",
                            "tools_used": [],
                            "session_id": "session-123",
                            "metadata": {"search_type": "hybrid"}
                        }
                        
                        # Make many requests
                        with TestClient(app) as client:
                            for i in range(100):
                                response = client.post("/chat", json={
                                    "message": f"Test message {i}",
                                    "search_type": "hybrid"
                                })
                                assert response.status_code == 200
                                
                                # Check memory every 20 requests
                                if i % 20 == 0:
                                    current_memory = process.memory_info().rss / 1024 / 1024
                                    memory_growth = current_memory - initial_memory
                                    
                                    # Memory growth should be reasonable (less than 100MB)
                                    assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Total memory growth should be reasonable
        assert total_growth < 200, f"Total memory growth too high: {total_growth}MB"


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database operation performance."""
    
    @patch('agent.core.database.connection.db_pool.acquire')
    def test_database_connection_pool_performance(self, mock_acquire):
        """Test database connection pool performance."""
        mock_conn = AsyncMock()
        mock_acquire.return_value.__aenter__.return_value = mock_conn
        mock_acquire.return_value.__aexit__.return_value = None
        
        # Simulate fast database responses
        mock_conn.fetchrow.return_value = {"id": "test-123"}
        mock_conn.fetch.return_value = [{"id": "doc-1"}, {"id": "doc-2"}]
        
        from agent.core.database.sessions import create_session, get_session
        from agent.core.database.documents import list_documents
        
        async def run_db_operations():
            start_time = time.time()
            
            # Run multiple database operations
            tasks = []
            for i in range(20):
                tasks.extend([
                    create_session(user_id=f"user-{i}"),
                    get_session(f"session-{i}"),
                    list_documents(limit=10, offset=i*10)
                ])
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            return end_time - start_time
        
        # Run the test
        total_time = asyncio.run(run_db_operations())
        
        # Should complete all operations quickly
        assert total_time < 5.0, f"Database operations too slow: {total_time}s"