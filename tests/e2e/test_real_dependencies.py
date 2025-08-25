"""End-to-end tests with real dependencies (when available)."""

import pytest
import os
from fastapi.testclient import TestClient

from agent.api.main import app


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("RUN_E2E_TESTS"), 
    reason="E2E tests require RUN_E2E_TESTS=true and real dependencies"
)
class TestRealDependencyIntegration:
    """Test with real database, graph DB, and LLM connections."""
    
    def test_complete_novel_workflow_real(self):
        """Test complete novel workflow with real dependencies."""
        with TestClient(app) as client:
            # Test health check first
            health_response = client.get("/health")
            if health_response.status_code != 200:
                pytest.skip("System not healthy for E2E testing")
            
            health_data = health_response.json()
            if not health_data.get("database") or not health_data.get("llm_connection"):
                pytest.skip("Required dependencies not available")
            
            # Create novel
            novel_response = client.post("/novels", params={
                "title": "E2E Test Novel",
                "author": "Test Author",
                "genre": "test",
                "summary": "A novel created during end-to-end testing"
            })
            
            assert novel_response.status_code == 200
            novel_data = novel_response.json()
            novel_id = novel_data["novel_id"]
            
            try:
                # Create character
                character_response = client.post(f"/novels/{novel_id}/characters", params={
                    "name": "E2E Test Hero",
                    "role": "protagonist",
                    "background": "A hero created for testing"
                })
                
                assert character_response.status_code == 200
                
                # Create chapter
                chapter_response = client.post(f"/novels/{novel_id}/chapters", params={
                    "chapter_number": 1,
                    "title": "Test Chapter",
                    "summary": "A chapter for testing"
                })
                
                assert chapter_response.status_code == 200
                
                # Test chat functionality
                chat_response = client.post("/chat", json={
                    "message": "Tell me about the novel creation process",
                    "search_type": "hybrid"
                })
                
                assert chat_response.status_code == 200
                chat_data = chat_response.json()
                assert len(chat_data["message"]) > 0
                
                # Test search functionality
                search_response = client.post("/search/vector", json={
                    "query": "novel creation",
                    "limit": 5
                })
                
                assert search_response.status_code == 200
                search_data = search_response.json()
                assert "results" in search_data
                
            finally:
                # Cleanup: In a real scenario, you'd clean up test data
                pass
    
    def test_real_llm_integration(self):
        """Test actual LLM integration if available."""
        with TestClient(app) as client:
            # Check if LLM is available
            health_response = client.get("/health")
            if health_response.status_code != 200:
                pytest.skip("System not healthy")
            
            health_data = health_response.json()
            if not health_data.get("llm_connection"):
                pytest.skip("LLM not available")
            
            # Test chat with real LLM
            response = client.post("/chat", json={
                "message": "What are the key elements of character development in novels?",
                "search_type": "hybrid"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Real LLM should provide meaningful response
            assert len(data["message"]) > 50  # Should be substantial
            assert "character" in data["message"].lower()
            assert "development" in data["message"].lower()
    
    def test_real_database_persistence(self):
        """Test actual database persistence."""
        with TestClient(app) as client:
            # Check database availability
            health_response = client.get("/health")
            if health_response.status_code != 200:
                pytest.skip("System not healthy")
            
            health_data = health_response.json()
            if not health_data.get("database"):
                pytest.skip("Database not available")
            
            # Create novel
            create_response = client.post("/novels", params={
                "title": "Persistence Test Novel",
                "author": "Test Author"
            })
            
            assert create_response.status_code == 200
            novel_id = create_response.json()["novel_id"]
            
            # Retrieve novel
            get_response = client.get(f"/novels/{novel_id}")
            
            assert get_response.status_code == 200
            novel_data = get_response.json()
            assert novel_data["title"] == "Persistence Test Novel"
            assert novel_data["author"] == "Test Author"
    
    def test_real_graph_database_integration(self):
        """Test actual graph database integration if available."""
        with TestClient(app) as client:
            # Check graph database availability
            health_response = client.get("/health")
            if health_response.status_code != 200:
                pytest.skip("System not healthy")
            
            health_data = health_response.json()
            if not health_data.get("graph_database"):
                pytest.skip("Graph database not available")
            
            # Test graph search
            response = client.post("/search/graph", json={
                "query": "character relationships and interactions",
                "limit": 5
            })
            
            # Should handle graph search (even if no results)
            assert response.status_code == 200
            data = response.json()
            assert "search_type" in data
            assert data["search_type"] == "graph"


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("RUN_PERFORMANCE_E2E"), 
    reason="Performance E2E tests require RUN_PERFORMANCE_E2E=true"
)
class TestRealPerformanceScenarios:
    """Performance testing with real dependencies."""
    
    def test_real_concurrent_load(self):
        """Test concurrent load with real dependencies."""
        import concurrent.futures
        import time
        
        def make_real_request(request_id):
            start_time = time.time()
            with TestClient(app) as client:
                response = client.post("/chat", json={
                    "message": f"What is the importance of plot structure in novel {request_id}?",
                    "search_type": "hybrid"
                })
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "response_length": len(response.json().get("message", "")) if response.status_code == 200 else 0
                }
        
        # Run 10 concurrent requests with real LLM
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_real_request, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        
        if len(successful_requests) > 0:
            import statistics
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            
            # Real LLM responses will be slower but should still be reasonable
            assert avg_response_time < 30.0, f"Real LLM responses too slow: {avg_response_time}s"
            
            # Responses should be substantial
            response_lengths = [r["response_length"] for r in successful_requests]
            avg_length = statistics.mean(response_lengths)
            assert avg_length > 20, "Real LLM responses too short"
    
    def test_real_system_stability(self):
        """Test system stability over extended period."""
        with TestClient(app) as client:
            # Run requests over 5 minutes
            start_time = time.time()
            request_count = 0
            errors = 0
            
            while time.time() - start_time < 300:  # 5 minutes
                try:
                    response = client.get("/health")
                    if response.status_code != 200:
                        errors += 1
                    request_count += 1
                    
                    # Make occasional chat requests
                    if request_count % 10 == 0:
                        chat_response = client.post("/chat", json={
                            "message": f"System stability test request {request_count}",
                            "search_type": "hybrid"
                        })
                        if chat_response.status_code != 200:
                            errors += 1
                    
                    time.sleep(1)  # 1 second between requests
                    
                except Exception as e:
                    errors += 1
            
            # System should remain stable
            error_rate = errors / request_count if request_count > 0 else 1
            assert error_rate < 0.05, f"Error rate too high: {error_rate}"
            assert request_count > 250, "Not enough requests completed"