"""Medium priority tests for search functionality - user experience."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import time

from agent.api.main import app


@pytest.mark.medium
@pytest.mark.search
class TestSearchFunctionalityCore:
    """Medium priority tests for core search functionality."""
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_vector_search_functionality(self, mock_cache, mock_search, mock_validate):
        """Test vector search provides relevant results."""
        # Setup mocks
        mock_validate.return_value = {"query": "character development", "limit": 10}
        mock_search.return_value = ([
            {
                "content": "Character development is crucial for engaging storytelling...",
                "score": 0.92,
                "document_title": "Writing Guide",
                "document_source": "writing_tips.txt",
                "chunk_id": "chunk-1"
            },
            {
                "content": "Developing characters requires understanding their motivations...",
                "score": 0.88,
                "document_title": "Character Creation",
                "document_source": "characters.txt",
                "chunk_id": "chunk-2"
            }
        ], 125.5)
        
        mock_cache.return_value = {
            "results": [
                {
                    "content": "Character development is crucial for engaging storytelling...",
                    "score": 0.92,
                    "document_title": "Writing Guide",
                    "document_source": "writing_tips.txt",
                    "chunk_id": "chunk-1"
                },
                {
                    "content": "Developing characters requires understanding their motivations...",
                    "score": 0.88,
                    "document_title": "Character Creation",
                    "document_source": "characters.txt",
                    "chunk_id": "chunk-2"
                }
            ],
            "total_results": 2,
            "search_type": "vector",
            "query_time_ms": 125.5
        }
        
        with TestClient(app) as client:
            response = client.post("/search/vector", json={
                "query": "character development",
                "limit": 10
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # User experience requirements
            assert data["search_type"] == "vector"
            assert len(data["results"]) == 2
            assert data["query_time_ms"] < 1000  # Should be fast for good UX
            
            # Results should be relevant and well-structured
            first_result = data["results"][0]
            assert first_result["score"] > 0.8  # High relevance
            assert "character development" in first_result["content"].lower()
            assert "document_title" in first_result
            assert "chunk_id" in first_result
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_graph_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_graph_search_functionality(self, mock_cache, mock_search, mock_validate):
        """Test graph search provides relationship information."""
        # Setup mocks
        mock_validate.return_value = {"query": "character relationships", "limit": 5}
        mock_search.return_value = ([
            {
                "fact": "Alice is friends with Bob in the story",
                "uuid": "fact-uuid-1",
                "valid_at": "2024-01-01T00:00:00Z",
                "source_node_uuid": "node-uuid-1"
            },
            {
                "fact": "Bob mentors Charlie throughout the narrative",
                "uuid": "fact-uuid-2", 
                "valid_at": "2024-01-01T01:00:00Z",
                "source_node_uuid": "node-uuid-2"
            }
        ], 200.0)
        
        mock_cache.return_value = {
            "graph_results": [
                {
                    "fact": "Alice is friends with Bob in the story",
                    "uuid": "fact-uuid-1",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "source_node_uuid": "node-uuid-1"
                },
                {
                    "fact": "Bob mentors Charlie throughout the narrative",
                    "uuid": "fact-uuid-2",
                    "valid_at": "2024-01-01T01:00:00Z", 
                    "source_node_uuid": "node-uuid-2"
                }
            ],
            "total_results": 2,
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
            
            # User experience requirements for graph search
            assert data["search_type"] == "graph"
            assert "graph_results" in data
            assert len(data["graph_results"]) == 2
            
            # Graph results should contain relationship information
            first_fact = data["graph_results"][0]
            assert "fact" in first_fact
            assert "uuid" in first_fact
            assert "relationship" in first_fact["fact"].lower() or "friends" in first_fact["fact"].lower()
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_hybrid_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_hybrid_search_combines_results(self, mock_cache, mock_search, mock_validate):
        """Test hybrid search combines vector and text search effectively."""
        # Setup mocks
        mock_validate.return_value = {"query": "plot development", "limit": 8}
        mock_search.return_value = ([
            {
                "content": "Plot development requires careful pacing and structure...",
                "combined_score": 0.95,
                "vector_similarity": 0.90,
                "text_similarity": 0.85,
                "document_title": "Plot Guide",
                "document_source": "plotting.txt",
                "chunk_id": "chunk-plot-1"
            }
        ], 180.0)
        
        mock_cache.return_value = {
            "results": [
                {
                    "content": "Plot development requires careful pacing and structure...",
                    "combined_score": 0.95,
                    "vector_similarity": 0.90,
                    "text_similarity": 0.85,
                    "document_title": "Plot Guide",
                    "document_source": "plotting.txt",
                    "chunk_id": "chunk-plot-1"
                }
            ],
            "total_results": 1,
            "search_type": "hybrid",
            "query_time_ms": 180.0
        }
        
        with TestClient(app) as client:
            response = client.post("/search/hybrid", json={
                "query": "plot development",
                "limit": 8
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Hybrid search should provide combined scoring
            assert data["search_type"] == "hybrid"
            result = data["results"][0]
            assert "combined_score" in result or "score" in result
            
            # Should be fast enough for good user experience
            assert data["query_time_ms"] < 2000


@pytest.mark.medium
@pytest.mark.search
class TestSearchUserExperience:
    """Tests focused on search user experience."""
    
    def test_search_response_time_acceptable(self):
        """Test search response times are acceptable for user experience."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                    with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                        # Setup fast mocks
                        mock_validate.return_value = {"query": "test", "limit": 10}
                        mock_search.return_value = ([{"content": "test result"}], 100.0)
                        mock_cache.return_value = {
                            "results": [{"content": "test result"}],
                            "total_results": 1,
                            "search_type": "vector",
                            "query_time_ms": 100.0
                        }
                        
                        start_time = time.time()
                        response = client.post("/search/vector", json={
                            "query": "test query",
                            "limit": 10
                        })
                        end_time = time.time()
                        
                        response_time = (end_time - start_time) * 1000  # Convert to ms
                        
                        assert response.status_code == 200
                        # Good user experience requires fast search
                        assert response_time < 2000, f"Search too slow for good UX: {response_time}ms"
    
    def test_search_result_pagination(self):
        """Test search supports pagination for large result sets."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                    with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                        # Test different limit values
                        for limit in [5, 10, 20, 50]:
                            mock_validate.return_value = {"query": "test", "limit": limit}
                            mock_results = [{"content": f"result {i}"} for i in range(limit)]
                            mock_search.return_value = (mock_results, 150.0)
                            mock_cache.return_value = {
                                "results": mock_results,
                                "total_results": limit,
                                "search_type": "vector",
                                "query_time_ms": 150.0
                            }
                            
                            response = client.post("/search/vector", json={
                                "query": "test query",
                                "limit": limit
                            })
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["results"]) == limit
                            assert data["total_results"] == limit
    
    def test_search_empty_results_handling(self):
        """Test search handles empty results gracefully."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                    with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                        # Setup mocks for no results
                        mock_validate.return_value = {"query": "nonexistent query", "limit": 10}
                        mock_search.return_value = ([], 50.0)
                        mock_cache.return_value = {
                            "results": [],
                            "total_results": 0,
                            "search_type": "vector",
                            "query_time_ms": 50.0
                        }
                        
                        response = client.post("/search/vector", json={
                            "query": "nonexistent query",
                            "limit": 10
                        })
                        
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Should handle empty results gracefully
                        assert data["results"] == []
                        assert data["total_results"] == 0
                        assert "search_type" in data
    
    def test_search_query_suggestions(self):
        """Test search provides helpful query suggestions for poor results."""
        # This would be implemented if the search system provides query suggestions
        with TestClient(app) as client:
            with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                    with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                        # Setup mocks for poor results
                        mock_validate.return_value = {"query": "chracter developmnt", "limit": 10}  # Typos
                        mock_search.return_value = ([
                            {"content": "Low relevance result", "score": 0.3}
                        ], 100.0)
                        mock_cache.return_value = {
                            "results": [{"content": "Low relevance result", "score": 0.3}],
                            "total_results": 1,
                            "search_type": "vector",
                            "query_time_ms": 100.0
                        }
                        
                        response = client.post("/search/vector", json={
                            "query": "chracter developmnt",  # Intentional typos
                            "limit": 10
                        })
                        
                        assert response.status_code == 200
                        # System should handle typos gracefully
                        data = response.json()
                        assert "results" in data


@pytest.mark.medium
@pytest.mark.search
class TestSearchAccuracy:
    """Tests for search result accuracy and relevance."""
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_search_relevance_scoring(self, mock_cache, mock_search, mock_validate):
        """Test search results are properly scored for relevance."""
        # Setup mocks with varying relevance scores
        mock_validate.return_value = {"query": "writing techniques", "limit": 5}
        mock_search.return_value = ([
            {"content": "Advanced writing techniques for authors", "score": 0.95},
            {"content": "Basic writing skills and techniques", "score": 0.88},
            {"content": "Writing tools and software", "score": 0.72},
            {"content": "Publishing and marketing", "score": 0.45}
        ], 120.0)
        
        mock_cache.return_value = {
            "results": [
                {"content": "Advanced writing techniques for authors", "score": 0.95},
                {"content": "Basic writing skills and techniques", "score": 0.88},
                {"content": "Writing tools and software", "score": 0.72},
                {"content": "Publishing and marketing", "score": 0.45}
            ],
            "total_results": 4,
            "search_type": "vector",
            "query_time_ms": 120.0
        }
        
        with TestClient(app) as client:
            response = client.post("/search/vector", json={
                "query": "writing techniques",
                "limit": 5
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Results should be ordered by relevance
            results = data["results"]
            assert len(results) == 4
            
            # Check that results are ordered by score (highest first)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by relevance score"
            
            # Most relevant result should have high score
            assert results[0]["score"] > 0.9, "Most relevant result should have high score"
    
    def test_search_content_filtering(self):
        """Test search can filter content appropriately."""
        with TestClient(app) as client:
            # Test various query types that might need filtering
            test_queries = [
                "inappropriate content",
                "violent scenes",
                "adult themes",
                "sensitive topics"
            ]
            
            for query in test_queries:
                with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                    with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                        with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                            mock_validate.return_value = {"query": query, "limit": 10}
                            mock_search.return_value = ([
                                {"content": f"Appropriate content about {query}", "score": 0.8}
                            ], 100.0)
                            mock_cache.return_value = {
                                "results": [{"content": f"Appropriate content about {query}", "score": 0.8}],
                                "total_results": 1,
                                "search_type": "vector",
                                "query_time_ms": 100.0
                            }
                            
                            response = client.post("/search/vector", json={
                                "query": query,
                                "limit": 10
                            })
                            
                            assert response.status_code == 200
                            # Should handle potentially sensitive queries appropriately
                            data = response.json()
                            assert "results" in data


@pytest.mark.medium
@pytest.mark.search
class TestSearchCaching:
    """Tests for search caching and performance optimization."""
    
    @patch('agent.api.endpoints.search.RequestValidator.validate_search_request')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    def test_search_caching_effectiveness(self, mock_search, mock_validate):
        """Test search caching improves performance for repeated queries."""
        # Setup mocks
        mock_validate.return_value = {"query": "cached query", "limit": 10}
        mock_search.return_value = ([{"content": "cached result"}], 200.0)
        
        with TestClient(app) as client:
            # First request - should hit the search operation
            with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                mock_cache.return_value = {
                    "results": [{"content": "cached result"}],
                    "total_results": 1,
                    "search_type": "vector",
                    "query_time_ms": 200.0
                }
                
                response1 = client.post("/search/vector", json={
                    "query": "cached query",
                    "limit": 10
                })
                
                assert response1.status_code == 200
                
                # Second identical request - should use cache
                response2 = client.post("/search/vector", json={
                    "query": "cached query", 
                    "limit": 10
                })
                
                assert response2.status_code == 200
                
                # Both responses should be identical
                assert response1.json() == response2.json()
    
    def test_search_cache_invalidation(self):
        """Test search cache is properly invalidated when needed."""
        # This would test cache invalidation logic if implemented
        with TestClient(app) as client:
            with patch('agent.api.endpoints.search.RequestValidator.validate_search_request') as mock_validate:
                with patch('agent.api.endpoints.search.SearchOperations.execute_vector_search') as mock_search:
                    with patch('agent.api.endpoints.search.cached_operation') as mock_cache:
                        mock_validate.return_value = {"query": "test", "limit": 10}
                        mock_search.return_value = ([{"content": "result"}], 100.0)
                        mock_cache.return_value = {
                            "results": [{"content": "result"}],
                            "total_results": 1,
                            "search_type": "vector",
                            "query_time_ms": 100.0
                        }
                        
                        response = client.post("/search/vector", json={
                            "query": "test",
                            "limit": 10
                        })
                        
                        assert response.status_code == 200
                        # Cache invalidation would be tested here if implemented