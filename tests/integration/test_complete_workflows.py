"""Integration tests for complete workflows."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from agent.api.main import app


@pytest.mark.integration
@pytest.mark.critical
class TestCompleteNovelWorkflow:
    """Test complete novel creation and management workflow."""
    
    @patch('agent.core.database.novels.create_novel_tables')
    @patch('agent.core.database.novels.create_novel')
    @patch('agent.core.database.novels.create_character')
    @patch('agent.core.database.novels.create_chapter')
    @patch('agent.core.database.novels.search_novel_content')
    def test_complete_novel_workflow(self, mock_search, mock_chapter, mock_character, mock_novel, mock_tables):
        """Test complete workflow from novel creation to content search."""
        # Setup mocks
        mock_tables.return_value = None
        mock_novel.return_value = "novel-123"
        mock_character.return_value = "character-456"
        mock_chapter.return_value = "chapter-789"
        mock_search.return_value = [
            {
                "scene_id": "scene-1",
                "title": "Opening Scene",
                "content": "The hero begins their journey...",
                "characters_present": ["Test Hero"],
                "emotional_tone": "hopeful",
                "chunk_type": "narration",
                "chapter_number": 1,
                "chapter_title": "The Beginning"
            }
        ]
        
        with TestClient(app) as client:
            # Step 1: Create novel
            novel_response = client.post("/novels", params={
                "title": "Epic Fantasy Novel",
                "author": "Test Author",
                "genre": "fantasy",
                "summary": "An epic tale of heroes and magic"
            })
            assert novel_response.status_code == 200
            novel_data = novel_response.json()
            novel_id = novel_data["novel_id"]
            
            # Step 2: Create main character
            character_response = client.post(f"/novels/{novel_id}/characters", params={
                "name": "Aragorn the Brave",
                "role": "protagonist",
                "personality_traits": ["brave", "loyal", "wise"],
                "background": "A ranger from the north"
            })
            assert character_response.status_code == 200
            character_data = character_response.json()
            
            # Step 3: Create first chapter
            chapter_response = client.post(f"/novels/{novel_id}/chapters", params={
                "chapter_number": 1,
                "title": "The Journey Begins",
                "summary": "Our hero starts their epic quest"
            })
            assert chapter_response.status_code == 200
            chapter_data = chapter_response.json()
            
            # Step 4: Search novel content
            search_response = client.get(f"/novels/{novel_id}/search", params={
                "query": "hero journey",
                "character_filter": "Aragorn the Brave",
                "limit": 10
            })
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            # Verify complete workflow
            assert novel_data["title"] == "Epic Fantasy Novel"
            assert character_data["name"] == "Aragorn the Brave"
            assert chapter_data["chapter_number"] == 1
            assert len(search_data["results"]) > 0


@pytest.mark.integration
@pytest.mark.critical
class TestChatToSearchWorkflow:
    """Test workflow from chat interaction to search results."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    @patch('agent.api.endpoints.chat.cached_operation')
    @patch('agent.api.endpoints.search.SearchOperations.execute_vector_search')
    @patch('agent.api.endpoints.search.cached_operation')
    def test_chat_to_search_workflow(self, mock_search_cache, mock_vector_search, 
                                   mock_chat_cache, mock_execute, mock_session, mock_validate):
        """Test workflow from chat interaction to follow-up search."""
        # Setup chat mocks
        mock_validate.return_value = {
            "message": "Tell me about character development",
            "user_id": "test-user"
        }
        mock_session.return_value = "session-123"
        mock_execute.return_value = ("Character development is crucial in storytelling...", [])
        mock_chat_cache.return_value = {
            "message": "Character development is crucial in storytelling...",
            "tools_used": [],
            "session_id": "session-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        # Setup search mocks
        mock_vector_search.return_value = ([
            {
                "content": "Character development involves growth and change...",
                "score": 0.95,
                "document_title": "Writing Guide",
                "document_source": "writing_tips.txt",
                "chunk_id": "chunk-1"
            }
        ], 125.5)
        mock_search_cache.return_value = {
            "results": [
                {
                    "content": "Character development involves growth and change...",
                    "score": 0.95,
                    "document_title": "Writing Guide",
                    "document_source": "writing_tips.txt",
                    "chunk_id": "chunk-1"
                }
            ],
            "total_results": 1,
            "search_type": "vector",
            "query_time_ms": 125.5
        }
        
        with TestClient(app) as client:
            # Step 1: Chat interaction
            chat_response = client.post("/chat", json={
                "message": "Tell me about character development",
                "search_type": "hybrid"
            })
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            
            # Step 2: Follow-up search based on chat response
            search_response = client.post("/search/vector", json={
                "query": "character development techniques",
                "limit": 5
            })
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            # Verify workflow
            assert "character development" in chat_data["message"].lower()
            assert len(search_data["results"]) > 0
            assert search_data["results"][0]["score"] > 0.9


@pytest.mark.integration
class TestErrorRecoveryWorkflow:
    """Test error recovery and fallback mechanisms."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    def test_chat_with_llm_failure_recovery(self, mock_execute, mock_session, mock_validate):
        """Test chat endpoint recovery when LLM fails."""
        # Setup mocks
        mock_validate.return_value = {
            "message": "Test message",
            "user_id": "test-user"
        }
        mock_session.return_value = "session-123"
        
        # Simulate LLM failure then recovery
        mock_execute.side_effect = [
            Exception("LLM API timeout"),  # First call fails
            ("Fallback response generated", [])  # Second call succeeds
        ]
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": "Test message",
                "search_type": "hybrid"
            })
            
            # Should either succeed with fallback or return appropriate error
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                # Should indicate fallback was used
                assert "fallback" in str(data).lower() or "error" in data.get("metadata", {})
    
    @patch('agent.core.database.connection.db_pool.acquire')
    def test_database_failure_recovery(self, mock_acquire):
        """Test API behavior when database is unavailable."""
        # Simulate database connection failure
        mock_acquire.side_effect = Exception("Database connection failed")
        
        with TestClient(app) as client:
            response = client.get("/health")
            
            # Health endpoint should handle database failure gracefully
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                # Should indicate database issue
                assert data.get("database") is False or "error" in str(data).lower()


@pytest.mark.integration
class TestConcurrentOperations:
    """Test concurrent operations and race conditions."""
    
    @patch('agent.core.database.novels.create_novel')
    @patch('agent.core.database.novels.create_novel_tables')
    def test_concurrent_novel_creation(self, mock_tables, mock_create):
        """Test concurrent novel creation doesn't cause issues."""
        import threading
        import time
        
        mock_tables.return_value = None
        mock_create.side_effect = lambda *args, **kwargs: f"novel-{int(time.time() * 1000000)}"
        
        results = []
        errors = []
        
        def create_novel_thread(novel_num):
            try:
                with TestClient(app) as client:
                    response = client.post("/novels", params={
                        "title": f"Concurrent Novel {novel_num}",
                        "author": f"Author {novel_num}",
                        "genre": "test"
                    })
                    results.append(response.json())
            except Exception as e:
                errors.append(str(e))
        
        # Create 5 novels concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_novel_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # All novels should have unique IDs
        novel_ids = [result["novel_id"] for result in results]
        assert len(set(novel_ids)) == 5, "Novel IDs should be unique"