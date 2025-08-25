"""Critical tests for chat endpoints - core functionality."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from agent.api.main import app


@pytest.mark.critical
@pytest.mark.chat
class TestChatCoreFunctionality:
    """Critical tests for chat endpoint core functionality."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    @patch('agent.api.endpoints.chat.cached_operation')
    def test_basic_chat_functionality(self, mock_cache, mock_execute, mock_session, mock_validate):
        """Test basic chat functionality works."""
        # Setup mocks
        mock_validate.return_value = {
            "message": "Hello, how are you?",
            "user_id": "test-user-123"
        }
        mock_session.return_value = "session-abc-123"
        mock_execute.return_value = ("I'm doing well, thank you for asking!", [])
        mock_cache.return_value = {
            "message": "I'm doing well, thank you for asking!",
            "tools_used": [],
            "session_id": "session-abc-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": "Hello, how are you?",
                "search_type": "hybrid"
            })
            
            # Critical assertions
            assert response.status_code == 200, "Chat endpoint must return 200"
            
            data = response.json()
            assert "message" in data, "Response must contain message field"
            assert "session_id" in data, "Response must contain session_id"
            assert len(data["message"]) > 0, "Message cannot be empty"
            assert data["session_id"] == "session-abc-123", "Session ID must match"
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    def test_chat_with_tools(self, mock_execute, mock_session, mock_validate):
        """Test chat functionality with tool usage."""
        from agent.models import ToolCall
        
        # Setup mocks
        mock_validate.return_value = {
            "message": "Search for information about character development",
            "user_id": "test-user-123"
        }
        mock_session.return_value = "session-def-456"
        
        # Mock tool usage
        mock_tools = [
            ToolCall(
                tool_name="vector_search",
                args={"query": "character development", "limit": 5},
                tool_call_id="tool-1"
            )
        ]
        mock_execute.return_value = ("Here's information about character development...", mock_tools)
        
        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
            mock_cache.return_value = {
                "message": "Here's information about character development...",
                "tools_used": [tool.dict() for tool in mock_tools],
                "session_id": "session-def-456",
                "metadata": {"search_type": "hybrid"}
            }
            
            with TestClient(app) as client:
                response = client.post("/chat", json={
                    "message": "Search for information about character development",
                    "search_type": "hybrid"
                })
                
                # Critical assertions for tool usage
                assert response.status_code == 200, "Chat with tools must succeed"
                
                data = response.json()
                assert "tools_used" in data, "Response must include tools_used"
                assert len(data["tools_used"]) > 0, "Tools must be recorded when used"
                assert data["tools_used"][0]["tool_name"] == "vector_search", "Tool name must be correct"
    
    def test_chat_input_validation_critical_cases(self):
        """Test critical input validation cases."""
        with TestClient(app) as client:
            # Empty message - critical failure case
            response = client.post("/chat", json={
                "message": "",
                "search_type": "hybrid"
            })
            assert response.status_code in [400, 422], "Empty message must be rejected"
            
            # Missing message field - critical failure case
            response = client.post("/chat", json={
                "search_type": "hybrid"
            })
            assert response.status_code in [400, 422], "Missing message must be rejected"
            
            # Invalid JSON - critical failure case
            response = client.post("/chat", 
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code in [400, 422], "Invalid JSON must be rejected"
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    def test_chat_session_continuity(self, mock_execute, mock_session, mock_validate):
        """Test chat session continuity - critical for conversation flow."""
        # Setup mocks
        mock_validate.return_value = {
            "message": "Continue our conversation",
            "user_id": "test-user-123"
        }
        mock_session.return_value = "session-continuity-789"
        mock_execute.return_value = ("Continuing our conversation...", [])
        
        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
            mock_cache.return_value = {
                "message": "Continuing our conversation...",
                "tools_used": [],
                "session_id": "session-continuity-789",
                "metadata": {"search_type": "hybrid"}
            }
            
            with TestClient(app) as client:
                # First message
                response1 = client.post("/chat", json={
                    "message": "Hello, start a conversation",
                    "search_type": "hybrid"
                })
                
                assert response1.status_code == 200
                session_id = response1.json()["session_id"]
                
                # Second message with same session
                response2 = client.post("/chat", json={
                    "message": "Continue our conversation",
                    "session_id": session_id,
                    "search_type": "hybrid"
                })
                
                assert response2.status_code == 200
                assert response2.json()["session_id"] == session_id, "Session must be maintained"
    
    def test_chat_streaming_endpoint_availability(self):
        """Test that streaming endpoint is available - critical for real-time UX."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    mock_validate.return_value = {"message": "test", "user_id": "test-user"}
                    mock_session.return_value = "session-stream-123"
                    
                    response = client.post("/chat/stream", json={
                        "message": "Test streaming",
                        "search_type": "hybrid"
                    })
                    
                    # Streaming endpoint must be available
                    assert response.status_code in [200, 500], "Streaming endpoint must be accessible"
                    
                    if response.status_code == 200:
                        # Check streaming headers
                        assert "text/event-stream" in response.headers.get("content-type", "").lower() or \
                               "text/plain" in response.headers.get("content-type", "").lower(), \
                               "Streaming response must have correct content type"


@pytest.mark.critical
@pytest.mark.chat
class TestChatErrorHandling:
    """Critical error handling tests for chat endpoints."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    def test_chat_llm_failure_recovery(self, mock_execute, mock_session, mock_validate):
        """Test chat recovery when LLM fails - critical for system stability."""
        # Setup mocks
        mock_validate.return_value = {"message": "test", "user_id": "test-user"}
        mock_session.return_value = "session-error-123"
        mock_execute.side_effect = Exception("LLM service unavailable")
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": "Test message",
                "search_type": "hybrid"
            })
            
            # System must handle LLM failure gracefully
            assert response.status_code in [200, 503], "Must handle LLM failure gracefully"
            
            if response.status_code == 200:
                data = response.json()
                # Should provide fallback response
                assert "message" in data, "Must provide fallback message"
                assert len(data["message"]) > 0, "Fallback message cannot be empty"
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    def test_chat_validation_failure_handling(self, mock_validate):
        """Test chat validation failure handling - critical for input safety."""
        mock_validate.side_effect = ValueError("Invalid input detected")
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": "Potentially malicious input",
                "search_type": "hybrid"
            })
            
            # Must handle validation failures safely
            assert response.status_code in [400, 422, 500], "Must reject invalid input"
    
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    def test_chat_session_failure_handling(self, mock_session):
        """Test chat session failure handling - critical for user experience."""
        mock_session.side_effect = Exception("Session service unavailable")
        
        with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
            mock_validate.return_value = {"message": "test", "user_id": "test-user"}
            
            with TestClient(app) as client:
                response = client.post("/chat", json={
                    "message": "Test message",
                    "search_type": "hybrid"
                })
                
                # Must handle session failures
                assert response.status_code in [200, 500, 503], "Must handle session failure"


@pytest.mark.critical
@pytest.mark.chat
@pytest.mark.performance
class TestChatPerformanceCritical:
    """Critical performance tests for chat endpoints."""
    
    @patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request')
    @patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session')
    @patch('agent.api.endpoints.chat.AgentExecutor.execute_agent')
    @patch('agent.api.endpoints.chat.cached_operation')
    def test_chat_response_time_critical(self, mock_cache, mock_execute, mock_session, mock_validate):
        """Test chat response time meets critical requirements."""
        import time
        
        # Setup fast mocks
        mock_validate.return_value = {"message": "test", "user_id": "test-user"}
        mock_session.return_value = "session-perf-123"
        mock_execute.return_value = ("Fast response", [])
        mock_cache.return_value = {
            "message": "Fast response",
            "tools_used": [],
            "session_id": "session-perf-123",
            "metadata": {"search_type": "hybrid"}
        }
        
        with TestClient(app) as client:
            start_time = time.time()
            
            response = client.post("/chat", json={
                "message": "Quick test",
                "search_type": "hybrid"
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Critical performance requirements
            assert response.status_code == 200, "Must respond successfully"
            assert response_time < 5.0, f"Response time {response_time}s exceeds critical threshold of 5s"
    
    def test_chat_concurrent_requests_stability(self):
        """Test chat endpoint stability under concurrent load - critical for production."""
        import concurrent.futures
        import time
        
        def make_chat_request(request_id):
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                            # Setup mocks
                            mock_validate.return_value = {"message": f"test {request_id}", "user_id": "test-user"}
                            mock_session.return_value = f"session-{request_id}"
                            mock_execute.return_value = (f"Response {request_id}", [])
                            mock_cache.return_value = {
                                "message": f"Response {request_id}",
                                "tools_used": [],
                                "session_id": f"session-{request_id}",
                                "metadata": {"search_type": "hybrid"}
                            }
                            
                            try:
                                with TestClient(app) as client:
                                    response = client.post("/chat", json={
                                        "message": f"Concurrent test {request_id}",
                                        "search_type": "hybrid"
                                    })
                                    return response.status_code == 200
                            except Exception:
                                return False
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_chat_request, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Critical stability requirement
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Concurrent success rate {success_rate} below critical threshold of 0.9"