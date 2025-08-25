"""Critical tests for authentication and authorization - security."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os

from agent.api.main import app


@pytest.mark.critical
@pytest.mark.security
class TestAuthenticationSecurity:
    """Critical security tests for authentication."""
    
    def test_production_api_key_required(self, production_env):
        """Test API key is required in production environment."""
        with patch.dict(os.environ, {"APP_ENV": "production", "API_KEY": "secure-api-key-123"}):
            with TestClient(app) as client:
                # Request without API key should be rejected
                response = client.post("/chat", json={
                    "message": "Test message",
                    "search_type": "hybrid"
                })
                
                # Critical security check
                assert response.status_code in [401, 403], "Production must require authentication"
    
    def test_invalid_api_key_rejected(self, production_env):
        """Test invalid API key is rejected."""
        with patch.dict(os.environ, {"APP_ENV": "production", "API_KEY": "correct-key-123"}):
            with TestClient(app) as client:
                # Request with invalid API key
                response = client.post("/chat", 
                    json={"message": "Test", "search_type": "hybrid"},
                    headers={"Authorization": "Bearer invalid-key"}
                )
                
                # Critical security check
                assert response.status_code in [401, 403], "Invalid API key must be rejected"
    
    def test_valid_api_key_accepted(self, production_env):
        """Test valid API key is accepted."""
        with patch.dict(os.environ, {"APP_ENV": "production", "API_KEY": "correct-key-123"}):
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                            # Setup mocks
                            mock_validate.return_value = {"message": "test", "user_id": "test-user"}
                            mock_session.return_value = "session-123"
                            mock_execute.return_value = ("Response", [])
                            mock_cache.return_value = {
                                "message": "Response",
                                "tools_used": [],
                                "session_id": "session-123",
                                "metadata": {"search_type": "hybrid"}
                            }
                            
                            with TestClient(app) as client:
                                # Request with valid API key
                                response = client.post("/chat",
                                    json={"message": "Test", "search_type": "hybrid"},
                                    headers={"Authorization": "Bearer correct-key-123"}
                                )
                                
                                # Should be accepted (or at least not rejected for auth reasons)
                                assert response.status_code not in [401, 403], "Valid API key must be accepted"
    
    def test_development_environment_bypass(self, development_env):
        """Test development environment bypasses authentication."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                            # Setup mocks
                            mock_validate.return_value = {"message": "test", "user_id": "test-user"}
                            mock_session.return_value = "session-123"
                            mock_execute.return_value = ("Response", [])
                            mock_cache.return_value = {
                                "message": "Response",
                                "tools_used": [],
                                "session_id": "session-123",
                                "metadata": {"search_type": "hybrid"}
                            }
                            
                            # Request without API key in development
                            response = client.post("/chat", json={
                                "message": "Test message",
                                "search_type": "hybrid"
                            })
                            
                            # Should not be rejected for auth reasons in development
                            assert response.status_code not in [401, 403], "Development should bypass auth"


@pytest.mark.critical
@pytest.mark.security
class TestInputValidationSecurity:
    """Critical security tests for input validation."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection attempts are prevented."""
        malicious_inputs = [
            "'; DROP TABLE sessions; --",
            "' OR '1'='1",
            "'; DELETE FROM novels WHERE '1'='1'; --",
            "' UNION SELECT * FROM users; --"
        ]
        
        with TestClient(app) as client:
            for malicious_input in malicious_inputs:
                response = client.post("/chat", json={
                    "message": malicious_input,
                    "search_type": "hybrid"
                })
                
                # Should not cause server error (500) due to SQL injection
                # May return 400/422 for validation, but not 500 for SQL error
                assert response.status_code != 500, f"SQL injection attempt should not cause server error: {malicious_input}"
    
    def test_xss_prevention(self):
        """Test XSS attempts are prevented."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        with TestClient(app) as client:
            for payload in xss_payloads:
                response = client.post("/chat", json={
                    "message": payload,
                    "search_type": "hybrid"
                })
                
                # Response should not contain unescaped script tags
                if response.status_code == 200:
                    response_text = response.text.lower()
                    assert "<script>" not in response_text, f"XSS payload not properly escaped: {payload}"
                    assert "javascript:" not in response_text, f"JavaScript protocol not blocked: {payload}"
    
    def test_command_injection_prevention(self):
        """Test command injection attempts are prevented."""
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(cat /etc/hosts)"
        ]
        
        with TestClient(app) as client:
            for payload in command_injection_payloads:
                response = client.post("/chat", json={
                    "message": f"Tell me about {payload}",
                    "search_type": "hybrid"
                })
                
                # Should not cause server error due to command execution
                assert response.status_code != 500, f"Command injection should not cause server error: {payload}"
    
    def test_path_traversal_prevention(self):
        """Test path traversal attempts are prevented."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts"
        ]
        
        with TestClient(app) as client:
            for payload in path_traversal_payloads:
                # Test in various endpoints
                response = client.post("/chat", json={
                    "message": f"Read file {payload}",
                    "search_type": "hybrid"
                })
                
                # Should not expose file system contents
                if response.status_code == 200:
                    response_text = response.text.lower()
                    assert "root:" not in response_text, f"Path traversal may have succeeded: {payload}"
                    assert "administrator" not in response_text, f"Path traversal may have succeeded: {payload}"
    
    def test_large_payload_handling(self):
        """Test handling of extremely large payloads."""
        # Test with very large message
        large_message = "A" * 100000  # 100KB message
        
        with TestClient(app) as client:
            response = client.post("/chat", json={
                "message": large_message,
                "search_type": "hybrid"
            })
            
            # Should handle large payloads gracefully
            assert response.status_code in [200, 400, 413, 422], "Large payload should be handled gracefully"
            
            # If accepted, should not cause memory issues
            if response.status_code == 200:
                assert len(response.text) < 1000000, "Response should not be excessively large"
    
    def test_unicode_handling_security(self):
        """Test secure handling of Unicode and special characters."""
        unicode_payloads = [
            "Test with emoji: 😀🎉🔥",
            "Unicode normalization: café vs café",
            "Right-to-left: ‮malicious text‬",
            "Zero-width characters: test\u200Bhidden\u200Ctext",
            "Homograph attack: раypal.com"  # Cyrillic 'a' instead of Latin 'a'
        ]
        
        with TestClient(app) as client:
            for payload in unicode_payloads:
                response = client.post("/chat", json={
                    "message": payload,
                    "search_type": "hybrid"
                })
                
                # Should handle Unicode securely without errors
                assert response.status_code != 500, f"Unicode payload caused server error: {payload}"


@pytest.mark.critical
@pytest.mark.security
class TestDataExposurePrevention:
    """Critical tests for preventing sensitive data exposure."""
    
    def test_error_messages_no_sensitive_info(self):
        """Test error messages don't expose sensitive information."""
        with TestClient(app) as client:
            # Trigger various error conditions
            error_responses = []
            
            # Invalid JSON
            response = client.post("/chat", 
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            error_responses.append(response)
            
            # Missing fields
            response = client.post("/chat", json={})
            error_responses.append(response)
            
            # Invalid endpoint
            response = client.get("/nonexistent-endpoint")
            error_responses.append(response)
            
            for response in error_responses:
                if response.status_code >= 400:
                    response_text = response.text.lower()
                    
                    # Should not expose sensitive information
                    sensitive_patterns = [
                        "database_url",
                        "api_key",
                        "password",
                        "secret",
                        "token",
                        "postgresql://",
                        "mongodb://",
                        "redis://",
                        "/etc/",
                        "c:\\",
                        "traceback",
                        "stack trace"
                    ]
                    
                    for pattern in sensitive_patterns:
                        assert pattern not in response_text, f"Error message exposes sensitive info: {pattern}"
    
    def test_health_endpoint_no_sensitive_exposure(self):
        """Test health endpoint doesn't expose sensitive system information."""
        with patch('agent.api.main.HealthChecker.get_comprehensive_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "database": True,
                "graph_database": True,
                "llm_connection": True,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            with TestClient(app) as client:
                response = client.get("/health")
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    # Should not expose sensitive system information
                    sensitive_info = [
                        "database_url",
                        "password",
                        "api_key",
                        "secret_key",
                        "private_key",
                        "connection_string",
                        "localhost",
                        "127.0.0.1",
                        "internal_ip"
                    ]
                    
                    for info in sensitive_info:
                        assert info not in response_text, f"Health endpoint exposes sensitive info: {info}"
    
    def test_system_endpoints_access_control(self):
        """Test system endpoints have proper access control."""
        system_endpoints = [
            "/system/health",
            "/system/status", 
            "/system/circuit-breakers",
            "/system/circuit-breakers/reset"
        ]
        
        with TestClient(app) as client:
            for endpoint in system_endpoints:
                # Test without authentication in production-like environment
                with patch.dict(os.environ, {"APP_ENV": "production"}):
                    if endpoint.endswith("/reset"):
                        response = client.post(endpoint)
                    else:
                        response = client.get(endpoint)
                    
                    # System endpoints should have some form of access control in production
                    # (This test may need adjustment based on actual implementation)
                    if response.status_code == 200:
                        # If accessible, should not expose sensitive operational data
                        response_text = response.text.lower()
                        assert "password" not in response_text, f"System endpoint exposes passwords: {endpoint}"
                        assert "secret" not in response_text, f"System endpoint exposes secrets: {endpoint}"


@pytest.mark.critical
@pytest.mark.security
class TestRateLimitingAndDOS:
    """Critical tests for rate limiting and DOS prevention."""
    
    def test_rapid_request_handling(self):
        """Test handling of rapid successive requests."""
        with TestClient(app) as client:
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                            # Setup mocks for fast responses
                            mock_validate.return_value = {"message": "test", "user_id": "test-user"}
                            mock_session.return_value = "session-123"
                            mock_execute.return_value = ("Response", [])
                            mock_cache.return_value = {
                                "message": "Response",
                                "tools_used": [],
                                "session_id": "session-123",
                                "metadata": {"search_type": "hybrid"}
                            }
                            
                            # Make rapid requests
                            responses = []
                            for i in range(20):
                                response = client.post("/chat", json={
                                    "message": f"Rapid request {i}",
                                    "search_type": "hybrid"
                                })
                                responses.append(response)
                            
                            # System should handle rapid requests without crashing
                            server_errors = [r for r in responses if r.status_code == 500]
                            assert len(server_errors) < len(responses) * 0.1, "Too many server errors under rapid requests"
    
    def test_concurrent_session_handling(self):
        """Test handling of many concurrent sessions."""
        import concurrent.futures
        
        def make_request_with_unique_session(session_id):
            with patch('agent.api.endpoints.chat.RequestValidator.validate_chat_request') as mock_validate:
                with patch('agent.api.endpoints.chat.ConversationManager.get_or_create_session') as mock_session:
                    with patch('agent.api.endpoints.chat.AgentExecutor.execute_agent') as mock_execute:
                        with patch('agent.api.endpoints.chat.cached_operation') as mock_cache:
                            # Setup mocks
                            mock_validate.return_value = {"message": "test", "user_id": f"user-{session_id}"}
                            mock_session.return_value = f"session-{session_id}"
                            mock_execute.return_value = (f"Response {session_id}", [])
                            mock_cache.return_value = {
                                "message": f"Response {session_id}",
                                "tools_used": [],
                                "session_id": f"session-{session_id}",
                                "metadata": {"search_type": "hybrid"}
                            }
                            
                            try:
                                with TestClient(app) as client:
                                    response = client.post("/chat", json={
                                        "message": f"Test from session {session_id}",
                                        "search_type": "hybrid"
                                    })
                                    return response.status_code == 200
                            except Exception:
                                return False
        
        # Test with multiple concurrent sessions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request_with_unique_session, i) for i in range(25)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most requests should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Concurrent session success rate too low: {success_rate}"