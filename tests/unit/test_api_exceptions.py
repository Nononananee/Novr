"""
Comprehensive unit tests for agent.api_exceptions module.
Tests all custom exception classes and their behavior.
"""

import pytest
from agent.api_exceptions import (
    APIException,
    ValidationError,
    NotFoundError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    InternalServerError,
    ServiceUnavailableError,
    GenerationError,
    ConsistencyError,
    DatabaseError,
    MemorySystemError
)


class TestAPIException:
    """Test base APIException class."""
    
    def test_api_exception_basic_creation(self):
        """Test basic APIException creation."""
        exc = APIException("Test message")
        
        assert str(exc) == "Test message"
        assert exc.message == "Test message"
        assert exc.status_code == 500  # Default
        assert exc.details == {}
    
    def test_api_exception_with_status_code(self):
        """Test APIException with custom status code."""
        exc = APIException("Test message", status_code=400)
        
        assert exc.message == "Test message"
        assert exc.status_code == 400
        assert exc.details == {}
    
    def test_api_exception_with_details(self):
        """Test APIException with details."""
        details = {"field": "value", "error_code": "E001"}
        exc = APIException("Test message", details=details)
        
        assert exc.message == "Test message"
        assert exc.status_code == 500
        assert exc.details == details
    
    def test_api_exception_full_parameters(self):
        """Test APIException with all parameters."""
        details = {"validation_errors": ["field1", "field2"]}
        exc = APIException("Complex error", status_code=422, details=details)
        
        assert exc.message == "Complex error"
        assert exc.status_code == 422
        assert exc.details == details
    
    def test_api_exception_inheritance(self):
        """Test APIException inherits from Exception."""
        exc = APIException("Test message")
        assert isinstance(exc, Exception)
        assert isinstance(exc, APIException)
    
    def test_api_exception_none_details(self):
        """Test APIException handles None details gracefully."""
        exc = APIException("Test message", details=None)
        assert exc.details == {}


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_basic(self):
        """Test basic ValidationError creation."""
        exc = ValidationError("Invalid input")
        
        assert str(exc) == "Invalid input"
        assert exc.message == "Invalid input"
        assert exc.status_code == 400
        assert exc.details == {}
    
    def test_validation_error_with_details(self):
        """Test ValidationError with validation details."""
        details = {
            "field_errors": {
                "email": "Invalid email format",
                "age": "Must be positive integer"
            }
        }
        exc = ValidationError("Validation failed", details=details)
        
        assert exc.message == "Validation failed"
        assert exc.status_code == 400
        assert exc.details == details
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from APIException."""
        exc = ValidationError("Test message")
        assert isinstance(exc, APIException)
        assert isinstance(exc, ValidationError)


class TestNotFoundError:
    """Test NotFoundError exception."""
    
    def test_not_found_error_basic(self):
        """Test basic NotFoundError creation."""
        exc = NotFoundError("Resource not found")
        
        assert exc.message == "Resource not found"
        assert exc.status_code == 404
        assert exc.details == {}
    
    def test_not_found_error_with_resource_info(self):
        """Test NotFoundError with resource information."""
        details = {
            "resource_type": "novel",
            "resource_id": "123",
            "query_params": {"title": "Missing Novel"}
        }
        exc = NotFoundError("Novel not found", details=details)
        
        assert exc.message == "Novel not found"
        assert exc.status_code == 404
        assert exc.details == details


class TestAuthenticationError:
    """Test AuthenticationError exception."""
    
    def test_authentication_error_basic(self):
        """Test basic AuthenticationError creation."""
        exc = AuthenticationError("Authentication required")
        
        assert exc.message == "Authentication required"
        assert exc.status_code == 401
        assert exc.details == {}
    
    def test_authentication_error_with_auth_details(self):
        """Test AuthenticationError with authentication details."""
        details = {
            "auth_method": "bearer_token",
            "error_type": "token_expired",
            "expires_at": "2024-01-01T00:00:00Z"
        }
        exc = AuthenticationError("Token expired", details=details)
        
        assert exc.message == "Token expired"
        assert exc.status_code == 401
        assert exc.details == details


class TestAuthorizationError:
    """Test AuthorizationError exception."""
    
    def test_authorization_error_basic(self):
        """Test basic AuthorizationError creation."""
        exc = AuthorizationError("Access denied")
        
        assert exc.message == "Access denied"
        assert exc.status_code == 403
        assert exc.details == {}
    
    def test_authorization_error_with_permission_details(self):
        """Test AuthorizationError with permission details."""
        details = {
            "required_permissions": ["novel:write", "character:modify"],
            "user_permissions": ["novel:read"],
            "resource": "novel_123"
        }
        exc = AuthorizationError("Insufficient permissions", details=details)
        
        assert exc.message == "Insufficient permissions"
        assert exc.status_code == 403
        assert exc.details == details


class TestRateLimitError:
    """Test RateLimitError exception."""
    
    def test_rate_limit_error_basic(self):
        """Test basic RateLimitError creation."""
        exc = RateLimitError("Rate limit exceeded")
        
        assert exc.message == "Rate limit exceeded"
        assert exc.status_code == 429
        assert exc.details == {}
    
    def test_rate_limit_error_with_limit_details(self):
        """Test RateLimitError with rate limit details."""
        details = {
            "limit": 100,
            "remaining": 0,
            "reset_time": "2024-01-01T01:00:00Z",
            "retry_after": 3600
        }
        exc = RateLimitError("API rate limit exceeded", details=details)
        
        assert exc.message == "API rate limit exceeded"
        assert exc.status_code == 429
        assert exc.details == details


class TestInternalServerError:
    """Test InternalServerError exception."""
    
    def test_internal_server_error_basic(self):
        """Test basic InternalServerError creation."""
        exc = InternalServerError("Internal server error")
        
        assert exc.message == "Internal server error"
        assert exc.status_code == 500
        assert exc.details == {}
    
    def test_internal_server_error_with_debug_info(self):
        """Test InternalServerError with debug information."""
        details = {
            "error_id": "ERR_001",
            "timestamp": "2024-01-01T00:00:00Z",
            "component": "generation_pipeline",
            "stack_trace": "Traceback..."
        }
        exc = InternalServerError("Pipeline failure", details=details)
        
        assert exc.message == "Pipeline failure"
        assert exc.status_code == 500
        assert exc.details == details


class TestServiceUnavailableError:
    """Test ServiceUnavailableError exception."""
    
    def test_service_unavailable_error_basic(self):
        """Test basic ServiceUnavailableError creation."""
        exc = ServiceUnavailableError("Service temporarily unavailable")
        
        assert exc.message == "Service temporarily unavailable"
        assert exc.status_code == 503
        assert exc.details == {}
    
    def test_service_unavailable_error_with_service_info(self):
        """Test ServiceUnavailableError with service information."""
        details = {
            "service": "llm_provider",
            "estimated_recovery": "2024-01-01T02:00:00Z",
            "retry_after": 1800,
            "status_page": "https://status.openai.com"
        }
        exc = ServiceUnavailableError("LLM service down", details=details)
        
        assert exc.message == "LLM service down"
        assert exc.status_code == 503
        assert exc.details == details


class TestGenerationError:
    """Test GenerationError exception."""
    
    def test_generation_error_basic(self):
        """Test basic GenerationError creation."""
        exc = GenerationError("Content generation failed")
        
        assert exc.message == "Content generation failed"
        assert exc.status_code == 422
        assert exc.details == {}
    
    def test_generation_error_with_generation_details(self):
        """Test GenerationError with generation details."""
        details = {
            "generation_type": "character_dialogue",
            "prompt_length": 1500,
            "model": "gpt-4-turbo-preview",
            "failure_reason": "context_length_exceeded",
            "suggested_action": "reduce_context_size"
        }
        exc = GenerationError("Context too long for generation", details=details)
        
        assert exc.message == "Context too long for generation"
        assert exc.status_code == 422
        assert exc.details == details


class TestConsistencyError:
    """Test ConsistencyError exception."""
    
    def test_consistency_error_basic(self):
        """Test basic ConsistencyError creation."""
        exc = ConsistencyError("Character consistency violation")
        
        assert exc.message == "Character consistency violation"
        assert exc.status_code == 422
        assert exc.details == {}
    
    def test_consistency_error_with_validation_details(self):
        """Test ConsistencyError with consistency validation details."""
        details = {
            "consistency_type": "character_traits",
            "character": "Alice",
            "conflicting_traits": ["shy", "outgoing"],
            "previous_context": "Chapter 1: Alice was described as shy",
            "current_content": "Alice confidently addressed the crowd",
            "confidence_score": 0.15
        }
        exc = ConsistencyError("Character trait inconsistency", details=details)
        
        assert exc.message == "Character trait inconsistency"
        assert exc.status_code == 422
        assert exc.details == details


class TestDatabaseError:
    """Test DatabaseError exception."""
    
    def test_database_error_basic(self):
        """Test basic DatabaseError creation."""
        exc = DatabaseError("Database connection failed")
        
        assert exc.message == "Database connection failed"
        assert exc.status_code == 500
        assert exc.details == {}
    
    def test_database_error_with_db_details(self):
        """Test DatabaseError with database details."""
        details = {
            "database": "postgresql",
            "operation": "vector_search",
            "table": "documents",
            "error_code": "23505",
            "connection_pool_status": "exhausted"
        }
        exc = DatabaseError("Vector search failed", details=details)
        
        assert exc.message == "Vector search failed"
        assert exc.status_code == 500
        assert exc.details == details


class TestMemorySystemError:
    """Test MemorySystemError exception."""
    
    def test_memory_system_error_basic(self):
        """Test basic MemorySystemError creation."""
        exc = MemorySystemError("Memory system initialization failed")
        
        assert exc.message == "Memory system initialization failed"
        assert exc.status_code == 500
        assert exc.details == {}
    
    def test_memory_system_error_with_memory_details(self):
        """Test MemorySystemError with memory system details."""
        details = {
            "memory_component": "emotional_memory",
            "operation": "store_emotional_context",
            "memory_usage": "95%",
            "available_tokens": 500,
            "required_tokens": 2000
        }
        exc = MemorySystemError("Memory capacity exceeded", details=details)
        
        assert exc.message == "Memory capacity exceeded"
        assert exc.status_code == 500
        assert exc.details == details


class TestExceptionChaining:
    """Test exception chaining and wrapping scenarios."""
    
    def test_exception_wrapping_with_cause(self):
        """Test wrapping another exception as a cause."""
        original_error = ValueError("Original validation error")
        
        try:
            raise ValidationError("Wrapped validation error") from original_error
        except ValidationError as exc:
            assert str(exc) == "Wrapped validation error"
            assert exc.status_code == 400
            assert exc.__cause__ is original_error
    
    def test_database_error_from_connection_error(self):
        """Test database error wrapping connection error."""
        import psycopg2
        
        try:
            # Simulate connection error
            original_error = psycopg2.OperationalError("Connection failed")
            raise DatabaseError(
                "Database connection failed",
                details={"original_error": str(original_error)}
            ) from original_error
        except DatabaseError as exc:
            assert exc.message == "Database connection failed"
            assert exc.status_code == 500
            assert "original_error" in exc.details


class TestExceptionSerialization:
    """Test exception serialization for API responses."""
    
    def test_exception_to_dict_basic(self):
        """Test converting exception to dictionary for API response."""
        exc = ValidationError("Validation failed")
        
        # Manual serialization (as would be done in API error handler)
        error_dict = {
            "error": exc.message,
            "status_code": exc.status_code,
            "details": exc.details
        }
        
        expected = {
            "error": "Validation failed",
            "status_code": 400,
            "details": {}
        }
        
        assert error_dict == expected
    
    def test_exception_to_dict_with_details(self):
        """Test converting exception with details to dictionary."""
        details = {
            "field_errors": {"name": "Required field"},
            "error_code": "VALIDATION_001"
        }
        exc = ValidationError("Validation failed", details=details)
        
        error_dict = {
            "error": exc.message,
            "status_code": exc.status_code,
            "details": exc.details
        }
        
        expected = {
            "error": "Validation failed",
            "status_code": 400,
            "details": details
        }
        
        assert error_dict == expected


class TestErrorScenarios:
    """Test specific error scenarios that might occur in the application."""
    
    def test_character_not_found_scenario(self):
        """Test character not found error scenario."""
        exc = NotFoundError(
            "Character not found",
            details={
                "character_name": "Alice",
                "novel_id": "novel_123",
                "search_attempted": ["by_name", "by_alias", "by_description"]
            }
        )
        
        assert exc.status_code == 404
        assert "character_name" in exc.details
        assert "Alice" in exc.details["character_name"]
    
    def test_content_generation_timeout_scenario(self):
        """Test content generation timeout scenario."""
        exc = GenerationError(
            "Content generation timed out",
            details={
                "timeout_seconds": 30,
                "generation_type": "chapter_continuation",
                "model": "gpt-4-turbo-preview",
                "prompt_tokens": 3500,
                "partial_response": "The character began to..."
            }
        )
        
        assert exc.status_code == 422
        assert exc.details["timeout_seconds"] == 30
        assert "partial_response" in exc.details
    
    def test_memory_consistency_violation_scenario(self):
        """Test memory consistency violation scenario."""
        exc = ConsistencyError(
            "Plot consistency violation detected",
            details={
                "violation_type": "timeline_conflict",
                "conflicting_events": [
                    {"event": "Character A meets Character B", "chapter": 3},
                    {"event": "Character A mentions never meeting Character B", "chapter": 5}
                ],
                "confidence_score": 0.92,
                "suggested_resolution": "Review character interactions in chapter 3"
            }
        )
        
        assert exc.status_code == 422
        assert exc.details["violation_type"] == "timeline_conflict"
        assert len(exc.details["conflicting_events"]) == 2
    
    def test_database_vector_search_failure_scenario(self):
        """Test database vector search failure scenario."""
        exc = DatabaseError(
            "Vector similarity search failed",
            details={
                "operation": "pgvector_similarity_search",
                "table": "document_embeddings",
                "query_vector_dimensions": 1536,
                "expected_dimensions": 1536,
                "error_type": "index_corruption",
                "recommended_action": "rebuild_vector_index"
            }
        )
        
        assert exc.status_code == 500
        assert exc.details["operation"] == "pgvector_similarity_search"
        assert "recommended_action" in exc.details
    
    def test_llm_provider_rate_limit_scenario(self):
        """Test LLM provider rate limit scenario."""
        exc = RateLimitError(
            "OpenAI API rate limit exceeded",
            details={
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "requests_per_minute": 60,
                "tokens_per_minute": 90000,
                "reset_time": "2024-01-01T01:00:00Z",
                "suggested_backoff": "exponential",
                "retry_after_seconds": 60
            }
        )
        
        assert exc.status_code == 429
        assert exc.details["provider"] == "openai"
        assert "retry_after_seconds" in exc.details