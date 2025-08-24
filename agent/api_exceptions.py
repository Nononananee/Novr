"""
Custom exceptions untuk API dengan comprehensive error handling.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class APIBaseException(Exception):
    """Base exception class untuk semua API exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERIC_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.now()
        self.request_id = str(uuid.uuid4())
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id
        }


class ValidationError(APIBaseException):
    """Exception untuk validation errors."""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, **(details or {})}
        )


class AuthenticationError(APIBaseException):
    """Exception untuk authentication errors."""
    
    def __init__(self, message: str = "Authentication required", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details
        )


class AuthorizationError(APIBaseException):
    """Exception untuk authorization errors."""
    
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR", 
            status_code=403,
            details=details
        )


class NotFoundError(APIBaseException):
    """Exception untuk resource not found errors."""
    
    def __init__(self, resource: str, resource_id: str = None, details: Optional[Dict[str, Any]] = None):
        message = f"{resource} not found"
        if resource_id:
            message += f" with id: {resource_id}"
        
        super().__init__(
            message=message,
            error_code="NOT_FOUND_ERROR",
            status_code=404,
            details={"resource": resource, "resource_id": resource_id, **(details or {})}
        )


class ConflictError(APIBaseException):
    """Exception untuk conflict errors."""
    
    def __init__(self, message: str, resource: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFLICT_ERROR",
            status_code=409,
            details={"resource": resource, **(details or {})}
        )


class RateLimitError(APIBaseException):
    """Exception untuk rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details={"retry_after": retry_after, **(details or {})}
        )


class ServiceUnavailableError(APIBaseException):
    """Exception untuk service unavailable errors."""
    
    def __init__(self, service: str, message: str = None, details: Optional[Dict[str, Any]] = None):
        default_message = f"{service} service is currently unavailable"
        super().__init__(
            message=message or default_message,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            status_code=503,
            details={"service": service, **(details or {})}
        )


class CircuitBreakerError(APIBaseException):
    """Exception untuk circuit breaker errors."""
    
    def __init__(self, circuit_name: str, message: str = None, details: Optional[Dict[str, Any]] = None):
        default_message = f"Circuit breaker '{circuit_name}' is open"
        super().__init__(
            message=message or default_message,
            error_code="CIRCUIT_BREAKER_ERROR",
            status_code=503,
            details={"circuit_name": circuit_name, **(details or {})}
        )


class GenerationError(APIBaseException):
    """Exception untuk content generation errors."""
    
    def __init__(self, message: str, generation_type: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="GENERATION_ERROR",
            status_code=500,
            details={"generation_type": generation_type, **(details or {})}
        )


class DatabaseError(APIBaseException):
    """Exception untuk database errors."""
    
    def __init__(self, message: str, operation: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={"operation": operation, **(details or {})}
        )


class ExternalServiceError(APIBaseException):
    """Exception untuk external service errors."""
    
    def __init__(self, service: str, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{service}: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status_code,
            details={"service": service, **(details or {})}
        )
