"""
Enhanced Error Handling Utilities for Production Ready System
Provides robust error handling, graceful degradation, and retry mechanisms.
"""

import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for appropriate handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    input_data: Dict[str, Any]
    severity: ErrorSeverity
    retry_count: int = 0
    max_retries: int = 3
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class GracefulDegradation:
    """Provides graceful degradation strategies for different operations."""
    
    @staticmethod
    async def get_validation_fallback(
        content: str, 
        validator_type: str, 
        context: ErrorContext
    ) -> Dict[str, Any]:
        """
        Provide contextual fallback for validation failures.
        
        Args:
            content: Content being validated
            validator_type: Type of validator that failed
            context: Error context information
            
        Returns:
            Fallback validation result
        """
        logger.warning(f"Using fallback for {validator_type} validation")
        
        # Basic content analysis for fallback score
        content_length = len(content.strip())
        word_count = len(content.split())
        
        # Calculate fallback score based on content characteristics
        if content_length == 0:
            score = 0.1
        elif word_count < 10:
            score = 0.3
        elif word_count < 50:
            score = 0.5
        else:
            score = 0.6
        
        fallback_result = {
            "score": score,
            "violations": [{
                "type": "fallback_validation",
                "message": f"Using fallback validation due to {validator_type} error",
                "severity": context.severity.value
            }],
            "suggestions": [
                f"Manual review recommended for {validator_type}",
                "Consider retrying with different parameters"
            ],
            "validator_type": validator_type,
            "is_fallback": True,
            "fallback_reason": f"Primary validation failed after {context.retry_count} attempts"
        }
        
        return fallback_result
    
    @staticmethod
    def get_validation_fallback_sync(
        content: str, 
        validator_type: str, 
        context: ErrorContext
    ) -> Dict[str, Any]:
        """
        Synchronous version of get_validation_fallback.
        Provide contextual fallback for validation failures.
        """
        logger.warning(f"Using fallback for {validator_type} validation")
        
        # Basic content analysis for fallback score
        content_length = len(content.strip())
        word_count = len(content.split())
        
        # Calculate fallback score based on content characteristics
        if content_length == 0:
            score = 0.1
        elif word_count < 10:
            score = 0.3
        elif word_count < 50:
            score = 0.5
        else:
            score = 0.6
        
        fallback_result = {
            "score": score,
            "violations": [{
                "type": "fallback_validation",
                "message": f"Using fallback validation due to {validator_type} error",
                "severity": context.severity.value
            }],
            "suggestions": [
                f"Manual review recommended for {validator_type}",
                "Consider retrying with different parameters"
            ],
            "validator_type": validator_type,
            "is_fallback": True,
            "fallback_reason": f"Primary validation failed after {context.retry_count} attempts"
        }
        
        return fallback_result
    
    @staticmethod
    async def get_generation_fallback(
        request: Dict[str, Any], 
        context: ErrorContext
    ) -> Dict[str, Any]:
        """
        Provide fallback for generation failures.
        
        Args:
            request: Generation request that failed
            context: Error context information
            
        Returns:
            Fallback generation result
        """
        logger.warning(f"Using generation fallback for operation: {context.operation}")
        
        # Simple template-based fallback
        fallback_content = f"[Generated content unavailable due to processing error. " \
                          f"Operation: {context.operation}. Please try again with different parameters.]"
        
        fallback_result = {
            "generated_content": fallback_content,
            "success": False,
            "is_fallback": True,
            "fallback_reason": f"Primary generation failed after {context.retry_count} attempts",
            "suggestions": [
                "Try simplifying the request",
                "Check input parameters",
                "Retry with shorter content"
            ],
            "error_context": {
                "operation": context.operation,
                "severity": context.severity.value,
                "retry_count": context.retry_count
            }
        }
        
        return fallback_result
    
    @staticmethod
    async def get_memory_fallback(
        operation: str, 
        context: ErrorContext
    ) -> Dict[str, Any]:
        """
        Provide fallback for memory operation failures.
        
        Args:
            operation: Memory operation that failed
            context: Error context information
            
        Returns:
            Fallback memory result
        """
        logger.warning(f"Using memory fallback for operation: {operation}")
        
        fallback_result = {
            "success": False,
            "is_fallback": True,
            "fallback_reason": f"Memory operation '{operation}' failed",
            "data": {},
            "suggestions": [
                "Check memory system status",
                "Try with smaller data sets",
                "Consider clearing cache"
            ]
        }
        
        return fallback_result


class RetryMechanism:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(
        self, 
        max_retries: int = 3, 
        base_delay: float = 1.0, 
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        error_context: ErrorContext = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry mechanism.
        
        Args:
            func: Function to execute
            *args: Function arguments
            error_context: Error context for logging
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises exception after max retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if error_context:
                    error_context.retry_count = attempt
                
                logger.debug(f"Attempting {func.__name__}, attempt {attempt + 1}/{self.max_retries + 1}")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Success on retry attempt {attempt + 1} for {func.__name__}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                    )
        
        # If we get here, all retries failed
        raise last_exception


def robust_error_handler(
    operation_name: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    max_retries: int = 3,
    use_fallback: bool = True
):
    """
    Decorator for robust error handling with retry and fallback mechanisms.
    
    Args:
        operation_name: Name of the operation for logging
        severity: Error severity level
        max_retries: Maximum number of retry attempts
        use_fallback: Whether to use fallback on final failure
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_context = ErrorContext(
                operation=operation_name,
                input_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                severity=severity,
                max_retries=max_retries
            )
            
            retry_mechanism = RetryMechanism(max_retries=max_retries)
            
            try:
                return await retry_mechanism.execute_with_retry(
                    func, *args, error_context=error_context, **kwargs
                )
            except Exception as e:
                logger.error(f"Operation {operation_name} failed after all retries: {str(e)}")
                logger.debug(f"Error traceback: {traceback.format_exc()}")
                
                if use_fallback:
                    # Determine fallback strategy based on operation type
                    if "validation" in operation_name.lower():
                        content = args[0] if args else ""
                        return await GracefulDegradation.get_validation_fallback(
                            content, operation_name, error_context
                        )
                    elif "generation" in operation_name.lower():
                        request = args[0] if args else {}
                        return await GracefulDegradation.get_generation_fallback(
                            request, error_context
                        )
                    elif "memory" in operation_name.lower():
                        return await GracefulDegradation.get_memory_fallback(
                            operation_name, error_context
                        )
                
                # Re-raise the exception if no fallback is available
                raise e
        
        return wrapper
    return decorator


class ErrorMetrics:
    """Track error metrics for monitoring and analysis."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_rates = {}
        self.last_errors = []
        self.max_history = 1000
    
    def record_error(self, operation: str, error_type: str, severity: ErrorSeverity):
        """Record an error occurrence."""
        timestamp = time.time()
        
        # Update error counts
        key = f"{operation}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Update error history
        error_record = {
            "timestamp": timestamp,
            "operation": operation,
            "error_type": error_type,
            "severity": severity.value
        }
        
        self.last_errors.append(error_record)
        
        # Keep history size manageable
        if len(self.last_errors) > self.max_history:
            self.last_errors = self.last_errors[-self.max_history:]
    
    def get_error_rate(self, operation: str, time_window: int = 3600) -> float:
        """Calculate error rate for an operation within time window."""
        current_time = time.time()
        window_start = current_time - time_window
        
        # Count errors in time window
        errors_in_window = [
            error for error in self.last_errors
            if error["operation"] == operation and error["timestamp"] >= window_start
        ]
        
        return len(errors_in_window) / max(time_window / 60, 1)  # errors per minute
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error metrics."""
        return {
            "total_errors": len(self.last_errors),
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.last_errors[-10:] if self.last_errors else []
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics (alias for get_error_summary)."""
        return self.get_error_summary()


# Global error metrics instance
error_metrics = ErrorMetrics()


class HealthChecker:
    """System health checking utilities."""
    
    @staticmethod
    async def check_component_health(component_name: str, check_func: Callable) -> Dict[str, Any]:
        """
        Check health of a system component.
        
        Args:
            component_name: Name of the component
            check_func: Function that performs the health check
            
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            return {
                "component": component_name,
                "status": "healthy",
                "response_time": response_time,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            
            logger.error(f"Health check failed for {component_name}: {str(e)}")
            
            return {
                "component": component_name,
                "status": "unhealthy",
                "response_time": response_time,
                "error": str(e),
                "timestamp": time.time()
            }


# Utility functions for common error handling patterns

async def safe_execute(
    func: Callable,
    *args,
    operation_name: str = "unknown_operation",
    default_return: Any = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with basic error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        operation_name: Name for logging
        default_return: Default value to return on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe execution failed for {operation_name}: {str(e)}")
        error_metrics.record_error(operation_name, type(e).__name__, ErrorSeverity.MEDIUM)
        return default_return


def validate_input(data: Any, validator: Callable, field_name: str = "input") -> bool:
    """
    Validate input data with error handling.
    
    Args:
        data: Data to validate
        validator: Validation function
        field_name: Name of the field being validated
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return validator(data)
    except Exception as e:
        logger.warning(f"Input validation failed for {field_name}: {str(e)}")
        error_metrics.record_error(f"input_validation_{field_name}", type(e).__name__, ErrorSeverity.LOW)
        return False
