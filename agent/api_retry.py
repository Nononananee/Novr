"""
Retry mechanism dengan exponential backoff dan circuit breaker integration.
"""

import asyncio
import logging
from typing import Any, Callable, Optional, Type, Union, List
from functools import wraps
import random
from datetime import datetime, timedelta

from .api_config import APIConfig
from .api_exceptions import (
    APIBaseException, 
    ServiceUnavailableError, 
    CircuitBreakerError,
    ExternalServiceError
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration untuk retry behavior."""
    
    def __init__(
        self,
        max_retries: int = None,
        initial_delay: float = None,
        backoff_factor: float = None,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_retries = max_retries or APIConfig.MAX_RETRIES
        self.initial_delay = initial_delay or APIConfig.RETRY_DELAY
        self.backoff_factor = backoff_factor or APIConfig.RETRY_BACKOFF
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            ServiceUnavailableError,
            ExternalServiceError
        ]


class RetryStats:
    """Statistics untuk retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.total_delay = 0.0
        self.last_retry_time = None
    
    def record_attempt(self, delay: float = 0.0):
        """Record retry attempt."""
        self.total_attempts += 1
        self.total_delay += delay
        self.last_retry_time = datetime.now()
    
    def record_success(self):
        """Record successful retry."""
        self.successful_retries += 1
    
    def record_failure(self):
        """Record failed retry."""
        self.failed_retries += 1
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "success_rate": (
                self.successful_retries / max(self.total_attempts, 1)
            ),
            "average_delay": (
                self.total_delay / max(self.total_attempts, 1)
            ),
            "last_retry_time": (
                self.last_retry_time.isoformat() if self.last_retry_time else None
            )
        }


# Global retry stats
retry_stats = RetryStats()


def calculate_delay(
    attempt: int, 
    initial_delay: float, 
    backoff_factor: float, 
    max_delay: float,
    jitter: bool = True
) -> float:
    """Calculate delay untuk retry attempt dengan exponential backoff."""
    delay = initial_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)
    
    if jitter:
        # Add jitter untuk avoid thundering herd
        jitter_factor = random.uniform(0.5, 1.5)
        delay *= jitter_factor
    
    return delay


def is_retryable_exception(
    exception: Exception, 
    retryable_exceptions: List[Type[Exception]]
) -> bool:
    """Check apakah exception dapat di-retry."""
    for exc_type in retryable_exceptions:
        if isinstance(exception, exc_type):
            return True
    
    # Special handling untuk circuit breaker
    if isinstance(exception, CircuitBreakerError):
        return False
    
    # Special handling untuk APIBaseException
    if isinstance(exception, APIBaseException):
        # Jangan retry untuk client errors (4xx)
        if 400 <= exception.status_code < 500:
            return False
        # Retry untuk server errors (5xx)
        return exception.status_code >= 500
    
    return False


async def retry_async(
    func: Callable,
    config: RetryConfig = None,
    operation_name: str = None
) -> Any:
    """
    Retry async function dengan exponential backoff.
    
    Args:
        func: Async function untuk retry
        config: Retry configuration
        operation_name: Nama operation untuk logging
    
    Returns:
        Function result
        
    Raises:
        Exception: Last exception jika semua retry gagal
    """
    config = config or RetryConfig()
    operation_name = operation_name or func.__name__
    
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            result = await func()
            
            if attempt > 0:
                retry_stats.record_success()
                logger.info(
                    f"✅ Operation '{operation_name}' succeeded after {attempt} retries"
                )
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check apakah exception dapat di-retry
            if not is_retryable_exception(e, config.retryable_exceptions):
                logger.warning(
                    f"❌ Operation '{operation_name}' failed with non-retryable exception: {e}"
                )
                raise e
            
            # Jika ini adalah attempt terakhir, jangan retry lagi
            if attempt >= config.max_retries:
                retry_stats.record_failure()
                logger.error(
                    f"❌ Operation '{operation_name}' failed after {config.max_retries} retries: {e}"
                )
                break
            
            # Calculate delay untuk retry
            delay = calculate_delay(
                attempt,
                config.initial_delay,
                config.backoff_factor,
                config.max_delay,
                config.jitter
            )
            
            retry_stats.record_attempt(delay)
            
            logger.warning(
                f"⚠️ Operation '{operation_name}' failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay:.2f}s: {e}"
            )
            
            await asyncio.sleep(delay)
    
    # Jika sampai sini, semua retry gagal
    raise last_exception


def retry_decorator(
    max_retries: int = None,
    initial_delay: float = None,
    backoff_factor: float = None,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator untuk auto-retry async functions.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries
        backoff_factor: Backoff multiplier
        retryable_exceptions: List of exceptions yang dapat di-retry
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                retryable_exceptions=retryable_exceptions
            )
            
            async def operation():
                return await func(*args, **kwargs)
            
            return await retry_async(operation, config, func.__name__)
        
        return wrapper
    return decorator


class RetryableOperation:
    """Context manager untuk retryable operations."""
    
    def __init__(
        self,
        operation_name: str,
        config: RetryConfig = None
    ):
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.attempt = 0
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = datetime.now()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and exc_val:
            duration = datetime.now() - self.start_time
            logger.error(
                f"❌ RetryableOperation '{self.operation_name}' failed "
                f"after {duration.total_seconds():.2f}s: {exc_val}"
            )
    
    async def execute(self, func: Callable) -> Any:
        """Execute function dengan retry logic."""
        return await retry_async(func, self.config, self.operation_name)


class BatchRetryManager:
    """Manager untuk batch operations dengan retry."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.results = []
        self.failures = []
    
    async def execute_batch(
        self,
        operations: List[Callable],
        max_concurrent: int = 3,
        fail_fast: bool = False
    ) -> dict:
        """
        Execute batch operations dengan retry dan concurrency control.
        
        Args:
            operations: List of async callables
            max_concurrent: Maximum concurrent operations
            fail_fast: Stop pada first failure
        
        Returns:
            Dict dengan results dan failures
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(operation: Callable, index: int):
            async with semaphore:
                try:
                    result = await retry_async(
                        operation, 
                        self.config, 
                        f"batch_operation_{index}"
                    )
                    return {"index": index, "success": True, "result": result}
                except Exception as e:
                    return {"index": index, "success": False, "error": str(e)}
        
        # Execute all operations
        tasks = [
            execute_single(op, i) 
            for i, op in enumerate(operations)
        ]
        
        if fail_fast:
            # Execute satu per satu dan stop pada first failure
            results = []
            for task in tasks:
                result = await task
                results.append(result)
                if not result["success"]:
                    break
        else:
            # Execute semua operations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    "success": False,
                    "error": str(result)
                })
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        return {
            "successful": successful_results,
            "failed": failed_results,
            "success_rate": len(successful_results) / len(operations) if operations else 0,
            "total_operations": len(operations)
        }
