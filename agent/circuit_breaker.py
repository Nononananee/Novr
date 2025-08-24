"""
Production-Grade Circuit Breaker Pattern Implementation
Advanced error recovery system for novel RAG components with state management,
fallback strategies, and intelligent recovery mechanisms.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import statistics
import weakref
import functools

# Import existing components
from .error_handling_utils import ErrorSeverity, error_metrics, robust_error_handler
from .advanced_system_monitor import record_custom_metric, MetricType, ComponentType

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"           # Normal operation, requests pass through
    OPEN = "open"              # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"    # Testing if service has recovered


class FailureType(Enum):
    """Types of failures tracked by circuit breaker."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    QUALITY_FAILURE = "quality_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    RATE_LIMIT = "rate_limit"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: float = 60.0     # Seconds before attempting recovery
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0              # Request timeout in seconds
    monitoring_window: float = 300.0   # Window for failure rate calculation (5 minutes)
    max_requests_half_open: int = 3    # Max requests in half-open state
    quality_threshold: float = 0.7     # Minimum quality score to consider success
    enable_quality_check: bool = True  # Whether to check quality scores
    enable_adaptive_timeout: bool = True  # Whether to adapt timeout based on performance


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker operation."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    current_state: CircuitState = CircuitState.CLOSED


@dataclass
class OperationResult:
    """Result of an operation through circuit breaker."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    quality_score: Optional[float] = None
    circuit_state: CircuitState = CircuitState.CLOSED
    from_cache: bool = False
    fallback_used: bool = False


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, circuit_name: str, state: CircuitState, last_failure: str):
        self.circuit_name = circuit_name
        self.state = state
        self.last_failure = last_failure
        super().__init__(f"Circuit breaker '{circuit_name}' is {state.value}. Last failure: {last_failure}")


class AdaptiveCircuitBreaker:
    """
    Advanced circuit breaker with adaptive behavior, quality monitoring,
    and intelligent fallback strategies for novel RAG systems.
    """
    
    def __init__(self, 
                 name: str, 
                 config: Optional[CircuitBreakerConfig] = None,
                 component_type: ComponentType = ComponentType.API_LAYER):
        """Initialize adaptive circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.component_type = component_type
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        
        # Metrics and monitoring
        self.metrics = CircuitBreakerMetrics()
        self.recent_requests: deque = deque(maxlen=100)  # Last 100 requests
        self.failure_history: deque = deque(maxlen=50)   # Last 50 failures
        
        # Adaptive behavior
        self.adaptive_timeout = self.config.timeout
        self.response_times: deque = deque(maxlen=50)    # For adaptive timeout
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Fallback cache
        self.fallback_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # Callbacks
        self.on_state_change: List[Callable] = []
        self.fallback_strategies: Dict[str, Callable] = {}
        
        logger.info(f"Adaptive circuit breaker '{name}' initialized for {component_type.value}")
    
    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Add callback for state changes."""
        self.on_state_change.append(callback)
    
    def register_fallback_strategy(self, operation_name: str, fallback_func: Callable):
        """Register fallback strategy for specific operations."""
        self.fallback_strategies[operation_name] = fallback_func
        logger.info(f"Registered fallback strategy for '{operation_name}' in circuit '{self.name}'")
    
    async def call(self, 
                   func: Callable,
                   *args,
                   operation_name: str = "default",
                   quality_check_func: Optional[Callable] = None,
                   cache_key: Optional[str] = None,
                   **kwargs) -> OperationResult:
        """
        Execute function through circuit breaker with comprehensive protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_name: Name of operation for metrics and fallbacks
            quality_check_func: Optional function to check result quality
            cache_key: Cache key for fallback caching
            **kwargs: Function keyword arguments
            
        Returns:
            OperationResult with execution details
        """
        
        start_time = time.time()
        
        with self.lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if not self._should_attempt_recovery():
                    # Circuit is open and not ready for recovery
                    fallback_result = await self._execute_fallback(
                        operation_name, args, kwargs, cache_key
                    )
                    
                    self._record_metric("circuit_breaker_blocked_request", 1)
                    
                    if fallback_result is not None:
                        return OperationResult(
                            success=True,
                            result=fallback_result,
                            execution_time=(time.time() - start_time) * 1000,
                            circuit_state=self.state,
                            from_cache=True,
                            fallback_used=True
                        )
                    else:
                        raise CircuitBreakerException(
                            self.name, 
                            self.state, 
                            self.failure_history[-1] if self.failure_history else "Unknown"
                        )
                else:
                    # Transition to half-open for recovery attempt
                    self._change_state(CircuitState.HALF_OPEN)
            
            elif self.state == CircuitState.HALF_OPEN:
                # In half-open state, limit concurrent requests
                if self.success_count + self.failure_count >= self.config.max_requests_half_open:
                    # Too many requests in half-open, fail fast
                    fallback_result = await self._execute_fallback(
                        operation_name, args, kwargs, cache_key
                    )
                    
                    if fallback_result is not None:
                        return OperationResult(
                            success=True,
                            result=fallback_result,
                            execution_time=(time.time() - start_time) * 1000,
                            circuit_state=self.state,
                            from_cache=True,
                            fallback_used=True
                        )
                    else:
                        raise CircuitBreakerException(self.name, self.state, "Half-open limit exceeded")
        
        # Execute the function
        try:
            result = await self._execute_with_timeout(func, args, kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Check quality if quality check function provided
            quality_score = None
            quality_passed = True
            
            if quality_check_func and self.config.enable_quality_check:
                try:
                    quality_score = await self._check_quality(quality_check_func, result)
                    quality_passed = quality_score >= self.config.quality_threshold
                except Exception as e:
                    logger.warning(f"Quality check failed: {e}")
                    quality_passed = True  # Don't fail on quality check errors
            
            if quality_passed:
                # Success
                await self._record_success(execution_time, cache_key, result)
                
                return OperationResult(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    circuit_state=self.state,
                    from_cache=False,
                    fallback_used=False
                )
            else:
                # Quality failure
                await self._record_failure(
                    FailureType.QUALITY_FAILURE, 
                    f"Quality score {quality_score} below threshold {self.config.quality_threshold}",
                    execution_time
                )
                
                # Try fallback for quality failures
                fallback_result = await self._execute_fallback(
                    operation_name, args, kwargs, cache_key
                )
                
                if fallback_result is not None:
                    return OperationResult(
                        success=True,
                        result=fallback_result,
                        execution_time=execution_time,
                        quality_score=quality_score,
                        circuit_state=self.state,
                        from_cache=True,
                        fallback_used=True
                    )
                else:
                    return OperationResult(
                        success=False,
                        error=Exception(f"Quality failure: score {quality_score}"),
                        execution_time=execution_time,
                        quality_score=quality_score,
                        circuit_state=self.state
                    )
        
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            await self._record_failure(FailureType.TIMEOUT, "Operation timeout", execution_time)
            
            # Try fallback for timeouts
            fallback_result = await self._execute_fallback(operation_name, args, kwargs, cache_key)
            
            if fallback_result is not None:
                return OperationResult(
                    success=True,
                    result=fallback_result,
                    execution_time=execution_time,
                    circuit_state=self.state,
                    from_cache=True,
                    fallback_used=True
                )
            else:
                return OperationResult(
                    success=False,
                    error=asyncio.TimeoutError("Operation timeout"),
                    execution_time=execution_time,
                    circuit_state=self.state
                )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            await self._record_failure(FailureType.EXCEPTION, str(e), execution_time)
            
            # Try fallback for exceptions
            fallback_result = await self._execute_fallback(operation_name, args, kwargs, cache_key)
            
            if fallback_result is not None:
                return OperationResult(
                    success=True,
                    result=fallback_result,
                    execution_time=execution_time,
                    circuit_state=self.state,
                    from_cache=True,
                    fallback_used=True
                )
            else:
                return OperationResult(
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    circuit_state=self.state
                )
    
    async def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with adaptive timeout."""
        
        # Use adaptive timeout if enabled
        timeout = self.adaptive_timeout if self.config.enable_adaptive_timeout else self.config.timeout
        
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            # For sync functions, run in thread pool
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=timeout
            )
    
    async def _check_quality(self, quality_func: Callable, result: Any) -> float:
        """Check quality of result."""
        if asyncio.iscoroutinefunction(quality_func):
            return await quality_func(result)
        else:
            return quality_func(result)
    
    async def _record_success(self, execution_time: float, cache_key: Optional[str], result: Any):
        """Record successful operation."""
        
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            
            # Update response times for adaptive timeout
            self.response_times.append(execution_time)
            if self.config.enable_adaptive_timeout:
                self._update_adaptive_timeout()
            
            # Record in recent requests
            self.recent_requests.append({
                "timestamp": datetime.now(),
                "success": True,
                "execution_time": execution_time
            })
            
            # Update cache if cache key provided
            if cache_key and result is not None:
                self.fallback_cache[cache_key] = result
                self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=30)
            
            # State management
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on successful operation
                self.failure_count = max(0, self.failure_count - 1)
            
            # Update metrics
            self._update_metrics()
            
            # Record monitoring metrics
            self._record_metric("circuit_breaker_success", 1)
            self._record_metric("circuit_breaker_response_time", execution_time)
    
    async def _record_failure(self, failure_type: FailureType, error_message: str, execution_time: float):
        """Record failed operation."""
        
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            
            if failure_type == FailureType.TIMEOUT:
                self.metrics.timeouts += 1
            
            # Add to failure history
            self.failure_history.append(error_message)
            
            # Record in recent requests
            self.recent_requests.append({
                "timestamp": datetime.now(),
                "success": False,
                "execution_time": execution_time,
                "failure_type": failure_type.value,
                "error": error_message
            })
            
            # State management
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._change_state(CircuitState.OPEN)
                    self.metrics.circuit_open_count += 1
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._change_state(CircuitState.OPEN)
                self.failure_count += 1
                self.success_count = 0
                self.metrics.circuit_open_count += 1
            
            # Update metrics
            self._update_metrics()
            
            # Record monitoring metrics
            self._record_metric("circuit_breaker_failure", 1)
            self._record_metric(f"circuit_breaker_failure_{failure_type.value}", 1)
    
    def _should_attempt_recovery(self) -> bool:
        """Check if circuit should attempt recovery."""
        if self.state != CircuitState.OPEN:
            return False
        
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        
        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
        
        # Record state change metric
        self._record_metric(f"circuit_breaker_state_{new_state.value}", 1)
        
        # Notify callbacks
        for callback in self.on_state_change:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")
    
    def _update_adaptive_timeout(self):
        """Update adaptive timeout based on recent response times."""
        if len(self.response_times) < 5:
            return  # Need minimum data points
        
        try:
            # Calculate 95th percentile of response times
            sorted_times = sorted(self.response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_time = sorted_times[p95_index] / 1000  # Convert to seconds
            
            # Adaptive timeout is 95th percentile + 50% buffer
            new_timeout = p95_time * 1.5
            
            # Clamp to reasonable bounds
            min_timeout = self.config.timeout * 0.5
            max_timeout = self.config.timeout * 3.0
            
            self.adaptive_timeout = max(min_timeout, min(new_timeout, max_timeout))
            
            logger.debug(f"Adaptive timeout updated to {self.adaptive_timeout:.2f}s (95th percentile: {p95_time:.2f}s)")
            
        except Exception as e:
            logger.warning(f"Failed to update adaptive timeout: {e}")
    
    def _update_metrics(self):
        """Update internal metrics."""
        
        # Update current state
        self.metrics.current_state = self.state
        
        # Calculate failure rate
        total_requests = self.metrics.total_requests
        if total_requests > 0:
            self.metrics.failure_rate = self.metrics.failed_requests / total_requests
        else:
            self.metrics.failure_rate = 0.0
        
        # Calculate average response time
        if self.response_times:
            self.metrics.average_response_time = statistics.mean(self.response_times)
        else:
            self.metrics.average_response_time = 0.0
    
    async def _execute_fallback(self, 
                              operation_name: str, 
                              args: tuple, 
                              kwargs: dict, 
                              cache_key: Optional[str]) -> Any:
        """Execute fallback strategy."""
        
        # Try cache first
        if cache_key and cache_key in self.fallback_cache:
            # Check if cache entry is still valid
            if cache_key in self.cache_ttl and datetime.now() < self.cache_ttl[cache_key]:
                logger.info(f"Using cached fallback for '{operation_name}' in circuit '{self.name}'")
                return self.fallback_cache[cache_key]
            else:
                # Remove expired cache entry
                self.fallback_cache.pop(cache_key, None)
                self.cache_ttl.pop(cache_key, None)
        
        # Try registered fallback strategy
        if operation_name in self.fallback_strategies:
            try:
                fallback_func = self.fallback_strategies[operation_name]
                
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(*args, **kwargs)
                else:
                    result = fallback_func(*args, **kwargs)
                
                logger.info(f"Fallback strategy executed for '{operation_name}' in circuit '{self.name}'")
                
                # Cache successful fallback result
                if cache_key and result is not None:
                    self.fallback_cache[cache_key] = result
                    self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=10)
                
                return result
                
            except Exception as e:
                logger.error(f"Fallback strategy failed for '{operation_name}': {e}")
        
        # No fallback available
        logger.warning(f"No fallback available for '{operation_name}' in circuit '{self.name}'")
        return None
    
    def _record_metric(self, metric_name: str, value: Union[int, float]):
        """Record monitoring metric."""
        try:
            record_custom_metric(
                f"{self.name}_{metric_name}",
                value,
                MetricType.COUNTER if isinstance(value, int) else MetricType.GAUGE,
                "count" if isinstance(value, int) else "ms",
                {"circuit": self.name, "component": self.component_type.value}
            )
        except Exception as e:
            logger.debug(f"Failed to record metric {metric_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "failure_rate": self.metrics.failure_rate,
                "average_response_time": self.metrics.average_response_time,
                "adaptive_timeout": self.adaptive_timeout,
                "circuit_open_count": self.metrics.circuit_open_count,
                "cached_fallbacks": len(self.fallback_cache),
                "registered_fallbacks": len(self.fallback_strategies)
            }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.metrics = CircuitBreakerMetrics()
            self.recent_requests.clear()
            self.failure_history.clear()
            self.response_times.clear()
            self.adaptive_timeout = self.config.timeout
            
            logger.info(f"Circuit breaker '{self.name}' reset to initial state")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers with global monitoring."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.global_config = CircuitBreakerConfig()
        
        logger.info("Circuit breaker manager initialized")
    
    def create_circuit_breaker(self, 
                             name: str, 
                             config: Optional[CircuitBreakerConfig] = None,
                             component_type: ComponentType = ComponentType.API_LAYER) -> AdaptiveCircuitBreaker:
        """Create and register a new circuit breaker."""
        
        if name in self.circuit_breakers:
            logger.warning(f"Circuit breaker '{name}' already exists, returning existing instance")
            return self.circuit_breakers[name]
        
        circuit_breaker = AdaptiveCircuitBreaker(
            name=name,
            config=config or self.global_config,
            component_type=component_type
        )
        
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Created circuit breaker '{name}' for {component_type.value}")
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[AdaptiveCircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: circuit.get_status()
            for name, circuit in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for circuit in self.circuit_breakers.values():
            circuit.reset()
        
        logger.info("All circuit breakers reset")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global circuit breaker statistics."""
        
        total_circuits = len(self.circuit_breakers)
        open_circuits = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        half_open_circuits = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN)
        closed_circuits = total_circuits - open_circuits - half_open_circuits
        
        total_requests = sum(cb.metrics.total_requests for cb in self.circuit_breakers.values())
        total_failures = sum(cb.metrics.failed_requests for cb in self.circuit_breakers.values())
        
        overall_failure_rate = (total_failures / total_requests) if total_requests > 0 else 0.0
        
        return {
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "closed_circuits": closed_circuits,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_failure_rate": overall_failure_rate,
            "circuit_names": list(self.circuit_breakers.keys())
        }


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Decorator for circuit breaker protection
def circuit_breaker_protected(circuit_name: str, 
                            config: Optional[CircuitBreakerConfig] = None,
                            component_type: ComponentType = ComponentType.API_LAYER,
                            operation_name: Optional[str] = None,
                            quality_check_func: Optional[Callable] = None,
                            cache_key_func: Optional[Callable] = None):
    """
    Decorator to protect functions with circuit breaker.
    
    Args:
        circuit_name: Name of circuit breaker
        config: Optional circuit breaker configuration
        component_type: Component type for monitoring
        operation_name: Operation name for fallbacks
        quality_check_func: Function to check result quality
        cache_key_func: Function to generate cache key from args
    """
    def decorator(func):
        # Get or create circuit breaker
        circuit_breaker = circuit_breaker_manager.create_circuit_breaker(
            circuit_name, config, component_type
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate operation name and cache key
            op_name = operation_name or func.__name__
            cache_key = None
            if cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Cache key generation failed: {e}")
            
            # Execute through circuit breaker
            result = await circuit_breaker.call(
                func, *args,
                operation_name=op_name,
                quality_check_func=quality_check_func,
                cache_key=cache_key,
                **kwargs
            )
            
            if result.success:
                return result.result
            else:
                raise result.error or Exception("Circuit breaker execution failed")
        
        # For sync functions
        def sync_wrapper(*args, **kwargs):
            # Convert sync function to async for circuit breaker
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Run async wrapper
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions

def create_circuit_breaker(name: str, 
                         config: Optional[CircuitBreakerConfig] = None,
                         component_type: ComponentType = ComponentType.API_LAYER) -> AdaptiveCircuitBreaker:
    """Create a new circuit breaker."""
    return circuit_breaker_manager.create_circuit_breaker(name, config, component_type)


def get_circuit_breaker(name: str) -> Optional[AdaptiveCircuitBreaker]:
    """Get circuit breaker by name."""
    return circuit_breaker_manager.get_circuit_breaker(name)


def get_all_circuit_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return circuit_breaker_manager.get_all_status()


def get_circuit_breaker_stats() -> Dict[str, Any]:
    """Get global circuit breaker statistics."""
    return circuit_breaker_manager.get_global_stats()
