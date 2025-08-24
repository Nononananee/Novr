"""
Performance Monitor for Database and Concurrent Operations
Addresses concurrent access bottlenecks and database performance issues.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

# Import enhanced memory monitoring
try:
    from .enhanced_memory_monitor import memory_monitor, MemoryProfiler
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    ENHANCED_MEMORY_AVAILABLE = False
    memory_monitor = None
    MemoryProfiler = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionPoolStats:
    """Database connection pool statistics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    avg_wait_time_ms: float
    max_wait_time_ms: float


class PerformanceMonitor:
    """Monitor system performance and database operations."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.operation_stats = defaultdict(list)
        self.connection_stats = defaultdict(list)
        
        # Concurrent operation tracking
        self.active_operations = {}
        self.operation_lock = threading.RLock()
        
        # Performance thresholds
        self.thresholds = {
            "max_response_time_ms": 5000,
            "max_concurrent_operations": 15,
            "max_db_wait_time_ms": 1000,
            "min_success_rate": 0.95
        }
        
        # Alert tracking
        self.alerts = deque(maxlen=100)
        self.last_alert_time = {}
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager to monitor operation performance."""
        
        operation_id = f"{operation_name}_{time.time()}"
        start_time = time.time()
        
        with self.operation_lock:
            self.active_operations[operation_id] = {
                "name": operation_name,
                "start_time": start_time,
                "metadata": metadata or {}
            }
        
        try:
            yield operation_id
            
            # Success
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=True,
                metadata=metadata or {}
            )
            
            self._record_metric(metric)
            
        except Exception as e:
            # Failure
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata=metadata or {}
            )
            
            self._record_metric(metric)
            raise
            
        finally:
            with self.operation_lock:
                self.active_operations.pop(operation_id, None)
    
    def _record_metric(self, metric: PerformanceMetrics):
        """Record performance metric."""
        
        self.metrics_history.append(metric)
        self.operation_stats[metric.operation_name].append(metric)
        
        # Check for performance alerts
        self._check_performance_alerts(metric)
        
        # Log slow operations
        if metric.duration_ms > self.thresholds["max_response_time_ms"]:
            logger.warning(f"Slow operation detected: {metric.operation_name} took {metric.duration_ms:.2f}ms")
    
    def _check_performance_alerts(self, metric: PerformanceMetrics):
        """Check for performance alert conditions."""
        
        current_time = time.time()
        
        # Slow operation alert
        if metric.duration_ms > self.thresholds["max_response_time_ms"]:
            alert_key = f"slow_{metric.operation_name}"
            if current_time - self.last_alert_time.get(alert_key, 0) > 300:  # 5 minutes
                self._send_alert(
                    "slow_operation",
                    f"Slow operation: {metric.operation_name} took {metric.duration_ms:.2f}ms"
                )
                self.last_alert_time[alert_key] = current_time
        
        # High concurrent operations alert
        with self.operation_lock:
            active_count = len(self.active_operations)
            
        if active_count > self.thresholds["max_concurrent_operations"]:
            alert_key = "high_concurrency"
            if current_time - self.last_alert_time.get(alert_key, 0) > 60:  # 1 minute
                self._send_alert(
                    "high_concurrency",
                    f"High concurrent operations: {active_count} active"
                )
                self.last_alert_time[alert_key] = current_time
        
        # Operation failure alert
        if not metric.success:
            alert_key = f"failure_{metric.operation_name}"
            if current_time - self.last_alert_time.get(alert_key, 0) > 60:  # 1 minute
                self._send_alert(
                    "operation_failure",
                    f"Operation failed: {metric.operation_name} - {metric.error_message}"
                )
                self.last_alert_time[alert_key] = current_time
    
    def _send_alert(self, alert_type: str, message: str):
        """Send performance alert."""
        
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message
        }
        
        self.alerts.append(alert)
        logger.warning(f"PERFORMANCE ALERT [{alert_type}]: {message}")
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.start_time >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Calculate summary statistics
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        success_rate = successful_operations / total_operations
        
        durations = [m.duration_ms for m in recent_metrics]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Operation breakdown
        operation_breakdown = defaultdict(lambda: {"count": 0, "success": 0, "avg_duration": 0})
        
        for metric in recent_metrics:
            op_stats = operation_breakdown[metric.operation_name]
            op_stats["count"] += 1
            if metric.success:
                op_stats["success"] += 1
            op_stats["avg_duration"] = (op_stats["avg_duration"] * (op_stats["count"] - 1) + metric.duration_ms) / op_stats["count"]
        
        # Current active operations
        with self.operation_lock:
            active_ops = len(self.active_operations)
            active_details = [
                {
                    "name": op["name"],
                    "duration_seconds": time.time() - op["start_time"]
                }
                for op in self.active_operations.values()
            ]
        
        return {
            "time_window_minutes": time_window_minutes,
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": success_rate,
                "avg_duration_ms": avg_duration,
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration
            },
            "operation_breakdown": dict(operation_breakdown),
            "current_state": {
                "active_operations": active_ops,
                "active_operation_details": active_details
            },
            "recent_alerts": [
                {
                    "timestamp": alert["timestamp"].isoformat(),
                    "type": alert["type"],
                    "message": alert["message"]
                }
                for alert in list(self.alerts)[-5:]  # Last 5 alerts
            ]
        }
    
    async def monitor_database_pool(self, db_pool) -> ConnectionPoolStats:
        """Monitor database connection pool performance."""
        
        try:
            # Get pool statistics (this would need to be adapted based on your actual pool implementation)
            # For asyncpg pools, you might access internal attributes
            
            # Mock implementation - replace with actual pool stats
            stats = ConnectionPoolStats(
                total_connections=20,  # From your pool config
                active_connections=5,   # Would get from pool._holders
                idle_connections=15,    # Would calculate
                waiting_requests=0,     # Would get from pool._queue
                avg_wait_time_ms=10.0,  # Would calculate from metrics
                max_wait_time_ms=50.0   # Would track
            )
            
            # Record connection stats
            self.connection_stats["pool_stats"].append({
                "timestamp": time.time(),
                "active": stats.active_connections,
                "idle": stats.idle_connections,
                "waiting": stats.waiting_requests
            })
            
            # Alert on high connection usage
            if stats.active_connections > 15:  # 75% of max
                self._send_alert(
                    "high_db_connections",
                    f"High database connection usage: {stats.active_connections}/20"
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error monitoring database pool: {e}")
            return ConnectionPoolStats(0, 0, 0, 0, 0.0, 0.0)
    
    def get_concurrent_operation_limit_recommendation(self) -> Dict[str, Any]:
        """Analyze performance data to recommend concurrent operation limits."""
        
        # Analyze recent performance under different concurrency levels
        recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 operations
        
        # Group by concurrent operation count (approximate)
        concurrency_performance = defaultdict(list)
        
        for metric in recent_metrics:
            # Estimate concurrency based on overlapping operations
            concurrent_ops = sum(
                1 for other in recent_metrics
                if (other.start_time <= metric.start_time <= other.end_time or
                    other.start_time <= metric.end_time <= other.end_time)
            )
            
            concurrency_performance[min(concurrent_ops, 20)].append(metric.duration_ms)
        
        # Calculate performance degradation
        recommendations = {}
        
        for concurrency_level, durations in concurrency_performance.items():
            if len(durations) >= 5:  # Enough data points
                avg_duration = sum(durations) / len(durations)
                success_rate = sum(1 for d in durations if d < self.thresholds["max_response_time_ms"]) / len(durations)
                
                recommendations[concurrency_level] = {
                    "avg_duration_ms": avg_duration,
                    "success_rate": success_rate,
                    "sample_size": len(durations)
                }
        
        # Find optimal concurrency level
        optimal_level = 10  # Default
        for level, stats in recommendations.items():
            if stats["success_rate"] >= self.thresholds["min_success_rate"]:
                optimal_level = max(optimal_level, level)
        
        return {
            "current_threshold": self.thresholds["max_concurrent_operations"],
            "recommended_threshold": optimal_level,
            "performance_by_concurrency": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }


class DatabasePerformanceOptimizer:
    """Optimize database operations for better concurrent performance."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """Initialize database performance optimizer."""
        self.monitor = performance_monitor
        self.query_cache = {}
        self.connection_semaphore = asyncio.Semaphore(15)  # Limit concurrent DB operations
    
    @asynccontextmanager
    async def optimized_db_operation(self, operation_name: str, use_cache: bool = True):
        """Context manager for optimized database operations."""
        
        async with self.connection_semaphore:  # Limit concurrency
            async with self.monitor.monitor_operation(operation_name) as op_id:
                yield op_id
    
    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        base_delay: float = 0.1
    ) -> Any:
        """Execute database operation with retry logic."""
        
        for attempt in range(max_retries):
            try:
                return await operation()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
    
    async def batch_database_operations(
        self,
        operations: List[Callable],
        batch_size: int = 5
    ) -> List[Any]:
        """Execute database operations in batches to reduce load."""
        
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            async with self.optimized_db_operation(f"batch_operation_{i//batch_size}"):
                batch_results = await asyncio.gather(*[op() for op in batch], return_exceptions=True)
                results.extend(batch_results)
                
                # Brief pause between batches
                if i + batch_size < len(operations):
                    await asyncio.sleep(0.01)
        
        return results


# Factory functions
def create_performance_monitor() -> PerformanceMonitor:
    """Create performance monitor with default configuration."""
    return PerformanceMonitor()


def create_db_performance_optimizer(monitor: PerformanceMonitor) -> DatabasePerformanceOptimizer:
    """Create database performance optimizer."""
    return DatabasePerformanceOptimizer(monitor)


# Example usage
async def main():
    """Example usage of performance monitor."""
    
    monitor = create_performance_monitor()
    db_optimizer = create_db_performance_optimizer(monitor)
    
    # Simulate database operations
    async def mock_db_operation(op_id: int):
        await asyncio.sleep(0.01)  # Simulate DB time
        return f"result_{op_id}"
    
    # Test concurrent operations
    operations = [lambda i=i: mock_db_operation(i) for i in range(20)]
    
    async with monitor.monitor_operation("concurrent_test"):
        results = await db_optimizer.batch_database_operations(operations, batch_size=5)
    
    # Get performance summary
    summary = monitor.get_performance_summary(time_window_minutes=1)
    print(f"Performance test completed: {len(results)} operations")
    print(f"Success rate: {summary['summary']['success_rate']:.2%}")
    print(f"Average duration: {summary['summary']['avg_duration_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())