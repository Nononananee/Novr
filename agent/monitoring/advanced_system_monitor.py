"""
Advanced System Monitor for Novel RAG System
Refactored from large monolithic file into modular components.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any, Union

# Import all components from modular structure
from .components.metrics import (
    AdvancedMetricsCollector,
    MetricType,
    SystemMetric
)

from .components.health import (
    ComponentHealthChecker,
    HealthStatus,
    ComponentType,
    HealthCheckResult
)

from .components.alerts import (
    AlertManager,
    AlertRule,
    SystemAlert
)

from .components.monitor import AdvancedSystemMonitor

logger = logging.getLogger(__name__)

# Global monitor instance
system_monitor = AdvancedSystemMonitor()


# Convenience functions for backward compatibility

async def start_system_monitoring(interval: float = 30.0) -> None:
    """Start the global system monitor."""
    global system_monitor
    system_monitor.monitoring_interval = interval
    await system_monitor.start_monitoring()


async def stop_system_monitoring() -> None:
    """Stop the global system monitor."""
    global system_monitor
    await system_monitor.stop_monitoring()


async def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    global system_monitor
    return await system_monitor.get_system_status()


def record_custom_metric(name: str, 
                        value: Union[int, float], 
                        metric_type: MetricType = MetricType.GAUGE,
                        unit: str = "",
                        tags: Optional[Dict[str, str]] = None) -> None:
    """Record a custom metric."""
    global system_monitor
    system_monitor.metrics_collector.record_metric(name, value, metric_type, unit, tags)


# Context manager for operation monitoring
class MonitoredOperation:
    """Context manager for monitoring specific operations."""
    
    def __init__(self, operation_name: str, component: ComponentType = ComponentType.API_LAYER):
        self.operation_name = operation_name
        self.component = component
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        record_custom_metric(
            f"operation_{self.operation_name}_started",
            1,
            MetricType.COUNTER,
            "count",
            {"operation": self.operation_name, "component": self.component.value}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000
            record_custom_metric(
                f"operation_{self.operation_name}_duration",
                duration,
                MetricType.TIMER,
                "ms",
                {"operation": self.operation_name, "component": self.component.value}
            )
            
            # Record success/failure
            success = exc_type is None
            record_custom_metric(
                f"operation_{self.operation_name}_success",
                1 if success else 0,
                MetricType.COUNTER,
                "count",
                {
                    "operation": self.operation_name, 
                    "component": self.component.value,
                    "status": "success" if success else "failure"
                }
            )


# Decorator for monitoring functions
def monitor_operation(operation_name: str, component: ComponentType = ComponentType.API_LAYER):
    """Decorator to monitor function operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with MonitoredOperation(operation_name, component):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we'll use a simplified approach
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    record_custom_metric(
                        f"operation_{operation_name}_duration",
                        duration,
                        MetricType.TIMER,
                        "ms",
                        {"operation": operation_name, "component": component.value}
                    )
                    record_custom_metric(
                        f"operation_{operation_name}_success",
                        1,
                        MetricType.COUNTER,
                        "count",
                        {"operation": operation_name, "component": component.value, "status": "success"}
                    )
                    return result
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    record_custom_metric(
                        f"operation_{operation_name}_duration",
                        duration,
                        MetricType.TIMER,
                        "ms",
                        {"operation": operation_name, "component": component.value}
                    )
                    record_custom_metric(
                        f"operation_{operation_name}_success",
                        0,
                        MetricType.COUNTER,
                        "count",
                        {"operation": operation_name, "component": component.value, "status": "failure"}
                    )
                    raise
            return sync_wrapper
    return decorator


# Export all important classes and functions for backward compatibility
__all__ = [
    # Core classes
    "AdvancedSystemMonitor",
    "AdvancedMetricsCollector", 
    "ComponentHealthChecker",
    "AlertManager",
    
    # Enums
    "HealthStatus",
    "ComponentType", 
    "MetricType",
    
    # Data classes
    "HealthCheckResult",
    "SystemAlert",
    "AlertRule",
    "SystemMetric",
    
    # Global instance
    "system_monitor",
    
    # Convenience functions
    "start_system_monitoring",
    "stop_system_monitoring", 
    "get_system_health",
    "record_custom_metric",
    
    # Monitoring utilities
    "MonitoredOperation",
    "monitor_operation"
]