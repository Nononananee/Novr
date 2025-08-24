"""Component health checking system."""

import asyncio
import time
import logging
import traceback
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

from .metrics import AdvancedMetricsCollector, MetricType

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""
    GENERATION_PIPELINE = "generation_pipeline"
    CONTEXT_OPTIMIZER = "context_optimizer"
    CHUNKING_SYSTEM = "chunking_system"
    MEMORY_SYSTEM = "memory_system"
    DATABASE = "database"
    EMBEDDING_SYSTEM = "embedding_system"
    API_LAYER = "api_layer"
    GRAPH_DATABASE = "graph_database"
    CACHE_SYSTEM = "cache_system"
    FILE_SYSTEM = "file_system"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    execution_time_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    error_details: Optional[str] = None


class ComponentHealthChecker:
    """Health checker for individual system components."""
    
    def __init__(self, metrics_collector: AdvancedMetricsCollector):
        """Initialize component health checker."""
        self.metrics_collector = metrics_collector
        self.health_check_registry: Dict[ComponentType, Callable] = {}
        self.last_health_results: Dict[ComponentType, HealthCheckResult] = {}
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("Component health checker initialized")
    
    def register_health_check(self, 
                            component: ComponentType, 
                            health_check_func: Callable) -> None:
        """Register a health check function for a component."""
        self.health_check_registry[component] = health_check_func
        logger.info(f"Registered health check for {component.value}")
    
    async def check_component_health(self, component: ComponentType) -> HealthCheckResult:
        """Check health of a specific component."""
        
        start_time = time.time()
        
        if component not in self.health_check_registry:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {component.value}",
                timestamp=datetime.now(),
                execution_time_ms=0,
                recommendations=["Register a health check function for this component"]
            )
        
        try:
            health_check_func = self.health_check_registry[component]
            
            # Execute health check with timeout
            result = await asyncio.wait_for(
                health_check_func(),
                timeout=30.0  # 30 second timeout
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Ensure result is properly formatted
            if not isinstance(result, HealthCheckResult):
                result = HealthCheckResult(
                    component=component,
                    status=HealthStatus.HEALTHY if result else HealthStatus.CRITICAL,
                    message=str(result),
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time
                )
            else:
                result.execution_time_ms = execution_time
                result.timestamp = datetime.now()
            
            # Record health check metrics
            self.metrics_collector.record_metric(
                f"health_check_{component.value}_duration",
                execution_time,
                MetricType.TIMER,
                "ms",
                {"component": component.value}
            )
            
            self.metrics_collector.record_metric(
                f"health_check_{component.value}_status",
                1 if result.status == HealthStatus.HEALTHY else 0,
                MetricType.GAUGE,
                "status",
                {"component": component.value, "status": result.status.value}
            )
            
            # Cache result
            self.last_health_results[component] = result
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                component=component,
                status=HealthStatus.CRITICAL,
                message=f"Health check timeout after 30 seconds",
                timestamp=datetime.now(),
                execution_time_ms=execution_time,
                error_details="Health check function timed out",
                recommendations=["Investigate component performance", "Check for blocking operations"]
            )
            
            self.last_health_results[component] = result
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                component=component,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                execution_time_ms=execution_time,
                error_details=traceback.format_exc(),
                recommendations=["Check component logs", "Verify component dependencies"]
            )
            
            self.last_health_results[component] = result
            return result
    
    async def check_all_components(self) -> Dict[ComponentType, HealthCheckResult]:
        """Check health of all registered components."""
        
        tasks = []
        for component in self.health_check_registry.keys():
            task = asyncio.create_task(self.check_component_health(component))
            tasks.append((component, task))
        
        results = {}
        for component, task in tasks:
            try:
                result = await task
                results[component] = result
            except Exception as e:
                results[component] = HealthCheckResult(
                    component=component,
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to execute health check: {str(e)}",
                    timestamp=datetime.now(),
                    execution_time_ms=0,
                    error_details=str(e)
                )
        
        return results
    
    def _register_default_health_checks(self):
        """Register default health checks for core components."""
        
        # Memory System Health Check
        async def memory_health_check():
            try:
                from ..enhanced_memory_monitor import get_memory_health
                memory_status = get_memory_health()
                
                if memory_status["memory_usage_mb"] > 2000:  # 2GB threshold
                    return HealthCheckResult(
                        component=ComponentType.MEMORY_SYSTEM,
                        status=HealthStatus.WARNING,
                        message=f"High memory usage: {memory_status['memory_usage_mb']:.1f}MB",
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        metrics=memory_status,
                        recommendations=["Monitor memory usage", "Consider memory optimization"]
                    )
                elif memory_status["memory_usage_mb"] > 4000:  # 4GB critical
                    return HealthCheckResult(
                        component=ComponentType.MEMORY_SYSTEM,
                        status=HealthStatus.CRITICAL,
                        message=f"Critical memory usage: {memory_status['memory_usage_mb']:.1f}MB",
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        metrics=memory_status,
                        recommendations=["Immediate memory cleanup required", "Restart service if necessary"]
                    )
                else:
                    return HealthCheckResult(
                        component=ComponentType.MEMORY_SYSTEM,
                        status=HealthStatus.HEALTHY,
                        message=f"Memory usage normal: {memory_status['memory_usage_mb']:.1f}MB",
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        metrics=memory_status
                    )
                    
            except Exception as e:
                return HealthCheckResult(
                    component=ComponentType.MEMORY_SYSTEM,
                    status=HealthStatus.CRITICAL,
                    message=f"Memory health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    execution_time_ms=0,
                    error_details=str(e)
                )
        
        # File System Health Check
        async def filesystem_health_check():
            try:
                # Check available disk space
                workspace_path = Path.cwd()
                stat = os.statvfs(workspace_path) if hasattr(os, 'statvfs') else None
                
                if stat:
                    # Calculate available space in GB
                    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
                    usage_percent = ((total_gb - available_gb) / total_gb) * 100
                    
                    if usage_percent > 90:
                        status = HealthStatus.CRITICAL
                        message = f"Critical disk usage: {usage_percent:.1f}% ({available_gb:.1f}GB available)"
                        recommendations = ["Free up disk space immediately", "Archive old files"]
                    elif usage_percent > 80:
                        status = HealthStatus.WARNING
                        message = f"High disk usage: {usage_percent:.1f}% ({available_gb:.1f}GB available)"
                        recommendations = ["Monitor disk usage", "Clean up temporary files"]
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Disk usage normal: {usage_percent:.1f}% ({available_gb:.1f}GB available)"
                        recommendations = []
                    
                    return HealthCheckResult(
                        component=ComponentType.FILE_SYSTEM,
                        status=status,
                        message=message,
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        metrics={
                            "available_gb": available_gb,
                            "total_gb": total_gb,
                            "usage_percent": usage_percent
                        },
                        recommendations=recommendations
                    )
                else:
                    # Fallback for Windows or systems without statvfs
                    return HealthCheckResult(
                        component=ComponentType.FILE_SYSTEM,
                        status=HealthStatus.HEALTHY,
                        message="File system accessible",
                        timestamp=datetime.now(),
                        execution_time_ms=0
                    )
                    
            except Exception as e:
                return HealthCheckResult(
                    component=ComponentType.FILE_SYSTEM,
                    status=HealthStatus.CRITICAL,
                    message=f"File system health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    execution_time_ms=0,
                    error_details=str(e)
                )
        
        # API Layer Health Check
        async def api_health_check():
            try:
                # Check if critical modules are importable
                critical_modules = [
                    'agent.api',
                    'agent.pipeline.enhanced_generation_pipeline',
                    'agent.optimization.enhanced_context_optimizer'
                ]
                
                import_failures = []
                for module_name in critical_modules:
                    try:
                        __import__(module_name)
                    except ImportError as e:
                        import_failures.append(f"{module_name}: {str(e)}")
                
                if import_failures:
                    return HealthCheckResult(
                        component=ComponentType.API_LAYER,
                        status=HealthStatus.CRITICAL,
                        message=f"Module import failures: {len(import_failures)}",
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        error_details="; ".join(import_failures),
                        recommendations=["Check module dependencies", "Verify installation"]
                    )
                else:
                    return HealthCheckResult(
                        component=ComponentType.API_LAYER,
                        status=HealthStatus.HEALTHY,
                        message="All critical modules importable",
                        timestamp=datetime.now(),
                        execution_time_ms=0,
                        metrics={"modules_checked": len(critical_modules)}
                    )
                    
            except Exception as e:
                return HealthCheckResult(
                    component=ComponentType.API_LAYER,
                    status=HealthStatus.CRITICAL,
                    message=f"API health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    execution_time_ms=0,
                    error_details=str(e)
                )
        
        # Register the health checks
        self.register_health_check(ComponentType.MEMORY_SYSTEM, memory_health_check)
        self.register_health_check(ComponentType.FILE_SYSTEM, filesystem_health_check)
        self.register_health_check(ComponentType.API_LAYER, api_health_check)