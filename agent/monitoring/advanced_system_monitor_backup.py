"""
Advanced System Monitor for Novel RAG System
Enterprise-grade monitoring with comprehensive health checks, metrics, and alerting.
Designed for production-ready, highly available novel generation systems.
"""

import asyncio
import time
import logging
import threading
import json
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import weakref
import gc
import sys
import os
from pathlib import Path

# Import existing components
from .enhanced_memory_monitor import MemoryProfiler, get_memory_health
from .error_handling_utils import ErrorSeverity, error_metrics
from .performance_monitor import PerformanceMonitor

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


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"           # Monotonic increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Time-based measurements
    RATE = "rate"                # Rate per time unit


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


@dataclass
class SystemMetric:
    """Individual system metric with metadata."""
    name: str
    metric_type: MetricType
    value: Union[int, float, List[float]]
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.9", "< 100", "== 0"
    threshold: Union[int, float]
    severity: HealthStatus
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class SystemAlert:
    """System alert notification."""
    rule_name: str
    component: ComponentType
    severity: HealthStatus
    message: str
    metric_value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class AdvancedMetricsCollector:
    """Advanced metrics collection with aggregation and persistence."""
    
    def __init__(self, max_data_points: int = 10000):
        """Initialize metrics collector."""
        self.max_data_points = max_data_points
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.metric_metadata: Dict[str, SystemMetric] = {}
        self.collection_lock = threading.RLock()
        
        # Aggregation windows
        self.aggregation_windows = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24)
        }
        
        logger.info("Advanced metrics collector initialized")
    
    def record_metric(self, 
                     name: str, 
                     value: Union[int, float, List[float]], 
                     metric_type: MetricType,
                     unit: str = "",
                     tags: Optional[Dict[str, str]] = None,
                     description: str = "") -> None:
        """Record a metric with full metadata."""
        
        timestamp = datetime.now()
        
        metric = SystemMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=timestamp,
            tags=tags or {},
            description=description
        )
        
        with self.collection_lock:
            # Store metric data point
            self.metrics_storage[name].append({
                "timestamp": timestamp,
                "value": value,
                "tags": tags or {}
            })
            
            # Update metadata
            self.metric_metadata[name] = metric
        
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def get_metric_aggregation(self, 
                              name: str, 
                              window: str = "5m",
                              aggregation: str = "avg") -> Optional[Dict[str, Any]]:
        """Get aggregated metric data for a time window."""
        
        if name not in self.metrics_storage:
            return None
        
        if window not in self.aggregation_windows:
            window = "5m"
        
        window_duration = self.aggregation_windows[window]
        cutoff_time = datetime.now() - window_duration
        
        with self.collection_lock:
            # Filter data points within window
            window_data = [
                dp for dp in self.metrics_storage[name]
                if dp["timestamp"] >= cutoff_time
            ]
        
        if not window_data:
            return None
        
        values = [dp["value"] for dp in window_data if isinstance(dp["value"], (int, float))]
        
        if not values:
            return None
        
        # Calculate aggregation
        aggregated_value = None
        if aggregation == "avg":
            aggregated_value = statistics.mean(values)
        elif aggregation == "min":
            aggregated_value = min(values)
        elif aggregation == "max":
            aggregated_value = max(values)
        elif aggregation == "sum":
            aggregated_value = sum(values)
        elif aggregation == "count":
            aggregated_value = len(values)
        elif aggregation == "median":
            aggregated_value = statistics.median(values)
        elif aggregation == "p95":
            aggregated_value = statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]
        elif aggregation == "p99":
            aggregated_value = statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0]
        else:
            aggregated_value = statistics.mean(values)
        
        return {
            "metric_name": name,
            "window": window,
            "aggregation": aggregation,
            "value": aggregated_value,
            "data_points": len(values),
            "timestamp": datetime.now(),
            "window_start": cutoff_time
        }
    
    def get_metric_trend(self, name: str, window: str = "15m") -> Optional[Dict[str, Any]]:
        """Analyze metric trend over time."""
        
        aggregation_5m = self.get_metric_aggregation(name, "5m", "avg")
        aggregation_15m = self.get_metric_aggregation(name, "15m", "avg")
        
        if not aggregation_5m or not aggregation_15m:
            return None
        
        recent_value = aggregation_5m["value"]
        historical_value = aggregation_15m["value"]
        
        if historical_value == 0:
            trend_percent = 0
        else:
            trend_percent = ((recent_value - historical_value) / historical_value) * 100
        
        trend_direction = "stable"
        if trend_percent > 5:
            trend_direction = "increasing"
        elif trend_percent < -5:
            trend_direction = "decreasing"
        
        return {
            "metric_name": name,
            "recent_value": recent_value,
            "historical_value": historical_value,
            "trend_percent": trend_percent,
            "trend_direction": trend_direction,
            "timestamp": datetime.now()
        }


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
                    'agent.enhanced_generation_pipeline',
                    'agent.enhanced_context_optimizer',
                    'memory.enhanced_chunking_strategies'
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


class AlertManager:
    """Advanced alerting system with rules, cooldowns, and notifications."""
    
    def __init__(self, metrics_collector: AdvancedMetricsCollector):
        """Initialize alert manager."""
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.notification_handlers: List[Callable] = []
        
        # Default alert rules
        self._create_default_alert_rules()
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_handler(self, handler: Callable[[SystemAlert], None]) -> None:
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    async def evaluate_alerts(self) -> List[SystemAlert]:
        """Evaluate all alert rules and trigger notifications."""
        
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                cooldown_expires = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_expires:
                    continue
            
            # Get current metric value
            aggregation = self.metrics_collector.get_metric_aggregation(
                rule.metric_name, "1m", "avg"
            )
            
            if not aggregation:
                continue
            
            current_value = aggregation["value"]
            
            # Evaluate condition
            alert_triggered = self._evaluate_condition(current_value, rule.condition, rule.threshold)
            
            if alert_triggered:
                # Create alert
                alert = SystemAlert(
                    rule_name=rule_name,
                    component=ComponentType.MEMORY_SYSTEM,  # Default, should be configurable
                    severity=rule.severity,
                    message=f"Alert: {rule.metric_name} {rule.condition} {rule.threshold}, current: {current_value}",
                    metric_value=current_value,
                    threshold=rule.threshold,
                    timestamp=datetime.now()
                )
                
                # Add to active alerts
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                new_alerts.append(alert)
                
                # Update last triggered
                rule.last_triggered = datetime.now()
                
                # Send notifications
                await self._send_notifications(alert)
                
                logger.warning(f"Alert triggered: {alert.message}")
            
            else:
                # Check if alert should be resolved
                if rule_name in self.active_alerts:
                    alert = self.active_alerts[rule_name]
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    del self.active_alerts[rule_name]
                    
                    logger.info(f"Alert resolved: {rule_name}")
        
        return new_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == ">":
                return value > threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<":
                return value < threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 0.001  # Float equality
            elif condition == "!=":
                return abs(value - threshold) >= 0.001
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _send_notifications(self, alert: SystemAlert) -> None:
        """Send alert notifications to all handlers."""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def _create_default_alert_rules(self):
        """Create default alert rules for common issues."""
        
        # High memory usage alert
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="memory_usage_mb",
            condition=">",
            threshold=1500.0,  # 1.5GB
            severity=HealthStatus.WARNING,
            cooldown_minutes=5
        ))
        
        # Critical memory usage alert
        self.add_alert_rule(AlertRule(
            name="critical_memory_usage",
            metric_name="memory_usage_mb",
            condition=">",
            threshold=3000.0,  # 3GB
            severity=HealthStatus.CRITICAL,
            cooldown_minutes=2
        ))
        
        # Error rate alert
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="error_rate_per_minute",
            condition=">",
            threshold=10.0,  # 10 errors per minute
            severity=HealthStatus.WARNING,
            cooldown_minutes=3
        ))


class AdvancedSystemMonitor:
    """
    Advanced system monitor that orchestrates all monitoring components.
    Enterprise-grade monitoring for novel RAG systems.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 enable_background_monitoring: bool = True):
        """Initialize advanced system monitor."""
        
        self.monitoring_interval = monitoring_interval
        self.enable_background_monitoring = enable_background_monitoring
        
        # Initialize components
        self.metrics_collector = AdvancedMetricsCollector()
        self.health_checker = ComponentHealthChecker(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.system_start_time = datetime.now()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        
        # Setup default notification handler
        self.alert_manager.add_notification_handler(self._default_alert_handler)
        
        logger.info(f"Advanced system monitor initialized (interval: {monitoring_interval}s)")
    
    async def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        if self.enable_background_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Background monitoring loop started")
        
        try:
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    await self._collect_system_metrics()
                    
                    # Run health checks
                    await self._run_health_checks()
                    
                    # Evaluate alerts
                    await self.alert_manager.evaluate_alerts()
                    
                    # Cleanup old data
                    await self._cleanup_old_data()
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    traceback.print_exc()
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Monitoring loop crashed: {e}")
            raise
    
    async def _collect_system_metrics(self) -> None:
        """Collect various system metrics."""
        
        # Memory metrics
        try:
            memory_status = get_memory_health()
            self.metrics_collector.record_metric(
                "memory_usage_mb",
                memory_status["memory_usage_mb"],
                MetricType.GAUGE,
                "MB",
                description="Current memory usage"
            )
            
            self.metrics_collector.record_metric(
                "memory_baseline_mb",
                memory_status.get("baseline_memory_mb", 0),
                MetricType.GAUGE,
                "MB",
                description="Baseline memory usage"
            )
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
        
        # Error metrics
        try:
            error_stats = error_metrics.get_metrics()
            
            # Calculate error rate
            total_errors = sum(error_stats.get("error_counts", {}).values())
            self.metrics_collector.record_metric(
                "total_errors",
                total_errors,
                MetricType.COUNTER,
                "count",
                description="Total error count"
            )
            
            # Error rate per minute (simplified)
            error_rate_metric = self.metrics_collector.get_metric_aggregation("total_errors", "1m", "count")
            if error_rate_metric:
                self.metrics_collector.record_metric(
                    "error_rate_per_minute",
                    error_rate_metric["value"],
                    MetricType.RATE,
                    "errors/min",
                    description="Error rate per minute"
                )
        except Exception as e:
            logger.error(f"Failed to collect error metrics: {e}")
        
        # System uptime
        uptime_seconds = (datetime.now() - self.system_start_time).total_seconds()
        self.metrics_collector.record_metric(
            "system_uptime_seconds",
            uptime_seconds,
            MetricType.GAUGE,
            "seconds",
            description="System uptime"
        )
        
        # Garbage collection metrics
        try:
            gc_stats = gc.get_stats()
            if gc_stats:
                self.metrics_collector.record_metric(
                    "gc_collections",
                    sum(stat['collections'] for stat in gc_stats),
                    MetricType.COUNTER,
                    "count",
                    description="Total garbage collections"
                )
                
                self.metrics_collector.record_metric(
                    "gc_collected_objects",
                    sum(stat['collected'] for stat in gc_stats),
                    MetricType.COUNTER,
                    "count",
                    description="Total objects collected by GC"
                )
        except Exception as e:
            logger.error(f"Failed to collect GC metrics: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        try:
            health_results = await self.health_checker.check_all_components()
            
            # Record health check metrics
            for component, result in health_results.items():
                self.metrics_collector.record_metric(
                    f"health_status_{component.value}",
                    1 if result.status == HealthStatus.HEALTHY else 0,
                    MetricType.GAUGE,
                    "status",
                    {"component": component.value, "status": result.status.value},
                    f"Health status for {component.value}"
                )
                
                self.metrics_collector.record_metric(
                    f"health_check_duration_{component.value}",
                    result.execution_time_ms,
                    MetricType.TIMER,
                    "ms",
                    {"component": component.value},
                    f"Health check duration for {component.value}"
                )
        except Exception as e:
            logger.error(f"Failed to run health checks: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        try:
            # Limit alert history
            max_alert_history = 1000
            if len(self.alert_manager.alert_history) > max_alert_history:
                self.alert_manager.alert_history = self.alert_manager.alert_history[-max_alert_history:]
            
            # The metrics collector already limits data points via deque maxlen
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def _default_alert_handler(self, alert: SystemAlert) -> None:
        """Default alert notification handler."""
        severity_emoji = {
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🚨",
            HealthStatus.DEGRADED: "📉"
        }
        
        emoji = severity_emoji.get(alert.severity, "🔔")
        
        logger.warning(
            f"{emoji} ALERT [{alert.severity.value.upper()}] {alert.rule_name}: "
            f"{alert.message} (Component: {alert.component.value})"
        )
    
    # Public API methods
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Get latest health check results
        health_results = {}
        for component, result in self.health_checker.last_health_results.items():
            health_results[component.value] = {
                "status": result.status.value,
                "message": result.message,
                "last_check": result.timestamp.isoformat(),
                "execution_time_ms": result.execution_time_ms
            }
        
        # Get key metrics
        memory_metric = self.metrics_collector.get_metric_aggregation("memory_usage_mb", "1m", "avg")
        uptime_metric = self.metrics_collector.get_metric_aggregation("system_uptime_seconds", "1m", "avg")
        error_rate_metric = self.metrics_collector.get_metric_aggregation("error_rate_per_minute", "5m", "avg")
        
        # Overall system health
        healthy_components = sum(
            1 for result in self.health_checker.last_health_results.values()
            if result.status == HealthStatus.HEALTHY
        )
        total_components = len(self.health_checker.last_health_results)
        
        if total_components == 0:
            overall_health = HealthStatus.UNKNOWN
        elif healthy_components == total_components:
            overall_health = HealthStatus.HEALTHY
        elif healthy_components >= total_components * 0.7:
            overall_health = HealthStatus.WARNING
        else:
            overall_health = HealthStatus.CRITICAL
        
        return {
            "overall_health": overall_health.value,
            "monitoring_active": self.monitoring_active,
            "system_start_time": self.system_start_time.isoformat(),
            "components": health_results,
            "active_alerts": len(self.alert_manager.active_alerts),
            "total_alerts_history": len(self.alert_manager.alert_history),
            "metrics": {
                "memory_usage_mb": memory_metric["value"] if memory_metric else None,
                "uptime_hours": (uptime_metric["value"] / 3600) if uptime_metric else None,
                "error_rate_per_minute": error_rate_metric["value"] if error_rate_metric else None
            },
            "component_summary": {
                "total": total_components,
                "healthy": healthy_components,
                "unhealthy": total_components - healthy_components
            }
        }
    
    def get_metrics_summary(self, window: str = "15m") -> Dict[str, Any]:
        """Get summary of key metrics."""
        
        key_metrics = [
            "memory_usage_mb",
            "system_uptime_seconds", 
            "error_rate_per_minute",
            "total_errors"
        ]
        
        summary = {}
        
        for metric_name in key_metrics:
            aggregation = self.metrics_collector.get_metric_aggregation(metric_name, window, "avg")
            trend = self.metrics_collector.get_metric_trend(metric_name, window)
            
            summary[metric_name] = {
                "current_value": aggregation["value"] if aggregation else None,
                "trend": trend["trend_direction"] if trend else None,
                "trend_percent": trend["trend_percent"] if trend else None
            }
        
        return summary
    
    async def force_health_check(self) -> Dict[str, Any]:
        """Force immediate health check of all components."""
        results = await self.health_checker.check_all_components()
        
        return {
            component.value: {
                "status": result.status.value,
                "message": result.message,
                "execution_time_ms": result.execution_time_ms,
                "recommendations": result.recommendations
            }
            for component, result in results.items()
        }


# Global monitor instance
system_monitor = AdvancedSystemMonitor()


# Convenience functions

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
