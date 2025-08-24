"""Main system monitor orchestrator."""

import asyncio
import time
import logging
import traceback
import gc
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .metrics import AdvancedMetricsCollector, MetricType
from .health import ComponentHealthChecker, HealthStatus, ComponentType
from .alerts import AlertManager, SystemAlert
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


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
            from ..enhanced_memory_monitor import get_memory_health
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
            from ..error_handling_utils import error_metrics
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