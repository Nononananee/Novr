"""
Creative performance monitoring and quality metrics for novel writing assistance.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class CreativeMetrics:
    """Metrics for creative operations."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    creative_context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def operation_id(self) -> str:
        """Generate unique operation ID."""
        return f"{self.operation_name}_{int(self.start_time * 1000)}"


@dataclass
class QualityThresholds:
    """Quality thresholds for creative content."""
    character_consistency_min: float = 0.7
    plot_coherence_min: float = 0.6
    emotional_consistency_min: float = 0.65
    style_consistency_min: float = 0.7
    creativity_score_min: float = 0.6
    overall_quality_min: float = 0.65


class CreativePerformanceMonitor:
    """Monitor creative operations and quality metrics."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize creative performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_lock = asyncio.Lock()
        
        # Creative quality metrics
        self.creative_metrics = {
            'character_consistency': deque(maxlen=1000),
            'plot_coherence': deque(maxlen=1000),
            'emotional_consistency': deque(maxlen=1000),
            'style_consistency': deque(maxlen=1000),
            'creativity_score': deque(maxlen=1000)
        }
        
        # Performance counters
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.quality_alerts = deque(maxlen=100)
        
        # Quality thresholds
        self.thresholds = QualityThresholds()
        
        logger.info("Creative performance monitor initialized")
    
    @asynccontextmanager
    async def monitor_creative_operation(
        self, 
        operation_name: str, 
        creative_context: Dict[str, Any] = None
    ):
        """
        Monitor creative operations with quality metrics.
        
        Args:
            operation_name: Name of the creative operation
            creative_context: Context information for the operation
        """
        operation_id = f"creative_{operation_name}_{time.time()}"
        start_time = time.time()
        
        async with self.operation_lock:
            self.active_operations[operation_id] = {
                "name": operation_name,
                "start_time": start_time,
                "creative_context": creative_context or {},
                "type": "creative"
            }
        
        try:
            yield operation_id
            
            # Success - measure creative quality
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Assess creative quality
            quality_metrics = await self._assess_creative_quality(
                operation_name, creative_context or {}
            )
            
            metric = CreativeMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=True,
                quality_metrics=quality_metrics,
                creative_context=creative_context or {}
            )
            
            self._record_creative_metric(metric)
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            metric = CreativeMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                creative_context=creative_context or {}
            )
            
            self._record_creative_metric(metric)
            self.error_counts[operation_name] += 1
            raise
        finally:
            async with self.operation_lock:
                self.active_operations.pop(operation_id, None)
    
    async def _assess_creative_quality(
        self, 
        operation_name: str, 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Assess quality of creative output.
        
        Args:
            operation_name: Name of the operation
            context: Operation context
            
        Returns:
            Dictionary of quality metrics
        """
        # Simulate quality assessment based on operation type
        base_quality = 0.75
        
        quality_metrics = {}
        
        if "character" in operation_name.lower():
            # Character-related operations
            quality_metrics.update({
                'character_consistency': min(base_quality + 0.1, 0.95),
                'character_development_quality': min(base_quality + 0.05, 0.9),
                'dialogue_authenticity': min(base_quality, 0.85)
            })
        
        if "plot" in operation_name.lower():
            # Plot-related operations
            quality_metrics.update({
                'plot_coherence': min(base_quality + 0.05, 0.9),
                'logical_consistency': min(base_quality, 0.85),
                'narrative_flow': min(base_quality + 0.1, 0.9)
            })
        
        if "emotional" in operation_name.lower():
            # Emotional operations
            quality_metrics.update({
                'emotional_consistency': min(base_quality + 0.1, 0.9),
                'emotional_impact': min(base_quality + 0.15, 0.95),
                'emotional_authenticity': min(base_quality, 0.8)
            })
        
        if "scene" in operation_name.lower() or "generate" in operation_name.lower():
            # Scene generation operations
            quality_metrics.update({
                'creativity_score': min(base_quality + 0.1, 0.85),
                'narrative_quality': min(base_quality + 0.05, 0.8),
                'engagement_potential': min(base_quality + 0.2, 0.9)
            })
        
        # Add context-based adjustments
        if context.get('genre') == 'fantasy':
            quality_metrics['world_building_consistency'] = min(base_quality + 0.05, 0.85)
        
        if context.get('emotional_tone'):
            quality_metrics['tone_appropriateness'] = min(base_quality + 0.1, 0.9)
        
        # Overall quality score
        if quality_metrics:
            quality_metrics['overall_quality'] = statistics.mean(quality_metrics.values())
        else:
            quality_metrics['overall_quality'] = base_quality
        
        return quality_metrics
    
    def _record_creative_metric(self, metric: CreativeMetrics):
        """Record creative performance metric."""
        self.metrics_history.append(metric)
        self.operation_counts[metric.operation_name] += 1
        
        # Track creative quality metrics
        if metric.success and metric.quality_metrics:
            for metric_name, score in metric.quality_metrics.items():
                if metric_name in self.creative_metrics:
                    self.creative_metrics[metric_name].append(score)
                
                # Check for quality alerts
                threshold = getattr(self.thresholds, f"{metric_name}_min", 0.6)
                if score < threshold:
                    self._send_creative_alert(
                        f"Low {metric_name} in {metric.operation_name}: {score:.2f} (threshold: {threshold:.2f})"
                    )
    
    def _send_creative_alert(self, message: str):
        """Send creative quality alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "type": "quality_alert"
        }
        self.quality_alerts.append(alert)
        logger.warning(f"Creative Quality Alert: {message}")
    
    def get_creative_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get creative performance report for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Performance report dictionary
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.start_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return {
                "period_hours": hours,
                "total_operations": 0,
                "message": "No creative operations in the specified period"
            }
        
        # Calculate performance statistics
        successful_ops = [m for m in recent_metrics if m.success]
        failed_ops = [m for m in recent_metrics if not m.success]
        
        # Duration statistics
        durations = [m.duration_ms for m in successful_ops]
        avg_duration = statistics.mean(durations) if durations else 0
        
        # Quality statistics
        quality_stats = {}
        for metric_name in self.creative_metrics.keys():
            scores = []
            for m in successful_ops:
                if metric_name in m.quality_metrics:
                    scores.append(m.quality_metrics[metric_name])
            
            if scores:
                quality_stats[metric_name] = {
                    "average": statistics.mean(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Operation breakdown
        operation_breakdown = defaultdict(int)
        for m in recent_metrics:
            operation_breakdown[m.operation_name] += 1
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.quality_alerts
            if datetime.fromisoformat(alert["timestamp"]) >= 
               datetime.now() - timedelta(hours=hours)
        ]
        
        return {
            "period_hours": hours,
            "total_operations": len(recent_metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(recent_metrics) if recent_metrics else 0,
            "average_duration_ms": avg_duration,
            "quality_statistics": quality_stats,
            "operation_breakdown": dict(operation_breakdown),
            "recent_alerts": recent_alerts,
            "quality_trends": self._calculate_quality_trends()
        }
    
    def _calculate_quality_trends(self) -> Dict[str, str]:
        """Calculate quality trends over time."""
        trends = {}
        
        for metric_name, scores in self.creative_metrics.items():
            if len(scores) < 10:
                trends[metric_name] = "insufficient_data"
                continue
            
            # Compare recent vs older scores
            recent_scores = list(scores)[-5:]
            older_scores = list(scores)[-10:-5] if len(scores) >= 10 else list(scores)[:-5]
            
            if not older_scores:
                trends[metric_name] = "stable"
                continue
            
            recent_avg = statistics.mean(recent_scores)
            older_avg = statistics.mean(older_scores)
            
            diff = recent_avg - older_avg
            if diff > 0.05:
                trends[metric_name] = "improving"
            elif diff < -0.05:
                trends[metric_name] = "declining"
            else:
                trends[metric_name] = "stable"
        
        return trends
    
    def get_quality_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for improving creative quality."""
        recommendations = []
        
        # Analyze recent quality metrics
        for metric_name, scores in self.creative_metrics.items():
            if not scores:
                continue
            
            recent_avg = statistics.mean(list(scores)[-10:]) if len(scores) >= 10 else statistics.mean(scores)
            threshold = getattr(self.thresholds, f"{metric_name}_min", 0.6)
            
            if recent_avg < threshold:
                recommendations.append({
                    "metric": metric_name,
                    "current_score": recent_avg,
                    "threshold": threshold,
                    "priority": "high" if recent_avg < threshold - 0.1 else "medium",
                    "recommendation": self._get_metric_recommendation(metric_name)
                })
        
        # Check for error patterns
        high_error_ops = [
            op for op, count in self.error_counts.items() 
            if count > 5
        ]
        
        for op in high_error_ops:
            recommendations.append({
                "metric": "error_rate",
                "operation": op,
                "error_count": self.error_counts[op],
                "priority": "high",
                "recommendation": f"Investigate and fix errors in {op} operation"
            })
        
        return recommendations
    
    def _get_metric_recommendation(self, metric_name: str) -> str:
        """Get specific recommendation for a quality metric."""
        recommendations = {
            'character_consistency': "Review character development guidelines and ensure consistent personality traits across scenes",
            'plot_coherence': "Strengthen logical connections between plot points and ensure cause-and-effect relationships",
            'emotional_consistency': "Maintain consistent emotional tone and ensure character emotions match situations",
            'style_consistency': "Establish and maintain consistent writing style, voice, and narrative perspective",
            'creativity_score': "Explore more creative approaches to scene development and character interactions"
        }
        
        return recommendations.get(metric_name, f"Review and improve {metric_name} in creative operations")
    
    async def optimize_creative_performance(self) -> Dict[str, Any]:
        """Optimize creative performance based on collected metrics."""
        report = self.get_creative_performance_report(hours=168)  # Last week
        recommendations = self.get_quality_recommendations()
        
        # Implement automatic optimizations
        optimizations_applied = []
        
        # Adjust quality thresholds based on performance
        for metric_name, stats in report.get("quality_statistics", {}).items():
            if stats["count"] > 50:  # Enough data points
                current_avg = stats["average"]
                current_threshold = getattr(self.thresholds, f"{metric_name}_min", 0.6)
                
                # If consistently performing above threshold, raise it slightly
                if current_avg > current_threshold + 0.1:
                    new_threshold = min(current_threshold + 0.05, 0.9)
                    setattr(self.thresholds, f"{metric_name}_min", new_threshold)
                    optimizations_applied.append(f"Raised {metric_name} threshold to {new_threshold:.2f}")
                
                # If consistently below threshold, lower it slightly to reduce false alerts
                elif current_avg < current_threshold - 0.1:
                    new_threshold = max(current_threshold - 0.05, 0.4)
                    setattr(self.thresholds, f"{metric_name}_min", new_threshold)
                    optimizations_applied.append(f"Lowered {metric_name} threshold to {new_threshold:.2f}")
        
        return {
            "optimization_timestamp": datetime.now().isoformat(),
            "performance_report": report,
            "recommendations": recommendations,
            "optimizations_applied": optimizations_applied,
            "next_optimization": (datetime.now() + timedelta(hours=24)).isoformat()
        }


# Global creative performance monitor instance
creative_monitor = CreativePerformanceMonitor()


# Decorator for monitoring creative functions
def monitor_creative_quality(operation_name: str):
    """Decorator to monitor creative operation quality."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = kwargs.get('context', {})
            async with creative_monitor.monitor_creative_operation(operation_name, context):
                return await func(*args, **kwargs)
        return wrapper
    return decorator