"""Advanced metrics collection and aggregation."""

import time
import logging
import threading
import statistics
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"           # Monotonic increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Time-based measurements
    RATE = "rate"                # Rate per time unit


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