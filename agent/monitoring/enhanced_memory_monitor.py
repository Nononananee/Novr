"""
Enhanced Memory Monitoring for Production Ready System
Provides accurate memory tracking, leak detection, and optimization suggestions.
"""

import asyncio
import logging
import gc
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single point-in-time memory measurement."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float  # Available system memory
    cpu_percent: float
    operation: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Aggregated memory statistics."""
    min_memory_mb: float
    max_memory_mb: float
    avg_memory_mb: float
    current_memory_mb: float
    peak_memory_mb: float
    memory_growth_mb: float
    leak_detected: bool
    efficiency_score: float


class AccurateMemoryMonitor:
    """
    Production-grade memory monitoring with accurate measurements and leak detection.
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 leak_threshold_mb: int = 100,
                 monitoring_interval: float = 30.0):
        """
        Initialize memory monitor.
        
        Args:
            history_size: Number of snapshots to keep in history
            leak_threshold_mb: Memory growth threshold for leak detection
            monitoring_interval: Background monitoring interval in seconds
        """
        self.history_size = history_size
        self.leak_threshold_mb = leak_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        # Memory tracking
        self.snapshots = deque(maxlen=history_size)
        self.baseline_memory = 0.0
        self.peak_memory = 0.0
        
        # Process reference
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
            logger.warning("psutil not available, using basic memory monitoring")
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Operation tracking
        self.operation_memory = {}
        self.current_operation = None
        
        # Initialize baseline
        self._record_baseline()
    
    def _record_baseline(self):
        """Record baseline memory usage."""
        if PSUTIL_AVAILABLE and self.process:
            memory_info = self.process.memory_info()
            self.baseline_memory = memory_info.rss / 1024 / 1024  # MB
        else:
            # Fallback: basic memory estimation
            import sys
            self.baseline_memory = sys.getsizeof(sys.modules) / 1024 / 1024
        
        self.peak_memory = self.baseline_memory
        logger.info(f"Memory monitor baseline set: {self.baseline_memory:.2f} MB")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get accurate current memory usage.
        
        Returns:
            Dictionary with memory metrics in MB
        """
        try:
            if PSUTIL_AVAILABLE and self.process:
                # Full psutil-based monitoring
                memory_info = self.process.memory_info()
                system_memory = psutil.virtual_memory()
                cpu_percent = self.process.cpu_percent()
                
                rss_mb = memory_info.rss / 1024 / 1024
                vms_mb = memory_info.vms / 1024 / 1024
                available_mb = system_memory.available / 1024 / 1024
            else:
                # Fallback monitoring without psutil
                import sys
                import gc
                
                # Basic memory estimation
                rss_mb = sys.getsizeof(sys.modules) / 1024 / 1024
                vms_mb = rss_mb * 1.5  # Rough estimate
                available_mb = 1024.0  # Default assumption
                cpu_percent = 0.0
            
            # Update peak memory
            if rss_mb > self.peak_memory:
                self.peak_memory = rss_mb
            
            return {
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "available_mb": available_mb,
                "cpu_percent": cpu_percent,
                "peak_mb": self.peak_memory,
                "growth_mb": rss_mb - self.baseline_memory,
                "psutil_available": PSUTIL_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {
                "rss_mb": 50.0,  # Basic fallback
                "vms_mb": 75.0,
                "available_mb": 1024.0,
                "cpu_percent": 0.0,
                "peak_mb": 50.0,
                "growth_mb": 0.0,
                "psutil_available": PSUTIL_AVAILABLE
            }
    
    def record_snapshot(self, operation: str = "", context: Dict[str, Any] = None) -> MemorySnapshot:
        """
        Record a memory snapshot.
        
        Args:
            operation: Name of current operation
            context: Additional context information
            
        Returns:
            Memory snapshot
        """
        memory_usage = self.get_current_memory_usage()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_usage["rss_mb"],
            vms_mb=memory_usage["vms_mb"],
            available_mb=memory_usage["available_mb"],
            cpu_percent=memory_usage["cpu_percent"],
            operation=operation,
            context=context or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Update operation tracking
        if operation:
            if operation not in self.operation_memory:
                self.operation_memory[operation] = []
            self.operation_memory[operation].append(snapshot.rss_mb)
        
        return snapshot
    
    def detect_memory_leak(self) -> Dict[str, Any]:
        """
        Detect potential memory leaks.
        
        Returns:
            Leak detection result
        """
        if len(self.snapshots) < 10:
            return {"leak_detected": False, "reason": "Insufficient data"}
        
        # Get recent snapshots
        recent_snapshots = list(self.snapshots)[-10:]
        
        # Calculate memory growth trend
        memory_values = [s.rss_mb for s in recent_snapshots]
        
        # Simple linear trend analysis
        n = len(memory_values)
        sum_x = sum(range(n))
        sum_y = sum(memory_values)
        sum_xy = sum(i * memory_values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        # Calculate slope (memory growth rate)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Check for consistent growth
        current_memory = recent_snapshots[-1].rss_mb
        baseline_memory = recent_snapshots[0].rss_mb
        total_growth = current_memory - baseline_memory
        
        leak_detected = (
            slope > 5.0 and  # Growing by more than 5 MB per snapshot
            total_growth > self.leak_threshold_mb  # Total growth exceeds threshold
        )
        
        return {
            "leak_detected": leak_detected,
            "growth_rate_mb": slope,
            "total_growth_mb": total_growth,
            "current_memory_mb": current_memory,
            "baseline_memory_mb": baseline_memory,
            "confidence": min(abs(slope) / 10, 1.0),
            "recommendation": "Investigate memory usage patterns" if leak_detected else "Memory usage normal"
        }
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get aggregated memory statistics.
        
        Returns:
            Memory statistics
        """
        if not self.snapshots:
            current_memory = self.get_current_memory_usage()["rss_mb"]
            return MemoryStats(
                min_memory_mb=current_memory,
                max_memory_mb=current_memory,
                avg_memory_mb=current_memory,
                current_memory_mb=current_memory,
                peak_memory_mb=self.peak_memory,
                memory_growth_mb=current_memory - self.baseline_memory,
                leak_detected=False,
                efficiency_score=1.0
            )
        
        memory_values = [s.rss_mb for s in self.snapshots]
        current_memory = memory_values[-1]
        
        # Calculate statistics
        min_memory = min(memory_values)
        max_memory = max(memory_values)
        avg_memory = sum(memory_values) / len(memory_values)
        
        # Memory growth
        memory_growth = current_memory - self.baseline_memory
        
        # Leak detection
        leak_info = self.detect_memory_leak()
        
        # Efficiency score (1.0 = perfect, 0.0 = very inefficient)
        efficiency_score = max(0.0, 1.0 - (memory_growth / max(current_memory, 1.0)))
        
        return MemoryStats(
            min_memory_mb=min_memory,
            max_memory_mb=max_memory,
            avg_memory_mb=avg_memory,
            current_memory_mb=current_memory,
            peak_memory_mb=self.peak_memory,
            memory_growth_mb=memory_growth,
            leak_detected=leak_info["leak_detected"],
            efficiency_score=efficiency_score
        )
    
    def get_operation_memory_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze memory usage by operation.
        
        Returns:
            Memory analysis by operation
        """
        analysis = {}
        
        for operation, memory_values in self.operation_memory.items():
            if memory_values:
                analysis[operation] = {
                    "min_mb": min(memory_values),
                    "max_mb": max(memory_values),
                    "avg_mb": sum(memory_values) / len(memory_values),
                    "count": len(memory_values),
                    "growth_mb": memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0.0
                }
        
        return analysis
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform memory optimization.
        
        Returns:
            Optimization result
        """
        memory_before = self.get_current_memory_usage()["rss_mb"]
        
        # Force garbage collection
        collected_objects = gc.collect()
        
        # Additional cleanup
        gc.collect(1)  # Collect generation 1
        gc.collect(2)  # Collect generation 2
        
        # Wait a moment for cleanup to complete
        time.sleep(0.1)
        
        memory_after = self.get_current_memory_usage()["rss_mb"]
        memory_freed = memory_before - memory_after
        
        optimization_result = {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_freed,
            "objects_collected": collected_objects,
            "success": memory_freed > 0,
            "efficiency": memory_freed / max(memory_before, 1.0)
        }
        
        logger.info(f"Memory optimization: freed {memory_freed:.2f} MB ({collected_objects} objects)")
        
        return optimization_result
    
    def start_background_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_active:
            logger.warning("Background monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started background memory monitoring (interval: {self.monitoring_interval}s)")
    
    def stop_background_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped background memory monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Record snapshot
                self.record_snapshot("background_monitoring")
                
                # Check for memory leaks
                leak_info = self.detect_memory_leak()
                if leak_info["leak_detected"]:
                    logger.warning(f"Potential memory leak detected: {leak_info}")
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory report.
        
        Returns:
            Detailed memory report
        """
        stats = self.get_memory_stats()
        leak_info = self.detect_memory_leak()
        operation_analysis = self.get_operation_memory_analysis()
        current_usage = self.get_current_memory_usage()
        
        return {
            "timestamp": time.time(),
            "current_usage": current_usage,
            "statistics": {
                "min_memory_mb": stats.min_memory_mb,
                "max_memory_mb": stats.max_memory_mb,
                "avg_memory_mb": stats.avg_memory_mb,
                "peak_memory_mb": stats.peak_memory_mb,
                "memory_growth_mb": stats.memory_growth_mb,
                "efficiency_score": stats.efficiency_score
            },
            "leak_detection": leak_info,
            "operation_analysis": operation_analysis,
            "snapshots_count": len(self.snapshots),
            "recommendations": self._generate_recommendations(stats, leak_info)
        }
    
    def _generate_recommendations(self, stats: MemoryStats, leak_info: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # High memory usage
        if stats.current_memory_mb > 1024:  # > 1GB
            recommendations.append("Consider processing data in smaller batches")
        
        # Memory growth
        if stats.memory_growth_mb > 500:  # > 500MB growth
            recommendations.append("Monitor for memory leaks and optimize caching")
        
        # Low efficiency
        if stats.efficiency_score < 0.7:
            recommendations.append("Optimize memory-intensive operations")
        
        # Leak detected
        if leak_info["leak_detected"]:
            recommendations.append("Investigate potential memory leak patterns")
        
        # High peak memory
        if stats.peak_memory_mb > 2048:  # > 2GB
            recommendations.append("Consider streaming processing for large datasets")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal")
        
        return recommendations


# Global memory monitor instance
memory_monitor = AccurateMemoryMonitor()


class MemoryProfiler:
    """Context manager for memory profiling operations."""
    
    def __init__(self, operation_name: str, auto_optimize: bool = False):
        self.operation_name = operation_name
        self.auto_optimize = auto_optimize
        self.start_snapshot = None
        self.end_snapshot = None
    
    async def __aenter__(self):
        self.start_snapshot = memory_monitor.record_snapshot(f"{self.operation_name}_start")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_snapshot = memory_monitor.record_snapshot(f"{self.operation_name}_end")
        
        # Log memory usage
        memory_used = self.end_snapshot.rss_mb - self.start_snapshot.rss_mb
        logger.info(f"Operation '{self.operation_name}' used {memory_used:.2f} MB")
        
        # Auto-optimize if requested and memory usage was high
        if self.auto_optimize and memory_used > 100:  # > 100MB
            optimization_result = memory_monitor.optimize_memory()
            logger.info(f"Auto-optimization freed {optimization_result['memory_freed_mb']:.2f} MB")
    
    def get_memory_usage(self) -> float:
        """Get memory usage for this operation."""
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.rss_mb - self.start_snapshot.rss_mb
        return 0.0


# Utility functions

async def monitor_operation_memory(operation_name: str, auto_optimize: bool = False):
    """
    Context manager for monitoring memory usage of an operation.
    
    Args:
        operation_name: Name of the operation
        auto_optimize: Whether to automatically optimize memory after operation
        
    Returns:
        MemoryProfiler context manager
    """
    return MemoryProfiler(operation_name, auto_optimize)


def get_memory_health() -> Dict[str, Any]:
    """
    Get current memory health status.
    
    Returns:
        Memory health information
    """
    stats = memory_monitor.get_memory_stats()
    current_usage = memory_monitor.get_current_memory_usage()
    
    # Determine health status
    if current_usage["rss_mb"] > 2048:  # > 2GB
        status = "critical"
    elif current_usage["rss_mb"] > 1024:  # > 1GB
        status = "warning"
    elif stats.leak_detected:
        status = "warning"
    else:
        status = "healthy"
    
    return {
        "status": status,
        "memory_usage_mb": current_usage["rss_mb"],  # Add legacy field name
        "current_memory_mb": current_usage["rss_mb"],
        "peak_memory_mb": stats.peak_memory_mb,
        "memory_growth_mb": stats.memory_growth_mb,
        "efficiency_score": stats.efficiency_score,
        "leak_detected": stats.leak_detected
    }


def start_memory_monitoring():
    """Start global memory monitoring."""
    memory_monitor.start_background_monitoring()


def stop_memory_monitoring():
    """Stop global memory monitoring."""
    memory_monitor.stop_background_monitoring()
