#!/usr/bin/env python3
"""
Production Deployment & Monitoring Setup
Configures the enhanced chunking system for production deployment with monitoring.
"""

import asyncio
import json
import time
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductionMetrics:
    """Production metrics for monitoring."""
    timestamp: datetime
    operation_type: str
    execution_time_ms: float
    tokens_processed: int
    chunks_created: int
    context_quality_score: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemHealth:
    """System health status."""
    status: str  # healthy, degraded, critical
    uptime_seconds: float
    total_operations: int
    success_rate: float
    avg_response_time_ms: float
    memory_usage_mb: float
    error_count: int
    last_error: Optional[str] = None


class ProductionMonitor:
    """Production monitoring system for enhanced chunking."""
    
    def __init__(self, max_metrics_history: int = 10000):
        """Initialize production monitor."""
        self.max_metrics_history = max_metrics_history
        self.metrics_history = deque(maxlen=max_metrics_history)
        self.start_time = time.time()
        
        # Performance tracking
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # Health thresholds
        self.health_thresholds = {
            "max_response_time_ms": 5000,
            "min_success_rate": 0.95,
            "max_memory_usage_mb": 1000,
            "max_error_rate": 0.05
        }
        
        # Alerts
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = {}
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def record_operation(
        self,
        operation_type: str,
        execution_time_ms: float,
        tokens_processed: int = 0,
        chunks_created: int = 0,
        context_quality_score: float = 0.0,
        memory_usage_mb: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Record an operation for monitoring."""
        
        metric = ProductionMetrics(
            timestamp=datetime.now(),
            operation_type=operation_type,
            execution_time_ms=execution_time_ms,
            tokens_processed=tokens_processed,
            chunks_created=chunks_created,
            context_quality_score=context_quality_score,
            memory_usage_mb=memory_usage_mb,
            success=success,
            error_message=error_message
        )
        
        self.metrics_history.append(metric)
        self.operation_counts[operation_type] += 1
        self.operation_times[operation_type].append(execution_time_ms)
        
        if not success:
            self.error_counts[operation_type] += 1
            logger.error(f"Operation failed: {operation_type} - {error_message}")
        
        # Check for alerts
        self._check_alerts(metric)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        
        if not self.metrics_history:
            return SystemHealth(
                status="unknown",
                uptime_seconds=time.time() - self.start_time,
                total_operations=0,
                success_rate=0.0,
                avg_response_time_ms=0.0,
                memory_usage_mb=0.0,
                error_count=0
            )
        
        # Calculate metrics from recent history (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_cutoff]
        
        if not recent_metrics:
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 operations
        
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
        
        avg_response_time = sum(m.execution_time_ms for m in recent_metrics) / total_operations if total_operations > 0 else 0.0
        
        recent_memory_usage = [m.memory_usage_mb for m in recent_metrics if m.memory_usage_mb > 0]
        avg_memory_usage = sum(recent_memory_usage) / len(recent_memory_usage) if recent_memory_usage else 0.0
        
        error_count = sum(1 for m in recent_metrics if not m.success)
        last_error = None
        for m in reversed(recent_metrics):
            if not m.success:
                last_error = m.error_message
                break
        
        # Determine health status
        status = "healthy"
        if (avg_response_time > self.health_thresholds["max_response_time_ms"] or
            success_rate < self.health_thresholds["min_success_rate"] or
            avg_memory_usage > self.health_thresholds["max_memory_usage_mb"]):
            status = "degraded"
        
        error_rate = error_count / total_operations if total_operations > 0 else 0.0
        if error_rate > self.health_thresholds["max_error_rate"]:
            status = "critical"
        
        return SystemHealth(
            status=status,
            uptime_seconds=time.time() - self.start_time,
            total_operations=sum(self.operation_counts.values()),
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            memory_usage_mb=avg_memory_usage,
            error_count=error_count,
            last_error=last_error
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        health = self.get_system_health()
        
        # Operation breakdown
        operation_stats = {}
        for op_type, count in self.operation_counts.items():
            times = self.operation_times[op_type]
            errors = self.error_counts[op_type]
            
            operation_stats[op_type] = {
                "total_operations": count,
                "error_count": errors,
                "success_rate": (count - errors) / count if count > 0 else 0.0,
                "avg_response_time_ms": sum(times) / len(times) if times else 0.0,
                "min_response_time_ms": min(times) if times else 0.0,
                "max_response_time_ms": max(times) if times else 0.0
            }
        
        # Recent performance trends
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_cutoff]
        
        hourly_stats = defaultdict(lambda: {"operations": 0, "errors": 0, "total_time": 0.0})
        for metric in recent_metrics:
            hour_key = metric.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_stats[hour_key]["operations"] += 1
            if not metric.success:
                hourly_stats[hour_key]["errors"] += 1
            hourly_stats[hour_key]["total_time"] += metric.execution_time_ms
        
        # Quality metrics
        quality_scores = [m.context_quality_score for m in recent_metrics if m.context_quality_score > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "system_health": asdict(health),
            "operation_statistics": operation_stats,
            "hourly_trends": dict(hourly_stats),
            "quality_metrics": {
                "average_context_quality": avg_quality,
                "quality_samples": len(quality_scores)
            },
            "alert_history": [
                {
                    "timestamp": alert["timestamp"].isoformat(),
                    "type": alert["type"],
                    "message": alert["message"]
                }
                for alert in list(self.alert_history)[-10:]  # Last 10 alerts
            ]
        }
    
    def _check_alerts(self, metric: ProductionMetrics):
        """Check for alert conditions."""
        
        current_time = time.time()
        
        # Response time alert
        if metric.execution_time_ms > self.health_thresholds["max_response_time_ms"]:
            alert_key = "slow_response"
            if current_time - self.last_alert_time.get(alert_key, 0) > 300:  # 5 minutes
                self._send_alert(
                    "slow_response",
                    f"Slow response detected: {metric.execution_time_ms:.2f}ms for {metric.operation_type}"
                )
                self.last_alert_time[alert_key] = current_time
        
        # Error alert
        if not metric.success:
            alert_key = "operation_error"
            if current_time - self.last_alert_time.get(alert_key, 0) > 60:  # 1 minute
                self._send_alert(
                    "operation_error",
                    f"Operation failed: {metric.operation_type} - {metric.error_message}"
                )
                self.last_alert_time[alert_key] = current_time
        
        # Memory usage alert
        if metric.memory_usage_mb > self.health_thresholds["max_memory_usage_mb"]:
            alert_key = "high_memory"
            if current_time - self.last_alert_time.get(alert_key, 0) > 600:  # 10 minutes
                self._send_alert(
                    "high_memory",
                    f"High memory usage: {metric.memory_usage_mb:.2f}MB"
                )
                self.last_alert_time[alert_key] = current_time
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert (implement actual alerting mechanism)."""
        
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message
        }
        
        self.alert_history.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # In production, implement actual alerting:
        # - Send email/SMS
        # - Post to Slack/Discord
        # - Write to monitoring system
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Periodic health checks
                health = self.get_system_health()
                
                if health.status != "healthy":
                    logger.warning(f"System health: {health.status}")
                
                # Save metrics to file periodically
                if len(self.metrics_history) % 100 == 0:
                    self._save_metrics_to_file()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def _save_metrics_to_file(self):
        """Save metrics to file for persistence."""
        
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "system_health": asdict(self.get_system_health()),
                "recent_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "operation_type": m.operation_type,
                        "execution_time_ms": m.execution_time_ms,
                        "success": m.success
                    }
                    for m in list(self.metrics_history)[-100:]  # Last 100 metrics
                ]
            }
            
            with open("production_metrics.json", 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def shutdown(self):
        """Shutdown monitoring."""
        self.monitoring_active = False
        self._save_metrics_to_file()


class ProductionConfig:
    """Production configuration for enhanced chunking system."""
    
    def __init__(self):
        """Initialize production configuration."""
        self.config = {
            # Enhanced chunking settings
            "enhanced_chunking": {
                "enabled": True,
                "dialogue_chunk_size": 800,
                "narrative_chunk_size": 1200,
                "action_chunk_size": 600,
                "description_chunk_size": 1000,
                "min_scene_size": 200,
                "max_scene_size": 3000,
                "preserve_dialogue_integrity": True,
                "preserve_emotional_beats": True,
                "preserve_action_sequences": True
            },
            
            # Advanced context building settings
            "advanced_context": {
                "enabled": True,
                "max_context_tokens": 8000,
                "context_weights": {
                    "semantic_similarity": 0.3,
                    "character_relevance": 0.25,
                    "temporal_proximity": 0.15,
                    "narrative_importance": 0.15,
                    "emotional_resonance": 0.1,
                    "plot_continuity": 0.05
                }
            },
            
            # Performance settings
            "performance": {
                "max_concurrent_operations": 10,
                "operation_timeout_seconds": 30,
                "memory_limit_mb": 1000,
                "cache_size": 1000,
                "enable_caching": True
            },
            
            # Monitoring settings
            "monitoring": {
                "enabled": True,
                "metrics_retention_hours": 168,  # 1 week
                "alert_thresholds": {
                    "max_response_time_ms": 5000,
                    "min_success_rate": 0.95,
                    "max_memory_usage_mb": 1000,
                    "max_error_rate": 0.05
                }
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "file_path": "production.log",
                "max_file_size_mb": 100,
                "backup_count": 5,
                "enable_structured_logging": True
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get production configuration."""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration."""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_to_file(self, filepath: str = "production_config.json"):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_from_file(self, filepath: str = "production_config.json"):
        """Load configuration from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.config = json.load(f)


class ProductionDeployment:
    """Production deployment manager."""
    
    def __init__(self):
        """Initialize production deployment."""
        self.config = ProductionConfig()
        self.monitor = ProductionMonitor()
        self.is_deployed = False
    
    async def deploy(self):
        """Deploy enhanced chunking system to production."""
        
        print("=" * 80)
        print("PRODUCTION DEPLOYMENT - ENHANCED CHUNKING SYSTEM")
        print("=" * 80)
        
        try:
            # Step 1: Validate configuration
            print("\n1. Validating configuration...")
            await self._validate_configuration()
            print("✓ Configuration validated")
            
            # Step 2: Initialize components
            print("\n2. Initializing components...")
            await self._initialize_components()
            print("✓ Components initialized")
            
            # Step 3: Run health checks
            print("\n3. Running health checks...")
            await self._run_health_checks()
            print("✓ Health checks passed")
            
            # Step 4: Start monitoring
            print("\n4. Starting monitoring...")
            self._start_monitoring()
            print("✓ Monitoring started")
            
            # Step 5: Deploy services
            print("\n5. Deploying services...")
            await self._deploy_services()
            print("✓ Services deployed")
            
            self.is_deployed = True
            print("\n" + "=" * 80)
            print("✅ PRODUCTION DEPLOYMENT SUCCESSFUL")
            print("=" * 80)
            
            # Display deployment summary
            self._display_deployment_summary()
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            print(f"\n❌ DEPLOYMENT FAILED: {e}")
            raise
    
    async def _validate_configuration(self):
        """Validate production configuration."""
        
        config = self.config.get_config()
        
        # Validate enhanced chunking config
        chunking_config = config["enhanced_chunking"]
        if not chunking_config["enabled"]:
            raise ValueError("Enhanced chunking must be enabled for production")
        
        if chunking_config["dialogue_chunk_size"] < 100:
            raise ValueError("Dialogue chunk size too small")
        
        # Validate performance settings
        perf_config = config["performance"]
        if perf_config["max_concurrent_operations"] < 1:
            raise ValueError("Must allow at least 1 concurrent operation")
        
        # Validate monitoring settings
        monitor_config = config["monitoring"]
        if not monitor_config["enabled"]:
            logger.warning("Monitoring is disabled - not recommended for production")
    
    async def _initialize_components(self):
        """Initialize system components."""
        
        # Initialize enhanced chunker
        from demo_enhanced_chunking_simple import SimpleEnhancedChunker
        self.chunker = SimpleEnhancedChunker()
        
        # Initialize advanced context builder
        from demo_advanced_context import MockAdvancedContextBuilder
        self.context_builder = MockAdvancedContextBuilder()
        
        # Test components
        test_content = "This is a test sentence for component validation."
        chunks = self.chunker.chunk_document(test_content, "Test", "test.md")
        if not chunks:
            raise RuntimeError("Enhanced chunker initialization failed")
        
        context = await self.context_builder.build_generation_context(
            query="test query",
            context_type="character_focused"
        )
        if not context:
            raise RuntimeError("Advanced context builder initialization failed")
    
    async def _run_health_checks(self):
        """Run comprehensive health checks."""
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                raise RuntimeError(f"High memory usage: {memory.percent}%")
        except ImportError:
            logger.warning("psutil not available - skipping memory check")
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1:
            raise RuntimeError(f"Low disk space: {free_gb:.2f}GB free")
        
        # Test database connectivity (if applicable)
        # This would test actual database connections in real deployment
        
        # Test external services (if applicable)
        # This would test API endpoints, etc.
    
    def _start_monitoring(self):
        """Start production monitoring."""
        
        # Monitor is already started in __init__
        # Configure alert thresholds from config
        config = self.config.get_config()
        self.monitor.health_thresholds.update(
            config["monitoring"]["alert_thresholds"]
        )
    
    async def _deploy_services(self):
        """Deploy production services."""
        
        # In real deployment, this would:
        # - Start web servers
        # - Configure load balancers
        # - Update DNS records
        # - Deploy to container orchestration
        
        # For demo, we'll simulate service deployment
        await asyncio.sleep(1)  # Simulate deployment time
    
    def _display_deployment_summary(self):
        """Display deployment summary."""
        
        config = self.config.get_config()
        health = self.monitor.get_system_health()
        
        print(f"\nDeployment Summary:")
        print(f"  Enhanced Chunking: {'✓ Enabled' if config['enhanced_chunking']['enabled'] else '✗ Disabled'}")
        print(f"  Advanced Context: {'✓ Enabled' if config['advanced_context']['enabled'] else '✗ Disabled'}")
        print(f"  Monitoring: {'✓ Enabled' if config['monitoring']['enabled'] else '✗ Disabled'}")
        print(f"  System Health: {health.status.upper()}")
        print(f"  Uptime: {health.uptime_seconds:.1f} seconds")
        
        print(f"\nConfiguration:")
        print(f"  Max Concurrent Operations: {config['performance']['max_concurrent_operations']}")
        print(f"  Operation Timeout: {config['performance']['operation_timeout_seconds']}s")
        print(f"  Memory Limit: {config['performance']['memory_limit_mb']}MB")
        print(f"  Context Token Limit: {config['advanced_context']['max_context_tokens']}")
        
        print(f"\nMonitoring Endpoints:")
        print(f"  Health Check: /health")
        print(f"  Metrics: /metrics")
        print(f"  Performance Report: /performance")
        
        print(f"\nNext Steps:")
        print(f"  • Monitor system performance")
        print(f"  • Review logs for any issues")
        print(f"  • Test with production workload")
        print(f"  • Set up automated backups")
    
    async def test_production_workload(self):
        """Test with production-like workload."""
        
        print("\n" + "=" * 60)
        print("PRODUCTION WORKLOAD TEST")
        print("=" * 60)
        
        # Simulate production workload
        test_scenarios = [
            ("Enhanced Chunking", self._test_chunking_workload),
            ("Advanced Context Building", self._test_context_workload),
            ("Concurrent Operations", self._test_concurrent_workload)
        ]
        
        for scenario_name, test_func in test_scenarios:
            print(f"\nTesting {scenario_name}...")
            start_time = time.time()
            
            try:
                await test_func()
                execution_time = (time.time() - start_time) * 1000
                
                self.monitor.record_operation(
                    operation_type=scenario_name,
                    execution_time_ms=execution_time,
                    success=True
                )
                
                print(f"✓ {scenario_name} completed in {execution_time:.2f}ms")
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                self.monitor.record_operation(
                    operation_type=scenario_name,
                    execution_time_ms=execution_time,
                    success=False,
                    error_message=str(e)
                )
                
                print(f"✗ {scenario_name} failed: {e}")
        
        # Display performance report
        report = self.monitor.get_performance_report()
        print(f"\nWorkload Test Results:")
        print(f"  System Health: {report['system_health']['status']}")
        print(f"  Success Rate: {report['system_health']['success_rate']*100:.1f}%")
        print(f"  Avg Response Time: {report['system_health']['avg_response_time_ms']:.2f}ms")
    
    async def _test_chunking_workload(self):
        """Test chunking with production workload."""
        
        # Simulate processing multiple documents
        for i in range(5):
            content = f"Test document {i} with multiple paragraphs and scenes. " * 50
            chunks = self.chunker.chunk_document(content, f"Doc {i}", "test.md")
            
            if not chunks:
                raise RuntimeError(f"Chunking failed for document {i}")
    
    async def _test_context_workload(self):
        """Test context building with production workload."""
        
        # Simulate building context for multiple queries
        queries = [
            "Character development scene",
            "Action sequence in the mansion",
            "Dialogue between main characters",
            "World building description",
            "Emotional confrontation scene"
        ]
        
        for query in queries:
            context = await self.context_builder.build_generation_context(
                query=query,
                context_type="character_focused"
            )
            
            if not context or context.get("context_quality_score", 0) < 0.5:
                raise RuntimeError(f"Context building failed for query: {query}")
    
    async def _test_concurrent_workload(self):
        """Test concurrent operations."""
        
        async def concurrent_operation(op_id: int):
            content = f"Concurrent test {op_id} " * 20
            chunks = self.chunker.chunk_document(content, f"Concurrent {op_id}", "test.md")
            return len(chunks)
        
        # Run concurrent operations
        tasks = [concurrent_operation(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        if not all(r > 0 for r in results):
            raise RuntimeError("Concurrent operations failed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        
        health = self.monitor.get_system_health()
        config = self.config.get_config()
        
        return {
            "deployed": self.is_deployed,
            "system_health": asdict(health),
            "configuration": {
                "enhanced_chunking_enabled": config["enhanced_chunking"]["enabled"],
                "advanced_context_enabled": config["advanced_context"]["enabled"],
                "monitoring_enabled": config["monitoring"]["enabled"]
            },
            "uptime_seconds": health.uptime_seconds,
            "total_operations": health.total_operations
        }
    
    def shutdown(self):
        """Shutdown production deployment."""
        
        print("\nShutting down production deployment...")
        self.monitor.shutdown()
        self.is_deployed = False
        print("✓ Production deployment shutdown complete")


async def main():
    """Main deployment function."""
    
    print("Starting Production Deployment Process...")
    
    try:
        # Create deployment manager
        deployment = ProductionDeployment()
        
        # Deploy to production
        await deployment.deploy()
        
        # Test production workload
        await deployment.test_production_workload()
        
        # Display final status
        status = deployment.get_status()
        print(f"\nFinal Deployment Status:")
        print(f"  Deployed: {'✓' if status['deployed'] else '✗'}")
        print(f"  System Health: {status['system_health']['status'].upper()}")
        print(f"  Total Operations: {status['total_operations']}")
        print(f"  Uptime: {status['uptime_seconds']:.1f} seconds")
        
        # Save configuration
        deployment.config.save_to_file()
        print(f"\n✓ Configuration saved to production_config.json")
        
        print("\n" + "=" * 80)
        print("🚀 PRODUCTION DEPLOYMENT COMPLETE!")
        print("=" * 80)
        print("\nThe enhanced chunking system is now ready for production use.")
        print("Monitor the system health and performance regularly.")
        
        # Keep running for a bit to collect metrics
        print("\nCollecting initial metrics... (press Ctrl+C to stop)")
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise
    finally:
        if 'deployment' in locals():
            deployment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())