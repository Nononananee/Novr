"""Advanced alerting system with rules and notifications."""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from .metrics import AdvancedMetricsCollector
from .health import HealthStatus, ComponentType

logger = logging.getLogger(__name__)


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
                import asyncio
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