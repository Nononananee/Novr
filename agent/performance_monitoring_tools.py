"""
Performance monitoring tools for creative operations.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic_ai import RunContext

from .agent import rag_agent, AgentDependencies
from .creative_performance_monitor import creative_monitor

logger = logging.getLogger(__name__)


@rag_agent.tool
async def get_creative_performance_report(
    ctx: RunContext[AgentDependencies],
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get creative performance report for the specified time period.
    
    This tool provides insights into creative operation performance,
    quality metrics, and trends over time. Essential for monitoring
    and optimizing creative writing assistance quality.
    
    Args:
        hours: Number of hours to look back for the report (default: 24)
    
    Returns:
        Comprehensive performance report with quality metrics and trends
    """
    try:
        report = creative_monitor.get_creative_performance_report(hours=hours)
        
        return {
            "report_type": "creative_performance",
            "time_period_hours": hours,
            "performance_data": report,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate creative performance report for {hours} hours"
        }


@rag_agent.tool
async def get_quality_recommendations(
    ctx: RunContext[AgentDependencies]
) -> Dict[str, Any]:
    """
    Get recommendations for improving creative quality.
    
    This tool analyzes recent creative operations and provides specific
    recommendations for improving quality metrics and addressing
    performance issues.
    
    Returns:
        List of quality recommendations with priorities and specific actions
    """
    try:
        recommendations = creative_monitor.get_quality_recommendations()
        
        return {
            "recommendation_type": "creative_quality",
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate quality recommendations"
        }


@rag_agent.tool
async def optimize_creative_performance(
    ctx: RunContext[AgentDependencies]
) -> Dict[str, Any]:
    """
    Optimize creative performance based on collected metrics.
    
    This tool automatically adjusts quality thresholds and provides
    optimization suggestions based on historical performance data.
    Essential for maintaining and improving creative output quality.
    
    Returns:
        Optimization results with applied changes and recommendations
    """
    try:
        optimization_result = await creative_monitor.optimize_creative_performance()
        
        return {
            "optimization_type": "creative_performance",
            "optimization_result": optimization_result,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to optimize creative performance"
        }


@rag_agent.tool
async def analyze_creative_quality_trends(
    ctx: RunContext[AgentDependencies],
    metric_name: str = "all"
) -> Dict[str, Any]:
    """
    Analyze creative quality trends over time.
    
    This tool examines how creative quality metrics have changed over time,
    identifying improvement or decline patterns. Useful for understanding
    the effectiveness of creative assistance features.
    
    Args:
        metric_name: Specific metric to analyze (all, character_consistency, 
                    plot_coherence, emotional_consistency, etc.)
    
    Returns:
        Quality trend analysis with patterns and insights
    """
    try:
        # Get performance report for trend analysis
        report = creative_monitor.get_creative_performance_report(hours=168)  # Last week
        trends = creative_monitor._calculate_quality_trends()
        
        if metric_name != "all" and metric_name in trends:
            specific_trend = {metric_name: trends[metric_name]}
            quality_stats = report.get("quality_statistics", {}).get(metric_name, {})
        else:
            specific_trend = trends
            quality_stats = report.get("quality_statistics", {})
        
        return {
            "analysis_type": "creative_quality_trends",
            "metric_analyzed": metric_name,
            "trends": specific_trend,
            "quality_statistics": quality_stats,
            "analysis_period": "7 days",
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze quality trends for {metric_name}"
        }


@rag_agent.tool
async def monitor_creative_operation_health(
    ctx: RunContext[AgentDependencies]
) -> Dict[str, Any]:
    """
    Monitor the health of creative operations.
    
    This tool provides a health check of creative operations, identifying
    any issues with performance, quality, or reliability. Essential for
    maintaining system health and user experience.
    
    Returns:
        Health status report with alerts and recommendations
    """
    try:
        # Get recent performance data
        report = creative_monitor.get_creative_performance_report(hours=24)
        recommendations = creative_monitor.get_quality_recommendations()
        
        # Determine health status
        success_rate = report.get("success_rate", 0)
        total_operations = report.get("total_operations", 0)
        recent_alerts = report.get("recent_alerts", [])
        
        if success_rate >= 0.95 and len(recent_alerts) == 0:
            health_status = "excellent"
        elif success_rate >= 0.9 and len(recent_alerts) <= 2:
            health_status = "good"
        elif success_rate >= 0.8 and len(recent_alerts) <= 5:
            health_status = "fair"
        else:
            health_status = "poor"
        
        # Calculate quality health
        quality_stats = report.get("quality_statistics", {})
        quality_health = "good"
        
        for metric, stats in quality_stats.items():
            if stats.get("average", 0) < 0.6:
                quality_health = "poor"
                break
            elif stats.get("average", 0) < 0.7:
                quality_health = "fair"
        
        return {
            "health_check_type": "creative_operations",
            "overall_health": health_status,
            "quality_health": quality_health,
            "success_rate": success_rate,
            "total_operations_24h": total_operations,
            "active_alerts": len(recent_alerts),
            "high_priority_recommendations": len([r for r in recommendations if r.get("priority") == "high"]),
            "health_details": {
                "performance_health": health_status,
                "quality_health": quality_health,
                "reliability_health": "good" if success_rate >= 0.9 else "fair"
            },
            "recommendations": recommendations[:3],  # Top 3 recommendations
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to monitor creative operation health"
        }


@rag_agent.tool
async def get_creative_metrics_summary(
    ctx: RunContext[AgentDependencies],
    summary_type: str = "overview"
) -> Dict[str, Any]:
    """
    Get a summary of creative metrics and performance.
    
    This tool provides different types of summaries of creative performance
    metrics, from high-level overviews to detailed breakdowns.
    
    Args:
        summary_type: Type of summary (overview, detailed, quality_focused)
    
    Returns:
        Creative metrics summary based on the requested type
    """
    try:
        report = creative_monitor.get_creative_performance_report(hours=24)
        
        if summary_type == "overview":
            summary = {
                "total_operations": report.get("total_operations", 0),
                "success_rate": f"{report.get('success_rate', 0) * 100:.1f}%",
                "average_duration": f"{report.get('average_duration_ms', 0):.0f}ms",
                "quality_score": "Good" if report.get("quality_statistics") else "No data"
            }
        
        elif summary_type == "detailed":
            summary = {
                "performance_metrics": {
                    "total_operations": report.get("total_operations", 0),
                    "successful_operations": report.get("successful_operations", 0),
                    "failed_operations": report.get("failed_operations", 0),
                    "success_rate": report.get("success_rate", 0),
                    "average_duration_ms": report.get("average_duration_ms", 0)
                },
                "operation_breakdown": report.get("operation_breakdown", {}),
                "quality_statistics": report.get("quality_statistics", {}),
                "recent_alerts": len(report.get("recent_alerts", []))
            }
        
        elif summary_type == "quality_focused":
            quality_stats = report.get("quality_statistics", {})
            summary = {
                "quality_metrics": quality_stats,
                "quality_trends": creative_monitor._calculate_quality_trends(),
                "quality_alerts": report.get("recent_alerts", []),
                "quality_recommendations": creative_monitor.get_quality_recommendations()[:5]
            }
        
        else:
            summary = {"error": f"Unknown summary type: {summary_type}"}
        
        return {
            "summary_type": summary_type,
            "summary_data": summary,
            "generated_at": report.get("period_hours", 24),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate creative metrics summary of type {summary_type}"
        }