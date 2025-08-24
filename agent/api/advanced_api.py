"""
Advanced API endpoints for performance monitoring, testing, and optimization.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# Create router for advanced endpoints
router = APIRouter(prefix="/advanced", tags=["advanced"])


# Import necessary functions
async def create_session(metadata: Dict[str, Any] = None) -> str:
    """Create a session for agent execution."""
    from .db_utils import create_session as db_create_session
    return await db_create_session(metadata=metadata)


async def execute_agent(message: str, session_id: str, save_conversation: bool = True):
    """Execute agent with message."""
    from .api import execute_agent as api_execute_agent
    return await api_execute_agent(message, session_id, save_conversation=save_conversation)


# Performance Monitoring Endpoints
@router.get("/performance/report")
async def get_performance_report_endpoint(hours: int = 24):
    """Get creative performance report."""
    try:
        session_id = await create_session(metadata={"api_call": "performance_report"})
        
        response, tools_used = await execute_agent(
            message=f"Get creative performance report for the last {hours} hours",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "report_type": "creative_performance",
            "time_period_hours": hours,
            "report": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Performance report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality/recommendations")
async def get_quality_recommendations_endpoint():
    """Get quality improvement recommendations."""
    try:
        session_id = await create_session(metadata={"api_call": "quality_recommendations"})
        
        response, tools_used = await execute_agent(
            message="Get quality recommendations for improving creative output",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "recommendation_type": "creative_quality",
            "recommendations": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Quality recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/optimize")
async def optimize_system_performance_endpoint():
    """Optimize system performance."""
    try:
        session_id = await create_session(metadata={"api_call": "system_optimization"})
        
        response, tools_used = await execute_agent(
            message="Optimize creative performance based on collected metrics",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "optimization_type": "system_performance",
            "optimization_result": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/creative")
async def get_creative_health_endpoint():
    """Get creative operations health status."""
    try:
        session_id = await create_session(metadata={"api_call": "creative_health"})
        
        response, tools_used = await execute_agent(
            message="Monitor the health of creative operations",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "health_check_type": "creative_operations",
            "health_status": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Creative health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary")
async def get_metrics_summary_endpoint(summary_type: str = "overview"):
    """Get creative metrics summary."""
    try:
        session_id = await create_session(metadata={"api_call": "metrics_summary"})
        
        response, tools_used = await execute_agent(
            message=f"Get creative metrics summary of type {summary_type}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "summary_type": summary_type,
            "metrics_summary": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Testing Framework Endpoints
@router.post("/test/suite/{suite_name}")
async def run_test_suite_endpoint(suite_name: str):
    """Run a specific test suite."""
    try:
        from .testing_framework import novel_test_framework
        
        result = await novel_test_framework.run_test_suite(suite_name)
        
        return {
            "test_suite": suite_name,
            "test_results": result
        }
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/suites")
async def list_test_suites_endpoint():
    """List available test suites."""
    try:
        from .testing_framework import novel_test_framework
        
        suites = {}
        for name, suite in novel_test_framework.test_suites.items():
            suites[name] = {
                "name": suite.name,
                "test_count": len(suite.tests),
                "setup_required": suite.setup_required,
                "cleanup_required": suite.cleanup_required,
                "timeout_seconds": suite.timeout_seconds
            }
        
        return {
            "available_suites": suites,
            "total_suites": len(suites)
        }
        
    except Exception as e:
        logger.error(f"Test suite listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/report")
async def get_test_report_endpoint():
    """Get comprehensive test report."""
    try:
        from .testing_framework import novel_test_framework
        
        report = novel_test_framework.generate_test_report()
        
        return {
            "report_type": "comprehensive_testing",
            "test_report": report
        }
        
    except Exception as e:
        logger.error(f"Test report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory Management Endpoints
@router.get("/memory/report")
async def get_memory_report_endpoint():
    """Get memory usage report."""
    try:
        from .memory_optimization import novel_memory_manager
        
        report = await novel_memory_manager.get_memory_report()
        
        return {
            "report_type": "memory_usage",
            "memory_report": report
        }
        
    except Exception as e:
        logger.error(f"Memory report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/optimize")
async def optimize_memory_endpoint():
    """Optimize memory usage."""
    try:
        from .memory_optimization import novel_memory_manager
        
        optimization_result = await novel_memory_manager.optimize_memory_usage()
        
        return {
            "optimization_type": "memory_usage",
            "optimization_result": optimization_result
        }
        
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/preload/{novel_id}")
async def preload_novel_data_endpoint(novel_id: str, priority_content: List[str] = None):
    """Preload novel data into cache."""
    try:
        from .memory_optimization import novel_memory_manager
        
        await novel_memory_manager.preload_novel_data(novel_id, priority_content)
        
        return {
            "novel_id": novel_id,
            "preload_status": "completed",
            "priority_content": priority_content or ["chapters", "characters"]
        }
        
    except Exception as e:
        logger.error(f"Novel data preload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/cleanup/{novel_id}")
async def cleanup_novel_data_endpoint(novel_id: str):
    """Clean up cached data for a specific novel."""
    try:
        from .memory_optimization import novel_memory_manager
        
        await novel_memory_manager.cleanup_novel_data(novel_id)
        
        return {
            "novel_id": novel_id,
            "cleanup_status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Novel data cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Genre-Specific Analysis Endpoints
@router.post("/genre/analyze/fantasy/{novel_id}")
async def analyze_fantasy_endpoint(novel_id: str, focus_element: str = "magic_system"):
    """Analyze fantasy-specific elements."""
    try:
        session_id = await create_session(metadata={"api_call": "fantasy_analysis"})
        
        response, tools_used = await execute_agent(
            message=f"Analyze fantasy world-building for novel {novel_id} focusing on {focus_element}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "fantasy_world_building",
            "focus_element": focus_element,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Fantasy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genre/analyze/mystery/{novel_id}")
async def analyze_mystery_endpoint(novel_id: str, mystery_type: str = "detective"):
    """Analyze mystery-specific plot structure."""
    try:
        session_id = await create_session(metadata={"api_call": "mystery_analysis"})
        
        response, tools_used = await execute_agent(
            message=f"Analyze mystery plot structure for novel {novel_id} of type {mystery_type}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "mystery_plot_structure",
            "mystery_type": mystery_type,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Mystery analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genre/analyze/romance/{novel_id}")
async def analyze_romance_endpoint(novel_id: str, relationship_focus: str = "primary"):
    """Analyze romance-specific relationship development."""
    try:
        session_id = await create_session(metadata={"api_call": "romance_analysis"})
        
        response, tools_used = await execute_agent(
            message=f"Analyze romance relationship development for novel {novel_id} focusing on {relationship_focus}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "romance_relationship_development",
            "relationship_focus": relationship_focus,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Romance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genre/generate")
async def generate_genre_content_endpoint(
    genre: str,
    content_type: str,
    parameters: Dict[str, Any] = None
):
    """Generate genre-specific content."""
    try:
        session_id = await create_session(metadata={"api_call": "genre_content_generation"})
        
        params_str = f" with parameters {parameters}" if parameters else ""
        response, tools_used = await execute_agent(
            message=f"Generate {content_type} content for {genre} genre{params_str}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "genre": genre,
            "content_type": content_type,
            "parameters": parameters or {},
            "generated_content": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Genre content generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Status Endpoints
@router.get("/status/comprehensive")
async def get_comprehensive_status_endpoint():
    """Get comprehensive system status."""
    try:
        # Get various system reports
        from .memory_optimization import novel_memory_manager
        from .creative_performance_monitor import creative_monitor
        
        memory_report = await novel_memory_manager.get_memory_report()
        performance_report = creative_monitor.get_creative_performance_report(hours=24)
        
        return {
            "system_status": "operational",
            "memory_status": memory_report,
            "performance_status": performance_report,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Comprehensive status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))