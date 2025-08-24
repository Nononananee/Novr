"""System monitoring and management endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ...monitoring.advanced_system_monitor import (
    monitor_operation, ComponentType, get_system_health
)
from ..circuit_breaker import (
    get_all_circuit_status, get_circuit_breaker_stats, 
    circuit_breaker_manager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
@monitor_operation("system_health_check", ComponentType.API_LAYER)
async def system_health_endpoint():
    """Get comprehensive system health status."""
    try:
        health_status = await get_system_health()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "health": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/status")
@monitor_operation("system_status", ComponentType.API_LAYER)
async def system_status_endpoint():
    """Get overall system status dashboard."""
    try:
        # Get health status
        health_status = await get_system_health()
        
        # Calculate system score
        healthy_components = health_status["component_summary"]["healthy"]
        total_components = health_status["component_summary"]["total"]
        
        if total_components > 0:
            health_score = (healthy_components / total_components) * 100
        else:
            health_score = 0
        
        # System grade
        if health_score >= 95:
            system_grade = "A+"
        elif health_score >= 90:
            system_grade = "A"
        elif health_score >= 80:
            system_grade = "B"
        else:
            system_grade = "C"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "system_grade": system_grade,
            "health_score": health_score,
            "overall_health": health_status["overall_health"],
            "monitoring_active": health_status["monitoring_active"],
            "components": {
                "total": total_components,
                "healthy": healthy_components,
                "unhealthy": total_components - healthy_components
            }
        }
    except Exception as e:
        logger.error(f"System status failed: {e}")
        raise HTTPException(status_code=500, detail=f"System status failed: {str(e)}")


@router.get("/circuit-breakers")
@monitor_operation("circuit_breaker_status", ComponentType.API_LAYER)
async def circuit_breaker_status_endpoint():
    """Get status of all circuit breakers."""
    try:
        circuit_status = get_all_circuit_status()
        circuit_stats = get_circuit_breaker_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": circuit_status,
            "global_stats": circuit_stats
        }
    except Exception as e:
        logger.error(f"Circuit breaker status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker status failed: {str(e)}")


@router.post("/circuit-breakers/reset")
@monitor_operation("circuit_breaker_reset", ComponentType.API_LAYER)
async def reset_circuit_breakers_endpoint():
    """Reset all circuit breakers."""
    try:
        circuit_breaker_manager.reset_all()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": "All circuit breakers reset successfully"
        }
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker reset failed: {str(e)}")


@router.get("/circuit-breakers/{circuit_name}")
@monitor_operation("circuit_breaker_detail", ComponentType.API_LAYER)
async def circuit_breaker_detail_endpoint(circuit_name: str):
    """Get detailed status of a specific circuit breaker."""
    try:
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker(circuit_name)
        
        if not circuit_breaker:
            raise HTTPException(status_code=404, detail=f"Circuit breaker '{circuit_name}' not found")
        
        status = circuit_breaker.get_status()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "circuit_breaker": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Circuit breaker detail failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker detail failed: {str(e)}")