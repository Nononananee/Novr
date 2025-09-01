from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import logging

from backend.app.schemas.requests import GenerateRequest
from backend.app.schemas.responses import GenerateResponse
from backend.app.services.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chapter", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_chapter(request: GenerateRequest):
    """
    Generate a novel chapter using multi-agent system
    
    This endpoint queues a chapter generation job and returns immediately
    with a job ID that can be used to track progress.
    """
    try:
        logger.info(f"Received generation request for user {request.user_id}, project {request.project_id}")
        
        # Validate request
        if not request.prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Get orchestrator and enqueue job
        orchestrator = await get_orchestrator()
        job_id = await orchestrator.enqueue_job(request.dict())
        
        logger.info(f"Successfully queued job {job_id}")
        
        return GenerateResponse(
            status="queued",
            code=202,
            data={
                "job_id": job_id,
                "estimated_completion": "5-10 minutes"
            },
            message="Chapter generation job queued successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue generation job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue generation job: {str(e)}"
        )

@router.get("/queue/info")
async def get_queue_info():
    """Get information about the job queue"""
    try:
        orchestrator = await get_orchestrator()
        queue_info = await orchestrator.get_queue_info()
        
        return {
            "status": "success",
            "data": queue_info
        }
    except Exception as e:
        logger.error(f"Failed to get queue info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue info: {str(e)}"
        )