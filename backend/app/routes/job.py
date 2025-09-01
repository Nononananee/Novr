from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
import logging

from backend.app.schemas.responses import JobResponse
from backend.app.services.orchestrator import get_orchestrator
from backend.app.db.mongodb_client import get_mongodb

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a generation job
    
    Returns the current status, progress, and results (if completed) of a job.
    """
    try:
        logger.info(f"Getting status for job {job_id}")
        
        orchestrator = await get_orchestrator()
        job_status = await orchestrator.get_job_status(job_id)
        
        return JobResponse(**job_status)
        
    except ValueError as e:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.get("/user/{user_id}")
async def get_user_jobs(
    user_id: str,
    limit: int = Query(default=50, le=100, description="Maximum number of jobs to return")
):
    """Get all jobs for a specific user"""
    try:
        logger.info(f"Getting jobs for user {user_id}")
        
        mongodb = await get_mongodb()
        jobs = await mongodb.get_jobs_by_user(user_id, limit)
        
        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "jobs": jobs,
                "count": len(jobs)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get jobs for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user jobs: {str(e)}"
        )

@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued or running job"""
    try:
        logger.info(f"Cancelling job {job_id}")
        
        orchestrator = await get_orchestrator()
        
        # Check if job exists and is cancellable
        job_status = await orchestrator.get_job_status(job_id)
        
        if job_status["status"] in ["success", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {job_status['status']} state"
            )
        
        # Update job status to cancelled
        success = await orchestrator.update_job_status(job_id, {
            "state": "cancelled",
            "progress": 0.0,
            "error": "Job cancelled by user"
        })
        
        if success:
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel job"
            )
            
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )