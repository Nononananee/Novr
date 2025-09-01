import uuid
import logging
from typing import Dict, Any
from datetime import datetime, timezone
import redis
from rq import Queue
import json

from backend.app.config import settings
from backend.app.db.mongodb_client import get_mongodb

logger = logging.getLogger(__name__)

class JobOrchestrator:
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.queue = Queue('novel_generation', connection=self.redis_client)
        # Cache the get_mongodb function at instantiation time. Tests may patch
        # `get_mongodb` in the module while creating an instance; binding it
        # here ensures the instance will use the patched function even after
        # the patch context exits.
        self.get_mongodb = get_mongodb
    
    async def enqueue_job(self, request_data: Dict[str, Any]) -> str:
        """Enqueue a generation job"""
        job_id = str(uuid.uuid4())

        # Create job document in MongoDB
        mongodb = await self.get_mongodb()
        job_doc = {
            "job_id": job_id,
            "user_id": request_data["user_id"],
            "project_id": request_data["project_id"],
            "chapter_id": request_data.get("chapter_id"),
            "payload": request_data,
            "state": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
        }

        await mongodb.create_job(job_doc)
        logger.info(f"Created job document: {job_id}")

        # Enqueue job in Redis
        try:
            job = self.queue.enqueue(
                "workers.tasks.generate_task.run_generate_job",
                job_id,
                request_data,
                job_timeout="30m",
            )
            logger.info(f"Enqueued job in Redis: {job_id}")

            return job_id
        except Exception as e:
            logger.error(f"Failed to enqueue job {job_id}: {e}")
            # Update job status to failed
            await mongodb.update_job(job_id, {
                "state": "failed",
                "error": f"Failed to enqueue: {str(e)}",
            })
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from MongoDB"""
        mongodb = await self.get_mongodb()
        job = await mongodb.get_job(job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        return {
            "job_id": job["job_id"],
            "status": job["state"],
            "progress": job["progress"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "result": job.get("result"),
            "error": job.get("error")
        }
    
    async def update_job_status(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job status"""
        mongodb = await self.get_mongodb()
        return await mongodb.update_job(job_id, updates)
    
    async def get_queue_info(self) -> Dict[str, Any]:
        """Get queue information"""
        try:
            return {
                "queued_jobs": len(self.queue),
                "failed_jobs": len(self.queue.failed_job_registry),
                "workers": len(self.queue.workers),
                "queue_name": self.queue.name
            }
        except Exception as e:
            logger.error(f"Failed to get queue info: {e}")
            return {"error": str(e)}

# Lazy global orchestrator instance (created on demand)
orchestrator: JobOrchestrator | None = None


async def get_orchestrator() -> JobOrchestrator:
    """Get orchestrator instance, create lazily if needed"""
    global orchestrator
    if orchestrator is None:
        orchestrator = JobOrchestrator()
    return orchestrator