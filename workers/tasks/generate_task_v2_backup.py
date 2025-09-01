import os
import sys
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.novel_crew import NovelGenerationCrew
from embeddings.qdrant_client import QdrantClient
from backend.app.db.neo4j_client import Neo4jClient
from backend.app.db.mongodb_client import MongoDBClient
from backend.app.config import settings

logger = logging.getLogger(__name__)

class GenerationTaskV2:
    def __init__(self):
        self.mongodb_client = MongoDBClient()
        self.qdrant_client = QdrantClient(url=settings.qdrant_url)
        self.neo4j_client = Neo4jClient()
        self.novel_crew = None
    
    async def initialize(self):
        """Initialize all clients and crew"""
        await self.mongodb_client.connect()
        await self.qdrant_client.initialize()
        await self.neo4j_client.connect()
        
        # Initialize novel generation crew
        self.novel_crew = NovelGenerationCrew(
            qdrant_client=self.qdrant_client,
            neo4j_client=self.neo4j_client,
            mongodb_client=self.mongodb_client,
            openai_api_key=settings.openai_api_key
        )
        
        logger.info("Phase 2 generation task initialized")
    
    async def update_job_progress(self, job_id: str, progress: float, state: str = None, result: Dict = None, error: str = None):
        """Update job progress in MongoDB"""
        updates = {"progress": progress}
        
        if state:
            updates["state"] = state
        if result:
            updates["result"] = result
        if error:
            updates["error"] = error
            
        await self.mongodb_client.update_job(job_id, updates)
        logger.info(f"Job {job_id} progress updated: {progress:.2%}")
    
    async def save_chapter_version(self, project_id: str, chapter_id: str, content: str, qa_analysis: Dict[str, Any], revision_count: int) -> str:
        """Save chapter version to MongoDB with enhanced metadata"""
        logger.info("Saving chapter version with Phase 2 enhancements")
        
        # Get next version number
        latest_version = await self.mongodb_client.get_latest_chapter_version(project_id, chapter_id)
        version_number = (latest_version["version_number"] + 1) if latest_version else 1
        
        # Create enhanced version document
        version_data = {
            "project_id": project_id,
            "chapter_id": chapter_id,
            "version_number": version_number,
            "content": content,
            "word_count": len(content.split()),
            "character_count": len(content),
            "revision_count": revision_count,
            
            # Phase 2 enhancements
            "qa_analysis": qa_analysis,
            "overall_qa_score": qa_analysis.get("overall_score", 0),
            "agent_scores": qa_analysis.get("agent_scores", {}),
            "total_issues": qa_analysis.get("total_issues", 0),
            "severity_breakdown": qa_analysis.get("severity_breakdown", {}),
            "requires_revision": qa_analysis.get("requires_revision", False),
            
            # Workflow metadata
            "workflow_version": "2.0",
            "generation_method": "multi_agent_crew",
            "parallel_qa_enabled": settings.enable_parallel_qa,
            "context_retrieval_method": "hybrid_semantic_graph"
        }
        
        version_id = await self.mongodb_client.create_chapter_version(version_data)
        logger.info(f"Saved enhanced chapter version: {version_id}")
        
        return version_id

async def run_generate_job_v2(job_id: str, payload: Dict[str, Any]):
    """Enhanced Phase 2 job execution function with multi-agent crew"""
    task = GenerationTaskV2()
    
    try:
        # Initialize clients
        await task.initialize()
        
        # Update job status to running
        await task.update_job_progress(job_id, 0.05, "running")
        
        # Extract parameters
        user_id = payload["user_id"]
        project_id = payload["project_id"]
        chapter_id = payload.get("chapter_id", f"chapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        prompt = payload["prompt"]
        settings = payload.get("settings", {})
        
        logger.info(f"Starting Phase 2 generation job {job_id} for user {user_id}, project {project_id}")
        
        # Step 1: Execute novel generation workflow with multi-agent crew
        await task.update_job_progress(job_id, 0.1)
        
        workflow_result = await task.novel_crew.execute_generation_workflow(
            project_id=project_id,
            prompt=prompt,
            settings=settings
        )
        
        # Check for workflow errors
        if "error" in workflow_result:
            raise Exception(f"Workflow failed: {workflow_result['error']}")
        
        # Step 2: Extract results
        await task.update_job_progress(job_id, 0.9)
        
        final_content = workflow_result["content"]
        qa_analysis = workflow_result["qa_analysis"]
        revision_count = workflow_result["revision_count"]
        context_used = workflow_result["context_used"]
        
        # Step 3: Save final version with enhanced metadata
        version_id = await task.save_chapter_version(
            project_id, chapter_id, final_content, qa_analysis, revision_count
        )
        
        # Step 4: Complete job with comprehensive results
        result = {
            "version_id": version_id,
            "chapter_id": chapter_id,
            "content": final_content,
            "word_count": workflow_result["word_count"],
            "character_count": workflow_result["character_count"],
            "revision_count": revision_count,
            
            # Phase 2 enhancements
            "qa_analysis": qa_analysis,
            "overall_qa_score": qa_analysis.get("overall_score", 0),
            "agent_scores": qa_analysis.get("agent_scores", {}),
            "context_used": context_used,
            "workflow_metadata": workflow_result["generation_metadata"],
            
            # Performance metrics
            "parallel_qa_enabled": settings.enable_parallel_qa,
            "total_issues_found": qa_analysis.get("total_issues", 0),
            "high_severity_issues": qa_analysis.get("severity_breakdown", {}).get("high", 0),
            "requires_further_revision": qa_analysis.get("requires_revision", False)
        }
        
        await task.update_job_progress(job_id, 1.0, "success", result)
        logger.info(f"Phase 2 job {job_id} completed successfully - Score: {qa_analysis.get('overall_score', 0)}")
        
    except Exception as e:
        logger.error(f"Phase 2 job {job_id} failed: {e}")
        await task.update_job_progress(job_id, 0.0, "failed", error=str(e))
        raise
    
    finally:
        # Cleanup
        if task.mongodb_client:
            await task.mongodb_client.disconnect()
        if task.neo4j_client:
            await task.neo4j_client.disconnect()

# RQ job function (synchronous wrapper) - Enhanced for Phase 2
def run_generate_job_sync(job_id: str, payload: Dict[str, Any]):
    """Synchronous wrapper for RQ - Phase 2 version"""
    return asyncio.run(run_generate_job_v2(job_id, payload))

# Backward compatibility - Phase 1 function still available
def run_generate_job_v1(job_id: str, payload: Dict[str, Any]):
    """Phase 1 generation function for backward compatibility"""
    # Import Phase 1 implementation
    from workers.tasks.generate_task_v1 import run_generate_job as run_v1
    return run_v1(job_id, payload)

# Default to Phase 2
run_generate_job = run_generate_job_sync