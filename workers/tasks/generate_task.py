import os
import sys
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.generator_agent import GeneratorAgent
from agents.technical_qa import TechnicalQAAgent
from embeddings.qdrant_client import QdrantClient
from backend.app.db.mongodb_client import MongoDBClient
from backend.app.config import settings

logger = logging.getLogger(__name__)

class GenerationTaskPhase1:
    """Phase 1 implementation - Simple and solid foundation"""
    
    def __init__(self):
        self.mongodb_client = MongoDBClient()
        self.qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=getattr(settings, 'qdrant_key', None))
        self.generator_agent = None
        self.technical_qa_agent = None
    
    async def initialize(self):
        """Initialize all clients and agents"""
        await self.mongodb_client.connect()
        await self.qdrant_client.initialize()
        
        # Initialize agents with OpenRouter or Gemini
        api_key = getattr(settings, 'openrouter_api_key', None) or getattr(settings, 'openai_api_key', None)
        model = getattr(settings, 'openrouter_model', 'gpt-4o-mini')
        
        if not api_key:
            raise ValueError("No API key found for LLM. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        
        self.generator_agent = GeneratorAgent(api_key=api_key, model=model)
        self.technical_qa_agent = TechnicalQAAgent(api_key=api_key, model=model)
        
        logger.info("Phase 1 generation task initialized")
    
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
    
    async def retrieve_context(self, prompt: str, project_id: str, top_k: int = 4) -> str:
        """Retrieve relevant context from Qdrant"""
        try:
            logger.info(f"Retrieving context for project {project_id}")
            
            # Search for relevant chunks
            results = await self.qdrant_client.search_text(
                collection_name="novel_chunks",
                query_text=prompt,
                top_k=top_k,
                filter_conditions={"project_id": project_id}
            )
            
            if not results:
                logger.warning(f"No context found for project {project_id}")
                return "No relevant context found."
            
            # Combine context from search results
            context_parts = []
            for result in results:
                text = result.payload.get("text", "")
                source = result.payload.get("source", "unknown")
                score = result.score
                
                context_parts.append(f"[Source: {source}, Relevance: {score:.3f}]\n{text}")
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Retrieved {len(results)} context chunks, total length: {len(context)}")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return "Context retrieval failed."
    
    async def save_chapter_version(self, project_id: str, chapter_id: str, content: str, qa_score: int, qa_issues: list, revision_count: int) -> str:
        """Save chapter version to MongoDB - Phase 1 simple version"""
        logger.info("Saving chapter version (Phase 1)")
        
        # Get next version number
        latest_version = await self.mongodb_client.get_latest_chapter_version(project_id, chapter_id)
        version_number = (latest_version["version_number"] + 1) if latest_version else 1
        
        # Create version document
        version_data = {
            "project_id": project_id,
            "chapter_id": chapter_id,
            "version_number": version_number,
            "content": content,
            "word_count": len(content.split()),
            "character_count": len(content),
            "revision_count": revision_count,
            "qa_score": qa_score,
            "qa_issues": qa_issues,
            "workflow_version": "1.0",
            "generation_method": "simple_generator_qa"
        }
        
        version_id = await self.mongodb_client.create_chapter_version(version_data)
        logger.info(f"Saved chapter version: {version_id}")
        
        return version_id

async def run_generate_job_phase1(job_id: str, payload: Dict[str, Any]):
    """Phase 1 job execution function - Simple and solid"""
    task = GenerationTaskPhase1()
    
    try:
        # Initialize clients and agents
        await task.initialize()
        
        # Update job status to running
        await task.update_job_progress(job_id, 0.05, "running")
        
        # Extract parameters
        user_id = payload["user_id"]
        project_id = payload["project_id"]
        chapter_id = payload.get("chapter_id", f"chapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        prompt = payload["prompt"]
        settings_dict = payload.get("settings", {})
        
        # Get settings with defaults
        length_words = settings_dict.get("length_words", 1200)
        max_revision_rounds = settings_dict.get("max_revision_rounds", 2)
        qa_threshold = getattr(settings, 'technical_qa_threshold', 80)
        
        logger.info(f"Starting Phase 1 generation job {job_id} for user {user_id}, project {project_id}")
        
        # Step 1: Retrieve context from Qdrant
        await task.update_job_progress(job_id, 0.1)
        context = await task.retrieve_context(prompt, project_id)
        
        # Step 2: Generate initial draft
        await task.update_job_progress(job_id, 0.3)
        logger.info("Generating initial draft")
        
        draft = await task.generator_agent.generate(
            prompt=prompt,
            context=context,
            length_words=length_words,
            temperature=0.7
        )
        
        # Step 3: Run Technical QA
        await task.update_job_progress(job_id, 0.6)
        logger.info("Running technical QA")
        
        qa_result = await task.technical_qa_agent.review(draft)
        qa_score = qa_result.get("score", 0)
        qa_issues = qa_result.get("issues", [])
        
        logger.info(f"Initial QA score: {qa_score}")
        
        revision_count = 0
        final_content = draft
        
        # Step 4: Revise if needed (max 1 revision for Phase 1)
        if qa_score < qa_threshold and revision_count < max_revision_rounds:
            await task.update_job_progress(job_id, 0.75)
            logger.info(f"QA score {qa_score} below threshold {qa_threshold}, revising...")
            
            # Build revision feedback
            feedback_parts = []
            for issue in qa_issues:
                feedback_parts.append(f"- {issue.get('type', 'issue')}: {issue.get('issue', '')} (Suggestion: {issue.get('suggestion', '')})")
            
            feedback = "Please address the following issues:\n" + "\n".join(feedback_parts)
            
            # Generate revision
            revised_content = await task.generator_agent.revise_with_feedback(
                original_content=draft,
                feedback=feedback,
                context=context
            )
            
            # Re-run QA on revision
            revised_qa_result = await task.technical_qa_agent.review(revised_content)
            revised_qa_score = revised_qa_result.get("score", 0)
            
            logger.info(f"Revised QA score: {revised_qa_score}")
            
            # Use revision if it's better
            if revised_qa_score > qa_score:
                final_content = revised_content
                qa_score = revised_qa_score
                qa_issues = revised_qa_result.get("issues", [])
                revision_count = 1
                logger.info("Using revised content")
            else:
                logger.info("Keeping original content (revision didn't improve quality)")
        
        # Step 5: Save final version
        await task.update_job_progress(job_id, 0.9)
        version_id = await task.save_chapter_version(
            project_id, chapter_id, final_content, qa_score, qa_issues, revision_count
        )
        
        # Step 6: Complete job
        result = {
            "version_id": version_id,
            "chapter_id": chapter_id,
            "content": final_content,
            "word_count": len(final_content.split()),
            "character_count": len(final_content),
            "qa_score": qa_score,
            "qa_issues_count": len(qa_issues),
            "revision_count": revision_count,
            "workflow_version": "1.0"
        }
        
        await task.update_job_progress(job_id, 1.0, "success", result)
        logger.info(f"Phase 1 job {job_id} completed successfully - QA Score: {qa_score}")
        
    except Exception as e:
        logger.error(f"Phase 1 job {job_id} failed: {e}")
        await task.update_job_progress(job_id, 0.0, "failed", error=str(e))
        raise
    
    finally:
        # Cleanup
        if task.mongodb_client:
            await task.mongodb_client.disconnect()

# RQ job function (synchronous wrapper)
def run_generate_job(job_id: str, payload: Dict[str, Any]):
    """Synchronous wrapper for RQ - Phase 1 version"""
    return asyncio.run(run_generate_job_phase1(job_id, payload))