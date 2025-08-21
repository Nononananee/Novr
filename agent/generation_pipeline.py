"""
Advanced Generation Pipeline for Creative RAG System (Refactored)
Orchestrates content generation by leveraging the advanced IntegratedNovelMemorySystem.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import the new, advanced memory system and its data structures
from ..memory.integrated_memory_system import (
    IntegratedNovelMemorySystem,
    GenerationContext as NovelGenerationContext,
    GenerationResult as NovelGenerationResult
)

# Keep the existing data models for the API layer
from .models import ValidationResult

logger = logging.getLogger(__name__)

# --- ENUMS and DataClasses for the Pipeline's Public API ---
# These remain the same as they are the interface for the rest of the application.

class GenerationType(Enum):
    """Types of content generation"""
    NARRATIVE_CONTINUATION = "narrative_continuation"
    CHARACTER_DIALOGUE = "character_dialogue"
    SCENE_DESCRIPTION = "scene_description"
    CHAPTER_OPENING = "chapter_opening"
    CONFLICT_SCENE = "conflict_scene"
    RESOLUTION_SCENE = "resolution_scene"
    CHARACTER_INTRODUCTION = "character_introduction"
    WORLD_BUILDING = "world_building"

class GenerationMode(Enum):
    """Generation modes"""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"
    PREVIEW = "preview"

class QualityThreshold(Enum):
    """Quality thresholds for content"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"

@dataclass
class GenerationRequest:
    """Request for content generation"""
    generation_type: GenerationType
    generation_mode: GenerationMode = GenerationMode.SEMI_AUTOMATIC
    quality_threshold: QualityThreshold = QualityThreshold.MODERATE
    current_chapter: int = 1
    current_scene: Optional[str] = None
    target_characters: List[str] = field(default_factory=list)
    active_plot_threads: List[str] = field(default_factory=list)
    pov_character: Optional[str] = None
    target_word_count: int = 500
    user_prompt: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class GenerationResult:
    """Result of content generation"""
    request_id: str
    success: bool
    generated_content: str = ""
    consistency_score: float = 0.0
    originality_score: float = 0.0
    approval_status: str = "pending"
    proposal_id: Optional[str] = None
    total_processing_time_ms: float = 0.0
    word_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class AdvancedGenerationPipeline:
    """
    Refactored generation pipeline that acts as a lightweight orchestrator,
    delegating core logic to the IntegratedNovelMemorySystem.
    """
    
    def __init__(self, memory_system: IntegratedNovelMemorySystem):
        self.memory_system = memory_system
        self.active_generations: Dict[str, GenerationRequest] = {}

    def _map_request_to_novel_context(self, request: GenerationRequest) -> NovelGenerationContext:
        """Maps the pipeline's request to the memory system's context."""
        # Note: current_word_count is not available in the request, using a placeholder.
        # This could be added to the request if needed for more precise memory retrieval.
        return NovelGenerationContext(
            current_chapter=request.current_chapter,
            current_word_count=0, 
            target_characters=request.target_characters,
            active_plot_threads=request.active_plot_threads,
            generation_intent=request.generation_type.value,
            tone_requirements={},
            constraints=None,
            pov_character=request.pov_character,
            scene_location=request.current_scene
        )

    def _map_novel_result_to_pipeline_result(self, novel_result: NovelGenerationResult, request: GenerationRequest) -> GenerationResult:
        """Maps the memory system's result back to the pipeline's result format."""
        is_successful = bool(novel_result.generated_content) and not novel_result.generation_metadata.get("error")
        
        # Basic mapping of scores
        consistency_score = novel_result.quality_score
        originality_score = novel_result.originality_score
        
        # Determine approval status based on consistency issues
        approval_status = "approved" if not novel_result.consistency_issues else "pending_review"
        
        return GenerationResult(
            request_id=request.request_id,
            success=is_successful,
            generated_content=novel_result.generated_content,
            consistency_score=consistency_score,
            originality_score=originality_score,
            approval_status=approval_status,
            proposal_id=None,  # Approval logic is now internal to memory system, this might need adjustment
            word_count=len(novel_result.generated_content.split()),
            warnings=[str(issue) for issue in novel_result.consistency_issues],
            errors=[novel_result.generation_metadata.get("error")] if novel_result.generation_metadata.get("error") else []
        )

    async def generate_content(self, request: GenerationRequest) -> GenerationResult:
        """Main content generation method using the integrated memory system."""
        
        start_time = time.time()
        self.active_generations[request.request_id] = request
        
        try:
            # 1. Map the request to the context required by the memory system
            novel_context = self._map_request_to_novel_context(request)
            
            # 2. Delegate the entire generation process to the memory system
            novel_result = await self.memory_system.generate_with_full_context(
                generation_context=novel_context,
                prompt=request.user_prompt
            )
            
            # 3. Map the result back to the pipeline's public format
            pipeline_result = self._map_novel_result_to_pipeline_result(novel_result, request)
            
            total_time_ms = (time.time() - start_time) * 1000
            pipeline_result.total_processing_time_ms = total_time_ms
            
            return pipeline_result

        except Exception as e:
            logger.error(f"Error in refactored generation pipeline: {e}", exc_info=True)
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                errors=[str(e)],
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
        finally:
            self.active_generations.pop(request.request_id, None)

    # The public API methods are now simple wrappers that create the request object
    async def generate_narrative_continuation(self, **kwargs) -> GenerationResult:
        request = GenerationRequest(generation_type=GenerationType.NARRATIVE_CONTINUATION, **kwargs)
        return await self.generate_content(request)

    async def generate_character_dialogue(self, **kwargs) -> GenerationResult:
        request = GenerationRequest(generation_type=GenerationType.CHARACTER_DIALOGUE, **kwargs)
        return await self.generate_content(request)

    async def generate_scene_description(self, **kwargs) -> GenerationResult:
        request = GenerationRequest(generation_type=GenerationType.SCENE_DESCRIPTION, **kwargs)
        return await self.generate_content(request)

    # ... other public API methods would be similarly simplified ...