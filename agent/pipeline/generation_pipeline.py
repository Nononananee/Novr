"""
Novel-aware Generation Pipeline for Creative RAG System
Orchestrates content generation with enhanced narrative intelligence and emotional consistency.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

# Import novel-specific models
from .models import (
    ValidationResult, EmotionalTone, ChunkType, CharacterRole, PlotSignificance,
    Character, Scene, Chapter, Novel, EmotionalArc, NovelGenerationRequest,
    CharacterAnalysisRequest, EmotionalAnalysisRequest, PlotAnalysisRequest,
    CharacterAnalysisResponse, EmotionalAnalysisResponse, PlotAnalysisResponse,
    NovelConsistencyReport
)

# Import graph utilities for novel-specific operations
from .graph_utils import (
    search_character_arc, find_emotional_scenes, analyze_plot_structure,
    get_character_relationships, add_novel_content_to_graph
)

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


class NovelAwareGenerationPipeline:
    """
    Novel-aware generation pipeline with enhanced narrative intelligence.
    Provides comprehensive novel writing assistance with character consistency,
    emotional arc tracking, and plot coherence validation.
    """
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system  # Optional for now
        self.active_generations: Dict[str, GenerationRequest] = {}
        self.character_cache: Dict[str, Character] = {}
        self.emotional_context_cache: Dict[str, Dict[str, Any]] = {}
        self.plot_thread_cache: Dict[str, Dict[str, Any]] = {}

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
            scene_location=request.current_scene,
            # Add emotional context if available
            character_emotional_states=getattr(request, 'character_emotional_states', None),
            emotional_arc_requirements=getattr(request, 'emotional_arc_requirements', None),
            target_emotional_tone=getattr(request, 'target_emotional_tone', None)
        )

    def _map_novel_result_to_pipeline_result(self, novel_result: NovelGenerationResult, request: GenerationRequest) -> GenerationResult:
        """Maps the memory system's result back to the pipeline's result format."""
        is_successful = bool(novel_result.generated_content) and not novel_result.generation_metadata.get("error")
        
        # Basic mapping of scores
        consistency_score = novel_result.quality_score
        originality_score = novel_result.originality_score
        
        # Determine approval status based on consistency issues
        approval_status = "approved" if not novel_result.consistency_issues else "pending_review"
        
        # Include emotional analysis in warnings if available
        warnings = [str(issue) for issue in novel_result.consistency_issues]
        if hasattr(novel_result, 'emotional_consistency_score') and novel_result.emotional_consistency_score < 0.5:
            warnings.append(f"Low emotional consistency score: {novel_result.emotional_consistency_score:.2f}")
        
        return GenerationResult(
            request_id=request.request_id,
            success=is_successful,
            generated_content=novel_result.generated_content,
            consistency_score=consistency_score,
            originality_score=originality_score,
            approval_status=approval_status,
            proposal_id=None,  # Approval logic is now internal to memory system, this might need adjustment
            word_count=len(novel_result.generated_content.split()),
            warnings=warnings,
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

    # Novel-specific generation methods
    async def generate_with_emotional_context(
        self,
        request: GenerationRequest,
        target_emotion: EmotionalTone,
        emotional_intensity: float = 0.7
    ) -> GenerationResult:
        """Generate content with specific emotional context."""
        start_time = time.time()
        
        try:
            # Enhance request with emotional context
            enhanced_prompt = f"Generate content with {target_emotion.value} emotional tone (intensity: {emotional_intensity}). {request.user_prompt or ''}"
            
            # Get emotional context from previous scenes
            emotional_context = await self._get_emotional_context(
                request.target_characters, 
                request.current_chapter
            )
            
            # Simple generation for now (would integrate with memory system)
            generated_content = await self._generate_emotionally_aware_content(
                enhanced_prompt,
                target_emotion,
                emotional_intensity,
                emotional_context
            )
            
            # Validate emotional consistency
            consistency_score = await self._validate_emotional_consistency(
                generated_content,
                target_emotion,
                emotional_context
            )
            
            return GenerationResult(
                request_id=request.request_id,
                success=True,
                generated_content=generated_content,
                consistency_score=consistency_score,
                originality_score=0.8,  # Placeholder
                word_count=len(generated_content.split()),
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Error in emotional generation: {e}")
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                errors=[str(e)],
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def generate_character_consistent_dialogue(
        self,
        character_name: str,
        dialogue_context: str,
        novel_title: Optional[str] = None
    ) -> GenerationResult:
        """Generate dialogue that's consistent with character personality."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Get character development arc
            character_arc = await search_character_arc(character_name, novel_title)
            
            # Get character relationships
            relationships = await get_character_relationships(character_name, novel_title=novel_title)
            
            # Generate character-consistent dialogue
            dialogue = await self._generate_character_dialogue(
                character_name,
                dialogue_context,
                character_arc,
                relationships
            )
            
            # Validate character consistency
            consistency_score = await self._validate_character_consistency(
                dialogue,
                character_name,
                character_arc
            )
            
            return GenerationResult(
                request_id=request_id,
                success=True,
                generated_content=dialogue,
                consistency_score=consistency_score,
                originality_score=0.75,
                word_count=len(dialogue.split()),
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Error in character dialogue generation: {e}")
            return GenerationResult(
                request_id=request_id,
                success=False,
                errors=[str(e)],
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def analyze_character_development(
        self,
        request: CharacterAnalysisRequest
    ) -> CharacterAnalysisResponse:
        """Analyze character development across chapters."""
        try:
            # Get character arc from graph
            character_arc = await search_character_arc(
                request.character_name, 
                request.novel_id
            )
            
            # Get character relationships
            relationships = await get_character_relationships(
                request.character_name,
                novel_title=request.novel_id
            )
            
            # Analyze development patterns
            development_score = self._calculate_development_score(character_arc)
            consistency_score = self._calculate_character_consistency_score(character_arc)
            
            # Extract personality traits and dialogue patterns
            personality_traits = self._extract_personality_traits(character_arc)
            dialogue_patterns = self._extract_dialogue_patterns(character_arc)
            
            # Generate suggestions
            suggestions = self._generate_character_suggestions(
                character_arc,
                development_score,
                consistency_score
            )
            
            return CharacterAnalysisResponse(
                character_name=request.character_name,
                development_score=development_score,
                consistency_score=consistency_score,
                relationships=[rel for rel in relationships.get('relationships', [])],
                development_arc=[arc for arc in character_arc],
                personality_traits=personality_traits,
                dialogue_patterns=dialogue_patterns,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in character analysis: {e}")
            return CharacterAnalysisResponse(
                character_name=request.character_name,
                development_score=0.0,
                consistency_score=0.0,
                suggestions=[f"Error analyzing character: {str(e)}"]
            )
    
    async def analyze_emotional_arc(
        self,
        request: EmotionalAnalysisRequest
    ) -> EmotionalAnalysisResponse:
        """Analyze emotional arc of content."""
        try:
            # Analyze emotional content
            emotions = self._extract_emotions_from_content(request.content)
            emotional_intensity = self._calculate_emotional_intensity(request.content)
            emotional_progression = self._analyze_emotional_progression(request.content)
            
            # Calculate consistency with context
            consistency_score = 0.8  # Placeholder
            if request.context:
                consistency_score = self._calculate_emotional_context_consistency(
                    emotions,
                    request.context
                )
            
            # Generate suggestions
            suggestions = self._generate_emotional_suggestions(
                emotions,
                emotional_intensity,
                consistency_score
            )
            
            return EmotionalAnalysisResponse(
                dominant_emotions=emotions,
                emotional_intensity=emotional_intensity,
                emotional_progression=emotional_progression,
                consistency_score=consistency_score,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in emotional analysis: {e}")
            return EmotionalAnalysisResponse(
                dominant_emotions=[],
                emotional_intensity=0.0,
                consistency_score=0.0,
                suggestions=[f"Error analyzing emotions: {str(e)}"]
            )
    
    async def generate_consistency_report(
        self,
        novel_id: str
    ) -> NovelConsistencyReport:
        """Generate comprehensive consistency report for a novel."""
        try:
            # Analyze different aspects of consistency
            character_consistency = await self._analyze_novel_character_consistency(novel_id)
            plot_consistency = await self._analyze_novel_plot_consistency(novel_id)
            emotional_consistency = await self._analyze_novel_emotional_consistency(novel_id)
            style_consistency = await self._analyze_novel_style_consistency(novel_id)
            
            # Calculate overall score
            overall_score = (
                character_consistency + plot_consistency + 
                emotional_consistency + style_consistency
            ) / 4
            
            # Collect violations and suggestions
            violations = []
            suggestions = []
            
            if character_consistency < 0.7:
                violations.append({
                    "type": "character_consistency",
                    "score": character_consistency,
                    "description": "Character behavior inconsistencies detected"
                })
                suggestions.append("Review character development arcs for consistency")
            
            if plot_consistency < 0.7:
                violations.append({
                    "type": "plot_consistency", 
                    "score": plot_consistency,
                    "description": "Plot inconsistencies or holes detected"
                })
                suggestions.append("Review plot threads for logical consistency")
            
            return NovelConsistencyReport(
                novel_id=novel_id,
                overall_score=overall_score,
                character_consistency=character_consistency,
                plot_consistency=plot_consistency,
                emotional_consistency=emotional_consistency,
                style_consistency=style_consistency,
                violations=violations,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error generating consistency report: {e}")
            return NovelConsistencyReport(
                novel_id=novel_id,
                overall_score=0.0,
                character_consistency=0.0,
                plot_consistency=0.0,
                emotional_consistency=0.0,
                style_consistency=0.0,
                suggestions=[f"Error generating report: {str(e)}"]
            )
    
    # Helper methods for novel-specific functionality
    async def _get_emotional_context(
        self,
        characters: List[str],
        chapter: int
    ) -> Dict[str, Any]:
        """Get emotional context for characters in a chapter."""
        context = {}
        
        for character in characters:
            try:
                emotional_scenes = await find_emotional_scenes(
                    "any",  # Get all emotional content
                    novel_title=None  # Would be provided in real implementation
                )
                
                # Filter by character and chapter (simplified)
                character_emotions = [
                    scene for scene in emotional_scenes 
                    if character.lower() in scene.get('fact', '').lower()
                ]
                
                context[character] = {
                    'recent_emotions': character_emotions[:3],  # Last 3 emotional moments
                    'dominant_emotion': self._extract_dominant_emotion(character_emotions)
                }
                
            except Exception as e:
                logger.warning(f"Could not get emotional context for {character}: {e}")
                context[character] = {'recent_emotions': [], 'dominant_emotion': 'neutral'}
        
        return context
    
    async def _generate_emotionally_aware_content(
        self,
        prompt: str,
        target_emotion: EmotionalTone,
        intensity: float,
        context: Dict[str, Any]
    ) -> str:
        """Generate content with emotional awareness."""
        # This would integrate with an LLM in a real implementation
        # For now, return a placeholder that demonstrates the concept
        
        emotion_words = {
            EmotionalTone.JOYFUL: ["happy", "delighted", "cheerful", "bright"],
            EmotionalTone.MELANCHOLIC: ["sad", "wistful", "somber", "heavy"],
            EmotionalTone.TENSE: ["anxious", "worried", "on edge", "nervous"],
            EmotionalTone.DRAMATIC: ["intense", "powerful", "overwhelming", "striking"]
        }
        
        words = emotion_words.get(target_emotion, ["neutral", "calm", "steady"])
        
        return f"Generated content with {target_emotion.value} tone (intensity: {intensity}). " \
               f"The scene feels {words[0]} and {words[1]}, with characters experiencing " \
               f"{words[2]} emotions. Context: {len(context)} characters involved."
    
    async def _validate_emotional_consistency(
        self,
        content: str,
        target_emotion: EmotionalTone,
        context: Dict[str, Any]
    ) -> float:
        """Validate emotional consistency of generated content."""
        # Simplified validation - would use more sophisticated analysis
        emotion_keywords = {
            EmotionalTone.JOYFUL: ["happy", "joy", "smile", "laugh", "bright"],
            EmotionalTone.MELANCHOLIC: ["sad", "sorrow", "tears", "heavy", "dark"],
            EmotionalTone.TENSE: ["anxious", "worried", "fear", "nervous", "edge"],
            EmotionalTone.DRAMATIC: ["intense", "powerful", "dramatic", "overwhelming"]
        }
        
        keywords = emotion_keywords.get(target_emotion, [])
        content_lower = content.lower()
        
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        consistency_score = min(matches / len(keywords), 1.0) if keywords else 0.5
        
        return consistency_score


# Maintain backward compatibility
AdvancedGenerationPipeline = NovelAwareGenerationPipeline