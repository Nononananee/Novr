"""
Advanced Generation Pipeline for Creative Novel Writing

This module provides a sophisticated pipeline for generating creative content
with consistency validation, quality control, and human-in-the-loop approval.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import json

from .models import EmotionalTone, ChunkType, MessageRole
from .enhanced_context_builder import (
    EnhancedContextBuilder, ContextBuildRequest, ContextType, ContextBuildResult
)
from .consistency_validators import (
    CharacterConsistencyValidator, PlotConsistencyValidator, 
    WorldBuildingValidator, EmotionalToneValidator
)
from .providers import get_llm_client

logger = logging.getLogger(__name__)


class GenerationType(str, Enum):
    """Types of content generation."""
    NARRATIVE_CONTINUATION = "narrative_continuation"
    CHARACTER_DIALOGUE = "character_dialogue"
    SCENE_DESCRIPTION = "scene_description"
    CHAPTER_OPENING = "chapter_opening"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CHARACTER_INTRODUCTION = "character_introduction"
    WORLD_BUILDING = "world_building"
    PLOT_DEVELOPMENT = "plot_development"


class GenerationMode(str, Enum):
    """Generation modes."""
    CREATIVE = "creative"
    STRUCTURED = "structured"
    EXPERIMENTAL = "experimental"
    CONSERVATIVE = "conservative"


class QualityLevel(str, Enum):
    """Quality levels for generation."""
    DRAFT = "draft"
    REVIEW = "review"
    FINAL = "final"
    PUBLICATION = "publication"


@dataclass
class GenerationRequest:
    """Request for content generation."""
    prompt: str
    generation_type: GenerationType
    mode: GenerationMode = GenerationMode.CREATIVE
    quality_level: QualityLevel = QualityLevel.DRAFT
    max_tokens: int = 1000
    temperature: float = 0.8
    context_request: Optional[ContextBuildRequest] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    target_characters: List[str] = field(default_factory=list)
    scene_location: Optional[str] = None
    emotional_tone: Optional[EmotionalTone] = None
    narrative_constraints: Dict[str, Any] = field(default_factory=dict)
    require_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validator_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of content generation."""
    content: str
    generation_type: GenerationType
    mode: GenerationMode
    quality_score: float
    validation_results: List[ValidationResult] = field(default_factory=list)
    context_used: Optional[ContextBuildResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    token_count: int = 0
    requires_approval: bool = False
    approval_status: str = "pending"
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedGenerationPipeline:
    """Advanced pipeline for creative content generation."""
    
    def __init__(self, 
                 context_builder: Optional[EnhancedContextBuilder] = None,
                 enable_validation: bool = True,
                 enable_approval_workflow: bool = False):
        """Initialize the generation pipeline."""
        self.context_builder = context_builder or EnhancedContextBuilder()
        self.enable_validation = enable_validation
        self.enable_approval_workflow = enable_approval_workflow
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        if enable_validation:
            self.validators = {
                "character": CharacterConsistencyValidator(),
                "plot": PlotConsistencyValidator(),
                "world": WorldBuildingValidator(),
                "tone": EmotionalToneValidator()
            }
        else:
            self.validators = {}
        
        # Initialize LLM client
        self.llm_client = get_llm_client()
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate content based on the request."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting generation: {request.generation_type} - {request.prompt[:100]}...")
            
            # Step 1: Build context if needed
            context_result = None
            if request.context_request:
                context_result = await self.context_builder.build_context(request.context_request)
                self.logger.info(f"Built context with {len(context_result.elements)} elements")
            
            # Step 2: Prepare generation prompt
            generation_prompt = await self._prepare_generation_prompt(request, context_result)
            
            # Step 3: Generate content
            generated_content = await self._generate_content(generation_prompt, request)
            
            # Step 4: Validate content if enabled
            validation_results = []
            if self.enable_validation:
                validation_results = await self._validate_content(generated_content, request, context_result)
            
            # Step 5: Calculate quality score
            quality_score = self._calculate_quality_score(generated_content, validation_results, request)
            
            # Step 6: Create result
            generation_time = (datetime.now() - start_time).total_seconds()
            token_count = len(generated_content.split())  # Rough estimation
            
            result = GenerationResult(
                content=generated_content,
                generation_type=request.generation_type,
                mode=request.mode,
                quality_score=quality_score,
                validation_results=validation_results,
                context_used=context_result,
                metadata=request.metadata.copy(),
                generation_time=generation_time,
                token_count=token_count,
                requires_approval=request.require_approval,
                approval_status="pending" if request.require_approval else "approved"
            )
            
            self.logger.info(f"Generation completed: {quality_score:.2f} quality score, {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return GenerationResult(
                content=f"Generation failed: {str(e)}",
                generation_type=request.generation_type,
                mode=request.mode,
                quality_score=0.0,
                validation_results=[ValidationResult(
                    is_valid=False,
                    score=0.0,
                    issues=[f"Generation error: {str(e)}"],
                    validator_name="pipeline"
                )],
                generation_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate content as a stream."""
        try:
            # Build context first
            context_result = None
            if request.context_request:
                context_result = await self.context_builder.build_context(request.context_request)
            
            # Prepare prompt
            generation_prompt = await self._prepare_generation_prompt(request, context_result)
            
            # Stream generation
            async for chunk in self._generate_content_stream(generation_prompt, request):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Stream generation failed: {e}")
            yield f"Error: {str(e)}"
    
    async def _prepare_generation_prompt(self, 
                                       request: GenerationRequest, 
                                       context_result: Optional[ContextBuildResult]) -> str:
        """Prepare the generation prompt with context."""
        prompt_parts = []
        
        # Add system instructions based on generation type
        system_prompt = self._get_system_prompt(request)
        prompt_parts.append(system_prompt)
        
        # Add context if available
        if context_result and context_result.elements:
            prompt_parts.append("\n=== CONTEXT ===")
            
            # Add character profiles
            if context_result.character_profiles:
                prompt_parts.append("\nCharacter Information:")
                for char_name, profile in context_result.character_profiles.items():
                    prompt_parts.append(f"- {char_name}: {'; '.join(profile.get('context_snippets', [])[:2])}")
            
            # Add plot context
            if context_result.plot_threads:
                prompt_parts.append("\nPlot Context:")
                for thread in context_result.plot_threads[:3]:
                    prompt_parts.append(f"- {thread['content']}")
            
            # Add world building
            if context_result.world_elements:
                prompt_parts.append("\nWorld Building:")
                for location, descriptions in context_result.world_elements.items():
                    prompt_parts.append(f"- {location}: {descriptions[0] if descriptions else 'No description'}")
            
            # Add consistency notes
            if context_result.consistency_notes:
                prompt_parts.append("\nConsistency Notes:")
                for note in context_result.consistency_notes:
                    prompt_parts.append(f"- {note}")
            
            # Add generation hints
            if context_result.generation_hints:
                prompt_parts.append("\nGeneration Guidelines:")
                for hint in context_result.generation_hints:
                    prompt_parts.append(f"- {hint}")
        
        # Add specific constraints
        if request.emotional_tone:
            prompt_parts.append(f"\nEmotional Tone: {request.emotional_tone}")
        
        if request.target_characters:
            prompt_parts.append(f"\nFocus Characters: {', '.join(request.target_characters)}")
        
        if request.scene_location:
            prompt_parts.append(f"\nScene Location: {request.scene_location}")
        
        # Add the actual prompt
        prompt_parts.append(f"\n=== GENERATION REQUEST ===\n{request.prompt}")
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self, request: GenerationRequest) -> str:
        """Get system prompt based on generation type."""
        base_prompt = "You are a skilled creative writer specializing in novel writing."
        
        type_prompts = {
            GenerationType.NARRATIVE_CONTINUATION: 
                "Continue the narrative maintaining consistency with established characters, plot, and world building. Focus on smooth transitions and natural story progression.",
            
            GenerationType.CHARACTER_DIALOGUE:
                "Write authentic dialogue that reflects each character's unique voice, personality, and current emotional state. Ensure dialogue serves the story and reveals character.",
            
            GenerationType.SCENE_DESCRIPTION:
                "Create vivid, immersive scene descriptions that engage the senses and establish atmosphere. Balance detail with pacing.",
            
            GenerationType.CHAPTER_OPENING:
                "Craft a compelling chapter opening that hooks the reader while establishing setting, mood, and advancing the plot.",
            
            GenerationType.CONFLICT_RESOLUTION:
                "Develop meaningful conflict resolution that feels earned and consistent with character development and story themes.",
            
            GenerationType.CHARACTER_INTRODUCTION:
                "Introduce characters in a way that reveals personality, motivations, and role in the story through action and dialogue rather than exposition.",
            
            GenerationType.WORLD_BUILDING:
                "Expand the story world with rich, consistent details that enhance the narrative without overwhelming it.",
            
            GenerationType.PLOT_DEVELOPMENT:
                "Advance the plot through meaningful events that arise naturally from character actions and established story elements."
        }
        
        mode_instructions = {
            GenerationMode.CREATIVE: "Be imaginative and take creative risks while maintaining story consistency.",
            GenerationMode.STRUCTURED: "Follow established patterns and maintain tight narrative control.",
            GenerationMode.EXPERIMENTAL: "Explore new narrative techniques and unconventional approaches.",
            GenerationMode.CONSERVATIVE: "Prioritize consistency and traditional storytelling methods."
        }
        
        return f"{base_prompt}\n\n{type_prompts.get(request.generation_type, '')}\n\n{mode_instructions.get(request.mode, '')}"
    
    async def _generate_content(self, prompt: str, request: GenerationRequest) -> str:
        """Generate content using the LLM."""
        try:
            # Prepare generation parameters
            generation_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            # Adjust parameters based on mode
            if request.mode == GenerationMode.CONSERVATIVE:
                generation_params["temperature"] = min(0.6, request.temperature)
            elif request.mode == GenerationMode.EXPERIMENTAL:
                generation_params["temperature"] = min(1.0, request.temperature + 0.2)
            
            # Generate content
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": prompt}
                ],
                **generation_params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    async def _generate_content_stream(self, prompt: str, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate content as a stream."""
        try:
            generation_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True
            }
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": prompt}
                ],
                **generation_params
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Stream generation failed: {e}")
            yield f"Error: {str(e)}"
    
    async def _validate_content(self, 
                              content: str, 
                              request: GenerationRequest,
                              context_result: Optional[ContextBuildResult]) -> List[ValidationResult]:
        """Validate generated content."""
        validation_results = []
        
        try:
            # Character consistency validation
            if "character" in self.validators and request.target_characters:
                char_result = await self.validators["character"].validate(
                    content, request.target_characters, context_result
                )
                validation_results.append(char_result)
            
            # Plot consistency validation
            if "plot" in self.validators:
                plot_result = await self.validators["plot"].validate(
                    content, request.narrative_constraints, context_result
                )
                validation_results.append(plot_result)
            
            # World building validation
            if "world" in self.validators and request.scene_location:
                world_result = await self.validators["world"].validate(
                    content, request.scene_location, context_result
                )
                validation_results.append(world_result)
            
            # Emotional tone validation
            if "tone" in self.validators and request.emotional_tone:
                tone_result = await self.validators["tone"].validate(
                    content, request.emotional_tone, context_result
                )
                validation_results.append(tone_result)
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results.append(ValidationResult(
                is_valid=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                validator_name="validation_pipeline"
            ))
        
        return validation_results
    
    def _calculate_quality_score(self, 
                               content: str, 
                               validation_results: List[ValidationResult],
                               request: GenerationRequest) -> float:
        """Calculate overall quality score."""
        if not validation_results:
            # Basic quality metrics without validation
            base_score = 0.7
            
            # Length appropriateness
            word_count = len(content.split())
            target_words = request.max_tokens * 0.75  # Rough conversion
            length_score = min(1.0, word_count / target_words) if target_words > 0 else 0.5
            
            # Basic content quality (simple heuristics)
            content_score = 0.8 if len(content.strip()) > 50 else 0.3
            
            return (base_score + length_score + content_score) / 3
        
        # Calculate score from validation results
        valid_results = [r for r in validation_results if r.score > 0]
        if not valid_results:
            return 0.3
        
        avg_score = sum(r.score for r in valid_results) / len(valid_results)
        
        # Penalty for validation issues
        total_issues = sum(len(r.issues) for r in validation_results)
        issue_penalty = min(0.3, total_issues * 0.05)
        
        return max(0.0, avg_score - issue_penalty)


# Factory functions
def create_generation_pipeline(enable_validation: bool = True,
                             enable_approval: bool = False) -> AdvancedGenerationPipeline:
    """Create a generation pipeline instance."""
    return AdvancedGenerationPipeline(
        enable_validation=enable_validation,
        enable_approval_workflow=enable_approval
    )


async def quick_generate(prompt: str, 
                        generation_type: GenerationType = GenerationType.NARRATIVE_CONTINUATION,
                        max_tokens: int = 500) -> str:
    """Quick generation function for simple use cases."""
    pipeline = create_generation_pipeline(enable_validation=False)
    
    request = GenerationRequest(
        prompt=prompt,
        generation_type=generation_type,
        max_tokens=max_tokens
    )
    
    result = await pipeline.generate(request)
    return result.content