"""
Enhanced Context Builder for Creative Novel Generation

This module provides sophisticated context building capabilities for novel generation,
including hierarchical context assembly, character consistency tracking, and narrative flow management.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from .models import (
    ChunkResult, GraphSearchResult, DocumentMetadata,
    EmotionalTone, ChunkType, MessageRole
)
from .core.db_utils import vector_search, hybrid_search
from .core.graph_utils import search_knowledge_graph, get_entity_relationships

logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context for generation."""
    NARRATIVE_CONTINUATION = "narrative_continuation"
    CHARACTER_DIALOGUE = "character_dialogue"
    SCENE_DESCRIPTION = "scene_description"
    CHAPTER_OPENING = "chapter_opening"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CHARACTER_INTRODUCTION = "character_introduction"
    WORLD_BUILDING = "world_building"


class ContextPriority(str, Enum):
    """Priority levels for context elements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContextElement:
    """Individual context element with metadata."""
    content: str
    source: str
    priority: ContextPriority
    relevance_score: float
    context_type: ContextType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextBuildRequest:
    """Request for building context."""
    query: str
    context_type: ContextType
    max_tokens: int = 4000
    include_character_info: bool = True
    include_plot_context: bool = True
    include_world_building: bool = True
    emotional_tone: Optional[EmotionalTone] = None
    target_characters: List[str] = field(default_factory=list)
    scene_location: Optional[str] = None
    time_period: Optional[str] = None
    narrative_constraints: Dict[str, Any] = field(default_factory=dict)


class ContextBuildResult(BaseModel):
    """Result of context building operation."""
    elements: List[ContextElement]
    total_tokens: int
    context_summary: str
    character_profiles: Dict[str, Dict[str, Any]]
    plot_threads: List[Dict[str, Any]]
    world_elements: Dict[str, Any]
    consistency_notes: List[str]
    generation_hints: List[str]


class EnhancedContextBuilder:
    """Enhanced context builder for creative novel generation."""
    
    def __init__(self, max_context_tokens: int = 8000):
        """Initialize the enhanced context builder."""
        self.max_context_tokens = max_context_tokens
        self.logger = logging.getLogger(__name__)
        
    async def build_context(self, request: ContextBuildRequest) -> ContextBuildResult:
        """Build comprehensive context for generation."""
        try:
            self.logger.info(f"Building context for {request.context_type} with query: {request.query[:100]}...")
            
            # Collect context elements from multiple sources
            elements = []
            
            # 1. Vector search for relevant content
            vector_results = await self._get_vector_context(request)
            elements.extend(vector_results)
            
            # 2. Knowledge graph context
            graph_results = await self._get_graph_context(request)
            elements.extend(graph_results)
            
            # 3. Character-specific context
            if request.include_character_info:
                character_results = await self._get_character_context(request)
                elements.extend(character_results)
            
            # 4. Plot context
            if request.include_plot_context:
                plot_results = await self._get_plot_context(request)
                elements.extend(plot_results)
            
            # 5. World building context
            if request.include_world_building:
                world_results = await self._get_world_context(request)
                elements.extend(world_results)
            
            # Prioritize and filter elements
            filtered_elements = self._prioritize_elements(elements, request.max_tokens)
            
            # Build comprehensive result
            result = await self._build_result(filtered_elements, request)
            
            self.logger.info(f"Built context with {len(result.elements)} elements, {result.total_tokens} tokens")
            return result
            
        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            # Return minimal context on error
            return ContextBuildResult(
                elements=[],
                total_tokens=0,
                context_summary="Error building context",
                character_profiles={},
                plot_threads=[],
                world_elements={},
                consistency_notes=[f"Context building error: {str(e)}"],
                generation_hints=["Use general knowledge for generation"]
            )
    
    async def _get_vector_context(self, request: ContextBuildRequest) -> List[ContextElement]:
        """Get context from vector search."""
        try:
            # Perform vector search
            results = await vector_search(
                query=request.query,
                limit=10,
                threshold=0.7
            )
            
            elements = []
            for result in results:
                element = ContextElement(
                    content=result.content,
                    source=f"vector_search:{result.source}",
                    priority=self._determine_priority(result.score),
                    relevance_score=result.score,
                    context_type=request.context_type,
                    metadata={
                        "chunk_type": result.metadata.get("chunk_type", "unknown"),
                        "characters": result.metadata.get("characters", []),
                        "locations": result.metadata.get("locations", []),
                        "emotional_tone": result.metadata.get("emotional_tone")
                    }
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    async def _get_graph_context(self, request: ContextBuildRequest) -> List[ContextElement]:
        """Get context from knowledge graph."""
        try:
            # Search knowledge graph for entities
            graph_results = await search_knowledge_graph(
                query=request.query,
                limit=5
            )
            
            elements = []
            for result in graph_results:
                # Get relationships for each entity
                relationships = await get_entity_relationships(result.entity_name)
                
                # Build context from relationships
                context_content = self._build_graph_context_content(result, relationships)
                
                element = ContextElement(
                    content=context_content,
                    source=f"knowledge_graph:{result.entity_name}",
                    priority=ContextPriority.HIGH,
                    relevance_score=result.relevance_score,
                    context_type=request.context_type,
                    metadata={
                        "entity_type": result.entity_type,
                        "relationships": len(relationships),
                        "graph_data": result.properties
                    }
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Error in graph search: {e}")
            return []
    
    async def _get_character_context(self, request: ContextBuildRequest) -> List[ContextElement]:
        """Get character-specific context."""
        elements = []
        
        # If specific characters are targeted
        if request.target_characters:
            for character in request.target_characters:
                try:
                    # Search for character information
                    char_query = f"character {character} personality traits dialogue"
                    results = await vector_search(query=char_query, limit=3)
                    
                    for result in results:
                        element = ContextElement(
                            content=result.content,
                            source=f"character_context:{character}",
                            priority=ContextPriority.CRITICAL,
                            relevance_score=result.score,
                            context_type=request.context_type,
                            metadata={
                                "character_name": character,
                                "context_focus": "character_development"
                            }
                        )
                        elements.append(element)
                        
                except Exception as e:
                    self.logger.error(f"Error getting character context for {character}: {e}")
        
        return elements
    
    async def _get_plot_context(self, request: ContextBuildRequest) -> List[ContextElement]:
        """Get plot-related context."""
        try:
            # Search for plot-related content
            plot_query = f"{request.query} plot story arc conflict resolution"
            results = await vector_search(query=plot_query, limit=5)
            
            elements = []
            for result in results:
                element = ContextElement(
                    content=result.content,
                    source="plot_context",
                    priority=ContextPriority.HIGH,
                    relevance_score=result.score,
                    context_type=request.context_type,
                    metadata={
                        "context_focus": "plot_development",
                        "narrative_function": result.metadata.get("narrative_function")
                    }
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Error getting plot context: {e}")
            return []
    
    async def _get_world_context(self, request: ContextBuildRequest) -> List[ContextElement]:
        """Get world building context."""
        try:
            # Search for world building elements
            world_query = f"{request.query} setting location world building environment"
            if request.scene_location:
                world_query += f" {request.scene_location}"
            
            results = await vector_search(query=world_query, limit=3)
            
            elements = []
            for result in results:
                element = ContextElement(
                    content=result.content,
                    source="world_context",
                    priority=ContextPriority.MEDIUM,
                    relevance_score=result.score,
                    context_type=request.context_type,
                    metadata={
                        "context_focus": "world_building",
                        "locations": result.metadata.get("locations", [])
                    }
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Error getting world context: {e}")
            return []
    
    def _determine_priority(self, score: float) -> ContextPriority:
        """Determine priority based on relevance score."""
        if score >= 0.9:
            return ContextPriority.CRITICAL
        elif score >= 0.8:
            return ContextPriority.HIGH
        elif score >= 0.7:
            return ContextPriority.MEDIUM
        else:
            return ContextPriority.LOW
    
    def _prioritize_elements(self, elements: List[ContextElement], max_tokens: int) -> List[ContextElement]:
        """Prioritize and filter context elements by token limit."""
        # Sort by priority and relevance score
        priority_order = {
            ContextPriority.CRITICAL: 4,
            ContextPriority.HIGH: 3,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 1
        }
        
        sorted_elements = sorted(
            elements,
            key=lambda x: (priority_order[x.priority], x.relevance_score),
            reverse=True
        )
        
        # Filter by token limit (rough estimation)
        filtered_elements = []
        total_tokens = 0
        
        for element in sorted_elements:
            # Rough token estimation (4 chars per token)
            element_tokens = len(element.content) // 4
            
            if total_tokens + element_tokens <= max_tokens:
                filtered_elements.append(element)
                total_tokens += element_tokens
            else:
                break
        
        return filtered_elements
    
    def _build_graph_context_content(self, result: GraphSearchResult, relationships: List[Dict]) -> str:
        """Build context content from graph data."""
        content_parts = [f"Entity: {result.entity_name} ({result.entity_type})"]
        
        if result.properties:
            content_parts.append("Properties:")
            for key, value in result.properties.items():
                content_parts.append(f"  {key}: {value}")
        
        if relationships:
            content_parts.append("Relationships:")
            for rel in relationships[:5]:  # Limit to top 5 relationships
                content_parts.append(f"  {rel.get('type', 'related_to')} -> {rel.get('target', 'unknown')}")
        
        return "\n".join(content_parts)
    
    async def _build_result(self, elements: List[ContextElement], request: ContextBuildRequest) -> ContextBuildResult:
        """Build the final context result."""
        total_tokens = sum(len(elem.content) // 4 for elem in elements)
        
        # Extract character profiles
        character_profiles = {}
        for element in elements:
            if "character_name" in element.metadata:
                char_name = element.metadata["character_name"]
                if char_name not in character_profiles:
                    character_profiles[char_name] = {
                        "name": char_name,
                        "context_snippets": [],
                        "traits": [],
                        "relationships": []
                    }
                character_profiles[char_name]["context_snippets"].append(element.content[:200])
        
        # Extract plot threads
        plot_threads = []
        for element in elements:
            if element.metadata.get("context_focus") == "plot_development":
                plot_threads.append({
                    "content": element.content[:300],
                    "relevance": element.relevance_score,
                    "narrative_function": element.metadata.get("narrative_function")
                })
        
        # Extract world elements
        world_elements = {}
        for element in elements:
            if element.metadata.get("context_focus") == "world_building":
                locations = element.metadata.get("locations", [])
                for location in locations:
                    if location not in world_elements:
                        world_elements[location] = []
                    world_elements[location].append(element.content[:200])
        
        # Generate consistency notes
        consistency_notes = self._generate_consistency_notes(elements, request)
        
        # Generate generation hints
        generation_hints = self._generate_generation_hints(elements, request)
        
        # Create context summary
        context_summary = self._create_context_summary(elements, request)
        
        return ContextBuildResult(
            elements=elements,
            total_tokens=total_tokens,
            context_summary=context_summary,
            character_profiles=character_profiles,
            plot_threads=plot_threads,
            world_elements=world_elements,
            consistency_notes=consistency_notes,
            generation_hints=generation_hints
        )
    
    def _generate_consistency_notes(self, elements: List[ContextElement], request: ContextBuildRequest) -> List[str]:
        """Generate consistency notes for the context."""
        notes = []
        
        # Check for character consistency
        characters_mentioned = set()
        for element in elements:
            chars = element.metadata.get("characters", [])
            characters_mentioned.update(chars)
        
        if characters_mentioned:
            notes.append(f"Characters in context: {', '.join(characters_mentioned)}")
        
        # Check for emotional tone consistency
        tones = [elem.metadata.get("emotional_tone") for elem in elements if elem.metadata.get("emotional_tone")]
        if tones:
            dominant_tone = max(set(tones), key=tones.count)
            notes.append(f"Dominant emotional tone: {dominant_tone}")
        
        # Check for location consistency
        locations = set()
        for element in elements:
            locs = element.metadata.get("locations", [])
            locations.update(locs)
        
        if locations:
            notes.append(f"Locations mentioned: {', '.join(locations)}")
        
        return notes
    
    def _generate_generation_hints(self, elements: List[ContextElement], request: ContextBuildRequest) -> List[str]:
        """Generate hints for the generation process."""
        hints = []
        
        # Context type specific hints
        if request.context_type == ContextType.CHARACTER_DIALOGUE:
            hints.append("Focus on character voice consistency and personality traits")
            hints.append("Consider character relationships and emotional state")
        
        elif request.context_type == ContextType.SCENE_DESCRIPTION:
            hints.append("Include sensory details and atmospheric elements")
            hints.append("Maintain consistency with established world building")
        
        elif request.context_type == ContextType.NARRATIVE_CONTINUATION:
            hints.append("Maintain narrative flow and pacing")
            hints.append("Consider plot progression and character development")
        
        # Emotional tone hints
        if request.emotional_tone:
            hints.append(f"Maintain {request.emotional_tone} emotional tone throughout")
        
        # Character-specific hints
        if request.target_characters:
            hints.append(f"Focus on development of: {', '.join(request.target_characters)}")
        
        return hints
    
    def _create_context_summary(self, elements: List[ContextElement], request: ContextBuildRequest) -> str:
        """Create a summary of the context."""
        summary_parts = [
            f"Context built for {request.context_type}",
            f"Total elements: {len(elements)}",
            f"Query: {request.query[:100]}..."
        ]
        
        if request.target_characters:
            summary_parts.append(f"Target characters: {', '.join(request.target_characters)}")
        
        if request.scene_location:
            summary_parts.append(f"Scene location: {request.scene_location}")
        
        if request.emotional_tone:
            summary_parts.append(f"Emotional tone: {request.emotional_tone}")
        
        return " | ".join(summary_parts)


# Factory function for easy instantiation
def create_enhanced_context_builder(max_context_tokens: int = 8000) -> EnhancedContextBuilder:
    """Create an enhanced context builder instance."""
    return EnhancedContextBuilder(max_context_tokens=max_context_tokens)