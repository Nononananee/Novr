"""
Creative agent tools for character development and emotional analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic_ai import RunContext

from .agent import rag_agent, AgentDependencies

logger = logging.getLogger(__name__)


# Advanced Character Development Tools
@rag_agent.tool
async def analyze_character_consistency(
    ctx: RunContext[AgentDependencies],
    character_id: str,
    from_chapter: int = 1,
    to_chapter: int = None
) -> Dict[str, Any]:
    """
    Analyze character consistency across chapters.
    
    This tool examines how consistently a character is portrayed throughout
    the story, checking for personality consistency, dialogue patterns,
    and behavioral coherence. Essential for maintaining character integrity.
    
    Args:
        character_id: ID of the character to analyze
        from_chapter: Starting chapter for analysis (default: 1)
        to_chapter: Ending chapter for analysis (optional, analyzes all if not specified)
    
    Returns:
        Character consistency analysis with scores and recommendations
    """
    try:
        from .db_utils import get_character, get_character_arc
        from .creative_tools import analyze_character_consistency_internal
        
        # Get character details
        character = await get_character(character_id)
        if not character:
            return {"error": "Character not found", "character_id": character_id}
        
        # Get character arc data
        arc_data = await get_character_arc(character_id)
        
        if not arc_data:
            return {
                "character_id": character_id,
                "character_name": character["name"],
                "consistency_score": 0.0,
                "message": "No character appearances found for analysis"
            }
        
        # Filter by chapter range if specified
        if to_chapter:
            arc_data = [
                scene for scene in arc_data 
                if from_chapter <= scene["chapter_number"] <= to_chapter
            ]
        else:
            arc_data = [
                scene for scene in arc_data 
                if scene["chapter_number"] >= from_chapter
            ]
        
        # Analyze consistency
        analysis = await analyze_character_consistency_internal(character, arc_data)
        
        return {
            "character_id": character_id,
            "character_name": character["name"],
            "consistency_score": analysis["consistency_score"],
            "personality_consistency": analysis["personality_consistency"],
            "dialogue_consistency": analysis["dialogue_consistency"],
            "behavioral_consistency": analysis["behavioral_consistency"],
            "violations": analysis["violations"],
            "suggestions": analysis["suggestions"],
            "total_appearances": len(arc_data),
            "chapters_analyzed": f"{from_chapter}-{to_chapter or 'end'}"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze character consistency for {character_id}"
        }


@rag_agent.tool
async def generate_character_development(
    ctx: RunContext[AgentDependencies],
    character_id: str,
    target_development: str,
    current_chapter: int,
    development_type: str = "gradual"
) -> Dict[str, Any]:
    """
    Generate character development suggestions for a specific character.
    
    This tool creates development suggestions based on the character's current
    state and desired growth. Helps maintain realistic character progression
    and provides specific scenes or moments for development.
    
    Args:
        character_id: ID of the character to develop
        target_development: Description of desired character growth
        current_chapter: Current chapter number for context
        development_type: Type of development (gradual, dramatic, revelation)
    
    Returns:
        Character development plan with specific suggestions and scenes
    """
    try:
        from .db_utils import get_character, get_character_arc
        from .creative_tools import generate_character_development_internal
        
        # Get character details
        character = await get_character(character_id)
        if not character:
            return {"error": "Character not found", "character_id": character_id}
        
        # Get recent character arc for context
        arc_data = await get_character_arc(character_id)
        recent_appearances = [
            scene for scene in arc_data 
            if scene["chapter_number"] <= current_chapter
        ][-5:]  # Last 5 appearances
        
        # Generate development plan
        development_plan = await generate_character_development_internal(
            character, target_development, current_chapter, development_type, recent_appearances
        )
        
        return {
            "character_id": character_id,
            "character_name": character["name"],
            "target_development": target_development,
            "development_type": development_type,
            "current_chapter": current_chapter,
            "development_plan": development_plan["plan"],
            "suggested_scenes": development_plan["scenes"],
            "dialogue_suggestions": development_plan["dialogue"],
            "internal_changes": development_plan["internal_changes"],
            "external_manifestations": development_plan["external_manifestations"],
            "timeline": development_plan["timeline"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate character development for {character_id}"
        }


@rag_agent.tool
async def track_character_relationships(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    focus_character: str = None
) -> Dict[str, Any]:
    """
    Track and analyze character relationships within a novel.
    
    This tool maps relationships between characters, tracks how they evolve,
    and identifies relationship dynamics. Essential for maintaining consistent
    character interactions and developing compelling relationship arcs.
    
    Args:
        novel_id: ID of the novel to analyze
        focus_character: Optional character name to focus analysis on
    
    Returns:
        Character relationship map with dynamics and evolution tracking
    """
    try:
        from .db_utils import list_characters
        from .creative_tools import build_character_relationship_map
        
        # Get all characters in the novel
        characters = await list_characters(novel_id)
        
        if not characters:
            return {
                "novel_id": novel_id,
                "message": "No characters found in this novel"
            }
        
        # Build relationship map
        relationship_map = await build_character_relationship_map(
            novel_id, characters, focus_character
        )
        
        return {
            "novel_id": novel_id,
            "focus_character": focus_character,
            "total_characters": len(characters),
            "relationship_map": relationship_map["relationships"],
            "relationship_dynamics": relationship_map["dynamics"],
            "evolution_patterns": relationship_map["evolution"],
            "conflict_points": relationship_map["conflicts"],
            "alliance_patterns": relationship_map["alliances"],
            "suggestions": relationship_map["suggestions"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to track character relationships for novel {novel_id}"
        }


# Emotional Analysis Tools
@rag_agent.tool
async def analyze_emotional_content(
    ctx: RunContext[AgentDependencies],
    content: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze the emotional content and tone of a text passage.
    
    This tool examines emotional elements in narrative content, identifying
    dominant emotions, emotional intensity, and emotional progression.
    Essential for maintaining emotional consistency and impact.
    
    Args:
        content: Text content to analyze
        context: Optional context information (characters, scene type, etc.)
    
    Returns:
        Emotional analysis with dominant emotions, intensity, and suggestions
    """
    try:
        from .creative_tools import analyze_emotional_content_internal
        
        # Perform emotional analysis
        analysis = await analyze_emotional_content_internal(content, context or {})
        
        return {
            "content_length": len(content),
            "dominant_emotions": analysis["dominant_emotions"],
            "emotional_intensity": analysis["intensity"],
            "emotional_progression": analysis["progression"],
            "emotional_triggers": analysis["triggers"],
            "character_emotions": analysis["character_emotions"],
            "scene_mood": analysis["scene_mood"],
            "emotional_consistency": analysis["consistency_score"],
            "suggestions": analysis["suggestions"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to analyze emotional content"
        }


@rag_agent.tool
async def generate_emotional_scene(
    ctx: RunContext[AgentDependencies],
    emotional_tone: str,
    intensity: float = 0.7,
    characters: List[str] = None,
    scene_context: str = "",
    word_count: int = 300
) -> Dict[str, Any]:
    """
    Generate a scene with specific emotional tone and intensity.
    
    This tool creates narrative content designed to evoke specific emotions,
    with appropriate character reactions, dialogue, and descriptive elements.
    Useful for crafting emotionally impactful scenes.
    
    Args:
        emotional_tone: Target emotional tone (joyful, melancholic, tense, etc.)
        intensity: Emotional intensity from 0.0 to 1.0
        characters: List of character names to include (optional)
        scene_context: Context or setting for the scene
        word_count: Target word count for the generated scene
    
    Returns:
        Generated scene content with emotional analysis
    """
    try:
        from .creative_tools import generate_emotional_scene_internal
        
        # Generate emotional scene
        scene_content = await generate_emotional_scene_internal(
            emotional_tone, intensity, characters or [], scene_context, word_count
        )
        
        return {
            "emotional_tone": emotional_tone,
            "target_intensity": intensity,
            "characters_included": characters or [],
            "scene_context": scene_context,
            "generated_content": scene_content["content"],
            "actual_word_count": len(scene_content["content"].split()),
            "emotional_elements": scene_content["emotional_elements"],
            "character_reactions": scene_content["character_reactions"],
            "descriptive_techniques": scene_content["techniques"],
            "emotional_impact_score": scene_content["impact_score"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate emotional scene with tone '{emotional_tone}'"
        }


@rag_agent.tool
async def track_emotional_arc(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    entity_type: str = "character",
    entity_name: str = None
) -> Dict[str, Any]:
    """
    Track emotional arc progression for characters or the overall story.
    
    This tool analyzes how emotions develop and change throughout the narrative,
    identifying emotional peaks, valleys, and turning points. Essential for
    crafting satisfying emotional journeys.
    
    Args:
        novel_id: ID of the novel to analyze
        entity_type: Type of entity to track (character, story, chapter)
        entity_name: Name of specific character or entity (optional)
    
    Returns:
        Emotional arc analysis with progression, turning points, and patterns
    """
    try:
        from .creative_tools import track_emotional_arc_internal
        
        # Track emotional arc
        arc_analysis = await track_emotional_arc_internal(
            novel_id, entity_type, entity_name
        )
        
        return {
            "novel_id": novel_id,
            "entity_type": entity_type,
            "entity_name": entity_name,
            "emotional_progression": arc_analysis["progression"],
            "turning_points": arc_analysis["turning_points"],
            "emotional_peaks": arc_analysis["peaks"],
            "emotional_valleys": arc_analysis["valleys"],
            "overall_trajectory": arc_analysis["trajectory"],
            "consistency_score": arc_analysis["consistency"],
            "suggestions": arc_analysis["suggestions"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to track emotional arc for {entity_type} in novel {novel_id}"
        }