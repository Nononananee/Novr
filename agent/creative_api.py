"""
Creative Intelligence API endpoints for novel writing assistance.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# Create router for creative endpoints
router = APIRouter(prefix="/creative", tags=["creative"])


# Import necessary functions
async def create_session(metadata: Dict[str, Any] = None) -> str:
    """Create a session for agent execution."""
    from .db_utils import create_session as db_create_session
    return await db_create_session(metadata=metadata)


async def execute_agent(message: str, session_id: str, save_conversation: bool = True):
    """Execute agent with message."""
    from .api import execute_agent as api_execute_agent
    return await api_execute_agent(message, session_id, save_conversation=save_conversation)


# Creative Intelligence API Endpoints
@router.post("/novels/{novel_id}/analyze/consistency")
async def analyze_novel_consistency_endpoint(novel_id: str):
    """Generate comprehensive consistency report for a novel."""
    try:
        # Create a temporary session for agent execution
        session_id = await create_session(metadata={"api_call": "consistency_analysis"})
        
        # Execute agent with consistency analysis request
        response, tools_used = await execute_agent(
            message=f"Generate a comprehensive consistency report for novel {novel_id}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "consistency",
            "report": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Consistency analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/characters/{character_id}/analyze/consistency")
async def analyze_character_consistency_endpoint(
    character_id: str,
    from_chapter: int = 1,
    to_chapter: int = None
):
    """Analyze character consistency across chapters."""
    try:
        session_id = await create_session(metadata={"api_call": "character_consistency"})
        
        chapter_range = f"from chapter {from_chapter}"
        if to_chapter:
            chapter_range += f" to chapter {to_chapter}"
        
        response, tools_used = await execute_agent(
            message=f"Analyze character consistency for character {character_id} {chapter_range}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "character_id": character_id,
            "analysis_type": "character_consistency",
            "from_chapter": from_chapter,
            "to_chapter": to_chapter,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Character consistency analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/novels/{novel_id}/analyze/plot")
async def analyze_plot_consistency_endpoint(novel_id: str, focus_thread: str = None):
    """Analyze plot consistency and detect plot holes."""
    try:
        session_id = await create_session(metadata={"api_call": "plot_analysis"})
        
        focus_msg = f" focusing on {focus_thread}" if focus_thread else ""
        response, tools_used = await execute_agent(
            message=f"Analyze plot consistency for novel {novel_id}{focus_msg}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "plot_consistency",
            "focus_thread": focus_thread,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Plot analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/novels/{novel_id}/analyze/emotional-arc")
async def analyze_emotional_arc_endpoint(
    novel_id: str,
    entity_type: str = "character",
    entity_name: str = None
):
    """Track emotional arc progression."""
    try:
        session_id = await create_session(metadata={"api_call": "emotional_arc"})
        
        entity_msg = f" for {entity_type}"
        if entity_name:
            entity_msg += f" {entity_name}"
        
        response, tools_used = await execute_agent(
            message=f"Track emotional arc{entity_msg} in novel {novel_id}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "analysis_type": "emotional_arc",
            "entity_type": entity_type,
            "entity_name": entity_name,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Emotional arc analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/analyze/emotional")
async def analyze_emotional_content_endpoint(content: str, context: Dict[str, Any] = None):
    """Analyze emotional content of text."""
    try:
        session_id = await create_session(metadata={"api_call": "emotional_analysis"})
        
        context_msg = f" with context {context}" if context else ""
        response, tools_used = await execute_agent(
            message=f"Analyze the emotional content of this text{context_msg}: {content[:500]}...",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "content_length": len(content),
            "analysis_type": "emotional_content",
            "context": context,
            "analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Emotional content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/generate/emotional-scene")
async def generate_emotional_scene_endpoint(
    emotional_tone: str,
    intensity: float = 0.7,
    characters: List[str] = None,
    scene_context: str = "",
    word_count: int = 300
):
    """Generate a scene with specific emotional tone."""
    try:
        session_id = await create_session(metadata={"api_call": "scene_generation"})
        
        char_msg = f" with characters {', '.join(characters)}" if characters else ""
        context_msg = f" in context: {scene_context}" if scene_context else ""
        
        response, tools_used = await execute_agent(
            message=f"Generate a {emotional_tone} scene with intensity {intensity}{char_msg}{context_msg}. Target {word_count} words.",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "generation_type": "emotional_scene",
            "emotional_tone": emotional_tone,
            "intensity": intensity,
            "characters": characters or [],
            "scene_context": scene_context,
            "target_word_count": word_count,
            "generated_content": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Emotional scene generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/characters/{character_id}/develop")
async def generate_character_development_endpoint(
    character_id: str,
    target_development: str,
    current_chapter: int,
    development_type: str = "gradual"
):
    """Generate character development suggestions."""
    try:
        session_id = await create_session(metadata={"api_call": "character_development"})
        
        response, tools_used = await execute_agent(
            message=f"Generate {development_type} character development for character {character_id} towards '{target_development}' starting from chapter {current_chapter}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "character_id": character_id,
            "development_type": development_type,
            "target_development": target_development,
            "current_chapter": current_chapter,
            "development_plan": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Character development generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/novels/{novel_id}/relationships")
async def track_character_relationships_endpoint(novel_id: str, focus_character: str = None):
    """Track character relationships within a novel."""
    try:
        session_id = await create_session(metadata={"api_call": "relationship_tracking"})
        
        focus_msg = f" focusing on {focus_character}" if focus_character else ""
        response, tools_used = await execute_agent(
            message=f"Track character relationships in novel {novel_id}{focus_msg}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "focus_character": focus_character,
            "relationship_analysis": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Character relationship tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/novels/{novel_id}/validate/timeline")
async def validate_timeline_endpoint(novel_id: str, check_type: str = "chronological"):
    """Validate timeline consistency."""
    try:
        session_id = await create_session(metadata={"api_call": "timeline_validation"})
        
        response, tools_used = await execute_agent(
            message=f"Validate {check_type} timeline consistency for novel {novel_id}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "validation_type": "timeline",
            "check_type": check_type,
            "validation_results": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Timeline validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/novels/{novel_id}/validate/genre")
async def validate_genre_conventions_endpoint(novel_id: str, genre: str = None):
    """Validate adherence to genre conventions."""
    try:
        session_id = await create_session(metadata={"api_call": "genre_validation"})
        
        genre_msg = f" for {genre} genre" if genre else ""
        response, tools_used = await execute_agent(
            message=f"Validate genre conventions{genre_msg} for novel {novel_id}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "validation_type": "genre_conventions",
            "target_genre": genre,
            "validation_results": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Genre validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/novels/{novel_id}/suggest/plot")
async def suggest_plot_developments_endpoint(
    novel_id: str,
    current_chapter: int,
    plot_direction: str = "",
    genre_constraints: str = ""
):
    """Suggest plot developments and story directions."""
    try:
        session_id = await create_session(metadata={"api_call": "plot_suggestions"})
        
        direction_msg = f" towards {plot_direction}" if plot_direction else ""
        genre_msg = f" with {genre_constraints} constraints" if genre_constraints else ""
        
        response, tools_used = await execute_agent(
            message=f"Suggest plot developments for novel {novel_id} from chapter {current_chapter}{direction_msg}{genre_msg}",
            session_id=session_id,
            save_conversation=False
        )
        
        return {
            "novel_id": novel_id,
            "current_chapter": current_chapter,
            "plot_direction": plot_direction,
            "genre_constraints": genre_constraints,
            "suggestions": response,
            "tools_used": [tool.tool_name for tool in tools_used]
        }
        
    except Exception as e:
        logger.error(f"Plot suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))