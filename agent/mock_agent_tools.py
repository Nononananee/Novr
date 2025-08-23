"""
Mock agent tools for testing without pydantic_ai dependency.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class MockRunContext:
    """Mock run context for testing."""
    
    def __init__(self, dependencies: Dict[str, Any]):
        self.dependencies = dependencies


class MockAgentDependencies:
    """Mock agent dependencies for testing."""
    
    def __init__(self, session_id: str, user_id: str = None, **kwargs):
        self.session_id = session_id
        self.user_id = user_id
        self.search_preferences = kwargs.get('search_preferences', {})


# Mock agent tools for testing
async def analyze_character_consistency(
    ctx: MockRunContext,
    character_id: str,
    from_chapter: int = 1,
    to_chapter: int = None
) -> Dict[str, Any]:
    """Mock character consistency analysis."""
    await asyncio.sleep(0.01)  # Simulate processing time
    
    return {
        "character_id": character_id,
        "character_name": f"Character_{character_id}",
        "consistency_score": 0.85,
        "personality_consistency": 0.88,
        "dialogue_consistency": 0.82,
        "behavioral_consistency": 0.87,
        "violations": [],
        "suggestions": [
            "Maintain consistent personality traits across chapters",
            "Ensure dialogue patterns remain authentic"
        ],
        "total_appearances": 5,
        "chapters_analyzed": f"{from_chapter}-{to_chapter or 'end'}"
    }


async def generate_character_development(
    ctx: MockRunContext,
    character_id: str,
    target_development: str,
    current_chapter: int,
    development_type: str = "gradual"
) -> Dict[str, Any]:
    """Mock character development generation."""
    await asyncio.sleep(0.02)  # Simulate processing time
    
    return {
        "character_id": character_id,
        "character_name": f"Character_{character_id}",
        "target_development": target_development,
        "development_type": development_type,
        "current_chapter": current_chapter,
        "development_plan": f"Develop character towards: {target_development}",
        "suggested_scenes": [
            {
                "chapter": current_chapter + 1,
                "scene_type": "character_development",
                "description": f"Scene showing progress towards {target_development}"
            }
        ],
        "dialogue_suggestions": [
            "Internal monologue reflecting growth",
            "Conversation revealing new perspective"
        ],
        "internal_changes": ["Shift in mindset", "New emotional responses"],
        "external_manifestations": ["Changed behavior", "Different reactions"],
        "timeline": f"Development over 3 chapters"
    }


async def track_character_relationships(
    ctx: MockRunContext,
    novel_id: str,
    focus_character: str = None
) -> Dict[str, Any]:
    """Mock character relationship tracking."""
    await asyncio.sleep(0.015)  # Simulate processing time
    
    return {
        "novel_id": novel_id,
        "focus_character": focus_character,
        "total_characters": 4,
        "relationship_map": [
            {
                "character_1": "Hero",
                "character_2": "Mentor",
                "relationship_type": "mentorship",
                "interaction_count": 8,
                "relationship_strength": 0.9
            },
            {
                "character_1": "Hero",
                "character_2": "Villain",
                "relationship_type": "antagonistic",
                "interaction_count": 5,
                "relationship_strength": 0.8
            }
        ],
        "relationship_dynamics": ["mentorship", "rivalry", "friendship"],
        "evolution_patterns": ["strengthening", "developing"],
        "conflict_points": ["Ideological differences", "Personal history"],
        "alliance_patterns": ["Mentor-student bond", "Shared goals"],
        "suggestions": [
            "Develop more character interactions",
            "Add relationship metadata to scenes"
        ]
    }


async def analyze_emotional_content(
    ctx: MockRunContext,
    content: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Mock emotional content analysis."""
    await asyncio.sleep(0.01)  # Simulate processing time
    
    # Simple emotion detection based on keywords
    content_lower = content.lower()
    emotions = []
    
    if any(word in content_lower for word in ['happy', 'joy', 'smile', 'laugh']):
        emotions.append('joy')
    if any(word in content_lower for word in ['sad', 'sorrow', 'cry', 'tears']):
        emotions.append('sadness')
    if any(word in content_lower for word in ['angry', 'rage', 'fury', 'mad']):
        emotions.append('anger')
    if any(word in content_lower for word in ['afraid', 'scared', 'fear', 'terrified']):
        emotions.append('fear')
    
    dominant_emotion = emotions[0] if emotions else 'neutral'
    intensity = min(len(emotions) * 0.3, 1.0)
    
    return {
        "content_length": len(content),
        "dominant_emotions": [dominant_emotion],
        "emotional_intensity": intensity,
        "emotional_progression": "stable",
        "emotional_triggers": emotions,
        "character_emotions": context.get("characters", []) if context else [],
        "scene_mood": dominant_emotion,
        "emotional_consistency": 0.8,
        "suggestions": [
            "Consider varying emotional intensity",
            "Add more sensory details for emotional impact"
        ]
    }


async def generate_emotional_scene(
    ctx: MockRunContext,
    emotional_tone: str,
    intensity: float = 0.7,
    characters: List[str] = None,
    scene_context: str = "",
    word_count: int = 300
) -> Dict[str, Any]:
    """Mock emotional scene generation."""
    await asyncio.sleep(0.025)  # Simulate processing time
    
    character_list = ", ".join(characters) if characters else "the character"
    
    scene_templates = {
        "joyful": f"The warm sunlight filtered through the windows as {character_list} felt a surge of happiness.",
        "melancholic": f"A heavy silence settled over {character_list} as the weight of loss pressed down.",
        "tense": f"The air crackled with tension as {character_list} faced the approaching danger.",
        "romantic": f"The gentle evening breeze carried the scent of flowers as {character_list} shared a tender moment.",
        "mysterious": f"Strange shadows danced at the edges of vision as {character_list} ventured into the unknown."
    }
    
    base_content = scene_templates.get(emotional_tone, f"A scene unfolds with {character_list}.")
    
    if scene_context:
        base_content = f"In {scene_context}, {base_content.lower()}"
    
    # Add intensity-based descriptors
    if intensity > 0.8:
        base_content += " The emotions were overwhelming, consuming every thought."
    elif intensity > 0.5:
        base_content += " The feelings were strong and undeniable."
    else:
        base_content += " The emotions were subtle but present."
    
    return {
        "generation_type": "emotional_scene",
        "emotional_tone": emotional_tone,
        "intensity": intensity,
        "characters_included": characters or [],
        "scene_context": scene_context,
        "generated_content": base_content,
        "actual_word_count": len(base_content.split()),
        "emotional_elements": [emotional_tone, "descriptive_language"],
        "character_reactions": ["emotional_response", "physical_manifestation"],
        "descriptive_techniques": ["sensory_details", "atmosphere"],
        "emotional_impact_score": intensity
    }


async def track_emotional_arc(
    ctx: MockRunContext,
    novel_id: str,
    entity_type: str = "character",
    entity_name: str = None
) -> Dict[str, Any]:
    """Mock emotional arc tracking."""
    await asyncio.sleep(0.02)  # Simulate processing time
    
    # Mock emotional progression
    progression = [
        {"chapter": 1, "scene": "Opening", "dominant_emotion": "neutral", "intensity": 0.5},
        {"chapter": 2, "scene": "Inciting Incident", "dominant_emotion": "surprise", "intensity": 0.7},
        {"chapter": 3, "scene": "Rising Action", "dominant_emotion": "determination", "intensity": 0.8},
        {"chapter": 4, "scene": "Climax", "dominant_emotion": "fear", "intensity": 0.9},
        {"chapter": 5, "scene": "Resolution", "dominant_emotion": "relief", "intensity": 0.6}
    ]
    
    turning_points = [
        {"chapter": 2, "change": "neutral → surprise", "significance": "moderate"},
        {"chapter": 4, "change": "determination → fear", "significance": "high"}
    ]
    
    return {
        "novel_id": novel_id,
        "entity_type": entity_type,
        "entity_name": entity_name,
        "emotional_progression": progression,
        "turning_points": turning_points,
        "emotional_peaks": [p for p in progression if p["intensity"] > 0.7],
        "emotional_valleys": [p for p in progression if p["intensity"] < 0.4],
        "overall_trajectory": "rising_then_falling",
        "consistency_score": 0.75,
        "suggestions": [
            "Consider adding more emotional variety",
            "Develop stronger emotional peaks",
            "Ensure emotional changes are well-motivated"
        ]
    }


# Performance monitoring mock tools
async def get_creative_performance_report(
    ctx: MockRunContext,
    hours: int = 24
) -> Dict[str, Any]:
    """Mock performance report."""
    await asyncio.sleep(0.01)
    
    return {
        "report_type": "creative_performance",
        "time_period_hours": hours,
        "performance_data": {
            "total_operations": 15,
            "successful_operations": 14,
            "failed_operations": 1,
            "success_rate": 0.933,
            "average_duration_ms": 125.5,
            "quality_statistics": {
                "character_consistency": {"average": 0.85, "count": 5},
                "emotional_consistency": {"average": 0.78, "count": 8}
            }
        },
        "status": "success"
    }


async def get_quality_recommendations(
    ctx: MockRunContext
) -> Dict[str, Any]:
    """Mock quality recommendations."""
    await asyncio.sleep(0.005)
    
    return {
        "recommendation_type": "creative_quality",
        "total_recommendations": 3,
        "recommendations": [
            {
                "metric": "dialogue_consistency",
                "current_score": 0.65,
                "threshold": 0.7,
                "priority": "medium",
                "recommendation": "Review character dialogue patterns for consistency"
            },
            {
                "metric": "emotional_intensity",
                "current_score": 0.55,
                "threshold": 0.6,
                "priority": "low",
                "recommendation": "Consider increasing emotional impact in key scenes"
            }
        ],
        "status": "success"
    }


# Genre-specific mock tools
async def analyze_fantasy_world_building(
    ctx: MockRunContext,
    novel_id: str,
    focus_element: str = "magic_system"
) -> Dict[str, Any]:
    """Mock fantasy analysis."""
    await asyncio.sleep(0.02)
    
    return {
        "novel_id": novel_id,
        "novel_title": f"Fantasy Novel {novel_id}",
        "genre": "fantasy",
        "focus_element": focus_element,
        "fantasy_consistency_score": 0.82,
        "magic_system_analysis": {
            "consistency": 0.85,
            "rules_defined": True,
            "limitations_clear": True
        },
        "creature_consistency": {
            "consistency": 0.78,
            "mythical_creatures_present": ["dragons", "elves"],
            "creature_behavior_consistent": True
        },
        "world_geography": {
            "world_map_consistent": True,
            "location_descriptions_consistent": 0.8
        },
        "recommendations": [
            "Maintain consistent magic system rules",
            "Ensure creature behaviors align with established lore"
        ]
    }


# Export mock tools for testing
MOCK_TOOLS = {
    'analyze_character_consistency': analyze_character_consistency,
    'generate_character_development': generate_character_development,
    'track_character_relationships': track_character_relationships,
    'analyze_emotional_content': analyze_emotional_content,
    'generate_emotional_scene': generate_emotional_scene,
    'track_emotional_arc': track_emotional_arc,
    'get_creative_performance_report': get_creative_performance_report,
    'get_quality_recommendations': get_quality_recommendations,
    'analyze_fantasy_world_building': analyze_fantasy_world_building
}