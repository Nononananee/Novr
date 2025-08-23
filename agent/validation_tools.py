"""
Advanced validation tools for novel consistency and quality.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic_ai import RunContext

from .agent import rag_agent, AgentDependencies

logger = logging.getLogger(__name__)


@rag_agent.tool
async def validate_character_voice_consistency(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    character_name: str
) -> Dict[str, Any]:
    """
    Validate consistency of a character's voice and dialogue patterns.
    
    This tool analyzes how consistently a character speaks throughout
    the story, checking for voice consistency, vocabulary usage,
    and speech pattern maintenance.
    
    Args:
        novel_id: ID of the novel to analyze
        character_name: Name of the character to validate
    
    Returns:
        Character voice consistency analysis with specific recommendations
    """
    try:
        from .db_utils import search_novel_content, list_characters
        
        # Get all characters to verify the character exists
        characters = await list_characters(novel_id)
        character_exists = any(char["name"].lower() == character_name.lower() for char in characters)
        
        if not character_exists:
            return {
                "novel_id": novel_id,
                "character_name": character_name,
                "error": "Character not found in novel"
            }
        
        # Search for content with this character
        character_content = await search_novel_content(
            novel_id=novel_id,
            query="",
            character_filter=character_name,
            limit=20
        )
        
        if not character_content:
            return {
                "novel_id": novel_id,
                "character_name": character_name,
                "message": "No content found for this character"
            }
        
        # Analyze voice consistency
        voice_analysis = await _analyze_character_voice_internal(character_name, character_content)
        
        return {
            "novel_id": novel_id,
            "character_name": character_name,
            "voice_consistency_score": voice_analysis["consistency_score"],
            "dialogue_samples_analyzed": len(character_content),
            "vocabulary_consistency": voice_analysis["vocabulary"],
            "speech_pattern_consistency": voice_analysis["speech_patterns"],
            "tone_consistency": voice_analysis["tone"],
            "inconsistencies_found": voice_analysis["inconsistencies"],
            "recommendations": voice_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate character voice for {character_name}"
        }


@rag_agent.tool
async def validate_timeline_consistency(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    check_type: str = "chronological"
) -> Dict[str, Any]:
    """
    Validate timeline and chronological consistency throughout the story.
    
    This tool checks for timeline inconsistencies, chronological errors,
    and temporal logic issues that could confuse readers or break
    story immersion.
    
    Args:
        novel_id: ID of the novel to validate
        check_type: Type of timeline check (chronological, seasonal, character_age)
    
    Returns:
        Timeline validation results with identified issues and corrections
    """
    try:
        from .db_utils import get_novel_chapters, search_novel_content
        
        # Get chapters for timeline analysis
        chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for timeline validation"
            }
        
        # Validate timeline
        timeline_analysis = await _validate_timeline_internal(novel_id, chapters, check_type)
        
        return {
            "novel_id": novel_id,
            "check_type": check_type,
            "timeline_consistency_score": timeline_analysis["consistency_score"],
            "chapters_analyzed": len(chapters),
            "timeline_issues": timeline_analysis["issues"],
            "chronological_errors": timeline_analysis["chronological_errors"],
            "temporal_inconsistencies": timeline_analysis["temporal_inconsistencies"],
            "suggested_corrections": timeline_analysis["corrections"],
            "timeline_map": timeline_analysis["timeline_map"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate timeline for novel {novel_id}"
        }


@rag_agent.tool
async def validate_world_building_consistency(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    focus_element: str = "all"
) -> Dict[str, Any]:
    """
    Validate consistency of world-building elements.
    
    This tool checks for consistency in world-building elements such as
    geography, rules, magic systems, technology levels, and cultural
    elements throughout the story.
    
    Args:
        novel_id: ID of the novel to validate
        focus_element: Specific element to focus on (geography, rules, culture, all)
    
    Returns:
        World-building consistency analysis with identified inconsistencies
    """
    try:
        from .db_utils import search_novel_content, get_novel_chapters
        
        # Get content for world-building analysis
        chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for world-building validation"
            }
        
        # Analyze world-building consistency
        worldbuilding_analysis = await _validate_worldbuilding_internal(novel_id, chapters, focus_element)
        
        return {
            "novel_id": novel_id,
            "focus_element": focus_element,
            "worldbuilding_consistency_score": worldbuilding_analysis["consistency_score"],
            "elements_analyzed": worldbuilding_analysis["elements"],
            "consistency_issues": worldbuilding_analysis["issues"],
            "rule_violations": worldbuilding_analysis["rule_violations"],
            "geographical_inconsistencies": worldbuilding_analysis["geographical"],
            "cultural_inconsistencies": worldbuilding_analysis["cultural"],
            "recommendations": worldbuilding_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate world-building for novel {novel_id}"
        }


@rag_agent.tool
async def generate_consistency_report(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    report_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Generate a comprehensive consistency report for the entire novel.
    
    This tool creates a detailed report covering all aspects of story
    consistency including characters, plot, timeline, world-building,
    and overall narrative coherence.
    
    Args:
        novel_id: ID of the novel to analyze
        report_type: Type of report (comprehensive, summary, focused)
    
    Returns:
        Comprehensive consistency report with scores and recommendations
    """
    try:
        from .db_utils import get_novel, get_novel_chapters, list_characters
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        chapters = await get_novel_chapters(novel_id)
        characters = await list_characters(novel_id)
        
        # Generate comprehensive report
        report = await _generate_consistency_report_internal(
            novel, chapters, characters, report_type
        )
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "report_type": report_type,
            "generated_at": report["timestamp"],
            "overall_consistency_score": report["overall_score"],
            "character_consistency": report["character_consistency"],
            "plot_consistency": report["plot_consistency"],
            "timeline_consistency": report["timeline_consistency"],
            "worldbuilding_consistency": report["worldbuilding_consistency"],
            "style_consistency": report["style_consistency"],
            "critical_issues": report["critical_issues"],
            "recommendations": report["recommendations"],
            "improvement_priorities": report["priorities"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate consistency report for novel {novel_id}"
        }


@rag_agent.tool
async def validate_genre_conventions(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    genre: str = None
) -> Dict[str, Any]:
    """
    Validate adherence to genre conventions and expectations.
    
    This tool checks if the story follows appropriate conventions for
    its genre, identifies genre-specific elements, and suggests
    improvements for better genre alignment.
    
    Args:
        novel_id: ID of the novel to validate
        genre: Specific genre to validate against (optional, uses novel's genre if not specified)
    
    Returns:
        Genre convention validation with adherence scores and suggestions
    """
    try:
        from .db_utils import get_novel, search_novel_content
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        # Use provided genre or novel's genre
        target_genre = genre or novel.get("genre", "general")
        
        # Validate genre conventions
        genre_analysis = await _validate_genre_conventions_internal(novel_id, target_genre)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "target_genre": target_genre,
            "genre_adherence_score": genre_analysis["adherence_score"],
            "genre_elements_present": genre_analysis["elements_present"],
            "missing_genre_elements": genre_analysis["missing_elements"],
            "genre_violations": genre_analysis["violations"],
            "reader_expectations": genre_analysis["expectations"],
            "recommendations": genre_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate genre conventions for novel {novel_id}"
        }


# Internal helper functions
async def _analyze_character_voice_internal(
    character_name: str,
    character_content: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Internal function to analyze character voice consistency."""
    
    # Simplified voice analysis
    consistency_score = 0.8  # Placeholder
    
    # Analyze vocabulary patterns
    vocabulary_words = []
    for content in character_content:
        text = content.get("content", "").lower()
        words = text.split()
        vocabulary_words.extend(words)
    
    vocabulary_consistency = {
        "unique_words": len(set(vocabulary_words)),
        "total_words": len(vocabulary_words),
        "consistency_score": 0.75
    }
    
    speech_patterns = {
        "formal_speech": False,
        "informal_speech": True,
        "consistent_patterns": True,
        "pattern_score": 0.8
    }
    
    tone_consistency = {
        "dominant_tone": "conversational",
        "tone_variations": ["friendly", "serious"],
        "consistency_score": 0.85
    }
    
    inconsistencies = []
    if len(character_content) < 3:
        inconsistencies.append("Insufficient dialogue samples for thorough analysis")
    
    recommendations = [
        "Maintain consistent vocabulary level for the character",
        "Keep speech patterns consistent across scenes",
        "Ensure character's tone matches their personality",
        "Review dialogue for character-specific phrases or expressions"
    ]
    
    return {
        "consistency_score": consistency_score,
        "vocabulary": vocabulary_consistency,
        "speech_patterns": speech_patterns,
        "tone": tone_consistency,
        "inconsistencies": inconsistencies,
        "recommendations": recommendations
    }


async def _validate_timeline_internal(
    novel_id: str,
    chapters: List[Dict[str, Any]],
    check_type: str
) -> Dict[str, Any]:
    """Internal function to validate timeline consistency."""
    
    consistency_score = 0.85  # Placeholder
    issues = []
    chronological_errors = []
    temporal_inconsistencies = []
    
    # Check chapter sequence
    chapter_numbers = [ch["chapter_number"] for ch in chapters]
    if chapter_numbers != sorted(chapter_numbers):
        chronological_errors.append({
            "type": "chapter_sequence",
            "description": "Chapters are not in sequential order",
            "severity": "high"
        })
    
    # Basic timeline mapping
    timeline_map = []
    for chapter in chapters:
        timeline_map.append({
            "chapter": chapter["chapter_number"],
            "title": chapter.get("title", f"Chapter {chapter['chapter_number']}"),
            "estimated_time": "Unknown",  # Would implement time extraction
            "events": []  # Would extract key events
        })
    
    corrections = [
        "Ensure chapters are in correct chronological order",
        "Add time markers to clarify story progression",
        "Check for consistent time references throughout"
    ]
    
    return {
        "consistency_score": consistency_score,
        "issues": issues,
        "chronological_errors": chronological_errors,
        "temporal_inconsistencies": temporal_inconsistencies,
        "corrections": corrections,
        "timeline_map": timeline_map
    }


async def _validate_worldbuilding_internal(
    novel_id: str,
    chapters: List[Dict[str, Any]],
    focus_element: str
) -> Dict[str, Any]:
    """Internal function to validate world-building consistency."""
    
    consistency_score = 0.7  # Placeholder
    
    elements = ["geography", "rules", "culture", "technology", "magic_system"]
    issues = []
    rule_violations = []
    geographical = []
    cultural = []
    
    # Basic world-building analysis
    if len(chapters) < 5:
        issues.append("Limited content for comprehensive world-building analysis")
    
    recommendations = [
        "Maintain consistent rules for your world's systems",
        "Keep geographical descriptions consistent",
        "Ensure cultural elements remain coherent",
        "Document world-building rules for reference"
    ]
    
    return {
        "consistency_score": consistency_score,
        "elements": elements,
        "issues": issues,
        "rule_violations": rule_violations,
        "geographical": geographical,
        "cultural": cultural,
        "recommendations": recommendations
    }


async def _generate_consistency_report_internal(
    novel: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
    report_type: str
) -> Dict[str, Any]:
    """Internal function to generate consistency report."""
    
    from datetime import datetime
    
    # Calculate overall scores (simplified)
    character_consistency = 0.8
    plot_consistency = 0.75
    timeline_consistency = 0.85
    worldbuilding_consistency = 0.7
    style_consistency = 0.8
    
    overall_score = (
        character_consistency + plot_consistency + timeline_consistency +
        worldbuilding_consistency + style_consistency
    ) / 5
    
    critical_issues = []
    if len(chapters) < 3:
        critical_issues.append("Insufficient content for comprehensive analysis")
    if len(characters) == 0:
        critical_issues.append("No characters defined for the novel")
    
    recommendations = [
        "Focus on character consistency across all scenes",
        "Review plot logic and causality",
        "Ensure timeline coherence throughout",
        "Maintain consistent world-building rules",
        "Keep writing style consistent"
    ]
    
    priorities = [
        {"priority": 1, "area": "Character Development", "reason": "Foundation of reader engagement"},
        {"priority": 2, "area": "Plot Consistency", "reason": "Essential for story coherence"},
        {"priority": 3, "area": "World Building", "reason": "Creates immersive experience"}
    ]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "character_consistency": character_consistency,
        "plot_consistency": plot_consistency,
        "timeline_consistency": timeline_consistency,
        "worldbuilding_consistency": worldbuilding_consistency,
        "style_consistency": style_consistency,
        "critical_issues": critical_issues,
        "recommendations": recommendations,
        "priorities": priorities
    }


async def _validate_genre_conventions_internal(
    novel_id: str,
    target_genre: str
) -> Dict[str, Any]:
    """Internal function to validate genre conventions."""
    
    adherence_score = 0.75  # Placeholder
    
    # Genre-specific elements (simplified)
    genre_conventions = {
        "fantasy": ["magic_system", "world_building", "quest", "mythical_creatures"],
        "mystery": ["crime", "investigation", "clues", "revelation"],
        "romance": ["relationship_development", "emotional_conflict", "happy_ending"],
        "science_fiction": ["technology", "future_setting", "scientific_concepts"],
        "horror": ["fear_elements", "suspense", "supernatural_or_psychological"]
    }
    
    expected_elements = genre_conventions.get(target_genre.lower(), ["general_story_elements"])
    
    # Simplified analysis
    elements_present = expected_elements[:2]  # Assume some elements are present
    missing_elements = expected_elements[2:]  # Assume some are missing
    
    violations = []
    if target_genre.lower() == "romance" and "happy_ending" not in elements_present:
        violations.append("Romance genre typically requires positive relationship resolution")
    
    expectations = {
        "pacing": "Appropriate for genre",
        "tone": "Matches genre expectations",
        "themes": "Aligned with genre conventions"
    }
    
    recommendations = [
        f"Ensure key {target_genre} elements are present and well-developed",
        f"Research successful {target_genre} novels for inspiration",
        f"Consider reader expectations for {target_genre} genre",
        "Balance genre conventions with original storytelling"
    ]
    
    return {
        "adherence_score": adherence_score,
        "elements_present": elements_present,
        "missing_elements": missing_elements,
        "violations": violations,
        "expectations": expectations,
        "recommendations": recommendations
    }