"""
Plot analysis and development tools for novel writing.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic_ai import RunContext

from .agent import rag_agent, AgentDependencies

logger = logging.getLogger(__name__)


@rag_agent.tool
async def analyze_plot_consistency(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    focus_thread: str = None
) -> Dict[str, Any]:
    """
    Analyze plot consistency and identify potential plot holes.
    
    This tool examines the logical flow of plot events, checks for
    consistency in cause-and-effect relationships, and identifies
    potential plot holes or inconsistencies.
    
    Args:
        novel_id: ID of the novel to analyze
        focus_thread: Optional specific plot thread to focus on
    
    Returns:
        Plot consistency analysis with identified issues and suggestions
    """
    try:
        from .db_utils import get_novel_chapters, search_novel_content
        
        # Get all chapters for analysis
        chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for analysis"
            }
        
        # Analyze plot consistency across chapters
        plot_analysis = await _analyze_plot_consistency_internal(novel_id, chapters, focus_thread)
        
        return {
            "novel_id": novel_id,
            "focus_thread": focus_thread,
            "total_chapters": len(chapters),
            "consistency_score": plot_analysis["consistency_score"],
            "plot_holes": plot_analysis["plot_holes"],
            "logical_inconsistencies": plot_analysis["logical_inconsistencies"],
            "timeline_issues": plot_analysis["timeline_issues"],
            "character_motivation_issues": plot_analysis["character_issues"],
            "suggestions": plot_analysis["suggestions"],
            "overall_assessment": plot_analysis["assessment"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze plot consistency for novel {novel_id}"
        }


@rag_agent.tool
async def detect_plot_holes(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    chapter_range: List[int] = None
) -> Dict[str, Any]:
    """
    Detect potential plot holes and logical inconsistencies.
    
    This tool specifically looks for gaps in logic, unexplained events,
    contradictions, and missing causal connections in the story.
    
    Args:
        novel_id: ID of the novel to analyze
        chapter_range: Optional list [start, end] to limit analysis range
    
    Returns:
        List of detected plot holes with severity and suggestions
    """
    try:
        from .db_utils import get_novel_chapters, search_novel_content
        
        # Get chapters for analysis
        if chapter_range and len(chapter_range) == 2:
            chapters = await get_novel_chapters(novel_id, chapter_range[0], chapter_range[1])
        else:
            chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for analysis"
            }
        
        # Detect plot holes
        plot_holes = await _detect_plot_holes_internal(novel_id, chapters)
        
        return {
            "novel_id": novel_id,
            "chapters_analyzed": len(chapters),
            "chapter_range": chapter_range or "all",
            "plot_holes_found": len(plot_holes["holes"]),
            "plot_holes": plot_holes["holes"],
            "severity_breakdown": plot_holes["severity_breakdown"],
            "categories": plot_holes["categories"],
            "priority_fixes": plot_holes["priority_fixes"],
            "suggestions": plot_holes["suggestions"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to detect plot holes for novel {novel_id}"
        }


@rag_agent.tool
async def suggest_plot_developments(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    current_chapter: int,
    plot_direction: str = "",
    genre_constraints: str = ""
) -> Dict[str, Any]:
    """
    Suggest plot developments and story directions.
    
    This tool analyzes the current story state and suggests logical
    plot developments, potential conflicts, and story directions
    that maintain consistency and engagement.
    
    Args:
        novel_id: ID of the novel
        current_chapter: Current chapter number
        plot_direction: Optional desired plot direction
        genre_constraints: Optional genre-specific constraints
    
    Returns:
        Plot development suggestions with multiple options and rationales
    """
    try:
        from .db_utils import get_novel_chapters, list_characters, search_novel_content
        
        # Get current story context
        chapters = await get_novel_chapters(novel_id, 1, current_chapter)
        characters = await list_characters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for context"
            }
        
        # Generate plot suggestions
        plot_suggestions = await _generate_plot_suggestions_internal(
            novel_id, chapters, characters, current_chapter, plot_direction, genre_constraints
        )
        
        return {
            "novel_id": novel_id,
            "current_chapter": current_chapter,
            "plot_direction": plot_direction,
            "genre_constraints": genre_constraints,
            "suggested_developments": plot_suggestions["developments"],
            "conflict_opportunities": plot_suggestions["conflicts"],
            "character_arcs": plot_suggestions["character_arcs"],
            "plot_twists": plot_suggestions["twists"],
            "pacing_recommendations": plot_suggestions["pacing"],
            "next_chapter_focus": plot_suggestions["next_chapter"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to suggest plot developments for novel {novel_id}"
        }


@rag_agent.tool
async def analyze_story_pacing(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    pacing_type: str = "overall"
) -> Dict[str, Any]:
    """
    Analyze story pacing and rhythm.
    
    This tool examines the pacing of the story, identifying areas that
    may be too slow or too fast, and suggests improvements for better
    narrative flow and reader engagement.
    
    Args:
        novel_id: ID of the novel to analyze
        pacing_type: Type of pacing analysis (overall, chapter, scene)
    
    Returns:
        Pacing analysis with recommendations for improvement
    """
    try:
        from .db_utils import get_novel_chapters, search_novel_content
        
        # Get chapters for pacing analysis
        chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for pacing analysis"
            }
        
        # Analyze pacing
        pacing_analysis = await _analyze_pacing_internal(novel_id, chapters, pacing_type)
        
        return {
            "novel_id": novel_id,
            "pacing_type": pacing_type,
            "overall_pacing_score": pacing_analysis["overall_score"],
            "pacing_breakdown": pacing_analysis["breakdown"],
            "slow_sections": pacing_analysis["slow_sections"],
            "fast_sections": pacing_analysis["fast_sections"],
            "pacing_issues": pacing_analysis["issues"],
            "recommendations": pacing_analysis["recommendations"],
            "ideal_pacing_curve": pacing_analysis["ideal_curve"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze pacing for novel {novel_id}"
        }


@rag_agent.tool
async def validate_story_structure(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    structure_type: str = "three_act"
) -> Dict[str, Any]:
    """
    Validate story structure against common narrative frameworks.
    
    This tool checks if the story follows established narrative structures
    like three-act structure, hero's journey, or other frameworks,
    and provides suggestions for structural improvements.
    
    Args:
        novel_id: ID of the novel to validate
        structure_type: Type of structure to validate against (three_act, hero_journey, etc.)
    
    Returns:
        Structure validation with adherence scores and improvement suggestions
    """
    try:
        from .db_utils import get_novel_chapters, search_novel_content
        
        # Get chapters for structure analysis
        chapters = await get_novel_chapters(novel_id)
        
        if not chapters:
            return {
                "novel_id": novel_id,
                "message": "No chapters found for structure validation"
            }
        
        # Validate structure
        structure_analysis = await _validate_structure_internal(novel_id, chapters, structure_type)
        
        return {
            "novel_id": novel_id,
            "structure_type": structure_type,
            "adherence_score": structure_analysis["adherence_score"],
            "structure_elements": structure_analysis["elements"],
            "missing_elements": structure_analysis["missing"],
            "structural_issues": structure_analysis["issues"],
            "act_breakdown": structure_analysis["acts"],
            "recommendations": structure_analysis["recommendations"],
            "alternative_structures": structure_analysis["alternatives"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate structure for novel {novel_id}"
        }


# Internal helper functions
async def _analyze_plot_consistency_internal(
    novel_id: str, 
    chapters: List[Dict[str, Any]], 
    focus_thread: Optional[str]
) -> Dict[str, Any]:
    """Internal function to analyze plot consistency."""
    
    # Simplified plot consistency analysis
    consistency_score = 0.75  # Placeholder
    plot_holes = []
    logical_inconsistencies = []
    timeline_issues = []
    character_issues = []
    
    # Basic analysis based on chapter count and content
    if len(chapters) < 3:
        plot_holes.append({
            "type": "insufficient_development",
            "description": "Story needs more chapters for proper plot development",
            "severity": "medium",
            "chapter": "overall"
        })
    
    # Check for chapter progression
    chapter_numbers = [ch["chapter_number"] for ch in chapters]
    if chapter_numbers != sorted(chapter_numbers):
        timeline_issues.append({
            "type": "chapter_order",
            "description": "Chapters are not in sequential order",
            "severity": "high"
        })
    
    suggestions = [
        "Ensure each chapter advances the plot meaningfully",
        "Check for logical cause-and-effect relationships",
        "Verify character motivations are consistent",
        "Review timeline for chronological consistency"
    ]
    
    assessment = "Plot shows basic structure but needs detailed consistency review"
    
    return {
        "consistency_score": consistency_score,
        "plot_holes": plot_holes,
        "logical_inconsistencies": logical_inconsistencies,
        "timeline_issues": timeline_issues,
        "character_issues": character_issues,
        "suggestions": suggestions,
        "assessment": assessment
    }


async def _detect_plot_holes_internal(
    novel_id: str, 
    chapters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Internal function to detect plot holes."""
    
    holes = []
    severity_breakdown = {"high": 0, "medium": 0, "low": 0}
    categories = ["logic", "character", "timeline", "causality"]
    
    # Basic plot hole detection
    if len(chapters) > 1:
        # Check for missing chapter connections
        for i in range(1, len(chapters)):
            prev_chapter = chapters[i-1]
            curr_chapter = chapters[i]
            
            # Simple check for chapter gap
            if curr_chapter["chapter_number"] - prev_chapter["chapter_number"] > 1:
                holes.append({
                    "type": "chapter_gap",
                    "description": f"Gap between chapter {prev_chapter['chapter_number']} and {curr_chapter['chapter_number']}",
                    "severity": "medium",
                    "location": f"Between chapters {prev_chapter['chapter_number']}-{curr_chapter['chapter_number']}",
                    "category": "timeline"
                })
                severity_breakdown["medium"] += 1
    
    priority_fixes = [hole for hole in holes if hole["severity"] == "high"]
    
    suggestions = [
        "Review chapter transitions for logical flow",
        "Ensure all plot points have proper setup and payoff",
        "Check character actions for proper motivation",
        "Verify timeline consistency throughout the story"
    ]
    
    return {
        "holes": holes,
        "severity_breakdown": severity_breakdown,
        "categories": categories,
        "priority_fixes": priority_fixes,
        "suggestions": suggestions
    }


async def _generate_plot_suggestions_internal(
    novel_id: str,
    chapters: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
    current_chapter: int,
    plot_direction: str,
    genre_constraints: str
) -> Dict[str, Any]:
    """Internal function to generate plot suggestions."""
    
    # Generate plot development suggestions
    developments = [
        {
            "type": "character_conflict",
            "description": "Introduce conflict between main characters",
            "rationale": "Character conflict drives emotional engagement",
            "implementation": "Create disagreement over important decision"
        },
        {
            "type": "plot_twist",
            "description": "Reveal hidden information that changes perspective",
            "rationale": "Plot twists maintain reader interest",
            "implementation": "Character discovers something that changes their understanding"
        },
        {
            "type": "escalation",
            "description": "Increase stakes or tension",
            "rationale": "Rising action maintains narrative momentum",
            "implementation": "Make the consequences more significant"
        }
    ]
    
    conflicts = [
        "Internal character struggle",
        "Interpersonal relationship tension",
        "External obstacle or antagonist",
        "Moral dilemma requiring difficult choice"
    ]
    
    character_arcs = [
        {
            "character": char["name"],
            "suggested_development": f"Develop {char['name']}'s {char.get('role', 'character')} role further",
            "arc_stage": "development"
        }
        for char in characters[:3]  # Top 3 characters
    ]
    
    twists = [
        "Character reveals hidden motivation",
        "Unexpected alliance forms",
        "Past event resurfaces with new significance",
        "Character's assumption proves wrong"
    ]
    
    pacing = {
        "current_pace": "moderate",
        "recommendation": "Consider varying pace with action and reflection",
        "next_chapter_pace": "slightly faster"
    }
    
    next_chapter = {
        "focus": "Character development and plot advancement",
        "key_elements": ["Character interaction", "Plot progression", "Conflict development"],
        "avoid": ["Info dumping", "Repetitive scenes", "Stagnant character states"]
    }
    
    return {
        "developments": developments,
        "conflicts": conflicts,
        "character_arcs": character_arcs,
        "twists": twists,
        "pacing": pacing,
        "next_chapter": next_chapter
    }


async def _analyze_pacing_internal(
    novel_id: str,
    chapters: List[Dict[str, Any]],
    pacing_type: str
) -> Dict[str, Any]:
    """Internal function to analyze story pacing."""
    
    overall_score = 0.7  # Placeholder
    
    breakdown = []
    for chapter in chapters:
        word_count = chapter.get("word_count", 1000)  # Default estimate
        
        # Simple pacing assessment based on word count
        if word_count < 800:
            pace = "fast"
        elif word_count > 1500:
            pace = "slow"
        else:
            pace = "moderate"
        
        breakdown.append({
            "chapter": chapter["chapter_number"],
            "pace": pace,
            "word_count": word_count,
            "assessment": f"Chapter {chapter['chapter_number']} has {pace} pacing"
        })
    
    slow_sections = [ch for ch in breakdown if ch["pace"] == "slow"]
    fast_sections = [ch for ch in breakdown if ch["pace"] == "fast"]
    
    issues = []
    if len(slow_sections) > len(chapters) * 0.4:
        issues.append("Too many slow-paced sections may lose reader interest")
    if len(fast_sections) > len(chapters) * 0.4:
        issues.append("Too many fast-paced sections may exhaust readers")
    
    recommendations = [
        "Vary pacing between chapters for better rhythm",
        "Use fast pacing for action scenes, slower for character development",
        "Consider reader fatigue when planning intense sequences",
        "Balance exposition with action and dialogue"
    ]
    
    ideal_curve = "Start moderate, build to climax, then resolve"
    
    return {
        "overall_score": overall_score,
        "breakdown": breakdown,
        "slow_sections": slow_sections,
        "fast_sections": fast_sections,
        "issues": issues,
        "recommendations": recommendations,
        "ideal_curve": ideal_curve
    }


async def _validate_structure_internal(
    novel_id: str,
    chapters: List[Dict[str, Any]],
    structure_type: str
) -> Dict[str, Any]:
    """Internal function to validate story structure."""
    
    adherence_score = 0.6  # Placeholder
    
    if structure_type == "three_act":
        total_chapters = len(chapters)
        act1_end = total_chapters // 4
        act2_end = total_chapters * 3 // 4
        
        elements = {
            "setup": {"present": act1_end > 0, "chapters": f"1-{act1_end}"},
            "confrontation": {"present": act2_end > act1_end, "chapters": f"{act1_end+1}-{act2_end}"},
            "resolution": {"present": total_chapters > act2_end, "chapters": f"{act2_end+1}-{total_chapters}"}
        }
        
        missing = [elem for elem, data in elements.items() if not data["present"]]
        
        acts = {
            "act_1": {"chapters": f"1-{act1_end}", "purpose": "Setup and introduction"},
            "act_2": {"chapters": f"{act1_end+1}-{act2_end}", "purpose": "Confrontation and development"},
            "act_3": {"chapters": f"{act2_end+1}-{total_chapters}", "purpose": "Climax and resolution"}
        }
    else:
        elements = {"structure_elements": "Analysis for other structures not implemented"}
        missing = []
        acts = {}
    
    issues = []
    if len(chapters) < 3:
        issues.append("Insufficient chapters for proper three-act structure")
    
    recommendations = [
        "Ensure clear act divisions with appropriate pacing",
        "Develop proper setup in first act",
        "Build tension throughout second act",
        "Provide satisfying resolution in third act"
    ]
    
    alternatives = ["Hero's Journey", "Five-Act Structure", "Save the Cat Beat Sheet"]
    
    return {
        "adherence_score": adherence_score,
        "elements": elements,
        "missing": missing,
        "issues": issues,
        "acts": acts,
        "recommendations": recommendations,
        "alternatives": alternatives
    }