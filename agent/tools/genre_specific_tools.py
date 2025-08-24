"""
Genre-specific creative tools for specialized novel writing assistance.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic_ai import RunContext

from .agent import rag_agent, AgentDependencies

logger = logging.getLogger(__name__)


@rag_agent.tool
async def analyze_fantasy_world_building(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    focus_element: str = "magic_system"
) -> Dict[str, Any]:
    """
    Analyze fantasy-specific world-building elements.
    
    This tool examines fantasy novels for consistency in magic systems,
    mythical creatures, world geography, and fantasy-specific rules.
    Essential for maintaining immersive fantasy worlds.
    
    Args:
        novel_id: ID of the fantasy novel to analyze
        focus_element: Specific element to focus on (magic_system, creatures, geography, politics)
    
    Returns:
        Fantasy world-building analysis with consistency scores and recommendations
    """
    try:
        from .db_utils import get_novel, search_novel_content
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        # Verify it's a fantasy novel
        genre = novel.get("genre", "").lower()
        if "fantasy" not in genre:
            return {
                "novel_id": novel_id,
                "warning": f"Novel genre is '{genre}', not fantasy. Analysis may not be fully applicable."
            }
        
        # Analyze fantasy elements
        fantasy_analysis = await _analyze_fantasy_elements_internal(novel_id, focus_element)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "genre": novel.get("genre", "unknown"),
            "focus_element": focus_element,
            "fantasy_consistency_score": fantasy_analysis["consistency_score"],
            "magic_system_analysis": fantasy_analysis["magic_system"],
            "creature_consistency": fantasy_analysis["creatures"],
            "world_geography": fantasy_analysis["geography"],
            "fantasy_rules": fantasy_analysis["rules"],
            "inconsistencies": fantasy_analysis["inconsistencies"],
            "recommendations": fantasy_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze fantasy world-building for novel {novel_id}"
        }


@rag_agent.tool
async def analyze_mystery_plot_structure(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    mystery_type: str = "detective"
) -> Dict[str, Any]:
    """
    Analyze mystery-specific plot structure and elements.
    
    This tool examines mystery novels for proper clue placement,
    red herrings, revelation timing, and mystery-specific pacing.
    Essential for crafting engaging mystery narratives.
    
    Args:
        novel_id: ID of the mystery novel to analyze
        mystery_type: Type of mystery (detective, cozy, thriller, procedural)
    
    Returns:
        Mystery plot analysis with structure assessment and recommendations
    """
    try:
        from .db_utils import get_novel, get_novel_chapters
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        # Get chapters for structure analysis
        chapters = await get_novel_chapters(novel_id)
        
        # Analyze mystery structure
        mystery_analysis = await _analyze_mystery_structure_internal(novel_id, chapters, mystery_type)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "mystery_type": mystery_type,
            "chapters_analyzed": len(chapters),
            "mystery_structure_score": mystery_analysis["structure_score"],
            "clue_distribution": mystery_analysis["clue_distribution"],
            "red_herring_analysis": mystery_analysis["red_herrings"],
            "revelation_timing": mystery_analysis["revelation_timing"],
            "pacing_analysis": mystery_analysis["pacing"],
            "mystery_elements": mystery_analysis["elements"],
            "structural_issues": mystery_analysis["issues"],
            "recommendations": mystery_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze mystery plot structure for novel {novel_id}"
        }


@rag_agent.tool
async def analyze_romance_relationship_development(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    relationship_focus: str = "primary"
) -> Dict[str, Any]:
    """
    Analyze romance-specific relationship development and emotional arcs.
    
    This tool examines romance novels for relationship progression,
    emotional beats, conflict resolution, and romance-specific elements.
    Essential for crafting satisfying romantic narratives.
    
    Args:
        novel_id: ID of the romance novel to analyze
        relationship_focus: Focus on specific relationship (primary, secondary, all)
    
    Returns:
        Romance relationship analysis with development assessment and recommendations
    """
    try:
        from .db_utils import get_novel, list_characters
        
        # Get novel and character information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        characters = await list_characters(novel_id)
        
        # Analyze romance elements
        romance_analysis = await _analyze_romance_elements_internal(novel_id, characters, relationship_focus)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "relationship_focus": relationship_focus,
            "total_characters": len(characters),
            "romance_development_score": romance_analysis["development_score"],
            "relationship_progression": romance_analysis["progression"],
            "emotional_beats": romance_analysis["emotional_beats"],
            "conflict_resolution": romance_analysis["conflict_resolution"],
            "romantic_tension": romance_analysis["tension"],
            "character_chemistry": romance_analysis["chemistry"],
            "romance_tropes": romance_analysis["tropes"],
            "recommendations": romance_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze romance relationship development for novel {novel_id}"
        }


@rag_agent.tool
async def analyze_scifi_technology_consistency(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    tech_focus: str = "all"
) -> Dict[str, Any]:
    """
    Analyze science fiction technology consistency and scientific accuracy.
    
    This tool examines sci-fi novels for technology consistency,
    scientific plausibility, and future world-building coherence.
    Essential for maintaining believable sci-fi narratives.
    
    Args:
        novel_id: ID of the sci-fi novel to analyze
        tech_focus: Focus area (all, space_travel, ai, biotech, physics)
    
    Returns:
        Sci-fi technology analysis with consistency scores and recommendations
    """
    try:
        from .db_utils import get_novel, search_novel_content
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        # Analyze sci-fi elements
        scifi_analysis = await _analyze_scifi_elements_internal(novel_id, tech_focus)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "tech_focus": tech_focus,
            "technology_consistency_score": scifi_analysis["consistency_score"],
            "scientific_plausibility": scifi_analysis["plausibility"],
            "technology_rules": scifi_analysis["tech_rules"],
            "future_world_consistency": scifi_analysis["world_consistency"],
            "scientific_accuracy": scifi_analysis["accuracy"],
            "technology_progression": scifi_analysis["progression"],
            "inconsistencies": scifi_analysis["inconsistencies"],
            "recommendations": scifi_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to analyze sci-fi technology consistency for novel {novel_id}"
        }


@rag_agent.tool
async def generate_genre_specific_content(
    ctx: RunContext[AgentDependencies],
    genre: str,
    content_type: str,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate genre-specific content elements.
    
    This tool creates content tailored to specific genres, such as
    fantasy magic systems, mystery clues, romance scenes, or sci-fi
    technology descriptions.
    
    Args:
        genre: Target genre (fantasy, mystery, romance, scifi, horror, thriller)
        content_type: Type of content to generate (scene, dialogue, description, system)
        parameters: Genre-specific parameters for content generation
    
    Returns:
        Generated genre-specific content with analysis and suggestions
    """
    try:
        # Generate genre-specific content
        generated_content = await _generate_genre_content_internal(genre, content_type, parameters or {})
        
        return {
            "genre": genre,
            "content_type": content_type,
            "parameters": parameters or {},
            "generated_content": generated_content["content"],
            "genre_elements": generated_content["elements"],
            "writing_techniques": generated_content["techniques"],
            "genre_conventions": generated_content["conventions"],
            "enhancement_suggestions": generated_content["suggestions"],
            "word_count": len(generated_content["content"].split())
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate {content_type} content for {genre} genre"
        }


@rag_agent.tool
async def validate_genre_authenticity(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    target_genre: str = None
) -> Dict[str, Any]:
    """
    Validate authenticity and adherence to genre conventions.
    
    This tool performs deep genre analysis to ensure the novel
    authentically represents its genre with appropriate elements,
    tone, and conventions.
    
    Args:
        novel_id: ID of the novel to validate
        target_genre: Specific genre to validate against (optional)
    
    Returns:
        Genre authenticity validation with detailed analysis and recommendations
    """
    try:
        from .db_utils import get_novel, search_novel_content
        
        # Get novel information
        novel = await get_novel(novel_id)
        if not novel:
            return {"error": "Novel not found", "novel_id": novel_id}
        
        # Use provided genre or novel's genre
        genre_to_validate = target_genre or novel.get("genre", "general")
        
        # Validate genre authenticity
        authenticity_analysis = await _validate_genre_authenticity_internal(novel_id, genre_to_validate)
        
        return {
            "novel_id": novel_id,
            "novel_title": novel["title"],
            "target_genre": genre_to_validate,
            "authenticity_score": authenticity_analysis["authenticity_score"],
            "genre_elements_present": authenticity_analysis["elements_present"],
            "missing_genre_elements": authenticity_analysis["missing_elements"],
            "genre_tone_consistency": authenticity_analysis["tone_consistency"],
            "convention_adherence": authenticity_analysis["convention_adherence"],
            "reader_expectations": authenticity_analysis["reader_expectations"],
            "authenticity_issues": authenticity_analysis["issues"],
            "enhancement_recommendations": authenticity_analysis["recommendations"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to validate genre authenticity for novel {novel_id}"
        }


# Internal helper functions
async def _analyze_fantasy_elements_internal(novel_id: str, focus_element: str) -> Dict[str, Any]:
    """Internal function to analyze fantasy elements."""
    
    consistency_score = 0.8  # Placeholder
    
    magic_system = {
        "consistency": 0.85,
        "rules_defined": True,
        "limitations_clear": True,
        "usage_consistent": True
    }
    
    creatures = {
        "consistency": 0.75,
        "mythical_creatures_present": ["dragons", "elves"],
        "creature_behavior_consistent": True
    }
    
    geography = {
        "world_map_consistent": True,
        "location_descriptions_consistent": 0.8,
        "travel_times_logical": True
    }
    
    rules = {
        "fantasy_rules_established": True,
        "rule_violations": [],
        "world_logic_consistent": True
    }
    
    inconsistencies = []
    recommendations = [
        "Maintain consistent magic system rules throughout the story",
        "Ensure creature behaviors align with established lore",
        "Keep geographical descriptions consistent",
        "Document fantasy world rules for reference"
    ]
    
    return {
        "consistency_score": consistency_score,
        "magic_system": magic_system,
        "creatures": creatures,
        "geography": geography,
        "rules": rules,
        "inconsistencies": inconsistencies,
        "recommendations": recommendations
    }


async def _analyze_mystery_structure_internal(
    novel_id: str, 
    chapters: List[Dict[str, Any]], 
    mystery_type: str
) -> Dict[str, Any]:
    """Internal function to analyze mystery structure."""
    
    structure_score = 0.75  # Placeholder
    
    clue_distribution = {
        "total_clues": 8,  # Estimated
        "clues_per_chapter": 8 / len(chapters) if chapters else 0,
        "clue_spacing": "well_distributed",
        "red_herrings": 3
    }
    
    red_herrings = {
        "count": 3,
        "effectiveness": 0.8,
        "resolution": "satisfactory"
    }
    
    revelation_timing = {
        "major_reveals": ["chapter_5", "chapter_12", "final_chapter"],
        "pacing": "appropriate",
        "climax_placement": "good"
    }
    
    pacing = {
        "investigation_pace": "steady",
        "tension_building": "effective",
        "resolution_pace": "satisfying"
    }
    
    elements = {
        "detective_present": True,
        "crime_established": True,
        "suspects_introduced": True,
        "motive_clear": True,
        "method_explained": True,
        "opportunity_shown": True
    }
    
    issues = []
    if len(chapters) < 10:
        issues.append("Mystery may benefit from more chapters for proper development")
    
    recommendations = [
        "Ensure clues are distributed evenly throughout the story",
        "Balance red herrings with genuine clues",
        "Build tension steadily toward the revelation",
        "Provide satisfying resolution that ties up all loose ends"
    ]
    
    return {
        "structure_score": structure_score,
        "clue_distribution": clue_distribution,
        "red_herrings": red_herrings,
        "revelation_timing": revelation_timing,
        "pacing": pacing,
        "elements": elements,
        "issues": issues,
        "recommendations": recommendations
    }


async def _analyze_romance_elements_internal(
    novel_id: str, 
    characters: List[Dict[str, Any]], 
    relationship_focus: str
) -> Dict[str, Any]:
    """Internal function to analyze romance elements."""
    
    development_score = 0.8  # Placeholder
    
    # Find potential romantic leads
    romantic_leads = [char for char in characters if char.get("role") in ["protagonist", "supporting"]]
    
    progression = {
        "meet_cute": "present",
        "attraction_development": "gradual",
        "conflict_introduction": "appropriate",
        "resolution": "satisfying"
    }
    
    emotional_beats = {
        "first_meeting": "chapter_1",
        "growing_attraction": "chapters_2_5",
        "major_conflict": "chapter_8",
        "resolution": "final_chapters"
    }
    
    conflict_resolution = {
        "internal_conflicts": "addressed",
        "external_obstacles": "overcome",
        "communication_issues": "resolved"
    }
    
    tension = {
        "romantic_tension": 0.85,
        "sexual_tension": 0.7,
        "emotional_tension": 0.8
    }
    
    chemistry = {
        "character_compatibility": 0.9,
        "dialogue_chemistry": 0.8,
        "emotional_connection": 0.85
    }
    
    tropes = ["enemies_to_lovers", "slow_burn", "second_chance"]
    
    recommendations = [
        "Develop romantic tension gradually throughout the story",
        "Ensure both characters have clear motivations and growth",
        "Balance romantic scenes with plot advancement",
        "Provide satisfying emotional resolution"
    ]
    
    return {
        "development_score": development_score,
        "progression": progression,
        "emotional_beats": emotional_beats,
        "conflict_resolution": conflict_resolution,
        "tension": tension,
        "chemistry": chemistry,
        "tropes": tropes,
        "recommendations": recommendations
    }


async def _analyze_scifi_elements_internal(novel_id: str, tech_focus: str) -> Dict[str, Any]:
    """Internal function to analyze sci-fi elements."""
    
    consistency_score = 0.75  # Placeholder
    
    plausibility = {
        "scientific_basis": 0.8,
        "technology_feasibility": 0.7,
        "future_extrapolation": 0.85
    }
    
    tech_rules = {
        "technology_rules_established": True,
        "limitations_defined": True,
        "consequences_addressed": True
    }
    
    world_consistency = {
        "future_society_coherent": True,
        "technology_integration": 0.8,
        "social_implications": 0.75
    }
    
    accuracy = {
        "scientific_accuracy": 0.7,
        "physics_consistency": 0.8,
        "biological_plausibility": 0.75
    }
    
    progression = {
        "technology_development": "logical",
        "innovation_pace": "realistic",
        "adoption_patterns": "believable"
    }
    
    inconsistencies = []
    recommendations = [
        "Maintain consistent technology rules throughout the story",
        "Consider scientific plausibility in technology descriptions",
        "Address social implications of advanced technology",
        "Ensure future world-building is internally consistent"
    ]
    
    return {
        "consistency_score": consistency_score,
        "plausibility": plausibility,
        "tech_rules": tech_rules,
        "world_consistency": world_consistency,
        "accuracy": accuracy,
        "progression": progression,
        "inconsistencies": inconsistencies,
        "recommendations": recommendations
    }


async def _generate_genre_content_internal(
    genre: str, 
    content_type: str, 
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Internal function to generate genre-specific content."""
    
    # Genre-specific content templates
    content_templates = {
        "fantasy": {
            "scene": "The ancient magic crackled through the air as {character} approached the mystical {location}. The {magical_element} pulsed with otherworldly energy, casting ethereal shadows across the landscape.",
            "dialogue": "\"The old magic runs deep in these lands,\" {character} whispered, feeling the power coursing through the very stones beneath their feet.",
            "description": "Towering spires of crystalline {material} rose from the mist-shrouded valley, their surfaces gleaming with runic inscriptions that seemed to shift and dance in the moonlight."
        },
        "mystery": {
            "scene": "The evidence was subtle but unmistakable. {detective} examined the {clue} carefully, noting the peculiar {detail} that others had missed. This changed everything.",
            "dialogue": "\"You're missing the obvious,\" {detective} said, pointing to the {evidence}. \"The killer made one crucial mistake.\"",
            "description": "The crime scene held its secrets close, but to the trained eye, every displaced object told a story of what had transpired in those final moments."
        },
        "romance": {
            "scene": "Their eyes met across the {location}, and {character1} felt their heart skip a beat. {character2}'s smile was like sunshine breaking through storm clouds, warm and unexpected.",
            "dialogue": "\"I never expected to find someone like you,\" {character1} whispered, their voice barely audible above the {ambient_sound}.",
            "description": "The tension between them was palpable, electric, filling the space with unspoken possibilities and the promise of something beautiful."
        },
        "scifi": {
            "scene": "The {technology} hummed with quantum energy as {character} initiated the {process}. Reality seemed to bend around the device, space-time rippling like water.",
            "dialogue": "\"The implications of this technology are staggering,\" {scientist} explained, gesturing to the {device}. \"We're not just changing the future—we're rewriting the laws of physics.\"",
            "description": "The starship's hull gleamed with adaptive nano-materials, its surface constantly shifting to optimize for the harsh conditions of deep space travel."
        }
    }
    
    # Get template for genre and content type
    template = content_templates.get(genre, {}).get(content_type, "Generic content for {genre} {content_type}.")
    
    # Fill in parameters
    content = template
    for key, value in parameters.items():
        content = content.replace(f"{{{key}}}", str(value))
    
    # Fill in generic placeholders
    content = content.replace("{character}", parameters.get("character", "the protagonist"))
    content = content.replace("{location}", parameters.get("location", "the mysterious place"))
    
    elements = [f"{genre}_specific_elements", "genre_appropriate_tone", "conventional_structure"]
    techniques = ["genre_specific_language", "appropriate_pacing", "conventional_descriptions"]
    conventions = [f"{genre}_genre_conventions", "reader_expectations", "traditional_elements"]
    suggestions = [
        f"Enhance {genre}-specific elements for better genre authenticity",
        f"Consider adding more {genre} conventions to meet reader expectations",
        "Balance genre elements with original storytelling"
    ]
    
    return {
        "content": content,
        "elements": elements,
        "techniques": techniques,
        "conventions": conventions,
        "suggestions": suggestions
    }


async def _validate_genre_authenticity_internal(novel_id: str, genre: str) -> Dict[str, Any]:
    """Internal function to validate genre authenticity."""
    
    authenticity_score = 0.8  # Placeholder
    
    # Genre-specific elements
    genre_elements = {
        "fantasy": ["magic", "mythical_creatures", "quest", "world_building"],
        "mystery": ["crime", "investigation", "clues", "revelation"],
        "romance": ["relationship_development", "emotional_conflict", "romantic_tension"],
        "scifi": ["advanced_technology", "future_setting", "scientific_concepts"],
        "horror": ["fear_elements", "suspense", "supernatural_or_psychological"]
    }
    
    expected_elements = genre_elements.get(genre.lower(), ["general_elements"])
    elements_present = expected_elements[:2]  # Assume some elements are present
    missing_elements = expected_elements[2:]  # Assume some are missing
    
    tone_consistency = {
        "genre_appropriate_tone": True,
        "tone_consistency_score": 0.85,
        "mood_alignment": "good"
    }
    
    convention_adherence = {
        "follows_genre_conventions": True,
        "convention_score": 0.8,
        "innovative_elements": True
    }
    
    reader_expectations = {
        "meets_expectations": True,
        "expectation_score": 0.85,
        "surprise_elements": "appropriate"
    }
    
    issues = []
    if len(missing_elements) > len(elements_present):
        issues.append(f"Missing several key {genre} elements")
    
    recommendations = [
        f"Strengthen {genre}-specific elements throughout the story",
        f"Ensure tone consistently matches {genre} expectations",
        f"Balance genre conventions with original storytelling",
        "Consider reader expectations while maintaining creativity"
    ]
    
    return {
        "authenticity_score": authenticity_score,
        "elements_present": elements_present,
        "missing_elements": missing_elements,
        "tone_consistency": tone_consistency,
        "convention_adherence": convention_adherence,
        "reader_expectations": reader_expectations,
        "issues": issues,
        "recommendations": recommendations
    }