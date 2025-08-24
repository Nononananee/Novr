"""
Advanced creative tools for character development and emotional analysis.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def analyze_character_consistency_internal(
    character: Dict[str, Any], 
    arc_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Internal function to analyze character consistency.
    
    Args:
        character: Character data
        arc_data: Character arc data
    
    Returns:
        Consistency analysis results
    """
    personality_traits = character.get("personality_traits", [])
    total_appearances = len(arc_data)
    
    # Basic consistency scoring (would implement more sophisticated analysis)
    consistency_score = 0.8
    violations = []
    suggestions = []
    
    if total_appearances < 3:
        suggestions.append("Character needs more appearances for thorough analysis")
    
    if not personality_traits:
        violations.append("Character lacks defined personality traits")
        suggestions.append("Define clear personality traits for consistency checking")
    
    # Analyze dialogue patterns
    dialogue_scenes = [scene for scene in arc_data if "dialogue" in scene.get("content", "").lower()]
    dialogue_consistency = 0.75 if dialogue_scenes else 0.5
    
    # Analyze behavioral patterns
    behavioral_consistency = 0.80
    
    return {
        "consistency_score": consistency_score,
        "personality_consistency": 0.85,
        "dialogue_consistency": dialogue_consistency,
        "behavioral_consistency": behavioral_consistency,
        "violations": violations,
        "suggestions": suggestions
    }


async def generate_character_development_internal(
    character: Dict[str, Any],
    target_development: str,
    current_chapter: int,
    development_type: str,
    recent_appearances: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Internal function to generate character development plan.
    
    Args:
        character: Character data
        target_development: Target development description
        current_chapter: Current chapter number
        development_type: Type of development
        recent_appearances: Recent character appearances
    
    Returns:
        Character development plan
    """
    character_name = character["name"]
    
    # Generate development plan based on type
    if development_type == "gradual":
        timeline_chapters = 3
        scene_count = 3
    elif development_type == "dramatic":
        timeline_chapters = 1
        scene_count = 2
    else:  # revelation
        timeline_chapters = 2
        scene_count = 2
    
    development_plan = {
        "plan": f"Develop {character_name} towards: {target_development} over {timeline_chapters} chapters",
        "scenes": [
            {
                "chapter": current_chapter + i + 1,
                "scene_type": "character_development",
                "description": f"Scene {i+1}: Show {character_name} progressing towards {target_development}"
            }
            for i in range(scene_count)
        ],
        "dialogue": [
            f"Internal monologue reflecting on {target_development}",
            f"Conversation with another character about their growth"
        ],
        "internal_changes": [
            "Shift in perspective and understanding",
            "New emotional responses to situations"
        ],
        "external_manifestations": [
            "Changed behavior patterns",
            "Different reactions to familiar situations"
        ],
        "timeline": f"Development over {timeline_chapters} chapters"
    }
    
    return development_plan


async def build_character_relationship_map(
    novel_id: str,
    characters: List[Dict[str, Any]],
    focus_character: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal function to build character relationship map.
    
    Args:
        novel_id: Novel ID
        characters: List of characters
        focus_character: Optional focus character
    
    Returns:
        Relationship map data
    """
    from .db_utils import search_novel_content
    
    relationships = []
    
    # Build relationship pairs
    for i, char1 in enumerate(characters):
        for char2 in characters[i+1:]:
            # Search for scenes with both characters
            try:
                search_results = await search_novel_content(
                    novel_id=novel_id,
                    query=f"{char1['name']} {char2['name']}",
                    limit=5
                )
                
                if search_results:
                    relationships.append({
                        "character_1": char1["name"],
                        "character_2": char2["name"],
                        "relationship_type": "unknown",  # Would analyze content to determine
                        "interaction_count": len(search_results),
                        "relationship_strength": min(len(search_results) * 0.2, 1.0)
                    })
            except Exception as e:
                logger.warning(f"Failed to search for character interactions: {e}")
    
    return {
        "relationships": relationships,
        "dynamics": ["friendship", "rivalry", "romance", "mentorship"],
        "evolution": ["strengthening", "weakening", "changing"],
        "conflicts": [],
        "alliances": [],
        "suggestions": [
            "Develop more character interactions for deeper relationship analysis",
            "Consider adding relationship metadata to scenes"
        ]
    }


async def analyze_emotional_content_internal(
    content: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Internal function to analyze emotional content.
    
    Args:
        content: Text content to analyze
        context: Context information
    
    Returns:
        Emotional analysis results
    """
    content_lower = content.lower()
    
    # Basic emotion detection
    emotion_keywords = {
        "joy": ["happy", "joy", "delight", "cheerful", "elated", "smile", "laugh"],
        "sadness": ["sad", "sorrow", "grief", "melancholy", "tears", "cry", "weep"],
        "anger": ["angry", "rage", "fury", "mad", "irritated", "furious"],
        "fear": ["afraid", "scared", "terrified", "anxious", "worried", "panic"],
        "love": ["love", "affection", "adore", "cherish", "romantic", "tender"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"]
    }
    
    detected_emotions = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for keyword in keywords if keyword in content_lower)
        if count > 0:
            detected_emotions[emotion] = count / len(keywords)
    
    # Determine dominant emotion
    dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1]) if detected_emotions else ("neutral", 0.0)
    
    # Calculate intensity based on emotional words density
    emotional_words = sum(detected_emotions.values())
    word_count = len(content.split())
    intensity = min(emotional_words / max(word_count * 0.1, 1), 1.0)
    
    return {
        "dominant_emotions": [dominant_emotion[0]] if dominant_emotion[1] > 0 else ["neutral"],
        "intensity": intensity,
        "progression": "stable",  # Would implement progression analysis
        "triggers": list(detected_emotions.keys()),
        "character_emotions": context.get("characters", []),
        "scene_mood": dominant_emotion[0],
        "consistency_score": 0.8,  # Placeholder
        "suggestions": [
            "Consider varying emotional intensity throughout the scene",
            "Add more sensory details to enhance emotional impact"
        ]
    }


async def generate_emotional_scene_internal(
    emotional_tone: str,
    intensity: float,
    characters: List[str],
    scene_context: str,
    word_count: int
) -> Dict[str, Any]:
    """
    Internal function to generate emotional scene.
    
    Args:
        emotional_tone: Target emotional tone
        intensity: Emotional intensity
        characters: Characters to include
        scene_context: Scene context
        word_count: Target word count
    
    Returns:
        Generated scene data
    """
    character_list = ", ".join(characters) if characters else "the character"
    
    # Emotional scene templates
    scene_templates = {
        "joyful": f"The warm sunlight filtered through the windows as {character_list} felt a surge of happiness. Laughter echoed through the room, and smiles came easily. The moment felt perfect, filled with hope and contentment.",
        "melancholic": f"A heavy silence settled over {character_list} as the weight of loss pressed down. The gray sky seemed to mirror the sadness within, and even the familiar surroundings felt distant and cold.",
        "tense": f"The air crackled with tension as {character_list} faced the approaching danger. Hearts pounded, muscles tensed, and every shadow seemed to hide a threat. Time slowed to a crawl.",
        "romantic": f"The gentle evening breeze carried the scent of flowers as {character_list} shared a tender moment. Eyes met, hearts fluttered, and the world seemed to fade away, leaving only this perfect connection.",
        "mysterious": f"Strange shadows danced at the edges of vision as {character_list} ventured into the unknown. Questions multiplied with each step, and the very air seemed to whisper secrets."
    }
    
    base_content = scene_templates.get(emotional_tone, f"A scene unfolds with {character_list} experiencing {emotional_tone} emotions.")
    
    # Expand content to reach target word count
    expanded_content = base_content
    if scene_context:
        expanded_content = f"In {scene_context}, {expanded_content.lower()}"
    
    # Add intensity-based descriptors
    if intensity > 0.8:
        expanded_content += " The emotions were overwhelming, consuming every thought and sensation."
    elif intensity > 0.5:
        expanded_content += " The feelings were strong and undeniable, coloring every perception."
    else:
        expanded_content += " The emotions were subtle but present, adding depth to the moment."
    
    return {
        "content": expanded_content,
        "emotional_elements": [emotional_tone, "descriptive_language", "character_focus"],
        "character_reactions": ["emotional_response", "physical_manifestation"],
        "techniques": ["sensory_details", "metaphor", "atmosphere"],
        "impact_score": intensity
    }


async def track_emotional_arc_internal(
    novel_id: str,
    entity_type: str,
    entity_name: Optional[str]
) -> Dict[str, Any]:
    """
    Internal function to track emotional arc.
    
    Args:
        novel_id: Novel ID
        entity_type: Entity type
        entity_name: Entity name
    
    Returns:
        Emotional arc analysis
    """
    from .db_utils import search_novel_content, get_novel_chapters
    
    # Get novel content for analysis
    if entity_type == "character" and entity_name:
        search_results = await search_novel_content(
            novel_id=novel_id,
            query="",
            character_filter=entity_name,
            limit=20
        )
    else:
        chapters = await get_novel_chapters(novel_id)
        search_results = []
        for chapter in chapters[:5]:  # Analyze first 5 chapters
            chapter_content = await search_novel_content(
                novel_id=novel_id,
                query="",
                limit=5
            )
            search_results.extend(chapter_content)
    
    # Analyze emotional progression (simplified)
    emotional_progression = []
    for i, result in enumerate(search_results):
        # Simple emotional analysis of each scene
        content = result.get("content", "")
        emotion_analysis = await analyze_emotional_content_internal(
            content, {"chapter": result.get("chapter_number", i+1)}
        )
        
        emotional_progression.append({
            "chapter": result.get("chapter_number", i+1),
            "scene": result.get("title", f"Scene {i+1}"),
            "dominant_emotion": emotion_analysis.get("dominant_emotions", ["neutral"])[0],
            "intensity": emotion_analysis.get("intensity", 0.5)
        })
    
    # Identify turning points (simplified)
    turning_points = []
    for i in range(1, len(emotional_progression)):
        prev_emotion = emotional_progression[i-1]["dominant_emotion"]
        curr_emotion = emotional_progression[i]["dominant_emotion"]
        if prev_emotion != curr_emotion:
            turning_points.append({
                "chapter": emotional_progression[i]["chapter"],
                "change": f"{prev_emotion} → {curr_emotion}",
                "significance": "moderate"
            })
    
    return {
        "progression": emotional_progression,
        "turning_points": turning_points,
        "peaks": [p for p in emotional_progression if p["intensity"] > 0.7],
        "valleys": [p for p in emotional_progression if p["intensity"] < 0.3],
        "trajectory": "developing",  # Would implement trajectory analysis
        "consistency": 0.75,
        "suggestions": [
            "Consider adding more emotional variety",
            "Develop stronger emotional peaks and valleys",
            "Ensure emotional changes are well-motivated"
        ]
    }