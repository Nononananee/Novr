"""General graph operations and convenience functions."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .client import GraphitiClient
from .novel_operations import NovelGraphOperations

logger = logging.getLogger(__name__)

# Global Graphiti client instance
graph_client = GraphitiClient()
novel_ops = NovelGraphOperations(graph_client)


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


async def test_graph_connection() -> bool:
    """
    Test graph database connection.
    
    Returns:
        True if connection successful
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False


# Basic graph operations
async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add content to the knowledge graph.
    
    Args:
        content: Content to add
        source: Source of the content
        episode_id: Optional episode ID
        metadata: Optional metadata
    
    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    
    await graph_client.add_episode(
        episode_id=episode_id,
        content=content,
        source=source,
        metadata=metadata
    )
    
    return episode_id


async def search_knowledge_graph(
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.
    
    Args:
        query: Search query
    
    Returns:
        Search results
    """
    return await graph_client.search(query)


async def get_entity_relationships(
    entity: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Args:
        entity: Entity name
        depth: Maximum traversal depth
    
    Returns:
        Entity relationships
    """
    return await graph_client.get_related_entities(entity, depth=depth)


# Novel-specific convenience functions
async def add_novel_content_to_graph(
    content: str,
    novel_title: str,
    chapter: Optional[str] = None,
    characters: Optional[List[str]] = None,
    location: Optional[str] = None,
    emotional_tone: Optional[str] = None,
    episode_id: Optional[str] = None
) -> str:
    """
    Add novel content to the knowledge graph with narrative metadata.
    
    Args:
        content: Novel content to add
        novel_title: Title of the novel
        chapter: Chapter information
        characters: Characters present
        location: Location/setting
        emotional_tone: Emotional tone
        episode_id: Optional episode ID
    
    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"novel_{novel_title}_{chapter or 'unknown'}_{datetime.now(timezone.utc).isoformat()}"
    
    await novel_ops.add_novel_episode(
        episode_id=episode_id,
        content=content,
        novel_title=novel_title,
        chapter=chapter,
        characters=characters,
        location=location,
        emotional_tone=emotional_tone
    )
    
    return episode_id


async def search_character_arc(
    character_name: str,
    novel_title: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for character development arc.
    
    Args:
        character_name: Name of the character
        novel_title: Optional novel title
    
    Returns:
        Character development results
    """
    return await novel_ops.search_character_development(character_name, novel_title)


async def find_emotional_scenes(
    emotion_type: str,
    novel_title: Optional[str] = None,
    intensity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Find scenes with specific emotional content.
    
    Args:
        emotion_type: Type of emotion to search for
        novel_title: Optional novel title
        intensity_threshold: Minimum emotional intensity
    
    Returns:
        Emotional content results
    """
    return await novel_ops.search_emotional_content(
        emotion_type, intensity_threshold, novel_title
    )


async def analyze_plot_structure(
    plot_element: str,
    novel_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze plot structure and connections.
    
    Args:
        plot_element: Plot element to analyze
        novel_title: Optional novel title
    
    Returns:
        Plot analysis results
    """
    return await novel_ops.get_plot_connections(plot_element, novel_title)


async def get_character_relationships(
    character1: str,
    character2: Optional[str] = None,
    novel_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get character relationships and interactions.
    
    Args:
        character1: First character
        character2: Optional second character
        novel_title: Optional novel title
    
    Returns:
        Character relationship analysis
    """
    return await novel_ops.analyze_character_relationships(
        character1, character2, novel_title
    )