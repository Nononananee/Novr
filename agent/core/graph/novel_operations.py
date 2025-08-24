"""Novel-specific graph operations."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NovelGraphOperations:
    """Handles novel-specific graph operations."""
    
    def __init__(self, graph_client):
        self.graph_client = graph_client
    
    async def add_novel_episode(
        self,
        episode_id: str,
        content: str,
        novel_title: str,
        chapter: Optional[str] = None,
        characters: Optional[List[str]] = None,
        location: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        plot_significance: float = 0.5,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a novel episode with enhanced narrative metadata.
        
        Args:
            episode_id: Unique episode identifier
            content: Episode content
            novel_title: Title of the novel
            chapter: Chapter information
            characters: Characters present in this episode
            location: Location/setting
            emotional_tone: Emotional tone of the episode
            plot_significance: Significance score (0.0 to 1.0)
            timestamp: Episode timestamp
        """
        if not self.graph_client._initialized:
            await self.graph_client.initialize()
        
        # Enhance content with narrative context
        narrative_context = f"[Novel: {novel_title}]"
        if chapter:
            narrative_context += f" [Chapter: {chapter}]"
        if characters:
            narrative_context += f" [Characters: {', '.join(characters)}]"
        if location:
            narrative_context += f" [Location: {location}]"
        if emotional_tone:
            narrative_context += f" [Tone: {emotional_tone}]"
        
        enhanced_content = f"{narrative_context}\\n\\n{content}"
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        from graphiti_core.nodes import EpisodeType
        
        await self.graph_client.graphiti.add_episode(
            name=episode_id,
            episode_body=enhanced_content,
            source=EpisodeType.text,
            source_description=f"{novel_title} - Chapter {chapter or 'Unknown'}",
            reference_time=episode_timestamp
        )
        
        logger.info(f"Added novel episode {episode_id} for {novel_title}")
    
    async def search_character_development(
        self,
        character_name: str,
        novel_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for character development across the story.
        
        Args:
            character_name: Name of the character
            novel_title: Optional novel title to filter results
        
        Returns:
            Character development timeline
        """
        if not self.graph_client._initialized:
            await self.graph_client.initialize()
        
        query = f"character development arc for {character_name}"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graph_client.graphiti.search(query)
        
        return [
            {
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                "character": character_name,
                "novel": novel_title
            }
            for result in results
        ]
    
    async def search_emotional_content(
        self,
        emotion_type: str,
        intensity_threshold: float = 0.5,
        novel_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for content with specific emotional qualities.
        
        Args:
            emotion_type: Type of emotion to search for
            intensity_threshold: Minimum emotional intensity
            novel_title: Optional novel title to filter results
        
        Returns:
            Emotional content results
        """
        if not self.graph_client._initialized:
            await self.graph_client.initialize()
        
        query = f"emotional content with {emotion_type} feelings"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graph_client.graphiti.search(query)
        
        return [
            {
                "fact": result.fact,
                "uuid": str(result.uuid),
                "emotion_type": emotion_type,
                "estimated_intensity": intensity_threshold,  # Would be calculated in real implementation
                "novel": novel_title
            }
            for result in results
        ]
    
    async def get_plot_connections(
        self,
        plot_element: str,
        novel_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get connections between plot elements.
        
        Args:
            plot_element: Plot element to analyze
            novel_title: Optional novel title to filter results
        
        Returns:
            Plot connections and relationships
        """
        if not self.graph_client._initialized:
            await self.graph_client.initialize()
        
        query = f"plot connections and relationships involving {plot_element}"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graph_client.graphiti.search(query)
        
        connections = []
        for result in results:
            connections.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "plot_element": plot_element,
                "connection_type": "narrative_link"  # Would be analyzed in real implementation
            })
        
        return {
            "central_plot_element": plot_element,
            "connections": connections,
            "novel": novel_title
        }
    
    async def analyze_character_relationships(
        self,
        character1: str,
        character2: Optional[str] = None,
        novel_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationships between characters.
        
        Args:
            character1: First character
            character2: Optional second character (if None, gets all relationships)
            novel_title: Optional novel title to filter results
        
        Returns:
            Character relationship analysis
        """
        if not self.graph_client._initialized:
            await self.graph_client.initialize()
        
        if character2:
            query = f"relationship between {character1} and {character2}"
        else:
            query = f"all relationships involving {character1}"
        
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graph_client.graphiti.search(query)
        
        relationships = []
        for result in results:
            relationships.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "character1": character1,
                "character2": character2,
                "relationship_type": "character_interaction"  # Would be analyzed in real implementation
            })
        
        return {
            "primary_character": character1,
            "target_character": character2,
            "relationships": relationships,
            "novel": novel_title
        }