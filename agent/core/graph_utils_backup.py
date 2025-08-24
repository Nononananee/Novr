"""
Novel-aware graph utilities for Neo4j/Graphiti integration.
Enhanced for literary content processing with character, plot, and emotional tracking.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import asyncio
import re

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class NovelEntityType(Enum):
    """Types of novel entities."""
    CHARACTER = "character"
    LOCATION = "location"
    PLOT_POINT = "plot_point"
    THEME = "theme"
    OBJECT = "object"
    CONCEPT = "concept"
    SCENE = "scene"
    CHAPTER = "chapter"


class RelationshipType(Enum):
    """Types of relationships in novels."""
    CHARACTER_RELATIONSHIP = "character_relationship"
    CHARACTER_LOCATION = "character_at_location"
    CHARACTER_PLOT = "character_in_plot"
    PLOT_SEQUENCE = "plot_sequence"
    THEME_MANIFESTATION = "theme_manifestation"
    EMOTIONAL_CONNECTION = "emotional_connection"
    TEMPORAL_SEQUENCE = "temporal_sequence"


@dataclass
class NovelEntity:
    """Represents a novel entity with metadata."""
    name: str
    entity_type: NovelEntityType
    description: str
    first_appearance: Optional[str] = None
    significance_score: float = 0.0
    emotional_associations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.emotional_associations is None:
            self.emotional_associations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CharacterProfile:
    """Detailed character profile."""
    name: str
    personality_traits: List[str]
    relationships: Dict[str, str]
    development_arc: List[str]
    emotional_states: Dict[str, float]
    first_appearance: str
    significance_score: float
    dialogue_patterns: List[str] = None
    
    def __post_init__(self):
        if self.dialogue_patterns is None:
            self.dialogue_patterns = []

# Help from this PR for setting up the custom clients: https://github.com/getzep/graphiti/pull/601/files
class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize Graphiti client.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        # LLM configuration
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_choice = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
        
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        
        # Embedding configuration
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("VECTOR_DIMENSION", "1536"))
        
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")
        
        self.graphiti: Optional[Graphiti] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Graphiti client."""
        if self._initialized:
            return
        
        try:
            # Create LLMConfig
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,  # Can be the same as main model
                base_url=self.llm_base_url
            )
            
            # Create OpenAI LLM client
            llm_client = OpenAIClient(config=llm_config)
            
            # Create OpenAI embedder
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url
                )
            )
            
            # Initialize Graphiti with custom clients
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )
            
            # Build indices and constraints
            await self.graphiti.build_indices_and_constraints()
            
            self._initialized = True
            logger.info(f"Graphiti client initialized successfully with LLM: {self.llm_choice} and embedder: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise
    
    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")
    
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an episode to the knowledge graph.
        
        Args:
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata
        """
        if not self._initialized:
            await self.initialize()
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        # Import EpisodeType for proper source handling
        from graphiti_core.nodes import EpisodeType
        
        await self.graphiti.add_episode(
            name=episode_id,
            episode_body=content,
            source=EpisodeType.text,  # Always use text type for our content
            source_description=source,
            reference_time=episode_timestamp
        )
        
        logger.info(f"Added episode {episode_id} to knowledge graph")
    
    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search
        
        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's search method (simplified parameters)
            results = await self.graphiti.search(query)
            
            # Convert results to dictionaries
            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                    "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
                    "source_node_uuid": str(result.source_node_uuid) if hasattr(result, 'source_node_uuid') and result.source_node_uuid else None
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity using Graphiti search.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow (not used with Graphiti)
            depth: Maximum depth to traverse (not used with Graphiti)
        
        Returns:
            Related entities and relationships
        """
        if not self._initialized:
            await self.initialize()
        
        # Use Graphiti search to find related information about the entity
        results = await self.graphiti.search(f"relationships involving {entity_name}")
        
        # Extract entity information from the search results
        related_entities = set()
        facts = []
        
        for result in results:
            facts.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None
            })
            
            # Simple entity extraction from fact text (could be enhanced)
            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)
        
        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search"
        }
    
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.
        
        Args:
            entity_name: Name of the entity
            start_date: Start of time range (not currently used)
            end_date: End of time range (not currently used)
        
        Returns:
            Timeline of facts
        """
        if not self._initialized:
            await self.initialize()
        
        # Search for temporal information about the entity
        results = await self.graphiti.search(f"timeline history of {entity_name}")
        
        timeline = []
        for result in results:
            timeline.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None
            })
        
        # Sort by valid_at if available
        timeline.sort(key=lambda x: x.get('valid_at') or '', reverse=True)
        
        return timeline
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, return a simple search to verify the graph is working
        # More detailed statistics would require direct Neo4j access
        try:
            test_results = await self.graphiti.search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access"
            }
        except Exception as e:
            return {
                "graphiti_initialized": False,
                "error": str(e)
            }
    
    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's proper clear_data function with the driver
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            # Fallback: Close and reinitialize (this will create fresh indices)
            if self.graphiti:
                await self.graphiti.close()
            
            # Create OpenAI-compatible clients for reinitialization
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,
                base_url=self.llm_base_url
            )
            
            llm_client = OpenAIClient(config=llm_config)
            
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url
                )
            )
            
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )
            await self.graphiti.build_indices_and_constraints()
            
            logger.warning("Reinitialized Graphiti client (fresh indices created)")
    
    # Novel-specific methods
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
        if not self._initialized:
            await self.initialize()
        
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
        
        enhanced_content = f"{narrative_context}\n\n{content}"
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        from graphiti_core.nodes import EpisodeType
        
        await self.graphiti.add_episode(
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
        if not self._initialized:
            await self.initialize()
        
        query = f"character development arc for {character_name}"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graphiti.search(query)
        
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
        if not self._initialized:
            await self.initialize()
        
        query = f"emotional content with {emotion_type} feelings"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graphiti.search(query)
        
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
        if not self._initialized:
            await self.initialize()
        
        query = f"plot connections and relationships involving {plot_element}"
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graphiti.search(query)
        
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
        if not self._initialized:
            await self.initialize()
        
        if character2:
            query = f"relationship between {character1} and {character2}"
        else:
            query = f"all relationships involving {character1}"
        
        if novel_title:
            query += f" in {novel_title}"
        
        results = await self.graphiti.search(query)
        
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


# Global Graphiti client instance
graph_client = GraphitiClient()


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


# Convenience functions for common operations
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
    
    await graph_client.add_novel_episode(
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
    return await graph_client.search_character_development(character_name, novel_title)


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
    return await graph_client.search_emotional_content(
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
    return await graph_client.get_plot_connections(plot_element, novel_title)


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
    return await graph_client.analyze_character_relationships(
        character1, character2, novel_title
    )