"""
Main Pydantic AI agent for agentic RAG with knowledge graph.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import NOVEL_SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }


# Initialize the agent with flexible model configuration
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=NOVEL_SYSTEM_PROMPT
)


# Register tools with proper docstrings (no description parameter)
@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for relevant narrative content using semantic similarity.
    
    This tool performs vector similarity search across novel chunks
    to find semantically related content like similar scenes, character
    descriptions, emotional moments, or thematic elements.
    
    Args:
        query: Search query to find similar narrative content
        limit: Maximum number of results to return (1-50)
    
    Returns:
        List of matching narrative chunks ordered by similarity (best first)
    """
    input_data = VectorSearchInput(
        query=query,
        limit=limit
    )
    
    results = await vector_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]


@rag_agent.tool
async def graph_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for character relationships and story facts.
    
    This tool queries the knowledge graph to find specific facts about characters,
    their relationships, plot events, and temporal story information. Best for
    finding character connections, plot relationships, and story timeline data.
    
    Args:
        query: Search query to find character/story facts and relationships
    
    Returns:
        List of story facts with associated episodes and temporal data
    """
    input_data = GraphSearchInput(query=query)
    
    results = await graph_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "invalid_at": r.invalid_at,
            "source_node_uuid": r.source_node_uuid
        }
        for r in results
    ]


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform both vector and keyword search for comprehensive results.
    
    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores. Best for combining semantic and exact matching.
    
    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
    
    Returns:
        List of chunks ranked by combined relevance score
    """
    input_data = HybridSearchInput(
        query=query,
        limit=limit,
        text_weight=text_weight
    )
    
    results = await hybrid_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.
    
    This tool fetches the full document content along with all its chunks
    and metadata. Best for getting comprehensive information from a specific
    source when you need the complete context.
    
    Args:
        document_id: UUID of the document to retrieve
    
    Returns:
        Complete document data with content and metadata, or None if not found
    """
    input_data = DocumentInput(document_id=document_id)
    
    document = await get_document_tool(input_data)
    
    if document:
        # Format for agent consumption
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"]
        }
    
    return None


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.
    
    This tool provides an overview of all documents in the knowledge base,
    including titles, sources, and chunk counts. Best for understanding
    what information sources are available.
    
    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination
    
    Returns:
        List of documents with metadata and chunk counts
    """
    input_data = DocumentListInput(limit=limit, offset=offset)
    
    documents = await list_documents_tool(input_data)
    
    # Convert to dict for agent
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat()
        }
        for d in documents
    ]


@rag_agent.tool
async def get_entity_relationships(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get all relationships for a specific character or story entity.
    
    This tool explores the knowledge graph to find how a specific character,
    location, or story element relates to other entities. Best for
    understanding character relationships, location connections, and plot element interactions.
    
    Args:
        entity_name: Name of the character/entity to explore (e.g., "Aragorn", "Hogwarts")
        depth: Maximum traversal depth for relationships (1-5)
    
    Returns:
        Character/entity relationships and connected entities with relationship types
    """
    input_data = EntityRelationshipInput(
        entity_name=entity_name,
        depth=depth
    )
    
    return await get_entity_relationships_tool(input_data)


@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get the timeline of story events for a specific character or entity.
    
    This tool retrieves chronological information about a character or story element,
    showing how their story has developed over time. Best for understanding
    character development arcs, plot progression, and story timeline.
    
    Args:
        entity_name: Name of the character/entity (e.g., "Frodo", "The Ring")
        start_date: Start date in ISO format (YYYY-MM-DD), optional
        end_date: End date in ISO format (YYYY-MM-DD), optional
    
    Returns:
        Chronological list of story events about the character/entity with timestamps
    """
    input_data = EntityTimelineInput(
        entity_name=entity_name,
        start_date=start_date,
        end_date=end_date
    )
    
    return await get_entity_timeline_tool(input_data)


# Novel-specific tools
@rag_agent.tool
async def create_novel(
    ctx: RunContext[AgentDependencies],
    title: str,
    author: str = "",
    genre: str = "general",
    summary: str = ""
) -> Dict[str, Any]:
    """
    Create a new novel in the system.
    
    This tool creates a new novel entry with basic metadata. Use this when
    starting work on a new novel project or when organizing existing content
    into a novel structure.
    
    Args:
        title: Title of the novel
        author: Author name (optional)
        genre: Genre of the novel (e.g., fantasy, mystery, romance)
        summary: Brief summary of the novel (optional)
    
    Returns:
        Novel creation result with ID and metadata
    """
    try:
        from .db_utils import create_novel
        
        novel_id = await create_novel(
            title=title,
            author=author,
            genre=genre,
            summary=summary if summary else None
        )
        
        return {
            "novel_id": novel_id,
            "title": title,
            "author": author,
            "genre": genre,
            "status": "created",
            "message": f"Novel '{title}' created successfully"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "message": f"Failed to create novel '{title}'"
        }


@rag_agent.tool
async def list_novels(
    ctx: RunContext[AgentDependencies],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List all novels in the system.
    
    This tool retrieves a list of all novels with their basic information.
    Useful for getting an overview of available novels or selecting a novel
    to work with.
    
    Args:
        limit: Maximum number of novels to return (1-50)
    
    Returns:
        List of novels with their metadata
    """
    try:
        from .db_utils import list_novels
        
        novels = await list_novels(limit=min(limit, 50))
        
        return novels
    except Exception as e:
        return [{
            "error": str(e),
            "message": "Failed to retrieve novels"
        }]


@rag_agent.tool
async def create_character(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    name: str,
    personality_traits: List[str] = None,
    background: str = "",
    role: str = "minor"
) -> Dict[str, Any]:
    """
    Create a new character for a novel.
    
    This tool creates a character with specified traits and background.
    Characters are essential for maintaining consistency in storytelling
    and can be referenced throughout the novel.
    
    Args:
        novel_id: ID of the novel this character belongs to
        name: Character name
        personality_traits: List of personality traits (optional)
        background: Character background story (optional)
        role: Character role (protagonist, antagonist, supporting, minor)
    
    Returns:
        Character creation result with ID and details
    """
    try:
        from .db_utils import create_character
        
        character_id = await create_character(
            novel_id=novel_id,
            name=name,
            personality_traits=personality_traits or [],
            background=background,
            role=role
        )
        
        return {
            "character_id": character_id,
            "name": name,
            "role": role,
            "personality_traits": personality_traits or [],
            "background": background,
            "status": "created",
            "message": f"Character '{name}' created successfully"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "message": f"Failed to create character '{name}'"
        }


@rag_agent.tool
async def list_characters(
    ctx: RunContext[AgentDependencies],
    novel_id: str
) -> List[Dict[str, Any]]:
    """
    List all characters for a specific novel.
    
    This tool retrieves all characters associated with a novel, showing
    their roles, traits, and basic information. Useful for character
    consistency checks and story development.
    
    Args:
        novel_id: ID of the novel to get characters for
    
    Returns:
        List of characters with their details
    """
    try:
        from .db_utils import list_characters
        
        characters = await list_characters(novel_id=novel_id)
        
        return characters
    except Exception as e:
        return [{
            "error": str(e),
            "message": f"Failed to retrieve characters for novel {novel_id}"
        }]


@rag_agent.tool
async def search_novel_content(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    query: str,
    character_filter: str = None,
    emotional_tone_filter: str = None,
    content_type: str = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for specific content within a novel.
    
    This tool performs targeted searches within a novel's content with
    various filters. Useful for finding specific scenes, character moments,
    or thematic elements within the story.
    
    Args:
        novel_id: ID of the novel to search in
        query: Search query text
        character_filter: Filter by character name (optional)
        emotional_tone_filter: Filter by emotional tone (optional)
        content_type: Filter by content type (dialogue, narration, etc.)
        limit: Maximum number of results (1-20)
    
    Returns:
        List of matching content with context information
    """
    try:
        from .db_utils import search_novel_content
        
        results = await search_novel_content(
            novel_id=novel_id,
            query=query,
            character_filter=character_filter,
            emotional_tone_filter=emotional_tone_filter,
            content_type=content_type,
            limit=min(limit, 20)
        )
        
        return results
    except Exception as e:
        return [{
            "error": str(e),
            "message": f"Failed to search novel content for query '{query}'"
        }]


@rag_agent.tool
async def get_character_arc(
    ctx: RunContext[AgentDependencies],
    character_id: str
) -> List[Dict[str, Any]]:
    """
    Get the development arc of a character across the story.
    
    This tool traces a character's appearances and development throughout
    the novel, showing their progression across chapters and scenes.
    Essential for character consistency and development analysis.
    
    Args:
        character_id: ID of the character to analyze
    
    Returns:
        Character's development arc with scenes and progression
    """
    try:
        from .db_utils import get_character_arc
        
        arc_data = await get_character_arc(character_id=character_id)
        
        return arc_data
    except Exception as e:
        return [{
            "error": str(e),
            "message": f"Failed to retrieve character arc for character {character_id}"
        }]


@rag_agent.tool
async def create_chapter(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    chapter_number: int,
    title: str = None,
    summary: str = None
) -> Dict[str, Any]:
    """
    Create a new chapter for a novel.
    
    This tool creates a chapter structure within a novel. Chapters help
    organize the story and provide structure for scenes and narrative flow.
    
    Args:
        novel_id: ID of the novel this chapter belongs to
        chapter_number: Chapter number in sequence
        title: Chapter title (optional)
        summary: Brief chapter summary (optional)
    
    Returns:
        Chapter creation result with ID and details
    """
    try:
        from .db_utils import create_chapter
        
        chapter_id = await create_chapter(
            novel_id=novel_id,
            chapter_number=chapter_number,
            title=title,
            summary=summary
        )
        
        return {
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "title": title,
            "summary": summary,
            "status": "created",
            "message": f"Chapter {chapter_number} created successfully"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "message": f"Failed to create chapter {chapter_number}"
        }


@rag_agent.tool
async def get_novel_chapters(
    ctx: RunContext[AgentDependencies],
    novel_id: str,
    start_chapter: int = 1,
    end_chapter: int = None
) -> List[Dict[str, Any]]:
    """
    Get chapters of a novel within a specified range.
    
    This tool retrieves chapter information for a novel, optionally
    filtering by chapter range. Useful for reviewing story structure
    and chapter organization.
    
    Args:
        novel_id: ID of the novel to get chapters for
        start_chapter: Starting chapter number (default: 1)
        end_chapter: Ending chapter number (optional, gets all if not specified)
    
    Returns:
        List of chapters with their details and metadata
    """
    try:
        from .db_utils import get_novel_chapters
        
        chapters = await get_novel_chapters(
            novel_id=novel_id,
            start_chapter=start_chapter,
            end_chapter=end_chapter
        )
        
        return chapters
    except Exception as e:
        return [{
            "error": str(e),
            "message": f"Failed to retrieve chapters for novel {novel_id}"
        }]