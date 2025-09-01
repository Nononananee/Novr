import logging
from typing import Dict, Any, List, Optional
import asyncio
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ContextRetriever:
    def __init__(self, qdrant_client=None, neo4j_client=None):
        """
        Initialize context retriever with database clients
        
        Args:
            qdrant_client: Qdrant client for semantic search
            neo4j_client: Neo4j client for character relationships
        """
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        logger.info("Initialized ContextRetriever")
    
    async def retrieve_semantic_context(self, 
                                      project_id: str, 
                                      query: str, 
                                      top_k: int = 4) -> Dict[str, Any]:
        """
        Retrieve semantic context from Qdrant vector database
        
        Args:
            project_id: Project identifier
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with semantic context
        """
        try:
            if not self.qdrant_client:
                logger.warning("Qdrant client not available")
                return {"chunks": [], "context_text": ""}
            
            logger.info(f"Retrieving semantic context for project {project_id}")
            
            # Search for relevant chunks. The client may be an async client
            # returning a coroutine, or a MagicMock returning a plain value. We
            # support both by awaiting only when necessary.
            search_call = self.qdrant_client.search_text(
                collection_name="novel_chunks",
                query_text=query,
                top_k=top_k,
                filter_conditions={"project_id": project_id}
            )
            if asyncio.iscoroutine(search_call) or isinstance(search_call, asyncio.Future):
                results = await search_call
            else:
                results = search_call
            
            # Process results
            chunks = []
            context_parts = []
            
            for result in results:
                chunk_data = {
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0)
                }
                chunks.append(chunk_data)
                context_parts.append(chunk_data["text"])
            
            context_text = "\n\n".join(context_parts)
            
            logger.info(f"Retrieved {len(chunks)} semantic chunks ({len(context_text)} chars)")
            
            return {
                "chunks": chunks,
                "context_text": context_text,
                "total_chunks": len(chunks),
                "total_characters": len(context_text)
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve semantic context: {e}")
            return {"chunks": [], "context_text": "", "error": str(e)}
    
    async def retrieve_character_context(self, 
                                       project_id: str, 
                                       character_names: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve character context from Neo4j graph database
        
        Args:
            project_id: Project identifier
            character_names: List of character names to focus on
            
        Returns:
            Dictionary with character context
        """
        try:
            if not self.neo4j_client:
                logger.warning("Neo4j client not available")
                return {"characters": [], "relationships": []}
            
            logger.info(f"Retrieving character context for project {project_id}")
            
            # Get all characters if none specified
            if not character_names:
                characters = await self.neo4j_client.get_project_characters(project_id)
                character_ids = [char["character_id"] for char in characters]
            else:
                # Convert names to IDs (simplified - in practice you'd have a name->ID mapping)
                character_ids = [f"{project_id}_{name.lower().replace(' ', '_')}" for name in character_names]
            
            if not character_ids:
                return {"characters": [], "relationships": []}

            # Get character context (handle sync or async client)
            context_call = self.neo4j_client.get_character_context(project_id, character_ids)
            if asyncio.iscoroutine(context_call) or isinstance(context_call, asyncio.Future):
                context = await context_call
            else:
                context = context_call
            
            logger.info(f"Retrieved context for {len(context.get('characters', []))} characters")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve character context: {e}")
            return {"characters": [], "relationships": [], "error": str(e)}
    
    async def retrieve_comprehensive_context(self, 
                                           project_id: str, 
                                           query: str,
                                           character_names: List[str] = None,
                                           semantic_top_k: int = 4) -> Dict[str, Any]:
        """
        Retrieve comprehensive context combining semantic and character data
        
        Args:
            project_id: Project identifier
            query: Search query for semantic context
            character_names: Character names for character context
            semantic_top_k: Number of semantic results
            
        Returns:
            Combined context dictionary
        """
        try:
            logger.info(f"Retrieving comprehensive context for project {project_id}")
            
            # Retrieve both types of context in parallel
            semantic_task = self.retrieve_semantic_context(project_id, query, semantic_top_k)
            character_task = self.retrieve_character_context(project_id, character_names)
            
            semantic_context, character_context = await asyncio.gather(
                semantic_task, character_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(semantic_context, Exception):
                logger.error(f"Semantic context retrieval failed: {semantic_context}")
                semantic_context = {"chunks": [], "context_text": ""}
            
            if isinstance(character_context, Exception):
                logger.error(f"Character context retrieval failed: {character_context}")
                character_context = {"characters": [], "relationships": []}
            
            # Combine contexts
            combined_context = {
                "semantic": semantic_context,
                "characters": character_context,
                "combined_text": self._build_combined_context_text(semantic_context, character_context)
            }
            
            logger.info(f"Retrieved comprehensive context: "
                       f"{len(semantic_context.get('chunks', []))} semantic chunks, "
                       f"{len(character_context.get('characters', []))} characters")
            
            return combined_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve comprehensive context: {e}")
            return {
                "semantic": {"chunks": [], "context_text": ""},
                "characters": {"characters": [], "relationships": []},
                "combined_text": "",
                "error": str(e)
            }

    async def get_context(self, query: str, project_id: str, character_names: List[str] = None, semantic_top_k: int = 4) -> Dict[str, Any]:
        """
        Backwards-compatible helper used by tests and callers.
        Validates required clients and returns a simplified dict containing
        semantic_results, character_data, and retrieved_at timestamp.
        """
        # Raise if required clients are not present - tests expect this behavior
        if not self.qdrant_client or not self.neo4j_client:
            raise Exception("Missing database clients")

        context = await self.retrieve_comprehensive_context(project_id=project_id, query=query, character_names=character_names, semantic_top_k=semantic_top_k)

        semantic = context.get("semantic", {})
        characters = context.get("characters", {})

        return {
            "semantic_results": semantic.get("chunks", []),
            "character_data": characters,
            "character_relationships": characters.get("relationships", []),
            "combined_text": context.get("combined_text", ""),
            "retrieved_at": datetime.now(timezone.utc).timestamp()
        }
    
    def _build_combined_context_text(self, 
                                   semantic_context: Dict[str, Any], 
                                   character_context: Dict[str, Any]) -> str:
        """Build combined context text from semantic and character data"""
        parts = []
        
        # Add character information
        if character_context.get("characters"):
            parts.append("=== CHARACTER INFORMATION ===")
            for char in character_context["characters"]:
                char_info = f"Character: {char.get('name', 'Unknown')}"
                if char.get('description'):
                    char_info += f"\nDescription: {char['description']}"
                if char.get('traits'):
                    char_info += f"\nTraits: {', '.join(char['traits'])}"
                if char.get('role'):
                    char_info += f"\nRole: {char['role']}"
                parts.append(char_info)
        
        # Add relationship information
        if character_context.get("relationships"):
            parts.append("=== CHARACTER RELATIONSHIPS ===")
            for rel in character_context["relationships"]:
                rel_info = f"{rel.get('character', '')} {rel.get('relationship', '')} {rel.get('related_to', '')}"
                if rel.get('details'):
                    rel_info += f" ({rel['details']})"
                parts.append(rel_info)
        
        # Add semantic context
        if semantic_context.get("context_text"):
            parts.append("=== RELEVANT BACKGROUND ===")
            parts.append(semantic_context["context_text"])
        
        return "\n\n".join(parts)
    
    async def extract_character_names_from_text(self, text: str) -> List[str]:
        """
        Extract potential character names from text
        Simple implementation - in practice you'd use NER or more sophisticated methods
        """
        # Find capitalized words that could be names
        potential_names = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        
        # Filter out common words that aren't names
        common_words = {
            'The', 'This', 'That', 'They', 'There', 'Then', 'When', 'Where', 'What', 'Who',
            'How', 'Why', 'But', 'And', 'Or', 'So', 'Yet', 'For', 'Nor', 'After', 'Before',
            'During', 'While', 'Since', 'Until', 'Although', 'Because', 'However', 'Therefore',
            'Chapter', 'Book', 'Story', 'Novel', 'Page', 'Part', 'Section'
        }
        
        # Count occurrences and filter
        name_counts = {}
        for name in potential_names:
            if name not in common_words and len(name) > 2:
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Return names that appear more than once (likely to be character names)
        character_names = [name for name, count in name_counts.items() if count > 1]
        
        logger.info(f"Extracted potential character names: {character_names}")
        return character_names
    
    async def get_plot_context(self, project_id: str, chapter_id: str = None) -> Dict[str, Any]:
        """
        Retrieve plot context for story continuity
        This is a placeholder - in practice you'd query a plot database
        """
        try:
            # Placeholder implementation
            # In a full implementation, this would query MongoDB for plot outlines
            plot_context = {
                "current_chapter": chapter_id,
                "story_arc_progress": 0.3,
                "key_plot_points": [
                    "Protagonist discovers powers",
                    "First conflict with antagonist",
                    "Mentor appears"
                ],
                "unresolved_conflicts": [
                    "Mystery of protagonist's origin",
                    "Threat to the kingdom"
                ]
            }
            
            logger.info(f"Retrieved plot context for project {project_id}")
            return plot_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve plot context: {e}")
            return {"error": str(e)}