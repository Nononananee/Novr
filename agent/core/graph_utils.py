"""
Novel-aware graph utilities for Neo4j/Graphiti integration.
Refactored from large monolithic file into modular components.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import all components from modular structure
from .graph.client import GraphitiClient
from .graph.novel_operations import NovelGraphOperations
from .graph.models import (
    NovelEntityType,
    RelationshipType,
    NovelEntity,
    CharacterProfile
)
from .graph.operations import (
    graph_client,
    novel_ops,
    initialize_graph,
    close_graph,
    test_graph_connection,
    add_to_knowledge_graph,
    search_knowledge_graph,
    get_entity_relationships,
    add_novel_content_to_graph,
    search_character_arc,
    find_emotional_scenes,
    analyze_plot_structure,
    get_character_relationships
)

logger = logging.getLogger(__name__)

# For backward compatibility, expose all important functions and classes
__all__ = [
    # Core classes
    "GraphitiClient",
    "NovelGraphOperations",
    
    # Data models
    "NovelEntityType",
    "RelationshipType", 
    "NovelEntity",
    "CharacterProfile",
    
    # Global instances
    "graph_client",
    "novel_ops",
    
    # Connection management
    "initialize_graph",
    "close_graph",
    "test_graph_connection",
    
    # Basic operations
    "add_to_knowledge_graph",
    "search_knowledge_graph",
    "get_entity_relationships",
    
    # Novel-specific operations
    "add_novel_content_to_graph",
    "search_character_arc",
    "find_emotional_scenes",
    "analyze_plot_structure",
    "get_character_relationships"
]