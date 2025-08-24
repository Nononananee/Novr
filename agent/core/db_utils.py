"""
Database utilities for PostgreSQL connection and operations.
Refactored from large monolithic file into modular components.
"""

# Import all functions from modular database components
from .database.connection import (
    initialize_database,
    close_database,
    test_connection,
    execute_query,
    db_pool
)

from .database.sessions import (
    create_session,
    get_session,
    update_session,
    add_message,
    get_session_messages
)

from .database.documents import (
    get_document,
    list_documents,
    get_document_chunks
)

from .database.search import (
    vector_search,
    hybrid_search
)

from .database.novels import (
    create_novel_tables,
    create_novel,
    get_novel,
    list_novels,
    create_character,
    list_characters,
    create_chapter,
    get_novel_chapters,
    get_character_arc,
    search_novel_content
)

# For backward compatibility, expose all functions
__all__ = [
    # Connection management
    "initialize_database",
    "close_database", 
    "test_connection",
    "execute_query",
    "db_pool",
    
    # Session management
    "create_session",
    "get_session",
    "update_session",
    "add_message",
    "get_session_messages",
    
    # Document management
    "get_document",
    "list_documents",
    "get_document_chunks",
    
    # Search functions
    "vector_search",
    "hybrid_search",
    
    # Novel-specific functions
    "create_novel_tables",
    "create_novel",
    "get_novel",
    "list_novels",
    "create_character",
    "list_characters",
    "create_chapter",
    "get_novel_chapters",
    "get_character_arc",
    "search_novel_content"
]