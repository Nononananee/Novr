"""
Compatibility layer for database utilities.
Re-exports the modular database components under a legacy-friendly path.
"""

from .core.db_utils import (
    # Connection management
    initialize_database,
    close_database,
    test_connection,
    execute_query,
    db_pool,
    # Session management
    create_session,
    get_session,
    update_session,
    add_message,
    get_session_messages,
    # Document management
    get_document,
    list_documents,
    get_document_chunks,
    # Search
    vector_search,
    hybrid_search,
    # Novel-specific
    create_novel_tables,
    create_novel,
    get_novel,
    list_novels,
    create_character,
    list_characters,
    create_chapter,
    get_novel_chapters,
    get_character_arc,
    search_novel_content,
)

# Explicitly import DatabasePool for tests that reference it directly
from .core.database.connection import DatabasePool

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
    # Search
    "vector_search",
    "hybrid_search",
    # Novel-specific
    "create_novel_tables",
    "create_novel",
    "get_novel",
    "list_novels",
    "create_character",
    "list_characters",
    "create_chapter",
    "get_novel_chapters",
    "get_character_arc",
    "search_novel_content",
    # Pool class
    "DatabasePool",
]
