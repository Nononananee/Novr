"""Unit tests for core database functionality."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from agent.core.database.connection import DatabasePool, initialize_database, close_database
from agent.core.database.sessions import create_session, get_session, add_message
from agent.core.database.documents import get_document, list_documents


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseConnection:
    """Test database connection management."""
    
    async def test_database_pool_initialization(self):
        """Test database pool can be initialized."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            db_pool = DatabasePool("postgresql://test:test@localhost/test")
            await db_pool.initialize()
            
            assert db_pool.pool == mock_pool
            mock_create_pool.assert_called_once()
    
    async def test_database_pool_close(self):
        """Test database pool can be closed."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            db_pool = DatabasePool("postgresql://test:test@localhost/test")
            await db_pool.initialize()
            await db_pool.close()
            
            assert db_pool.pool is None
            mock_pool.close.assert_called_once()
    
    async def test_database_pool_acquire_context(self):
        """Test database connection acquisition context manager."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_create_pool.return_value = mock_pool
            
            db_pool = DatabasePool("postgresql://test:test@localhost/test")
            
            async with db_pool.acquire() as conn:
                assert conn == mock_conn


@pytest.mark.unit
@pytest.mark.database
class TestSessionManagement:
    """Test session management functionality."""
    
    async def test_create_session(self, mock_database):
        """Test session creation."""
        mock_database.fetchrow.return_value = {"id": "test-session-123"}
        
        session_id = await create_session(user_id="test-user", metadata={"test": "data"})
        
        assert session_id == "test-session-123"
        mock_database.fetchrow.assert_called_once()
    
    async def test_get_session_existing(self, mock_database):
        """Test retrieving existing session."""
        mock_session_data = {
            "id": "test-session-123",
            "user_id": "test-user",
            "metadata": '{"test": "data"}',
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "expires_at": None
        }
        mock_database.fetchrow.return_value = mock_session_data
        
        session = await get_session("test-session-123")
        
        assert session is not None
        assert session["id"] == "test-session-123"
        assert session["user_id"] == "test-user"
    
    async def test_get_session_not_found(self, mock_database):
        """Test retrieving non-existent session."""
        mock_database.fetchrow.return_value = None
        
        session = await get_session("non-existent-session")
        
        assert session is None
    
    async def test_add_message(self, mock_database):
        """Test adding message to session."""
        mock_database.fetchrow.return_value = {"id": "test-message-123"}
        
        message_id = await add_message(
            session_id="test-session-123",
            role="user",
            content="Test message",
            metadata={"test": "data"}
        )
        
        assert message_id == "test-message-123"
        mock_database.fetchrow.assert_called_once()


@pytest.mark.unit
@pytest.mark.database
class TestDocumentManagement:
    """Test document management functionality."""
    
    async def test_get_document_existing(self, mock_database):
        """Test retrieving existing document."""
        mock_doc_data = {
            "id": "test-doc-123",
            "title": "Test Document",
            "source": "test_source",
            "content": "Test content",
            "metadata": '{"test": "data"}',
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        mock_database.fetchrow.return_value = mock_doc_data
        
        document = await get_document("test-doc-123")
        
        assert document is not None
        assert document["id"] == "test-doc-123"
        assert document["title"] == "Test Document"
    
    async def test_get_document_not_found(self, mock_database):
        """Test retrieving non-existent document."""
        mock_database.fetchrow.return_value = None
        
        document = await get_document("non-existent-doc")
        
        assert document is None
    
    async def test_list_documents(self, mock_database):
        """Test listing documents."""
        mock_docs = [
            {
                "id": "doc-1",
                "title": "Document 1",
                "source": "source1",
                "metadata": '{}',
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "chunk_count": 5
            },
            {
                "id": "doc-2", 
                "title": "Document 2",
                "source": "source2",
                "metadata": '{}',
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "chunk_count": 3
            }
        ]
        mock_database.fetch.return_value = mock_docs
        
        documents = await list_documents(limit=10, offset=0)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "doc-1"
        assert documents[1]["id"] == "doc-2"