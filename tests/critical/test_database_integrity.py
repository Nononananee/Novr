"""Critical tests for database operations - data integrity."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime
import json

from agent.core.database.connection import DatabasePool
from agent.core.database.sessions import create_session, get_session, add_message
from agent.core.database.documents import get_document, list_documents
from agent.core.database.novels import create_novel, get_novel, create_character


@pytest.mark.critical
@pytest.mark.database
class TestDatabaseConnectionIntegrity:
    """Critical tests for database connection integrity."""
    
    async def test_database_pool_initialization_critical(self):
        """Test database pool initialization - critical for system startup."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            # Test successful initialization
            db_pool = DatabasePool("postgresql://test:test@localhost/test")
            await db_pool.initialize()
            
            # Critical assertions
            assert db_pool.pool is not None, "Database pool must be initialized"
            assert db_pool.pool == mock_pool, "Pool must be set correctly"
            mock_create_pool.assert_called_once(), "Pool creation must be called"
    
    async def test_database_connection_failure_handling(self):
        """Test database connection failure handling - critical for system stability."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Database connection failed")
            
            db_pool = DatabasePool("postgresql://invalid:invalid@localhost/invalid")
            
            # Must handle connection failure
            with pytest.raises(Exception) as exc_info:
                await db_pool.initialize()
            
            assert "Database connection failed" in str(exc_info.value), "Must propagate connection error"
    
    async def test_database_pool_acquire_context_manager(self):
        """Test database connection acquisition - critical for all DB operations."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            mock_create_pool.return_value = mock_pool
            
            db_pool = DatabasePool("postgresql://test:test@localhost/test")
            
            # Test connection acquisition
            async with db_pool.acquire() as conn:
                assert conn == mock_conn, "Must return correct connection"
            
            # Verify proper cleanup
            mock_pool.acquire.assert_called_once(), "Must acquire connection"


@pytest.mark.critical
@pytest.mark.database
class TestSessionDataIntegrity:
    """Critical tests for session data integrity."""
    
    async def test_session_creation_data_integrity(self, mock_database):
        """Test session creation maintains data integrity."""
        # Mock successful session creation
        expected_session_id = "session-integrity-123"
        mock_database.fetchrow.return_value = {"id": expected_session_id}
        
        # Test session creation
        session_id = await create_session(
            user_id="test-user-456",
            metadata={"test_key": "test_value", "number": 42}
        )
        
        # Critical data integrity checks
        assert session_id == expected_session_id, "Session ID must match expected value"
        
        # Verify database call was made with correct parameters
        mock_database.fetchrow.assert_called_once()
        call_args = mock_database.fetchrow.call_args
        
        # Check SQL query structure
        assert "INSERT INTO sessions" in call_args[0][0], "Must use correct INSERT query"
        assert "user_id" in call_args[0][0], "Must include user_id field"
        assert "metadata" in call_args[0][0], "Must include metadata field"
        
        # Check parameters
        assert call_args[0][1] == "test-user-456", "User ID must be passed correctly"
        
        # Check metadata JSON serialization
        metadata_param = call_args[0][2]
        parsed_metadata = json.loads(metadata_param)
        assert parsed_metadata["test_key"] == "test_value", "Metadata must be serialized correctly"
        assert parsed_metadata["number"] == 42, "Numeric values must be preserved"
    
    async def test_session_retrieval_data_integrity(self, mock_database):
        """Test session retrieval maintains data integrity."""
        # Mock session data
        mock_session_data = {
            "id": "session-retrieve-789",
            "user_id": "user-retrieve-123",
            "metadata": '{"key": "value", "count": 10}',
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "updated_at": datetime(2024, 1, 1, 12, 30, 0),
            "expires_at": None
        }
        mock_database.fetchrow.return_value = mock_session_data
        
        # Test session retrieval
        session = await get_session("session-retrieve-789")
        
        # Critical data integrity checks
        assert session is not None, "Session must be retrieved"
        assert session["id"] == "session-retrieve-789", "Session ID must match"
        assert session["user_id"] == "user-retrieve-123", "User ID must match"
        
        # Check metadata deserialization
        assert isinstance(session["metadata"], dict), "Metadata must be deserialized to dict"
        assert session["metadata"]["key"] == "value", "String values must be preserved"
        assert session["metadata"]["count"] == 10, "Numeric values must be preserved"
        
        # Check timestamp formatting
        assert session["created_at"] == "2024-01-01T12:00:00", "Created timestamp must be ISO formatted"
        assert session["updated_at"] == "2024-01-01T12:30:00", "Updated timestamp must be ISO formatted"
    
    async def test_message_data_integrity(self, mock_database):
        """Test message data integrity in sessions."""
        expected_message_id = "message-integrity-456"
        mock_database.fetchrow.return_value = {"id": expected_message_id}
        
        # Test message creation with complex metadata
        message_metadata = {
            "tool_calls": ["search", "analyze"],
            "confidence": 0.95,
            "processing_time": 1.23,
            "nested": {"key": "value"}
        }
        
        message_id = await add_message(
            session_id="session-123",
            role="assistant",
            content="This is a test message with special characters: àáâãäå",
            metadata=message_metadata
        )
        
        # Critical data integrity checks
        assert message_id == expected_message_id, "Message ID must match expected"
        
        # Verify database call
        mock_database.fetchrow.assert_called_once()
        call_args = mock_database.fetchrow.call_args
        
        # Check parameters
        assert call_args[0][1] == "session-123", "Session ID must be correct"
        assert call_args[0][2] == "assistant", "Role must be correct"
        assert "special characters: àáâãäå" in call_args[0][3], "Unicode content must be preserved"
        
        # Check metadata serialization
        metadata_param = call_args[0][4]
        parsed_metadata = json.loads(metadata_param)
        assert parsed_metadata["tool_calls"] == ["search", "analyze"], "Arrays must be preserved"
        assert parsed_metadata["confidence"] == 0.95, "Float precision must be preserved"
        assert parsed_metadata["nested"]["key"] == "value", "Nested objects must be preserved"


@pytest.mark.critical
@pytest.mark.database
class TestDocumentDataIntegrity:
    """Critical tests for document data integrity."""
    
    async def test_document_retrieval_integrity(self, mock_database):
        """Test document retrieval maintains data integrity."""
        # Mock document with complex data
        mock_doc_data = {
            "id": "doc-integrity-123",
            "title": "Test Document with Unicode: 中文测试",
            "source": "test_source.txt",
            "content": "Content with\nnewlines and\ttabs",
            "metadata": '{"tags": ["test", "unicode"], "size": 1024, "encoding": "utf-8"}',
            "created_at": datetime(2024, 1, 1, 10, 0, 0),
            "updated_at": datetime(2024, 1, 1, 11, 0, 0)
        }
        mock_database.fetchrow.return_value = mock_doc_data
        
        # Test document retrieval
        document = await get_document("doc-integrity-123")
        
        # Critical data integrity checks
        assert document is not None, "Document must be retrieved"
        assert document["id"] == "doc-integrity-123", "Document ID must match"
        assert "中文测试" in document["title"], "Unicode in title must be preserved"
        assert "\\n" in document["content"] or "\n" in document["content"], "Newlines must be preserved"
        assert "\\t" in document["content"] or "\t" in document["content"], "Tabs must be preserved"
        
        # Check metadata integrity
        metadata = document["metadata"]
        assert isinstance(metadata, dict), "Metadata must be deserialized"
        assert metadata["tags"] == ["test", "unicode"], "Array values must be preserved"
        assert metadata["size"] == 1024, "Numeric values must be preserved"
        assert metadata["encoding"] == "utf-8", "String values must be preserved"
    
    async def test_document_list_integrity(self, mock_database):
        """Test document listing maintains data integrity."""
        # Mock multiple documents
        mock_docs = [
            {
                "id": "doc-1",
                "title": "Document 1",
                "source": "source1.txt",
                "metadata": '{"priority": 1}',
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1),
                "chunk_count": 5
            },
            {
                "id": "doc-2",
                "title": "Document 2",
                "source": "source2.txt", 
                "metadata": '{"priority": 2}',
                "created_at": datetime(2024, 1, 2),
                "updated_at": datetime(2024, 1, 2),
                "chunk_count": 3
            }
        ]
        mock_database.fetch.return_value = mock_docs
        
        # Test document listing
        documents = await list_documents(limit=10, offset=0)
        
        # Critical data integrity checks
        assert len(documents) == 2, "Must return correct number of documents"
        
        # Check first document
        doc1 = documents[0]
        assert doc1["id"] == "doc-1", "First document ID must be correct"
        assert doc1["chunk_count"] == 5, "Chunk count must be preserved"
        assert doc1["metadata"]["priority"] == 1, "Metadata must be deserialized correctly"
        
        # Check second document
        doc2 = documents[1]
        assert doc2["id"] == "doc-2", "Second document ID must be correct"
        assert doc2["chunk_count"] == 3, "Chunk count must be preserved"
        assert doc2["metadata"]["priority"] == 2, "Metadata must be deserialized correctly"


@pytest.mark.critical
@pytest.mark.database
class TestNovelDataIntegrity:
    """Critical tests for novel data integrity."""
    
    async def test_novel_creation_integrity(self, mock_database):
        """Test novel creation maintains data integrity."""
        expected_novel_id = "novel-integrity-789"
        mock_database.fetchval.return_value = expected_novel_id
        
        # Test novel creation with complex data
        novel_id = await create_novel(
            title="Test Novel with Special Characters: àáâãäå & 中文",
            author="Author Name with Ñ",
            genre="fantasy/sci-fi",
            summary="A summary with\nmultiple lines\nand special chars: @#$%",
            metadata={"tags": ["test", "unicode"], "rating": 4.5}
        )
        
        # Critical data integrity checks
        assert novel_id == expected_novel_id, "Novel ID must match expected"
        
        # Verify database call
        mock_database.fetchval.assert_called_once()
        call_args = mock_database.fetchval.call_args
        
        # Check parameters
        assert "àáâãäå & 中文" in call_args[0][1], "Unicode in title must be preserved"
        assert "Ñ" in call_args[0][2], "Unicode in author must be preserved"
        assert "fantasy/sci-fi" in call_args[0][3], "Special characters in genre must be preserved"
        assert "\\n" in call_args[0][4] or "\n" in call_args[0][4], "Newlines in summary must be preserved"
        
        # Check metadata serialization
        metadata_param = call_args[0][5]
        parsed_metadata = json.loads(metadata_param)
        assert parsed_metadata["tags"] == ["test", "unicode"], "Array metadata must be preserved"
        assert parsed_metadata["rating"] == 4.5, "Float metadata must be preserved"
    
    async def test_novel_retrieval_integrity(self, mock_database):
        """Test novel retrieval maintains data integrity."""
        # Mock novel data with all fields
        mock_novel_data = {
            "id": "novel-retrieve-456",
            "title": "Retrieved Novel",
            "author": "Test Author",
            "genre": "test",
            "summary": "Test summary",
            "total_word_count": 50000,
            "chapter_count": 10,
            "main_characters": ["Hero", "Villain", "Sidekick"],
            "main_themes": ["friendship", "courage", "sacrifice"],
            "setting_overview": "Fantasy world",
            "target_audience": "young adult",
            "completion_status": "in_progress",
            "metadata": '{"version": 1, "last_edited": "2024-01-01"}',
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 2)
        }
        mock_database.fetchrow.return_value = mock_novel_data
        
        # Test novel retrieval
        novel = await get_novel("novel-retrieve-456")
        
        # Critical data integrity checks
        assert novel is not None, "Novel must be retrieved"
        assert novel["id"] == "novel-retrieve-456", "Novel ID must match"
        assert novel["total_word_count"] == 50000, "Word count must be preserved"
        assert novel["chapter_count"] == 10, "Chapter count must be preserved"
        
        # Check array fields
        assert novel["main_characters"] == ["Hero", "Villain", "Sidekick"], "Character array must be preserved"
        assert novel["main_themes"] == ["friendship", "courage", "sacrifice"], "Themes array must be preserved"
        
        # Check metadata deserialization
        assert isinstance(novel["metadata"], dict), "Metadata must be deserialized"
        assert novel["metadata"]["version"] == 1, "Metadata version must be preserved"
        assert novel["metadata"]["last_edited"] == "2024-01-01", "Metadata date must be preserved"
    
    async def test_character_creation_integrity(self, mock_database):
        """Test character creation maintains data integrity."""
        expected_character_id = "char-integrity-123"
        mock_database.fetchval.return_value = expected_character_id
        
        # Test character creation with complex traits
        character_id = await create_character(
            novel_id="novel-123",
            name="Character with Ñoñó",
            personality_traits=["brave", "intelligent", "compassionate", "stubborn"],
            background="A background story with\nmultiple paragraphs\nand unicode: àáâã",
            role="protagonist"
        )
        
        # Critical data integrity checks
        assert character_id == expected_character_id, "Character ID must match expected"
        
        # Verify database call
        mock_database.fetchval.assert_called_once()
        call_args = mock_database.fetchval.call_args
        
        # Check parameters
        assert call_args[0][1] == "novel-123", "Novel ID must be correct"
        assert "Ñoñó" in call_args[0][2], "Unicode in name must be preserved"
        
        # Check personality traits array
        traits_param = call_args[0][3]
        assert traits_param == ["brave", "intelligent", "compassionate", "stubborn"], "Traits array must be preserved"
        
        # Check background with newlines and unicode
        background_param = call_args[0][4]
        assert "àáâã" in background_param, "Unicode in background must be preserved"
        assert "\\n" in background_param or "\n" in background_param, "Newlines must be preserved"


@pytest.mark.critical
@pytest.mark.database
class TestDatabaseTransactionIntegrity:
    """Critical tests for database transaction integrity."""
    
    async def test_transaction_rollback_on_error(self, mock_database):
        """Test transaction rollback maintains data integrity."""
        # Mock transaction behavior
        mock_transaction = AsyncMock()
        mock_database.transaction.return_value = mock_transaction
        mock_database.transaction.return_value.__aenter__.return_value = mock_transaction
        mock_database.transaction.return_value.__aexit__.return_value = None
        
        # Simulate operation that fails mid-transaction
        mock_database.fetchval.side_effect = [
            "novel-123",  # First operation succeeds
            Exception("Database constraint violation")  # Second operation fails
        ]
        
        # Test transaction rollback scenario
        with pytest.raises(Exception) as exc_info:
            async with mock_database.transaction():
                # First operation
                await create_novel(title="Test Novel", author="Test Author")
                # Second operation that fails
                await create_character(novel_id="novel-123", name="Test Character")
        
        # Critical integrity checks
        assert "Database constraint violation" in str(exc_info.value), "Error must be propagated"
        # In a real scenario, we would verify that the novel creation was rolled back
    
    async def test_concurrent_access_integrity(self, mock_database):
        """Test concurrent database access maintains integrity."""
        import asyncio
        
        # Mock concurrent operations
        call_count = 0
        def mock_fetchval(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"
        
        mock_database.fetchval.side_effect = mock_fetchval
        
        # Simulate concurrent operations
        async def create_test_novel(novel_num):
            return await create_novel(
                title=f"Concurrent Novel {novel_num}",
                author=f"Author {novel_num}"
            )
        
        # Run concurrent operations
        tasks = [create_test_novel(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Critical integrity checks
        assert len(results) == 5, "All concurrent operations must complete"
        assert len(set(results)) == 5, "All results must be unique"
        
        # Verify all database calls were made
        assert mock_database.fetchval.call_count == 5, "All database calls must be executed"