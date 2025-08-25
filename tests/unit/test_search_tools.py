"""
Comprehensive unit tests for agent.tools.search_tools module.
Tests all search tool functions and input validation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError

from agent.tools.search_tools import (
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool
)


class TestSearchInputModels:
    """Test input model validation."""
    
    def test_vector_search_input_valid(self):
        """Test valid VectorSearchInput creation."""
        input_data = VectorSearchInput(
            query="test query",
            limit=5,
            threshold=0.8
        )
        
        assert input_data.query == "test query"
        assert input_data.limit == 5
        assert input_data.threshold == 0.8
    
    def test_vector_search_input_defaults(self):
        """Test VectorSearchInput with default values."""
        input_data = VectorSearchInput(query="test query")
        
        assert input_data.query == "test query"
        assert input_data.limit == 10  # Default
        assert input_data.threshold == 0.7  # Default
    
    def test_vector_search_input_validation_error(self):
        """Test VectorSearchInput validation errors."""
        with pytest.raises(ValidationError):
            VectorSearchInput()  # Missing required query
    
    def test_graph_search_input_valid(self):
        """Test valid GraphSearchInput creation."""
        input_data = GraphSearchInput(
            entity="Alice",
            relationship_type="knows",
            depth=3
        )
        
        assert input_data.entity == "Alice"
        assert input_data.relationship_type == "knows"
        assert input_data.depth == 3
    
    def test_graph_search_input_defaults(self):
        """Test GraphSearchInput with default values."""
        input_data = GraphSearchInput(entity="Alice")
        
        assert input_data.entity == "Alice"
        assert input_data.relationship_type is None  # Default
        assert input_data.depth == 2  # Default
    
    def test_hybrid_search_input_valid(self):
        """Test valid HybridSearchInput creation."""
        input_data = HybridSearchInput(
            query="test query",
            entity="Alice",
            limit=15,
            vector_weight=0.6
        )
        
        assert input_data.query == "test query"
        assert input_data.entity == "Alice"
        assert input_data.limit == 15
        assert input_data.vector_weight == 0.6
    
    def test_hybrid_search_input_defaults(self):
        """Test HybridSearchInput with default values."""
        input_data = HybridSearchInput(query="test query")
        
        assert input_data.query == "test query"
        assert input_data.entity is None  # Default
        assert input_data.limit == 10  # Default
        assert input_data.vector_weight == 0.7  # Default
    
    def test_document_input_valid(self):
        """Test valid DocumentInput creation."""
        input_data = DocumentInput(document_id="doc_123")
        
        assert input_data.document_id == "doc_123"
    
    def test_document_list_input_valid(self):
        """Test valid DocumentListInput creation."""
        input_data = DocumentListInput(limit=50, offset=10)
        
        assert input_data.limit == 50
        assert input_data.offset == 10
    
    def test_document_list_input_defaults(self):
        """Test DocumentListInput with default values."""
        input_data = DocumentListInput()
        
        assert input_data.limit == 20  # Default
        assert input_data.offset == 0  # Default
    
    def test_entity_relationship_input_valid(self):
        """Test valid EntityRelationshipInput creation."""
        input_data = EntityRelationshipInput(
            entity="Alice",
            relationship_types=["knows", "friend_of"]
        )
        
        assert input_data.entity == "Alice"
        assert input_data.relationship_types == ["knows", "friend_of"]
    
    def test_entity_relationship_input_defaults(self):
        """Test EntityRelationshipInput with default values."""
        input_data = EntityRelationshipInput(entity="Alice")
        
        assert input_data.entity == "Alice"
        assert input_data.relationship_types is None  # Default
    
    def test_entity_timeline_input_valid(self):
        """Test valid EntityTimelineInput creation."""
        input_data = EntityTimelineInput(
            entity="Alice",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        assert input_data.entity == "Alice"
        assert input_data.start_date == "2024-01-01"
        assert input_data.end_date == "2024-12-31"
    
    def test_entity_timeline_input_defaults(self):
        """Test EntityTimelineInput with default values."""
        input_data = EntityTimelineInput(entity="Alice")
        
        assert input_data.entity == "Alice"
        assert input_data.start_date is None  # Default
        assert input_data.end_date is None  # Default


class TestVectorSearchTool:
    """Test vector search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_vector_search_tool_basic(self):
        """Test basic vector search tool functionality."""
        ctx = Mock()
        input_data = VectorSearchInput(query="test query", limit=5, threshold=0.8)
        
        result = await vector_search_tool(ctx, input_data)
        
        assert result["query"] == "test query"
        assert result["search_type"] == "vector"
        assert "results" in result
        assert "total_results" in result
        assert len(result["results"]) == 1  # Mock returns 1 result
        assert result["results"][0]["score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_vector_search_tool_result_structure(self):
        """Test vector search tool result structure."""
        ctx = Mock()
        input_data = VectorSearchInput(query="character development")
        
        result = await vector_search_tool(ctx, input_data)
        
        # Check result structure
        assert "query" in result
        assert "results" in result
        assert "total_results" in result
        assert "search_type" in result
        
        # Check first result structure
        first_result = result["results"][0]
        assert "id" in first_result
        assert "content" in first_result
        assert "score" in first_result
        assert "metadata" in first_result
    
    @pytest.mark.asyncio
    async def test_vector_search_tool_query_in_content(self):
        """Test that query appears in mock result content."""
        ctx = Mock()
        input_data = VectorSearchInput(query="magical forest")
        
        result = await vector_search_tool(ctx, input_data)
        
        # Mock implementation includes query in content
        assert "magical forest" in result["results"][0]["content"]
    
    @pytest.mark.asyncio
    async def test_vector_search_tool_exception_handling(self):
        """Test vector search tool exception handling."""
        ctx = Mock()
        input_data = VectorSearchInput(query="test query")
        
        # Simulate an exception during processing
        with patch('agent.tools.search_tools.logger') as mock_logger:
            # The current implementation doesn't have external dependencies that can fail,
            # but we can test the exception handling structure
            result = await vector_search_tool(ctx, input_data)
            
            # Should return successful result in current implementation
            assert "error" not in result
            assert "results" in result


class TestGraphSearchTool:
    """Test graph search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_graph_search_tool_basic(self):
        """Test basic graph search tool functionality."""
        ctx = Mock()
        input_data = GraphSearchInput(entity="Alice", relationship_type="knows", depth=2)
        
        result = await graph_search_tool(ctx, input_data)
        
        assert result["entity"] == "Alice"
        assert result["search_type"] == "graph"
        assert result["depth"] == 2
        assert "relationships" in result
        assert len(result["relationships"]) == 1  # Mock returns 1 relationship
    
    @pytest.mark.asyncio
    async def test_graph_search_tool_default_relationship(self):
        """Test graph search tool with default relationship type."""
        ctx = Mock()
        input_data = GraphSearchInput(entity="Bob")  # No relationship_type specified
        
        result = await graph_search_tool(ctx, input_data)
        
        assert result["entity"] == "Bob"
        # Should use default relationship type
        assert result["relationships"][0]["relationship"] == "related_to"
    
    @pytest.mark.asyncio
    async def test_graph_search_tool_relationship_structure(self):
        """Test graph search tool relationship structure."""
        ctx = Mock()
        input_data = GraphSearchInput(entity="Alice", relationship_type="friend_of")
        
        result = await graph_search_tool(ctx, input_data)
        
        relationship = result["relationships"][0]
        assert "target" in relationship
        assert "relationship" in relationship
        assert "properties" in relationship
        assert relationship["relationship"] == "friend_of"
        assert "strength" in relationship["properties"]
    
    @pytest.mark.asyncio
    async def test_graph_search_tool_exception_handling(self):
        """Test graph search tool exception handling."""
        ctx = Mock()
        input_data = GraphSearchInput(entity="Alice")
        
        with patch('agent.tools.search_tools.logger') as mock_logger:
            result = await graph_search_tool(ctx, input_data)
            
            # Should return successful result in current implementation
            assert "error" not in result
            assert "relationships" in result


class TestHybridSearchTool:
    """Test hybrid search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_basic(self):
        """Test basic hybrid search tool functionality."""
        ctx = Mock()
        input_data = HybridSearchInput(
            query="adventure story",
            entity="Alice",
            limit=5,
            vector_weight=0.8
        )
        
        result = await hybrid_search_tool(ctx, input_data)
        
        assert result["query"] == "adventure story"
        assert result["entity"] == "Alice"
        assert result["search_type"] == "hybrid"
        assert "vector_results" in result
        assert "graph_results" in result
        assert "combined_score" in result
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_no_entity(self):
        """Test hybrid search tool without entity."""
        ctx = Mock()
        input_data = HybridSearchInput(query="mystery novel")
        
        result = await hybrid_search_tool(ctx, input_data)
        
        assert result["query"] == "mystery novel"
        assert result["entity"] is None
        # Should handle None entity gracefully
        assert result["graph_results"][0]["entity"] == "unknown"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_result_structure(self):
        """Test hybrid search tool result structure."""
        ctx = Mock()
        input_data = HybridSearchInput(query="test", entity="Alice")
        
        result = await hybrid_search_tool(ctx, input_data)
        
        # Check vector results structure
        vector_result = result["vector_results"][0]
        assert "id" in vector_result
        assert "content" in vector_result
        assert "score" in vector_result
        
        # Check graph results structure
        graph_result = result["graph_results"][0]
        assert "entity" in graph_result
        assert "relationships" in graph_result
        
        # Check combined score
        assert isinstance(result["combined_score"], float)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_query_in_vector_result(self):
        """Test that query appears in vector result content."""
        ctx = Mock()
        input_data = HybridSearchInput(query="dragon battle")
        
        result = await hybrid_search_tool(ctx, input_data)
        
        # Mock implementation includes query in vector result content
        assert "dragon battle" in result["vector_results"][0]["content"]


class TestDocumentTools:
    """Test document-related tools."""
    
    @pytest.mark.asyncio
    async def test_get_document_tool_basic(self):
        """Test basic document retrieval."""
        ctx = Mock()
        input_data = DocumentInput(document_id="doc_123")
        
        result = await get_document_tool(ctx, input_data)
        
        assert result["document_id"] == "doc_123"
        assert "content" in result
        assert "metadata" in result
        assert "doc_123" in result["content"]  # Mock includes ID in content
    
    @pytest.mark.asyncio
    async def test_get_document_tool_metadata_structure(self):
        """Test document tool metadata structure."""
        ctx = Mock()
        input_data = DocumentInput(document_id="novel_chapter_1")
        
        result = await get_document_tool(ctx, input_data)
        
        metadata = result["metadata"]
        assert "title" in metadata
        assert "source" in metadata
        assert "created_at" in metadata
        assert metadata["source"] == "mock"
    
    @pytest.mark.asyncio
    async def test_list_documents_tool_basic(self):
        """Test basic document listing."""
        ctx = Mock()
        input_data = DocumentListInput(limit=3, offset=0)
        
        result = await list_documents_tool(ctx, input_data)
        
        assert "documents" in result
        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert result["limit"] == 3
        assert result["offset"] == 0
        assert len(result["documents"]) == 3  # Should respect limit
    
    @pytest.mark.asyncio
    async def test_list_documents_tool_pagination(self):
        """Test document listing pagination."""
        ctx = Mock()
        input_data = DocumentListInput(limit=2, offset=5)
        
        result = await list_documents_tool(ctx, input_data)
        
        # Check pagination
        assert result["limit"] == 2
        assert result["offset"] == 5
        assert len(result["documents"]) == 2
        
        # Check document IDs respect offset
        first_doc = result["documents"][0]
        assert first_doc["id"] == "doc_6"  # offset 5 + 1
    
    @pytest.mark.asyncio
    async def test_list_documents_tool_document_structure(self):
        """Test document listing structure."""
        ctx = Mock()
        input_data = DocumentListInput(limit=1)
        
        result = await list_documents_tool(ctx, input_data)
        
        document = result["documents"][0]
        assert "id" in document
        assert "title" in document
        assert "summary" in document
        assert "created_at" in document
    
    @pytest.mark.asyncio
    async def test_list_documents_tool_limit_constraint(self):
        """Test document listing respects mock limit constraint."""
        ctx = Mock()
        input_data = DocumentListInput(limit=10)  # Request more than mock limit
        
        result = await list_documents_tool(ctx, input_data)
        
        # Mock implementation limits to 5 documents
        assert len(result["documents"]) == 5


class TestEntityTools:
    """Test entity-related tools."""
    
    @pytest.mark.asyncio
    async def test_get_entity_relationships_tool_basic(self):
        """Test basic entity relationships retrieval."""
        ctx = Mock()
        input_data = EntityRelationshipInput(entity="Alice")
        
        result = await get_entity_relationships_tool(ctx, input_data)
        
        assert result["entity"] == "Alice"
        assert "relationships" in result
        assert "relationship_types" in result
        assert len(result["relationships"]) == 2  # Mock returns 2 relationships
    
    @pytest.mark.asyncio
    async def test_get_entity_relationships_tool_with_types(self):
        """Test entity relationships with specific types."""
        ctx = Mock()
        input_data = EntityRelationshipInput(
            entity="Bob",
            relationship_types=["friend_of", "colleague_of"]
        )
        
        result = await get_entity_relationships_tool(ctx, input_data)
        
        assert result["entity"] == "Bob"
        assert result["relationship_types"] == ["friend_of", "colleague_of"]
    
    @pytest.mark.asyncio
    async def test_get_entity_relationships_tool_relationship_structure(self):
        """Test entity relationships structure."""
        ctx = Mock()
        input_data = EntityRelationshipInput(entity="Alice")
        
        result = await get_entity_relationships_tool(ctx, input_data)
        
        relationship = result["relationships"][0]
        assert "target" in relationship
        assert "type" in relationship
        assert "properties" in relationship
        assert "strength" in relationship["properties"]
    
    @pytest.mark.asyncio
    async def test_get_entity_timeline_tool_basic(self):
        """Test basic entity timeline retrieval."""
        ctx = Mock()
        input_data = EntityTimelineInput(entity="Alice")
        
        result = await get_entity_timeline_tool(ctx, input_data)
        
        assert result["entity"] == "Alice"
        assert "timeline" in result
        assert "start_date" in result
        assert "end_date" in result
        assert len(result["timeline"]) == 2  # Mock returns 2 timeline events
    
    @pytest.mark.asyncio
    async def test_get_entity_timeline_tool_with_dates(self):
        """Test entity timeline with date range."""
        ctx = Mock()
        input_data = EntityTimelineInput(
            entity="Bob",
            start_date="2024-01-01",
            end_date="2024-06-01"
        )
        
        result = await get_entity_timeline_tool(ctx, input_data)
        
        assert result["entity"] == "Bob"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-06-01"
    
    @pytest.mark.asyncio
    async def test_get_entity_timeline_tool_event_structure(self):
        """Test entity timeline event structure."""
        ctx = Mock()
        input_data = EntityTimelineInput(entity="Alice")
        
        result = await get_entity_timeline_tool(ctx, input_data)
        
        event = result["timeline"][0]
        assert "date" in event
        assert "event" in event
        assert "context" in event
        # Event should mention the entity
        assert "Alice" in event["event"]


class TestToolErrorHandling:
    """Test error handling across all tools."""
    
    @pytest.mark.asyncio
    async def test_vector_search_tool_with_invalid_context(self):
        """Test vector search tool with various context scenarios."""
        # Test with None context
        input_data = VectorSearchInput(query="test")
        result = await vector_search_tool(None, input_data)
        
        # Should handle gracefully
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_tools_with_empty_strings(self):
        """Test tools with empty string inputs."""
        ctx = Mock()
        
        # Vector search with empty query
        vector_input = VectorSearchInput(query="")
        vector_result = await vector_search_tool(ctx, vector_input)
        assert "results" in vector_result
        
        # Graph search with empty entity
        graph_input = GraphSearchInput(entity="")
        graph_result = await graph_search_tool(ctx, graph_input)
        assert "relationships" in graph_result
        
        # Document with empty ID
        doc_input = DocumentInput(document_id="")
        doc_result = await get_document_tool(ctx, doc_input)
        assert "content" in doc_result
    
    @pytest.mark.asyncio
    async def test_tools_with_extreme_values(self):
        """Test tools with extreme input values."""
        ctx = Mock()
        
        # Very large limit
        list_input = DocumentListInput(limit=1000000, offset=999999)
        list_result = await list_documents_tool(ctx, list_input)
        assert "documents" in list_result
        
        # Very deep graph search
        graph_input = GraphSearchInput(entity="Alice", depth=100)
        graph_result = await graph_search_tool(ctx, graph_input)
        assert graph_result["depth"] == 100
        
        # Very high threshold
        vector_input = VectorSearchInput(query="test", threshold=1.0)
        vector_result = await vector_search_tool(ctx, vector_input)
        assert vector_result["results"]  # Should still return results


class TestToolIntegration:
    """Test tool integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_search_workflow_scenario(self):
        """Test a complete search workflow scenario."""
        ctx = Mock()
        
        # 1. Start with vector search
        vector_input = VectorSearchInput(query="character Alice")
        vector_result = await vector_search_tool(ctx, vector_input)
        
        # 2. Follow up with graph search for Alice
        graph_input = GraphSearchInput(entity="Alice")
        graph_result = await graph_search_tool(ctx, graph_input)
        
        # 3. Get Alice's timeline
        timeline_input = EntityTimelineInput(entity="Alice")
        timeline_result = await get_entity_timeline_tool(ctx, timeline_input)
        
        # 4. Combine with hybrid search
        hybrid_input = HybridSearchInput(query="character Alice", entity="Alice")
        hybrid_result = await hybrid_search_tool(ctx, hybrid_input)
        
        # All searches should be successful
        assert "results" in vector_result
        assert "relationships" in graph_result
        assert "timeline" in timeline_result
        assert "vector_results" in hybrid_result
        assert "graph_results" in hybrid_result
    
    @pytest.mark.asyncio
    async def test_document_discovery_workflow(self):
        """Test document discovery workflow."""
        ctx = Mock()
        
        # 1. List documents
        list_input = DocumentListInput(limit=5)
        list_result = await list_documents_tool(ctx, list_input)
        
        # 2. Get specific document
        if list_result["documents"]:
            doc_id = list_result["documents"][0]["id"]
            doc_input = DocumentInput(document_id=doc_id)
            doc_result = await get_document_tool(ctx, doc_input)
            
            assert doc_result["document_id"] == doc_id
            assert "content" in doc_result
    
    @pytest.mark.asyncio
    async def test_entity_analysis_workflow(self):
        """Test entity analysis workflow."""
        ctx = Mock()
        
        # 1. Get entity relationships
        rel_input = EntityRelationshipInput(entity="Alice")
        rel_result = await get_entity_relationships_tool(ctx, rel_input)
        
        # 2. Get entity timeline
        timeline_input = EntityTimelineInput(entity="Alice")
        timeline_result = await get_entity_timeline_tool(ctx, timeline_input)
        
        # 3. Search for related entities
        if rel_result["relationships"]:
            related_entity = rel_result["relationships"][0]["target"]
            related_timeline_input = EntityTimelineInput(entity=related_entity)
            related_timeline_result = await get_entity_timeline_tool(ctx, related_timeline_input)
            
            assert related_timeline_result["entity"] == related_entity
        
        assert rel_result["entity"] == "Alice"
        assert timeline_result["entity"] == "Alice"