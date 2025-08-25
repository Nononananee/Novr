"""
Tests for Enhanced Context Builder
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from agent.enhanced_context_builder import (
    EnhancedContextBuilder,
    ContextBuildRequest,
    ContextType,
    ContextPriority,
    ContextElement,
    ContextBuildResult,
    create_enhanced_context_builder
)
from agent.models import EmotionalTone


@pytest.mark.unit
class TestContextElement:
    """Test ContextElement dataclass."""
    
    def test_context_element_creation(self):
        """Test creating a context element."""
        element = ContextElement(
            content="Test content",
            source="test_source",
            priority=ContextPriority.HIGH,
            relevance_score=0.8,
            context_type=ContextType.NARRATIVE_CONTINUATION
        )
        
        assert element.content == "Test content"
        assert element.source == "test_source"
        assert element.priority == ContextPriority.HIGH
        assert element.relevance_score == 0.8
        assert element.context_type == ContextType.NARRATIVE_CONTINUATION
        assert isinstance(element.timestamp, datetime)


@pytest.mark.unit
class TestContextBuildRequest:
    """Test ContextBuildRequest dataclass."""
    
    def test_context_build_request_creation(self):
        """Test creating a context build request."""
        request = ContextBuildRequest(
            query="Test query",
            context_type=ContextType.CHARACTER_DIALOGUE,
            max_tokens=2000,
            target_characters=["Alice", "Bob"]
        )
        
        assert request.query == "Test query"
        assert request.context_type == ContextType.CHARACTER_DIALOGUE
        assert request.max_tokens == 2000
        assert request.target_characters == ["Alice", "Bob"]
        assert request.include_character_info is True  # Default value


@pytest.mark.unit
class TestEnhancedContextBuilder:
    """Test EnhancedContextBuilder class."""
    
    @pytest.fixture
    def context_builder(self):
        """Create a context builder for testing."""
        return EnhancedContextBuilder(max_context_tokens=4000)
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample context build request."""
        return ContextBuildRequest(
            query="Generate dialogue between Alice and Bob",
            context_type=ContextType.CHARACTER_DIALOGUE,
            max_tokens=1000,
            target_characters=["Alice", "Bob"],
            emotional_tone=EmotionalTone.JOYFUL
        )
    
    def test_context_builder_initialization(self, context_builder):
        """Test context builder initialization."""
        assert context_builder.max_context_tokens == 4000
        assert context_builder.logger is not None
    
    @pytest.mark.asyncio
    async def test_build_context_basic(self, context_builder, sample_request):
        """Test basic context building."""
        result = await context_builder.build_context(sample_request)
        
        assert isinstance(result, ContextBuildResult)
        assert isinstance(result.elements, list)
        assert isinstance(result.total_tokens, int)
        assert isinstance(result.context_summary, str)
        assert isinstance(result.character_profiles, dict)
        assert isinstance(result.plot_threads, list)
        assert isinstance(result.world_elements, dict)
        assert isinstance(result.consistency_notes, list)
        assert isinstance(result.generation_hints, list)
    
    @pytest.mark.asyncio
    async def test_build_context_with_mocked_search(self, context_builder, sample_request):
        """Test context building with mocked search results."""
        # Mock vector search results
        mock_vector_results = [
            type('MockResult', (), {
                'content': 'Alice smiled warmly at Bob.',
                'score': 0.9,
                'source': 'chapter1.txt',
                'metadata': {
                    'characters': ['Alice', 'Bob'],
                    'emotional_tone': 'joy',
                    'chunk_type': 'dialogue'
                }
            })()
        ]
        
        with patch('agent.enhanced_context_builder.vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_vector_results
            
            result = await context_builder.build_context(sample_request)
            
            assert len(result.elements) > 0
            assert result.elements[0].content == 'Alice smiled warmly at Bob.'
            assert result.elements[0].relevance_score == 0.9
            assert result.elements[0].priority == ContextPriority.CRITICAL  # High score should give critical priority
    
    def test_determine_priority(self, context_builder):
        """Test priority determination based on score."""
        assert context_builder._determine_priority(0.95) == ContextPriority.CRITICAL
        assert context_builder._determine_priority(0.85) == ContextPriority.HIGH
        assert context_builder._determine_priority(0.75) == ContextPriority.MEDIUM
        assert context_builder._determine_priority(0.65) == ContextPriority.LOW
    
    def test_prioritize_elements(self, context_builder):
        """Test element prioritization and filtering."""
        elements = [
            ContextElement(
                content="Low priority content",
                source="test1",
                priority=ContextPriority.LOW,
                relevance_score=0.6,
                context_type=ContextType.NARRATIVE_CONTINUATION
            ),
            ContextElement(
                content="High priority content",
                source="test2",
                priority=ContextPriority.HIGH,
                relevance_score=0.9,
                context_type=ContextType.NARRATIVE_CONTINUATION
            ),
            ContextElement(
                content="Critical priority content",
                source="test3",
                priority=ContextPriority.CRITICAL,
                relevance_score=0.95,
                context_type=ContextType.NARRATIVE_CONTINUATION
            )
        ]
        
        filtered = context_builder._prioritize_elements(elements, max_tokens=100)
        
        # Should be sorted by priority and relevance
        assert len(filtered) > 0
        assert filtered[0].priority == ContextPriority.CRITICAL
        assert filtered[0].relevance_score == 0.95
    
    @pytest.mark.asyncio
    async def test_get_vector_context(self, context_builder, sample_request):
        """Test vector context retrieval."""
        # This will use the mock function since real dependencies aren't available
        elements = await context_builder._get_vector_context(sample_request)
        
        # Should return empty list from mock
        assert isinstance(elements, list)
    
    @pytest.mark.asyncio
    async def test_get_character_context(self, context_builder, sample_request):
        """Test character context retrieval."""
        elements = await context_builder._get_character_context(sample_request)
        
        assert isinstance(elements, list)
        # With mock functions, should return empty list
    
    def test_create_context_summary(self, context_builder, sample_request):
        """Test context summary creation."""
        elements = [
            ContextElement(
                content="Test content",
                source="test",
                priority=ContextPriority.HIGH,
                relevance_score=0.8,
                context_type=ContextType.CHARACTER_DIALOGUE
            )
        ]
        
        summary = context_builder._create_context_summary(elements, sample_request)
        
        assert isinstance(summary, str)
        assert "CHARACTER_DIALOGUE" in summary
        assert "Alice" in summary
        assert "Bob" in summary
        assert "joy" in summary.lower()
    
    def test_generate_consistency_notes(self, context_builder, sample_request):
        """Test consistency notes generation."""
        elements = [
            ContextElement(
                content="Test content",
                source="test",
                priority=ContextPriority.HIGH,
                relevance_score=0.8,
                context_type=ContextType.CHARACTER_DIALOGUE,
                metadata={
                    'characters': ['Alice', 'Bob'],
                    'emotional_tone': 'joy',
                    'locations': ['garden']
                }
            )
        ]
        
        notes = context_builder._generate_consistency_notes(elements, sample_request)
        
        assert isinstance(notes, list)
        assert any('Alice' in note for note in notes)
        assert any('joy' in note for note in notes)
    
    def test_generate_generation_hints(self, context_builder, sample_request):
        """Test generation hints creation."""
        elements = []
        
        hints = context_builder._generate_generation_hints(elements, sample_request)
        
        assert isinstance(hints, list)
        assert any('character voice' in hint.lower() for hint in hints)
        assert any('alice' in hint.lower() for hint in hints)
        assert any('joyful' in hint.lower() for hint in hints)


@pytest.mark.unit
class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_enhanced_context_builder(self):
        """Test creating enhanced context builder."""
        builder = create_enhanced_context_builder(max_context_tokens=5000)
        
        assert isinstance(builder, EnhancedContextBuilder)
        assert builder.max_context_tokens == 5000
    
    def test_create_enhanced_context_builder_default(self):
        """Test creating enhanced context builder with defaults."""
        builder = create_enhanced_context_builder()
        
        assert isinstance(builder, EnhancedContextBuilder)
        assert builder.max_context_tokens == 8000  # Default value


@pytest.mark.integration
class TestContextBuilderIntegration:
    """Integration tests for context builder."""
    
    @pytest.mark.asyncio
    async def test_full_context_building_workflow(self):
        """Test complete context building workflow."""
        builder = create_enhanced_context_builder()
        
        request = ContextBuildRequest(
            query="Write a scene where the protagonist faces their greatest fear",
            context_type=ContextType.SCENE_DESCRIPTION,
            max_tokens=2000,
            include_character_info=True,
            include_plot_context=True,
            include_world_building=True,
            emotional_tone=EmotionalTone.FEARFUL,
            scene_location="dark forest"
        )
        
        result = await builder.build_context(request)
        
        # Verify result structure
        assert isinstance(result, ContextBuildResult)
        assert result.total_tokens >= 0
        assert len(result.context_summary) > 0
        
        # Verify hints are appropriate for scene description
        hints = result.generation_hints
        assert any('sensory details' in hint.lower() for hint in hints)
        assert any('fear' in hint.lower() for hint in hints)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in context building."""
        builder = create_enhanced_context_builder()
        
        # Create a request that might cause issues
        request = ContextBuildRequest(
            query="",  # Empty query
            context_type=ContextType.NARRATIVE_CONTINUATION,
            max_tokens=-1  # Invalid token count
        )
        
        # Should not raise exception, but return error in result
        result = await builder.build_context(request)
        
        assert isinstance(result, ContextBuildResult)
        # Should handle gracefully even with invalid input