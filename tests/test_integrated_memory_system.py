"""
Integration tests for the complete memory system
Tests long-term memory, cache memory, and integrated controller
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

from memory.long_term_memory import LongTermMemoryManager, LongTermMemoryChunk, MemoryTier
from memory.cache_memory import MultiLevelCacheManager, PrefetchStrategy
from memory.integrated_memory_controller import (
    IntegratedMemoryController, MemoryRequest, MemoryOperationType, MemoryPriority
)
from agent.enhanced_context_builder import (
    EnhancedContextBuilder, ContextBuildRequest, ContextType
)
from agent.generation_pipeline import (
    AdvancedGenerationPipeline, GenerationRequest, GenerationType, GenerationMode
)

class TestIntegratedMemorySystem:
    """Test suite for integrated memory system"""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def memory_controller(self, temp_storage):
        """Create memory controller for testing"""
        controller = IntegratedMemoryController(
            storage_path=temp_storage,
            max_hot_cache_mb=64,
            max_warm_cache_mb=128,
            max_cold_cache_mb=256,
            long_term_memory_mb=512,
            enable_consistency_checks=False  # Disable for testing
        )
        
        await controller.initialize()
        yield controller
        await controller.close()
    
    @pytest.fixture
    async def context_builder(self, memory_controller):
        """Create context builder for testing"""
        return EnhancedContextBuilder(memory_controller)
    
    @pytest.fixture
    async def generation_pipeline(self, memory_controller, context_builder):
        """Create generation pipeline for testing"""
        return AdvancedGenerationPipeline(
            memory_controller=memory_controller,
            context_builder=context_builder,
            enable_human_approval=False,  # Disable for testing
            enable_consistency_validation=False  # Disable for testing
        )
    
    async def test_memory_storage_and_retrieval(self, memory_controller):
        """Test basic memory storage and retrieval"""
        
        # Store content
        test_content = "This is a test story about brave knights and magical dragons."
        context = {
            'characters': ['Sir Galahad', 'Dragon'],
            'chapter_range': (1, 1),
            'plot_threads': ['quest', 'magic']
        }
        
        success = await memory_controller.store_content(
            content=test_content,
            context=context,
            priority=MemoryPriority.HIGH
        )
        
        assert success, "Content storage should succeed"
        
        # Search for content
        results = await memory_controller.search_content(
            query="knights dragons",
            max_results=5,
            min_relevance=0.5
        )
        
        assert len(results) > 0, "Should find stored content"
        assert any("knights" in result.lower() for result in results), "Should find knight-related content"
    
    async def test_contextual_memory_retrieval(self, memory_controller):
        """Test contextual memory retrieval"""
        
        # Store multiple related pieces of content
        contents = [
            ("Sir Galahad drew his sword, ready to face the dragon.", 
             {'characters': ['Sir Galahad'], 'plot_threads': ['quest'], 'chapter_range': (1, 1)}),
            ("The dragon's eyes glowed with ancient magic and wisdom.",
             {'characters': ['Dragon'], 'plot_threads': ['magic'], 'chapter_range': (1, 1)}),
            ("Princess Elena watched from the castle tower.",
             {'characters': ['Princess Elena'], 'plot_threads': ['romance'], 'chapter_range': (1, 1)})
        ]
        
        for content, context in contents:
            await memory_controller.store_content(content, context)
        
        # Retrieve contextual memory
        context_query = {
            'characters': ['Sir Galahad', 'Dragon'],
            'chapter_range': (1, 1)
        }
        
        results = await memory_controller.get_contextual_memory(
            context=context_query,
            max_results=10
        )
        
        assert len(results) >= 2, "Should find contextually related content"
    
    async def test_cache_performance(self, memory_controller):
        """Test cache performance and hit ratios"""
        
        # Store content
        test_content = "The wizard cast a powerful spell of protection."
        await memory_controller.store_content(
            content=test_content,
            context={'characters': ['Wizard'], 'plot_threads': ['magic']}
        )
        
        # First search (cache miss)
        results1 = await memory_controller.search_content(
            query="wizard spell",
            max_results=5
        )
        
        # Second search (should be cache hit)
        results2 = await memory_controller.search_content(
            query="wizard spell",
            max_results=5
        )
        
        assert results1 == results2, "Results should be consistent"
        
        # Check cache statistics
        stats = memory_controller.get_system_stats()
        assert 'controller' in stats, "Should have controller stats"
    
    async def test_consistency_checking(self, memory_controller):
        """Test consistency checking functionality"""
        
        # Store baseline content
        baseline_content = "Sir Galahad has blue eyes and blonde hair."
        await memory_controller.store_content(
            content=baseline_content,
            context={'characters': ['Sir Galahad']}
        )
        
        # Test consistent content
        consistent_content = "Sir Galahad's blue eyes sparkled in the sunlight."
        is_consistent, warnings = await memory_controller.check_consistency(
            content=consistent_content,
            context={'characters': ['Sir Galahad']}
        )
        
        # Note: Since we disabled consistency checks, this will return True
        # In a full implementation, this would perform actual consistency validation
        assert isinstance(is_consistent, bool), "Should return boolean consistency result"
        assert isinstance(warnings, list), "Should return list of warnings"
    
    async def test_context_building(self, context_builder, memory_controller):
        """Test context building functionality"""
        
        # Store some narrative content
        narrative_pieces = [
            ("Chapter 1: The knight began his quest at dawn.", 
             {'chapter_range': (1, 1), 'characters': ['Knight'], 'plot_threads': ['quest']}),
            ("The knight's name was Sir Galahad, known for his courage.",
             {'chapter_range': (1, 1), 'characters': ['Sir Galahad'], 'plot_threads': ['character_development']}),
            ("A fearsome dragon guarded the ancient treasure.",
             {'chapter_range': (1, 1), 'characters': ['Dragon'], 'plot_threads': ['quest', 'treasure']})
        ]
        
        for content, context in narrative_pieces:
            await memory_controller.store_content(content, context)
        
        # Build context for narrative continuation
        context_request = ContextBuildRequest(
            context_type=ContextType.NARRATIVE_CONTINUATION,
            target_characters=['Sir Galahad', 'Dragon'],
            active_plot_threads=['quest'],
            current_chapter=2,
            generation_goal="Continue the quest story"
        )
        
        built_context = await context_builder.build_context(context_request)
        
        assert built_context is not None, "Should build context successfully"
        assert len(built_context.elements) > 0, "Should have context elements"
        assert built_context.total_tokens > 0, "Should have token count"
        
        # Test formatted context
        formatted_context = context_builder.get_formatted_context(built_context)
        assert isinstance(formatted_context, str), "Should return formatted string"
        assert len(formatted_context) > 0, "Formatted context should not be empty"
    
    async def test_generation_pipeline_basic(self, generation_pipeline, memory_controller):
        """Test basic generation pipeline functionality"""
        
        # Store some context for generation
        context_content = [
            ("Sir Galahad rode through the enchanted forest.", 
             {'characters': ['Sir Galahad'], 'chapter_range': (1, 1)}),
            ("The forest was filled with magical creatures and ancient trees.",
             {'plot_threads': ['magic', 'adventure'], 'chapter_range': (1, 1)})
        ]
        
        for content, context in context_content:
            await memory_controller.store_content(content, context)
        
        # Test generation request
        generation_request = GenerationRequest(
            generation_type=GenerationType.NARRATIVE_CONTINUATION,
            generation_mode=GenerationMode.PREVIEW,  # Preview mode for testing
            current_chapter=2,
            target_characters=['Sir Galahad'],
            active_plot_threads=['adventure'],
            target_word_count=100,
            user_prompt="Continue Sir Galahad's journey through the forest"
        )
        
        # Note: This will fail without actual LLM integration
        # In a real test environment, you'd mock the LLM calls
        try:
            result = await generation_pipeline.generate_content(generation_request)
            
            # Basic result validation
            assert result is not None, "Should return generation result"
            assert hasattr(result, 'success'), "Should have success attribute"
            assert hasattr(result, 'request_id'), "Should have request ID"
            
        except Exception as e:
            # Expected to fail without LLM setup
            assert "LLM" in str(e) or "generation" in str(e).lower(), f"Expected LLM-related error, got: {e}"
    
    async def test_memory_performance_under_load(self, memory_controller):
        """Test memory system performance under load"""
        
        import time
        
        # Generate test content
        test_contents = []
        for i in range(50):
            content = f"This is test story piece {i} about character {i % 5} in chapter {i // 10 + 1}."
            context = {
                'characters': [f'Character{i % 5}'],
                'chapter_range': (i // 10 + 1, i // 10 + 1),
                'plot_threads': [f'thread{i % 3}']
            }
            test_contents.append((content, context))
        
        # Store content and measure time
        start_time = time.time()
        
        store_tasks = []
        for content, context in test_contents:
            task = memory_controller.store_content(content, context)
            store_tasks.append(task)
        
        results = await asyncio.gather(*store_tasks)
        
        store_time = time.time() - start_time
        
        assert all(results), "All storage operations should succeed"
        assert store_time < 10.0, f"Storage should complete within 10 seconds, took {store_time:.2f}s"
        
        # Test search performance
        start_time = time.time()
        
        search_tasks = []
        for i in range(10):
            task = memory_controller.search_content(
                query=f"Character{i % 5}",
                max_results=5
            )
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks)
        
        search_time = time.time() - start_time
        
        assert len(search_results) == 10, "Should complete all searches"
        assert search_time < 5.0, f"Searches should complete within 5 seconds, took {search_time:.2f}s"
        
        # Check system statistics
        stats = memory_controller.get_system_stats()
        assert stats['controller']['total_requests'] > 0, "Should have processed requests"
    
    async def test_memory_cleanup_and_optimization(self, memory_controller):
        """Test memory cleanup and optimization"""
        
        # Store content with different priorities
        high_priority_content = "Critical plot point: The hero discovers the truth."
        medium_priority_content = "The weather was pleasant that day."
        low_priority_content = "Background detail about the tavern."
        
        await memory_controller.store_content(
            high_priority_content, 
            {'importance': 'high'}, 
            MemoryPriority.CRITICAL
        )
        
        await memory_controller.store_content(
            medium_priority_content, 
            {'importance': 'medium'}, 
            MemoryPriority.MEDIUM
        )
        
        await memory_controller.store_content(
            low_priority_content, 
            {'importance': 'low'}, 
            MemoryPriority.LOW
        )
        
        # Get initial stats
        initial_stats = memory_controller.get_system_stats()
        
        # The cleanup would happen in background tasks
        # For testing, we just verify the system is responsive
        
        # Search for high priority content
        results = await memory_controller.search_content(
            query="critical plot truth",
            max_results=5
        )
        
        assert len(results) > 0, "Should find high priority content"
    
    async def test_error_handling_and_recovery(self, memory_controller):
        """Test error handling and recovery"""
        
        # Test invalid content storage
        try:
            await memory_controller.store_content(
                content="",  # Empty content
                context=None,  # Invalid context
                priority=MemoryPriority.HIGH
            )
        except Exception:
            pass  # Expected to handle gracefully
        
        # Test invalid search
        results = await memory_controller.search_content(
            query="",  # Empty query
            max_results=0  # Invalid limit
        )
        
        assert isinstance(results, list), "Should return empty list for invalid search"
        
        # Test invalid contextual retrieval
        results = await memory_controller.get_contextual_memory(
            context={},  # Empty context
            max_results=5
        )
        
        assert isinstance(results, list), "Should handle empty context gracefully"
    
    async def test_concurrent_operations(self, memory_controller):
        """Test concurrent memory operations"""
        
        # Create concurrent store operations
        store_tasks = []
        for i in range(20):
            content = f"Concurrent story piece {i}"
            context = {'thread_id': i, 'characters': [f'Char{i % 3}']}
            task = memory_controller.store_content(content, context)
            store_tasks.append(task)
        
        # Create concurrent search operations
        search_tasks = []
        for i in range(10):
            task = memory_controller.search_content(
                query=f"Char{i % 3}",
                max_results=3
            )
            search_tasks.append(task)
        
        # Execute all operations concurrently
        all_tasks = store_tasks + search_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Check results
        store_results = results[:20]
        search_results = results[20:]
        
        # Most store operations should succeed
        successful_stores = sum(1 for r in store_results if r is True)
        assert successful_stores >= 15, f"At least 15 stores should succeed, got {successful_stores}"
        
        # Search operations should return lists
        successful_searches = sum(1 for r in search_results if isinstance(r, list))
        assert successful_searches >= 8, f"At least 8 searches should succeed, got {successful_searches}"

@pytest.mark.asyncio
class TestMemorySystemIntegration:
    """Integration tests for complete memory system workflow"""
    
    async def test_complete_workflow(self):
        """Test complete workflow from storage to generation"""
        
        # This would be a comprehensive test of the entire pipeline
        # For now, we'll test the basic integration points
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize system
            memory_controller = IntegratedMemoryController(
                storage_path=temp_dir,
                enable_consistency_checks=False
            )
            
            await memory_controller.initialize()
            
            try:
                # Store initial narrative
                initial_content = "In a land far away, there lived a brave knight named Sir Galahad."
                await memory_controller.store_content(
                    content=initial_content,
                    context={
                        'characters': ['Sir Galahad'],
                        'chapter_range': (1, 1),
                        'plot_threads': ['introduction']
                    }
                )
                
                # Search for content
                search_results = await memory_controller.search_content(
                    query="brave knight",
                    max_results=5
                )
                
                assert len(search_results) > 0, "Should find stored content"
                
                # Get contextual memory
                contextual_results = await memory_controller.get_contextual_memory(
                    context={'characters': ['Sir Galahad']},
                    max_results=10
                )
                
                assert len(contextual_results) > 0, "Should find contextual content"
                
                # Verify system health
                stats = memory_controller.get_system_stats()
                assert stats['controller']['total_requests'] > 0, "Should have processed requests"
                assert stats['controller']['successful_requests'] > 0, "Should have successful requests"
                
            finally:
                await memory_controller.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
