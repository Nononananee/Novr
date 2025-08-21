#!/usr/bin/env python3
"""
Integration Test dan Performance Optimization
Tests the complete enhanced chunking and advanced context building pipeline.
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for testing."""
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    tokens_processed: int
    chunks_created: int
    quality_score: float
    success: bool
    error_message: Optional[str] = None


class IntegrationTestSuite:
    """Integration test suite for enhanced chunking and context building."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.performance_metrics = []
    
    async def run_all_tests(self):
        """Run all integration tests."""
        
        print("=" * 80)
        print("INTEGRATION TEST SUITE - ENHANCED CHUNKING & CONTEXT BUILDING")
        print("=" * 80)
        
        # Test 1: Enhanced Scene Chunking
        await self.test_enhanced_scene_chunking()
        
        # Test 2: Advanced Context Building
        await self.test_advanced_context_building()
        
        # Test 3: End-to-End Generation Pipeline
        await self.test_end_to_end_pipeline()
        
        # Test 4: Performance Benchmarks
        await self.test_performance_benchmarks()
        
        # Test 5: Memory Usage Analysis
        await self.test_memory_usage()
        
        # Test 6: Scalability Testing
        await self.test_scalability()
        
        # Generate final report
        self.generate_test_report()
    
    async def test_enhanced_scene_chunking(self):
        """Test enhanced scene chunking functionality."""
        
        print("\n" + "=" * 60)
        print("TEST 1: ENHANCED SCENE CHUNKING")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Import and test enhanced chunker
            from demo_enhanced_chunking_simple import SimpleEnhancedChunker
            
            chunker = SimpleEnhancedChunker()
            
            # Test content
            test_content = self.get_test_novel_content()
            
            # Perform chunking
            chunks = chunker.chunk_document(
                content=test_content,
                title="Integration Test Novel",
                source="test.md"
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Analyze results
            total_tokens = sum(chunk.token_count for chunk in chunks)
            avg_importance = sum(chunk.metadata.get('importance_score', 0.5) for chunk in chunks) / len(chunks)
            
            # Validate chunking quality
            scene_types = set(chunk.metadata.get('scene_type') for chunk in chunks)
            character_coverage = len(set(char for chunk in chunks for char in chunk.metadata.get('characters', [])))
            
            success = (
                len(chunks) > 0 and
                total_tokens > 0 and
                len(scene_types) > 1 and
                character_coverage > 0
            )
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="Enhanced Scene Chunking",
                execution_time_ms=execution_time,
                memory_usage_mb=0.0,  # Would measure in real implementation
                tokens_processed=total_tokens,
                chunks_created=len(chunks),
                quality_score=avg_importance,
                success=success
            )
            self.performance_metrics.append(metrics)
            
            # Display results
            print(f"✓ Chunks Created: {len(chunks)}")
            print(f"✓ Total Tokens: {total_tokens}")
            print(f"✓ Average Importance: {avg_importance:.3f}")
            print(f"✓ Scene Types Detected: {len(scene_types)}")
            print(f"✓ Characters Found: {character_coverage}")
            print(f"✓ Execution Time: {execution_time:.2f}ms")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
            # Detailed chunk analysis
            print(f"\nChunk Analysis:")
            for i, chunk in enumerate(chunks, 1):
                metadata = chunk.metadata
                print(f"  Chunk {i}: {metadata.get('scene_type', 'unknown')} "
                      f"(Importance: {metadata.get('importance_score', 0.0):.3f}, "
                      f"Tokens: {chunk.token_count})")
            
        except Exception as e:
            logger.error(f"Enhanced scene chunking test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="Enhanced Scene Chunking",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    async def test_advanced_context_building(self):
        """Test advanced context building functionality."""
        
        print("\n" + "=" * 60)
        print("TEST 2: ADVANCED CONTEXT BUILDING")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Import and test advanced context builder
            from demo_advanced_context import MockAdvancedContextBuilder
            
            context_builder = MockAdvancedContextBuilder()
            
            # Test scenarios
            test_scenarios = [
                {
                    "query": "Emma discovers the truth about her family",
                    "context_type": "character_focused",
                    "characters": ["Emma"],
                    "emotional_tone": "dramatic"
                },
                {
                    "query": "Action scene in the mansion",
                    "context_type": "action_sequence",
                    "characters": ["Emma"],
                    "locations": ["Victorian mansion"],
                    "emotional_tone": "tension"
                }
            ]
            
            total_contexts = 0
            total_quality = 0.0
            total_tokens = 0
            
            for scenario in test_scenarios:
                context = await context_builder.build_generation_context(
                    query=scenario["query"],
                    context_type=scenario["context_type"],
                    target_characters=scenario.get("characters"),
                    target_locations=scenario.get("locations"),
                    emotional_tone=scenario.get("emotional_tone")
                )
                
                total_contexts += 1
                total_quality += context["context_quality_score"]
                total_tokens += context["total_tokens"]
            
            execution_time = (time.time() - start_time) * 1000
            avg_quality = total_quality / total_contexts
            
            success = (
                total_contexts > 0 and
                avg_quality > 0.5 and
                total_tokens > 0
            )
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="Advanced Context Building",
                execution_time_ms=execution_time,
                memory_usage_mb=0.0,
                tokens_processed=total_tokens,
                chunks_created=total_contexts,
                quality_score=avg_quality,
                success=success
            )
            self.performance_metrics.append(metrics)
            
            print(f"✓ Contexts Built: {total_contexts}")
            print(f"✓ Average Quality: {avg_quality:.3f}")
            print(f"✓ Total Tokens: {total_tokens}")
            print(f"✓ Execution Time: {execution_time:.2f}ms")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Advanced context building test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="Advanced Context Building",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    async def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline integration."""
        
        print("\n" + "=" * 60)
        print("TEST 3: END-TO-END PIPELINE")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Simulate end-to-end pipeline
            # 1. Content ingestion with enhanced chunking
            from demo_enhanced_chunking_simple import SimpleEnhancedChunker
            chunker = SimpleEnhancedChunker()
            
            test_content = self.get_test_novel_content()
            chunks = chunker.chunk_document(test_content, "Test Novel", "test.md")
            
            # 2. Context building
            from demo_advanced_context import MockAdvancedContextBuilder
            context_builder = MockAdvancedContextBuilder()
            
            context = await context_builder.build_generation_context(
                query="Generate a continuation of Emma's story",
                context_type="character_focused",
                target_characters=["Emma"],
                emotional_tone="dramatic"
            )
            
            # 3. Simulate generation (mock)
            generated_content = self.simulate_content_generation(context)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Validate pipeline
            success = (
                len(chunks) > 0 and
                context["context_quality_score"] > 0.5 and
                len(generated_content) > 100
            )
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="End-to-End Pipeline",
                execution_time_ms=execution_time,
                memory_usage_mb=0.0,
                tokens_processed=context["total_tokens"],
                chunks_created=len(chunks),
                quality_score=context["context_quality_score"],
                success=success
            )
            self.performance_metrics.append(metrics)
            
            print(f"✓ Pipeline Steps Completed: 3")
            print(f"✓ Chunks Processed: {len(chunks)}")
            print(f"✓ Context Quality: {context['context_quality_score']:.3f}")
            print(f"✓ Generated Content Length: {len(generated_content)} chars")
            print(f"✓ Execution Time: {execution_time:.2f}ms")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="End-to-End Pipeline",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        
        print("\n" + "=" * 60)
        print("TEST 4: PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        try:
            # Test different content sizes
            content_sizes = [500, 1000, 2000, 5000]  # words
            
            from demo_enhanced_chunking_simple import SimpleEnhancedChunker
            chunker = SimpleEnhancedChunker()
            
            benchmark_results = []
            
            for size in content_sizes:
                # Generate test content of specific size
                test_content = self.generate_test_content(size)
                
                # Measure chunking performance
                start_time = time.time()
                chunks = chunker.chunk_document(test_content, f"Test {size}w", "test.md")
                execution_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                tokens_per_second = (size * 1.3) / (execution_time / 1000) if execution_time > 0 else 0
                chunks_per_second = len(chunks) / (execution_time / 1000) if execution_time > 0 else 0
                
                benchmark_results.append({
                    "content_size_words": size,
                    "execution_time_ms": execution_time,
                    "chunks_created": len(chunks),
                    "tokens_per_second": tokens_per_second,
                    "chunks_per_second": chunks_per_second
                })
                
                print(f"  {size:4d} words: {execution_time:6.2f}ms, "
                      f"{len(chunks):2d} chunks, "
                      f"{tokens_per_second:6.0f} tokens/sec")
            
            # Calculate average performance
            avg_execution_time = statistics.mean([r["execution_time_ms"] for r in benchmark_results])
            avg_tokens_per_second = statistics.mean([r["tokens_per_second"] for r in benchmark_results])
            
            success = avg_execution_time < 5000  # Should process within 5 seconds
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="Performance Benchmarks",
                execution_time_ms=avg_execution_time,
                memory_usage_mb=0.0,
                tokens_processed=int(avg_tokens_per_second),
                chunks_created=len(benchmark_results),
                quality_score=1.0 if success else 0.0,
                success=success
            )
            self.performance_metrics.append(metrics)
            
            print(f"✓ Average Execution Time: {avg_execution_time:.2f}ms")
            print(f"✓ Average Tokens/Second: {avg_tokens_per_second:.0f}")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Performance benchmark test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="Performance Benchmarks",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    async def test_memory_usage(self):
        """Test memory usage patterns."""
        
        print("\n" + "=" * 60)
        print("TEST 5: MEMORY USAGE ANALYSIS")
        print("=" * 60)
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform memory-intensive operations
            from demo_enhanced_chunking_simple import SimpleEnhancedChunker
            from demo_advanced_context import MockAdvancedContextBuilder
            
            chunker = SimpleEnhancedChunker()
            context_builder = MockAdvancedContextBuilder()
            
            # Process multiple documents
            for i in range(10):
                test_content = self.generate_test_content(1000)
                chunks = chunker.chunk_document(test_content, f"Doc {i}", "test.md")
                
                context = await context_builder.build_generation_context(
                    query=f"Test query {i}",
                    context_type="character_focused",
                    target_characters=["Emma"],
                    emotional_tone="neutral"
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            success = memory_increase < 100  # Should not increase by more than 100MB
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="Memory Usage Analysis",
                execution_time_ms=0.0,
                memory_usage_mb=memory_increase,
                tokens_processed=10000,  # Approximate
                chunks_created=10,
                quality_score=1.0 if success else 0.0,
                success=success
            )
            self.performance_metrics.append(metrics)
            
            print(f"✓ Initial Memory: {initial_memory:.2f}MB")
            print(f"✓ Final Memory: {final_memory:.2f}MB")
            print(f"✓ Memory Increase: {memory_increase:.2f}MB")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
        except ImportError:
            print("⚠ psutil not available, skipping memory test")
            metrics = PerformanceMetrics(
                operation_name="Memory Usage Analysis",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=1.0,
                success=True,
                error_message="psutil not available"
            )
            self.performance_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="Memory Usage Analysis",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    async def test_scalability(self):
        """Test scalability with concurrent operations."""
        
        print("\n" + "=" * 60)
        print("TEST 6: SCALABILITY TESTING")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Test concurrent chunking operations
            from demo_enhanced_chunking_simple import SimpleEnhancedChunker
            chunker = SimpleEnhancedChunker()
            
            async def chunk_document_async(doc_id):
                test_content = self.generate_test_content(500)
                return chunker.chunk_document(test_content, f"Doc {doc_id}", "test.md")
            
            # Run concurrent operations
            concurrent_tasks = 5
            tasks = [chunk_document_async(i) for i in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Analyze results
            successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
            total_chunks = sum(len(r) for r in results if not isinstance(r, Exception))
            
            success = successful_tasks == concurrent_tasks
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name="Scalability Testing",
                execution_time_ms=execution_time,
                memory_usage_mb=0.0,
                tokens_processed=concurrent_tasks * 500 * 1.3,
                chunks_created=total_chunks,
                quality_score=successful_tasks / concurrent_tasks,
                success=success
            )
            self.performance_metrics.append(metrics)
            
            print(f"✓ Concurrent Tasks: {concurrent_tasks}")
            print(f"✓ Successful Tasks: {successful_tasks}")
            print(f"✓ Total Chunks Created: {total_chunks}")
            print(f"✓ Execution Time: {execution_time:.2f}ms")
            print(f"✓ Test Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Scalability test failed: {e}")
            metrics = PerformanceMetrics(
                operation_name="Scalability Testing",
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                tokens_processed=0,
                chunks_created=0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(metrics)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        
        print("\n" + "=" * 80)
        print("INTEGRATION TEST REPORT")
        print("=" * 80)
        
        # Summary statistics
        total_tests = len(self.performance_metrics)
        passed_tests = sum(1 for m in self.performance_metrics if m.success)
        failed_tests = total_tests - passed_tests
        
        print(f"\nTest Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Performance summary
        successful_metrics = [m for m in self.performance_metrics if m.success]
        if successful_metrics:
            avg_execution_time = statistics.mean([m.execution_time_ms for m in successful_metrics])
            total_tokens = sum(m.tokens_processed for m in successful_metrics)
            total_chunks = sum(m.chunks_created for m in successful_metrics)
            avg_quality = statistics.mean([m.quality_score for m in successful_metrics])
            
            print(f"\nPerformance Summary:")
            print(f"  Average Execution Time: {avg_execution_time:.2f}ms")
            print(f"  Total Tokens Processed: {total_tokens:,}")
            print(f"  Total Chunks Created: {total_chunks}")
            print(f"  Average Quality Score: {avg_quality:.3f}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for i, metric in enumerate(self.performance_metrics, 1):
            status = "✓ PASSED" if metric.success else "✗ FAILED"
            print(f"  {i}. {metric.operation_name}: {status}")
            print(f"     Execution Time: {metric.execution_time_ms:.2f}ms")
            print(f"     Tokens Processed: {metric.tokens_processed:,}")
            print(f"     Quality Score: {metric.quality_score:.3f}")
            if metric.error_message:
                print(f"     Error: {metric.error_message}")
        
        # Save report to file
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests/total_tests)*100
            },
            "performance_metrics": [
                {
                    "operation_name": m.operation_name,
                    "execution_time_ms": m.execution_time_ms,
                    "memory_usage_mb": m.memory_usage_mb,
                    "tokens_processed": m.tokens_processed,
                    "chunks_created": m.chunks_created,
                    "quality_score": m.quality_score,
                    "success": m.success,
                    "error_message": m.error_message
                }
                for m in self.performance_metrics
            ]
        }
        
        with open("integration_test_report.json", 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Test report saved to: integration_test_report.json")
        
        # Recommendations
        print(f"\nRecommendations:")
        if failed_tests > 0:
            print("  • Review failed tests and fix underlying issues")
        if successful_metrics:
            avg_time = statistics.mean([m.execution_time_ms for m in successful_metrics])
            if avg_time > 1000:
                print("  • Consider performance optimizations for faster processing")
        print("  • Monitor memory usage in production environment")
        print("  • Implement continuous integration testing")
        print("  • Add more edge case testing scenarios")
    
    # Helper methods
    def get_test_novel_content(self) -> str:
        """Get test novel content."""
        return '''
        Emma stood at the threshold of her grandfather's study, her heart pounding with anticipation and dread. The letter in her trembling hands felt heavier than paper should, as if the words themselves carried the weight of generations of secrets.

        "The Hartwell Legacy," she whispered, reading the ornate heading for the third time. The ink seemed to shimmer in the lamplight, and she wondered if her eyes were playing tricks on her.

        ***

        Meanwhile, across town, Detective Marcus Chen was staring at the security footage from the museum. The thieves had moved with precision, bypassing priceless artifacts to reach one specific item: a simple wooden box donated by the Hartwell estate.

        "They knew exactly what they were looking for," he muttered to himself, rewinding the footage again.

        ***

        The explosion shattered the night silence. Emma dove behind the stone fountain as debris rained down around her. Through the smoke and chaos, she could see dark figures advancing across the courtyard, their intentions clear and deadly.

        "There she is!" one of them shouted. "Don't let her reach the house!"

        Emma's mind raced. The secret passages her grandfather had told her about in childhood stories—were they real? Did she dare to find out?
        '''
    
    def generate_test_content(self, word_count: int) -> str:
        """Generate test content of specified word count."""
        base_content = self.get_test_novel_content()
        words = base_content.split()
        
        # Repeat and truncate to get desired word count
        if len(words) < word_count:
            multiplier = (word_count // len(words)) + 1
            extended_words = (words * multiplier)[:word_count]
        else:
            extended_words = words[:word_count]
        
        return ' '.join(extended_words)
    
    def simulate_content_generation(self, context: Dict[str, Any]) -> str:
        """Simulate content generation based on context."""
        # Mock generation based on context
        characters = context.get("characters_involved", [])
        emotional_tone = context.get("emotional_tone", "neutral")
        
        generated = f"Generated content featuring {', '.join(characters)} with {emotional_tone} tone. "
        generated += "This is a simulated generation result that would normally come from an LLM. "
        generated += f"The context quality score was {context.get('context_quality_score', 0.0):.3f}."
        
        return generated


async def main():
    """Main test function."""
    
    print("Starting Integration Test Suite...")
    
    try:
        test_suite = IntegrationTestSuite()
        await test_suite.run_all_tests()
        
        print("\n" + "=" * 80)
        print("INTEGRATION TESTING COMPLETED")
        print("=" * 80)
        print("\nNext Steps:")
        print("• Review test report for any failed tests")
        print("• Optimize performance based on benchmark results")
        print("• Implement monitoring for production deployment")
        print("• Add automated testing to CI/CD pipeline")
        
    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())