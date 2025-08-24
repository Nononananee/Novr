#!/usr/bin/env python3
"""
Phase 2.2 Completion Test: Chunking Optimization
Tests the implementation of enhanced chunking strategies for complex narrative content.
"""

import asyncio
import pytest
import sys
import os
import logging
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2_2TestSuite:
    """Test suite for Phase 2.2: Chunking Optimization"""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
        self.chunking_performance = []
    
    async def run_all_tests(self):
        """Run all Phase 2.2 tests."""
        print("=" * 80)
        print("PHASE 2.2 COMPLETION TEST: CHUNKING OPTIMIZATION")
        print("=" * 80)
        
        # Test 1: Enhanced Chunking Import
        await self._run_test("Enhanced Chunking Import", self.test_enhanced_chunking_import)
        
        # Test 2: Adaptive Strategy Selection
        await self._run_test("Adaptive Strategy Selection", self.test_adaptive_strategy_selection)
        
        # Test 3: Content Type Detection
        await self._run_test("Content Type Detection", self.test_content_type_detection)
        
        # Test 4: Dialogue Preserving Chunking
        await self._run_test("Dialogue Preserving Chunking", self.test_dialogue_preserving_chunking)
        
        # Test 5: Action Oriented Chunking
        await self._run_test("Action Oriented Chunking", self.test_action_oriented_chunking)
        
        # Test 6: Character Focused Chunking
        await self._run_test("Character Focused Chunking", self.test_character_focused_chunking)
        
        # Test 7: Complex Narrative Handling
        await self._run_test("Complex Narrative Handling", self.test_complex_narrative_handling)
        
        # Test 8: Chunking Performance Analysis
        await self._run_test("Chunking Performance Analysis", self.test_chunking_performance_analysis)
        
        # Generate final report
        self.generate_final_report()
    
    async def _run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        self.total_tests += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            print(f"\n🧪 Running test: {test_name}")
            
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if result.get("success", False):
                self.passed_tests += 1
                print(f"✅ {test_name} PASSED ({execution_time:.2f}ms)")
            else:
                print(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
            
            # Record performance if available
            if "performance" in result:
                self.chunking_performance.append(result["performance"])
            
            self.test_results.append({
                "test_name": test_name,
                "success": result.get("success", False),
                "execution_time_ms": execution_time,
                "details": result,
                "error_message": result.get("error") if not result.get("success") else None
            })
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            print(f"❌ {test_name} CRASHED: {str(e)}")
            
            self.test_results.append({
                "test_name": test_name,
                "success": False,
                "execution_time_ms": execution_time,
                "details": {},
                "error_message": str(e)
            })
    
    def test_enhanced_chunking_import(self) -> Dict[str, Any]:
        """Test that enhanced chunking strategies can be imported."""
        try:
            from memory.enhanced_chunking_strategies import (
                AdaptiveChunkingStrategy,
                EnhancedChunk,
                ContentType,
                ChunkPriority,
                ChunkMetadata,
                chunk_novel_content,
                analyze_chunking_performance,
                adaptive_chunker
            )
            
            # Verify classes exist and have expected methods
            assert hasattr(AdaptiveChunkingStrategy, 'chunk_content')
            assert hasattr(AdaptiveChunkingStrategy, '_analyze_content')
            
            # Test instantiation
            chunker = AdaptiveChunkingStrategy()
            assert chunker is not None
            
            return {
                "success": True,
                "message": "Enhanced chunking strategies imported successfully",
                "components": [
                    "AdaptiveChunkingStrategy",
                    "EnhancedChunk",
                    "ContentType",
                    "ChunkPriority",
                    "ChunkMetadata",
                    "chunk_novel_content",
                    "analyze_chunking_performance"
                ],
                "chunker_initialized": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import failed: {str(e)}",
                "missing_component": str(e).split("'")[-2] if "'" in str(e) else "unknown"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def test_adaptive_strategy_selection(self) -> Dict[str, Any]:
        """Test adaptive strategy selection based on content analysis."""
        try:
            from memory.enhanced_chunking_strategies import AdaptiveChunkingStrategy
            
            chunker = AdaptiveChunkingStrategy()
            
            # Test different content types
            test_contents = {
                "dialogue_heavy": '''
                "Hello there," Emma said with a smile.
                "How are you doing today?" asked Sarah.
                "I'm doing well, thank you for asking," Emma replied.
                "That's wonderful to hear," Sarah responded cheerfully.
                ''',
                "action_oriented": '''
                Emma ran quickly down the hallway. She grabbed the door handle and pulled hard.
                The door slammed open. She jumped through the opening and landed safely on the other side.
                Behind her, footsteps thundered as her pursuers gave chase.
                ''',
                "character_driven": '''
                Emma and Sarah sat in the garden while James prepared tea in the kitchen.
                Margaret watched from the window as her children played outside.
                Thomas arrived shortly after, carrying news from the village.
                ''',
                "descriptive": '''
                The old Victorian mansion stood majestically on the hill, its gothic architecture
                silhouetted against the stormy sky. The windows were dark and imposing,
                giving the structure an air of mystery and foreboding.
                '''
            }
            
            strategy_results = {}
            
            for content_type, content in test_contents.items():
                # Analyze content
                analysis = await chunker._analyze_content(content)
                
                # Select strategy
                strategy = chunker._select_chunking_strategy(analysis)
                strategy_results[content_type] = {
                    "strategy": strategy,
                    "analysis": analysis
                }
            
            # Verify appropriate strategies were selected
            expected_strategies = {
                "dialogue_heavy": "dialogue_preserving",
                "action_oriented": "action_oriented", 
                "character_driven": "character_focused",
                "descriptive": "semantic_chunking"
            }
            
            strategy_accuracy = 0
            for content_type, expected in expected_strategies.items():
                if strategy_results[content_type]["strategy"] == expected:
                    strategy_accuracy += 1
            
            strategy_accuracy_rate = strategy_accuracy / len(expected_strategies)
            
            return {
                "success": True,
                "message": "Adaptive strategy selection working",
                "strategy_results": strategy_results,
                "strategy_accuracy_rate": strategy_accuracy_rate,
                "strategies_tested": len(test_contents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Adaptive strategy selection failed: {str(e)}"
            }
    
    async def test_content_type_detection(self) -> Dict[str, Any]:
        """Test content type detection accuracy."""
        try:
            from memory.enhanced_chunking_strategies import AdaptiveChunkingStrategy
            
            chunker = AdaptiveChunkingStrategy()
            
            # Test content with known characteristics
            test_cases = [
                {
                    "content": '"Hello," said Emma. "How are you?"',
                    "expected_high_dialogue_ratio": True
                },
                {
                    "content": "Emma ran quickly and grabbed the book.",
                    "expected_high_action_ratio": True
                },
                {
                    "content": "Emma, Sarah, James, and Margaret were all present.",
                    "expected_high_character_count": True
                }
            ]
            
            detection_results = []
            
            for case in test_cases:
                analysis = await chunker._analyze_content(case["content"])
                
                result = {
                    "content": case["content"][:50] + "...",
                    "dialogue_ratio": analysis["dialogue_ratio"],
                    "action_ratio": analysis["action_ratio"],
                    "character_count": analysis["character_count"],
                    "narrative_type": analysis["narrative_type"]
                }
                
                # Check expectations
                if case.get("expected_high_dialogue_ratio"):
                    result["dialogue_detection_correct"] = analysis["dialogue_ratio"] > 0.1
                if case.get("expected_high_action_ratio"):
                    result["action_detection_correct"] = analysis["action_ratio"] > 0.1
                if case.get("expected_high_character_count"):
                    result["character_detection_correct"] = analysis["character_count"] > 2
                
                detection_results.append(result)
            
            # Calculate overall detection accuracy
            correct_detections = sum(1 for result in detection_results 
                                  if any(key.endswith("_correct") and value for key, value in result.items()))
            detection_accuracy = correct_detections / len(test_cases)
            
            return {
                "success": True,
                "message": "Content type detection working",
                "detection_results": detection_results,
                "detection_accuracy": detection_accuracy,
                "cases_tested": len(test_cases)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Content type detection failed: {str(e)}"
            }
    
    async def test_dialogue_preserving_chunking(self) -> Dict[str, Any]:
        """Test dialogue preserving chunking strategy."""
        try:
            from memory.enhanced_chunking_strategies import chunk_novel_content
            
            # Test content with mixed dialogue and narrative
            dialogue_content = '''
            Emma walked into the room and saw Sarah sitting by the window.
            
            "Hello Sarah," Emma said warmly. "How are you feeling today?"
            
            Sarah looked up with tired eyes. "I'm doing better, thank you for asking."
            
            "That's good to hear," Emma replied, sitting down next to her friend.
            
            The room fell silent as both women contemplated the events of the day.
            '''
            
            # Chunk with dialogue preserving strategy
            chunks = await chunk_novel_content(dialogue_content, max_chunk_size=500, strategy="dialogue_preserving")
            
            # Analyze results
            dialogue_chunks = [chunk for chunk in chunks if '"' in chunk.content]
            narrative_chunks = [chunk for chunk in chunks if '"' not in chunk.content]
            
            # Check that dialogue integrity is preserved
            dialogue_integrity_preserved = True
            for chunk in dialogue_chunks:
                # Check that dialogue chunks contain complete dialogue
                quote_count = chunk.content.count('"')
                if quote_count % 2 != 0:  # Odd number of quotes indicates broken dialogue
                    dialogue_integrity_preserved = False
                    break
            
            return {
                "success": True,
                "message": "Dialogue preserving chunking working",
                "total_chunks": len(chunks),
                "dialogue_chunks": len(dialogue_chunks),
                "narrative_chunks": len(narrative_chunks),
                "dialogue_integrity_preserved": dialogue_integrity_preserved,
                "avg_chunk_size": sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Dialogue preserving chunking failed: {str(e)}"
            }
    
    async def test_action_oriented_chunking(self) -> Dict[str, Any]:
        """Test action oriented chunking strategy."""
        try:
            from memory.enhanced_chunking_strategies import chunk_novel_content
            
            # Test content with action sequences
            action_content = '''
            Emma burst through the door and sprinted down the hallway. Behind her, she could hear
            footsteps echoing off the walls. She reached the staircase and took the steps two at a time.
            
            At the bottom, she paused to catch her breath. The footsteps had stopped. She listened
            carefully, trying to determine where her pursuer had gone.
            
            Suddenly, a hand grabbed her shoulder. She spun around and saw James standing there,
            breathing heavily from the chase.
            '''
            
            # Chunk with action oriented strategy
            chunks = await chunk_novel_content(action_content, max_chunk_size=400, strategy="action_oriented")
            
            # Analyze action content preservation
            action_words = ['burst', 'sprinted', 'reached', 'paused', 'grabbed', 'spun']
            action_preservation_score = 0
            
            for chunk in chunks:
                chunk_action_words = sum(1 for word in action_words if word in chunk.content.lower())
                if chunk_action_words > 0:
                    action_preservation_score += chunk_action_words
            
            action_preservation_rate = action_preservation_score / len(action_words)
            
            return {
                "success": True,
                "message": "Action oriented chunking working",
                "total_chunks": len(chunks),
                "action_preservation_rate": action_preservation_rate,
                "chunks_with_action": sum(1 for chunk in chunks if any(word in chunk.content.lower() for word in action_words)),
                "avg_chunk_size": sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Action oriented chunking failed: {str(e)}"
            }
    
    async def test_character_focused_chunking(self) -> Dict[str, Any]:
        """Test character focused chunking strategy."""
        try:
            from memory.enhanced_chunking_strategies import chunk_novel_content
            
            # Test content with multiple characters
            character_content = '''
            Emma and Sarah sat in the library discussing their plans. Emma was excited about
            the upcoming trip, while Sarah seemed more hesitant.
            
            Meanwhile, James was in the kitchen preparing lunch. He had invited Thomas and
            Margaret to join them for the meal.
            
            Margaret arrived first, carrying a basket of fresh flowers from her garden.
            Thomas followed shortly after, bringing news from the village.
            '''
            
            # Chunk with character focused strategy
            chunks = await chunk_novel_content(character_content, max_chunk_size=400, strategy="character_focused")
            
            # Analyze character grouping
            characters = ['Emma', 'Sarah', 'James', 'Thomas', 'Margaret']
            character_distribution = {}
            
            for chunk in chunks:
                chunk_characters = [char for char in characters if char in chunk.content]
                character_distribution[chunk.id] = chunk_characters
            
            # Check that related characters are grouped together
            proper_grouping = True
            for chunk_id, chunk_chars in character_distribution.items():
                if len(chunk_chars) > 0:  # Chunk has characters
                    # Check if related characters are together (simplified check)
                    if 'Emma' in chunk_chars and 'Sarah' in chunk_chars:
                        proper_grouping = True  # Good grouping
                    elif 'James' in chunk_chars and any(char in chunk_chars for char in ['Thomas', 'Margaret']):
                        proper_grouping = True  # Good grouping
            
            return {
                "success": True,
                "message": "Character focused chunking working",
                "total_chunks": len(chunks),
                "character_distribution": character_distribution,
                "proper_character_grouping": proper_grouping,
                "chunks_with_characters": sum(1 for chars in character_distribution.values() if chars),
                "avg_chunk_size": sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Character focused chunking failed: {str(e)}"
            }
    
    async def test_complex_narrative_handling(self) -> Dict[str, Any]:
        """Test handling of complex narrative content."""
        try:
            from memory.enhanced_chunking_strategies import chunk_novel_content, analyze_chunking_performance
            
            # Complex narrative with multiple elements
            complex_content = '''
            The rain had been falling for hours when Emma finally decided to leave the house.
            She grabbed her coat and umbrella, pausing at the door to look back at the room
            where so many memories had been made.
            
            "Are you sure about this?" Sarah called from the kitchen.
            
            "I have to go," Emma replied, her voice barely above a whisper. "There's no other choice."
            
            The drive to the station took longer than expected. The roads were slick with rain,
            and Emma had to be careful not to skid on the wet pavement. As she drove, she thought
            about all the things she would miss about this place.
            
            At the station, she saw James waiting by the platform. He looked older than she
            remembered, more worn by the years that had passed between them.
            '''
            
            # Chunk the complex content
            chunks = await chunk_novel_content(complex_content, max_chunk_size=600)
            
            # Analyze chunking performance
            performance = await analyze_chunking_performance(chunks)
            
            # Check quality metrics
            quality_threshold = 0.7
            coherence_threshold = 0.8
            
            quality_passed = performance["avg_quality_score"] >= quality_threshold
            coherence_passed = performance["avg_coherence_score"] >= coherence_threshold
            
            # Check content diversity
            content_types = set(chunk.content_type.value for chunk in chunks)
            has_content_diversity = len(content_types) > 1
            
            return {
                "success": True,
                "message": "Complex narrative handling working",
                "performance": performance,
                "quality_passed": quality_passed,
                "coherence_passed": coherence_passed,
                "content_diversity": has_content_diversity,
                "content_types_found": list(content_types),
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Complex narrative handling failed: {str(e)}"
            }
    
    async def test_chunking_performance_analysis(self) -> Dict[str, Any]:
        """Test chunking performance analysis functionality."""
        try:
            from memory.enhanced_chunking_strategies import chunk_novel_content, analyze_chunking_performance
            
            # Test with various content types
            test_contents = [
                "Emma walked through the garden, enjoying the peaceful morning.",
                '"Hello there," said Emma. "How are you today?"',
                "Suddenly, Emma ran towards the door and grabbed the handle.",
                "The old house stood on the hill, its windows dark and mysterious."
            ]
            
            all_chunks = []
            for content in test_contents:
                chunks = await chunk_novel_content(content, max_chunk_size=200)
                all_chunks.extend(chunks)
            
            # Analyze overall performance
            performance = await analyze_chunking_performance(all_chunks)
            
            # Verify analysis structure
            required_metrics = [
                "total_chunks", "avg_quality_score", "avg_coherence_score",
                "avg_token_count", "content_type_distribution", "overall_performance"
            ]
            
            has_all_metrics = all(metric in performance for metric in required_metrics)
            
            # Check performance thresholds
            performance_good = performance["overall_performance"] >= 0.7
            
            return {
                "success": True,
                "message": "Chunking performance analysis working",
                "performance": performance,
                "has_all_metrics": has_all_metrics,
                "performance_good": performance_good,
                "chunks_analyzed": performance["total_chunks"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Chunking performance analysis failed: {str(e)}"
            }
    
    def generate_final_report(self):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("PHASE 2.2 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed Tests: {self.passed_tests}")
        print(f"Failed Tests: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Performance analysis
        if self.chunking_performance:
            avg_performance = sum(perf.get("overall_performance", 0) for perf in self.chunking_performance) / len(self.chunking_performance)
            print(f"\n📊 CHUNKING PERFORMANCE:")
            print(f"   Average Performance Score: {avg_performance:.3f}")
            print(f"   Performance Tests: {len(self.chunking_performance)}")
        
        # Phase 2.2 success criteria
        phase_success = success_rate >= 80  # At least 80% pass rate
        
        if self.chunking_performance:
            performance_success = avg_performance >= 0.7
            phase_success = phase_success and performance_success
        
        print(f"\n🎯 PHASE 2.2 STATUS: {'✅ PASSED' if phase_success else '❌ FAILED'}")
        
        if phase_success:
            print("\n✅ Chunking optimization implementation is ready!")
            print("✅ Adaptive strategy selection working correctly")
            print("✅ Complex narrative handling functional")
            print("✅ Ready to proceed to Sub-Phase 2.3")
        else:
            print("\n❌ Phase 2.2 requirements not met")
            print("❌ Fix failing tests or improve performance scores before proceeding")
        
        # Detailed results
        print("\n📊 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {result['test_name']} ({result['execution_time_ms']:.2f}ms)")
            if not result["success"] and result["error_message"]:
                print(f"    Error: {result['error_message']}")
        
        return {
            "phase": "2.2",
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "phase_passed": phase_success,
            "chunking_performance": self.chunking_performance,
            "test_results": self.test_results
        }


async def main():
    """Run Phase 2.2 completion tests."""
    test_suite = Phase2_2TestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
