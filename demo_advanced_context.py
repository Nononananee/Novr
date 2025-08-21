#!/usr/bin/env python3
"""
Demo for Advanced Context Building with Hierarchical Retrieval
Demonstrates the enhanced context building capabilities for creative content generation.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simplified models for demonstration
@dataclass
class MockChunkResult:
    """Mock chunk result for demonstration."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_title: str
    document_source: str


@dataclass
class MockGraphSearchResult:
    """Mock graph search result for demonstration."""
    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None


class MockAdvancedContextBuilder:
    """Mock advanced context builder for demonstration."""
    
    def __init__(self):
        """Initialize mock context builder."""
        # Sample chunks database
        self.sample_chunks = [
            MockChunkResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                content='''Emma stood in the doorway of her grandfather's study, her heart racing. The letter in her hands felt heavier than it should, as if the words themselves carried the weight of generations. "The Hartwell Legacy," she whispered, reading the heading for the third time.''',
                score=0.95,
                metadata={
                    "scene_type": "emotional_beat",
                    "characters": ["Emma", "grandfather"],
                    "locations": ["study"],
                    "emotional_tone": "anxiety",
                    "tension_level": 0.7,
                    "importance_score": 0.9,
                    "dialogue_ratio": 0.1,
                    "action_ratio": 0.2,
                    "description_ratio": 0.3,
                    "content_types": ["narrative", "internal_monologue"]
                },
                document_title="The Hartwell Mystery",
                document_source="chapter_1.md"
            ),
            MockChunkResult(
                chunk_id="chunk_2",
                document_id="doc_1",
                content='''"Detective Chen, we have a problem," Officer Martinez said, bursting into the office. "The museum theft wasn't random. Someone knew exactly what they were looking for."

Chen looked up from his files. "What makes you say that?"

"The security footage. They bypassed three more valuable artifacts to get to that wooden box."''',
                score=0.88,
                metadata={
                    "scene_type": "dialogue_scene",
                    "characters": ["Detective Chen", "Officer Martinez"],
                    "locations": ["office"],
                    "emotional_tone": "tension",
                    "tension_level": 0.6,
                    "importance_score": 0.8,
                    "dialogue_ratio": 0.8,
                    "action_ratio": 0.1,
                    "description_ratio": 0.1,
                    "content_types": ["dialogue"]
                },
                document_title="The Hartwell Mystery",
                document_source="chapter_2.md"
            ),
            MockChunkResult(
                chunk_id="chunk_3",
                document_id="doc_1",
                content='''The Victorian mansion loomed against the storm clouds, its Gothic spires piercing the gray sky like ancient fingers reaching toward heaven. Ivy crawled up the weathered stone walls, and the windows seemed to watch with dark, knowing eyes. This was the Hartwell estate, keeper of secrets for over two centuries.''',
                score=0.82,
                metadata={
                    "scene_type": "opening",
                    "characters": [],
                    "locations": ["Victorian mansion", "Hartwell estate"],
                    "emotional_tone": "mysterious",
                    "tension_level": 0.4,
                    "importance_score": 0.7,
                    "dialogue_ratio": 0.0,
                    "action_ratio": 0.0,
                    "description_ratio": 0.9,
                    "content_types": ["description", "world_building"]
                },
                document_title="The Hartwell Mystery",
                document_source="prologue.md"
            ),
            MockChunkResult(
                chunk_id="chunk_4",
                document_id="doc_1",
                content='''Emma's phone buzzed. A text from an unknown number: "Stop looking for the box. Some secrets are meant to stay buried." Her blood ran cold. How did they know? She hadn't told anyone about her search except...

She thought about Detective Chen. Could she trust him? Or was someone watching her every move?''',
                score=0.91,
                metadata={
                    "scene_type": "conflict",
                    "characters": ["Emma", "Detective Chen"],
                    "locations": [],
                    "emotional_tone": "fear",
                    "tension_level": 0.9,
                    "importance_score": 0.95,
                    "dialogue_ratio": 0.2,
                    "action_ratio": 0.3,
                    "description_ratio": 0.1,
                    "content_types": ["narrative", "internal_monologue", "action"]
                },
                document_title="The Hartwell Mystery",
                document_source="chapter_3.md"
            ),
            MockChunkResult(
                chunk_id="chunk_5",
                document_id="doc_1",
                content='''The explosion shattered the night silence. Emma dove behind the stone fountain as debris rained down around her. Through the smoke, she could see dark figures advancing across the courtyard.

"There!" one of them shouted. "She's by the fountain!"

Emma's heart pounded. She had to reach the main house, but they were between her and safety.''',
                score=0.87,
                metadata={
                    "scene_type": "action_sequence",
                    "characters": ["Emma"],
                    "locations": ["courtyard", "fountain", "main house"],
                    "emotional_tone": "fear",
                    "tension_level": 0.95,
                    "importance_score": 0.9,
                    "dialogue_ratio": 0.2,
                    "action_ratio": 0.8,
                    "description_ratio": 0.2,
                    "content_types": ["action", "dialogue"]
                },
                document_title="The Hartwell Mystery",
                document_source="chapter_4.md"
            )
        ]
        
        # Sample graph facts
        self.sample_facts = [
            MockGraphSearchResult(
                fact="Emma Hartwell is the granddaughter of Professor Edmund Hartwell",
                uuid="fact_1",
                valid_at="2024-01-01T00:00:00Z"
            ),
            MockGraphSearchResult(
                fact="The Hartwell family has protected an ancient secret for generations",
                uuid="fact_2",
                valid_at="2024-01-01T00:00:00Z"
            ),
            MockGraphSearchResult(
                fact="Detective Chen specializes in art theft and museum security",
                uuid="fact_3",
                valid_at="2024-01-01T00:00:00Z"
            ),
            MockGraphSearchResult(
                fact="The wooden box contains a key to the Hartwell family vault",
                uuid="fact_4",
                valid_at="2024-01-01T00:00:00Z"
            ),
            MockGraphSearchResult(
                fact="The Victorian mansion has secret passages built in the 1800s",
                uuid="fact_5",
                valid_at="2024-01-01T00:00:00Z"
            )
        ]
    
    async def build_generation_context(
        self,
        query: str,
        context_type: str,
        target_characters: Optional[List[str]] = None,
        target_locations: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        narrative_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build generation context using hierarchical retrieval."""
        
        logger.info(f"Building context for query: {query}")
        logger.info(f"Context type: {context_type}")
        logger.info(f"Target characters: {target_characters}")
        logger.info(f"Target locations: {target_locations}")
        logger.info(f"Emotional tone: {emotional_tone}")
        
        # Step 1: Primary semantic retrieval
        primary_chunks = self._retrieve_primary_chunks(query, context_type)
        
        # Step 2: Character-focused retrieval
        character_chunks = []
        if target_characters:
            character_chunks = self._retrieve_character_context(target_characters)
        
        # Step 3: Location/world context
        world_chunks = []
        if target_locations:
            world_chunks = self._retrieve_world_context(target_locations)
        
        # Step 4: Supporting context
        supporting_chunks = self._retrieve_supporting_context(context_type, emotional_tone)
        
        # Step 5: Graph facts
        graph_facts = self._retrieve_graph_context(target_characters, target_locations)
        
        # Step 6: Calculate metrics
        all_chunks = primary_chunks + character_chunks + world_chunks + supporting_chunks
        total_tokens = sum(len(chunk.content.split()) * 1.3 for chunk in all_chunks)
        
        # Calculate quality score
        avg_importance = sum(chunk.metadata.get('importance_score', 0.5) for chunk in all_chunks) / len(all_chunks)
        character_coverage = min(len(target_characters or []) / 3.0, 1.0)
        content_diversity = len(set(ct for chunk in all_chunks for ct in chunk.metadata.get('content_types', []))) / 4.0
        quality_score = (0.5 * avg_importance + 0.3 * character_coverage + 0.2 * content_diversity)
        
        context = {
            "query": query,
            "context_type": context_type,
            "primary_chunks": primary_chunks,
            "character_context": character_chunks,
            "world_context": world_chunks,
            "supporting_chunks": supporting_chunks,
            "graph_facts": graph_facts,
            "total_tokens": int(total_tokens),
            "context_quality_score": quality_score,
            "characters_involved": target_characters or [],
            "locations_involved": target_locations or [],
            "emotional_tone": emotional_tone or "neutral"
        }
        
        return context
    
    def _retrieve_primary_chunks(self, query: str, context_type: str) -> List[MockChunkResult]:
        """Retrieve primary chunks based on semantic similarity."""
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.sample_chunks:
            content_words = set(chunk.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1) * chunk.score
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:3]]
    
    def _retrieve_character_context(self, characters: List[str]) -> List[MockChunkResult]:
        """Retrieve chunks focused on specific characters."""
        character_chunks = []
        for chunk in self.sample_chunks:
            chunk_characters = chunk.metadata.get('characters', [])
            if any(char in chunk_characters for char in characters):
                character_chunks.append(chunk)
        
        # Sort by importance
        character_chunks.sort(key=lambda x: x.metadata.get('importance_score', 0.5), reverse=True)
        return character_chunks[:2]
    
    def _retrieve_world_context(self, locations: List[str]) -> List[MockChunkResult]:
        """Retrieve chunks focused on world-building and locations."""
        world_chunks = []
        for chunk in self.sample_chunks:
            chunk_locations = chunk.metadata.get('locations', [])
            content_types = chunk.metadata.get('content_types', [])
            
            if (any(loc in chunk_locations for loc in locations) or 
                'description' in content_types or 
                'world_building' in content_types):
                world_chunks.append(chunk)
        
        # Sort by description ratio
        world_chunks.sort(key=lambda x: x.metadata.get('description_ratio', 0.0), reverse=True)
        return world_chunks[:2]
    
    def _retrieve_supporting_context(self, context_type: str, emotional_tone: Optional[str]) -> List[MockChunkResult]:
        """Retrieve supporting context based on type and tone."""
        supporting_chunks = []
        
        for chunk in self.sample_chunks:
            # Match context type
            if context_type == "character_focused" and chunk.metadata.get('scene_type') in ['character_introduction', 'emotional_beat']:
                supporting_chunks.append(chunk)
            elif context_type == "action_sequence" and chunk.metadata.get('action_ratio', 0.0) > 0.5:
                supporting_chunks.append(chunk)
            elif context_type == "dialogue_heavy" and chunk.metadata.get('dialogue_ratio', 0.0) > 0.5:
                supporting_chunks.append(chunk)
            
            # Match emotional tone
            if emotional_tone and chunk.metadata.get('emotional_tone') == emotional_tone:
                if chunk not in supporting_chunks:
                    supporting_chunks.append(chunk)
        
        return supporting_chunks[:2]
    
    def _retrieve_graph_context(self, characters: Optional[List[str]], locations: Optional[List[str]]) -> List[MockGraphSearchResult]:
        """Retrieve relevant facts from knowledge graph."""
        relevant_facts = []
        
        for fact in self.sample_facts:
            # Check if fact mentions target characters or locations
            fact_text = fact.fact.lower()
            
            if characters:
                for char in characters:
                    if char.lower() in fact_text:
                        relevant_facts.append(fact)
                        break
            
            if locations:
                for loc in locations:
                    if loc.lower() in fact_text:
                        if fact not in relevant_facts:
                            relevant_facts.append(fact)
                        break
        
        return relevant_facts[:3]


def demonstrate_context_building():
    """Demonstrate advanced context building capabilities."""
    
    print("=" * 80)
    print("ADVANCED CONTEXT BUILDING DEMONSTRATION")
    print("=" * 80)
    
    # Create mock context builder
    context_builder = MockAdvancedContextBuilder()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Character-Focused Generation",
            "query": "Emma discovers a hidden message in her grandfather's letter",
            "context_type": "character_focused",
            "target_characters": ["Emma"],
            "emotional_tone": "anxiety"
        },
        {
            "name": "Action Sequence Generation",
            "query": "Emma escapes from the attackers in the courtyard",
            "context_type": "action_sequence",
            "target_characters": ["Emma"],
            "target_locations": ["courtyard"],
            "emotional_tone": "fear"
        },
        {
            "name": "Dialogue Scene Generation",
            "query": "Detective Chen questions Emma about the theft",
            "context_type": "dialogue_heavy",
            "target_characters": ["Detective Chen", "Emma"],
            "emotional_tone": "tension"
        },
        {
            "name": "World Building Generation",
            "query": "Describe the secret passages in the Victorian mansion",
            "context_type": "world_building",
            "target_locations": ["Victorian mansion"],
            "emotional_tone": "mysterious"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*60}")
        
        # Build context
        context = asyncio.run(context_builder.build_generation_context(
            query=scenario["query"],
            context_type=scenario["context_type"],
            target_characters=scenario.get("target_characters"),
            target_locations=scenario.get("target_locations"),
            emotional_tone=scenario.get("emotional_tone")
        ))
        
        # Display results
        print(f"\nQuery: {context['query']}")
        print(f"Context Type: {context['context_type']}")
        print(f"Total Tokens: {context['total_tokens']}")
        print(f"Quality Score: {context['context_quality_score']:.3f}")
        print(f"Characters Involved: {context['characters_involved']}")
        print(f"Locations Involved: {context['locations_involved']}")
        print(f"Emotional Tone: {context['emotional_tone']}")
        
        print(f"\nPrimary Chunks ({len(context['primary_chunks'])}):")
        for j, chunk in enumerate(context['primary_chunks'], 1):
            print(f"  {j}. [{chunk.metadata['scene_type']}] Score: {chunk.score:.3f}")
            print(f"     Characters: {chunk.metadata.get('characters', [])}")
            print(f"     Preview: {chunk.content[:100]}...")
        
        print(f"\nCharacter Context ({len(context['character_context'])}):")
        for j, chunk in enumerate(context['character_context'], 1):
            print(f"  {j}. [{chunk.metadata['scene_type']}] Importance: {chunk.metadata['importance_score']:.3f}")
            print(f"     Characters: {chunk.metadata.get('characters', [])}")
            print(f"     Preview: {chunk.content[:100]}...")
        
        print(f"\nWorld Context ({len(context['world_context'])}):")
        for j, chunk in enumerate(context['world_context'], 1):
            print(f"  {j}. [{chunk.metadata['scene_type']}] Description Ratio: {chunk.metadata.get('description_ratio', 0.0):.3f}")
            print(f"     Locations: {chunk.metadata.get('locations', [])}")
            print(f"     Preview: {chunk.content[:100]}...")
        
        print(f"\nGraph Facts ({len(context['graph_facts'])}):")
        for j, fact in enumerate(context['graph_facts'], 1):
            print(f"  {j}. {fact.fact}")


def demonstrate_context_optimization():
    """Demonstrate context optimization and ranking."""
    
    print("\n" + "=" * 80)
    print("CONTEXT OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    context_builder = MockAdvancedContextBuilder()
    
    # Build context for a complex scenario
    context = asyncio.run(context_builder.build_generation_context(
        query="Emma and Detective Chen work together to solve the mystery while being pursued by unknown enemies",
        context_type="plot_driven",
        target_characters=["Emma", "Detective Chen"],
        target_locations=["Victorian mansion", "office"],
        emotional_tone="tension"
    ))
    
    print(f"Query: {context['query']}")
    print(f"Context Optimization Results:")
    print(f"  Total Chunks Retrieved: {len(context['primary_chunks'] + context['character_context'] + context['world_context'] + context['supporting_chunks'])}")
    print(f"  Total Tokens: {context['total_tokens']}")
    print(f"  Context Quality Score: {context['context_quality_score']:.3f}")
    
    # Analyze content distribution
    all_chunks = context['primary_chunks'] + context['character_context'] + context['world_context'] + context['supporting_chunks']
    
    content_types = {}
    scene_types = {}
    emotional_tones = {}
    
    for chunk in all_chunks:
        # Count content types
        for ct in chunk.metadata.get('content_types', []):
            content_types[ct] = content_types.get(ct, 0) + 1
        
        # Count scene types
        st = chunk.metadata.get('scene_type', 'unknown')
        scene_types[st] = scene_types.get(st, 0) + 1
        
        # Count emotional tones
        et = chunk.metadata.get('emotional_tone', 'neutral')
        emotional_tones[et] = emotional_tones.get(et, 0) + 1
    
    print(f"\nContent Type Distribution:")
    for ct, count in content_types.items():
        print(f"  {ct}: {count}")
    
    print(f"\nScene Type Distribution:")
    for st, count in scene_types.items():
        print(f"  {st}: {count}")
    
    print(f"\nEmotional Tone Distribution:")
    for et, count in emotional_tones.items():
        print(f"  {et}: {count}")


def save_context_analysis():
    """Save context analysis results to JSON."""
    
    print("\n" + "=" * 80)
    print("SAVING CONTEXT ANALYSIS")
    print("=" * 80)
    
    context_builder = MockAdvancedContextBuilder()
    
    # Build context for analysis
    context = asyncio.run(context_builder.build_generation_context(
        query="Generate a climactic scene where Emma confronts the truth about her family's legacy",
        context_type="emotional_scene",
        target_characters=["Emma"],
        target_locations=["Victorian mansion"],
        emotional_tone="dramatic"
    ))
    
    # Prepare for JSON serialization
    serializable_context = {
        "query": context["query"],
        "context_type": context["context_type"],
        "total_tokens": context["total_tokens"],
        "context_quality_score": context["context_quality_score"],
        "characters_involved": context["characters_involved"],
        "locations_involved": context["locations_involved"],
        "emotional_tone": context["emotional_tone"],
        "chunks": []
    }
    
    # Add chunk information
    all_chunks = context['primary_chunks'] + context['character_context'] + context['world_context'] + context['supporting_chunks']
    
    for chunk in all_chunks:
        chunk_info = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "score": chunk.score,
            "metadata": chunk.metadata,
            "document_title": chunk.document_title
        }
        serializable_context["chunks"].append(chunk_info)
    
    # Add graph facts
    serializable_context["graph_facts"] = [
        {"fact": fact.fact, "uuid": fact.uuid} 
        for fact in context["graph_facts"]
    ]
    
    # Save to file
    with open("advanced_context_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(serializable_context, f, indent=2, ensure_ascii=False)
    
    print("Context analysis saved to: advanced_context_analysis.json")
    print(f"Total chunks analyzed: {len(all_chunks)}")
    print(f"Graph facts included: {len(context['graph_facts'])}")


def main():
    """Main demonstration function."""
    
    print("Starting Advanced Context Building Demonstration...")
    
    try:
        demonstrate_context_building()
        demonstrate_context_optimization()
        save_context_analysis()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Benefits of Advanced Context Building:")
        print("• Hierarchical retrieval prioritizes most relevant content")
        print("• Character-focused context improves character consistency")
        print("• Location-aware retrieval enhances world-building accuracy")
        print("• Emotional tone matching creates better narrative flow")
        print("• Graph integration provides factual consistency")
        print("• Quality scoring ensures high-relevance context")
        print("• Token optimization maximizes context efficiency")
        
        print("\nNext Steps for Implementation:")
        print("• Enable advanced context in GenerationRequest")
        print("• Set use_advanced_context=True for enhanced retrieval")
        print("• Configure context weights based on generation type")
        print("• Monitor context quality scores and adjust thresholds")
        print("• Test with different narrative scenarios")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()