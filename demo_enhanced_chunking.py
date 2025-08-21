#!/usr/bin/env python3
"""
Demo script for Enhanced Scene-Level Chunking
Demonstrates the new scene-aware chunking capabilities for creative content.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

from ingestion.enhanced_scene_chunker import (
    EnhancedSceneChunker,
    EnhancedChunkingConfig,
    ContentType,
    SceneType,
    create_enhanced_chunker
)
from ingestion.chunker import ChunkingConfig, create_chunker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_novel_content() -> str:
    """Create sample novel content for demonstration."""
    return '''
# Chapter 1: The Mysterious Letter

The old Victorian house stood silently against the stormy sky, its weathered shutters creaking in the wind. Rain pelted the windows with increasing intensity, creating rivulets that traced intricate patterns down the glass. Emma had always found comfort in storms, but tonight felt different—ominous, as if the very air held secrets waiting to be unveiled.

She sat in her grandfather's study, surrounded by towering bookshelves that reached toward the coffered ceiling. The leather-bound volumes seemed to watch her with ancient wisdom, their spines bearing titles in languages she couldn't read. A single lamp cast a warm circle of light across the mahogany desk where an unopened letter lay waiting.

***

"Emma, you must read this," her grandfather had said just hours before his passing. His voice, usually strong and commanding, had been reduced to a whisper. "The letter... it explains everything."

She remembered the weight of his hand on hers, the urgency in his eyes. Now, alone in the study, she finally understood why he had been so insistent. The envelope bore her name in elegant script, but the return address was smudged beyond recognition.

With trembling fingers, she broke the wax seal.

***

Meanwhile, across town, Detective Marcus Chen was investigating a break-in at the local museum. The thieves had been selective—only one item was missing from the extensive collection of artifacts. A small, unremarkable wooden box that had been donated just last week by the estate of Professor Edmund Hartwell.

"Any idea what was in the box?" Chen asked the curator, Dr. Sarah Mitchell.

"According to the donation records, it contained personal papers and a few family heirlooms," she replied, adjusting her glasses nervously. "Nothing that seemed particularly valuable."

Chen made a note in his pad. Something about this case felt off. Professional thieves didn't usually target items of sentimental value unless they knew something the rest of the world didn't.

***

Back at the Victorian house, Emma had finished reading the letter. Her hands shook as she set it down on the desk. The words seemed impossible, yet they explained so much about her family's history—and her own strange dreams.

"The Hartwell Legacy," she whispered to herself, reading the letter's heading once more.

According to the letter, her grandfather had been the guardian of an ancient secret, one that had been passed down through generations of their family. The wooden box mentioned in the letter—the same one that had been stolen from the museum—contained the key to unlocking a mystery that stretched back centuries.

Emma stood up abruptly, her chair scraping against the hardwood floor. She needed to find that box before whoever had stolen it discovered its true significance. But first, she had to understand what she was dealing with.

She pulled out her phone and dialed a number she hadn't called in years.

"Hello, Marcus? It's Emma Hartwell. I think we need to talk."

***

The next morning brought crisp autumn air and the scent of burning leaves. Emma met Detective Chen at a small café downtown, far from the prying eyes and ears that seemed to follow her everywhere since her grandfather's death.

"I know this sounds crazy," she began, sliding a photocopy of the letter across the table, "but I think the museum theft is connected to my grandfather's death."

Chen read the letter carefully, his expression growing more serious with each paragraph. When he finished, he looked up at Emma with new understanding.

"Your grandfather was Professor Edmund Hartwell," he said. It wasn't a question.

"You knew him?"

"Only by reputation. He was a respected historian, specialized in medieval manuscripts and ancient artifacts." Chen paused, considering his next words carefully. "Emma, there's something you should know about the break-in. It wasn't random."

He pulled out a file from his briefcase and opened it on the table. Crime scene photos showed the museum's security system had been disabled with professional precision. The thieves had known exactly where to find the box and how to avoid the cameras.

"This was planned," Chen continued. "Someone knew about that box and what it contained. The question is: how?"

Emma felt a chill run down her spine. If the letter was true, then her family had been protecting something for generations. Something that others were willing to kill for.

"Marcus," she said quietly, "I think my grandfather's death wasn't natural."

***

As they talked, neither Emma nor Detective Chen noticed the figure watching them from across the street. Hidden behind the tinted windows of a black sedan, the observer made a phone call.

"They're together," the voice said. "The granddaughter and the detective. Should we proceed?"

The response was immediate and cold: "Not yet. Let them lead us to what we're looking for. But keep them under surveillance. We can't afford any more mistakes."

The sedan pulled away from the curb, disappearing into the morning traffic. The hunt for the Hartwell Legacy had begun in earnest, and Emma was about to discover that some family secrets were worth dying for.
'''


def analyze_chunking_comparison():
    """Compare traditional chunking vs enhanced scene chunking."""
    
    print("=" * 80)
    print("ENHANCED SCENE-LEVEL CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample content
    content = create_sample_novel_content()
    
    print(f"\nSample Content Length: {len(content)} characters")
    print(f"Word Count: {len(content.split())} words")
    
    # Traditional chunking
    print("\n" + "=" * 50)
    print("TRADITIONAL CHUNKING")
    print("=" * 50)
    
    traditional_config = ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        use_semantic_splitting=True
    )
    traditional_chunker = create_chunker(traditional_config)
    
    traditional_chunks = traditional_chunker.chunk_document(
        content=content,
        title="Sample Novel Chapter",
        source="demo.md"
    )
    
    print(f"Traditional Chunks Created: {len(traditional_chunks)}")
    for i, chunk in enumerate(traditional_chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Size: {chunk.token_count} tokens")
        print(f"  Type: {chunk.metadata.get('chunk_type', 'unknown')}")
        print(f"  Preview: {chunk.content[:100]}...")
    
    # Enhanced scene chunking
    print("\n" + "=" * 50)
    print("ENHANCED SCENE CHUNKING")
    print("=" * 50)
    
    enhanced_config = EnhancedChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        dialogue_chunk_size=800,
        narrative_chunk_size=1200,
        action_chunk_size=600,
        description_chunk_size=1000,
        min_scene_size=200,
        preserve_dialogue_integrity=True,
        preserve_emotional_beats=True
    )
    enhanced_chunker = create_enhanced_chunker(enhanced_config)
    
    enhanced_chunks = enhanced_chunker.chunk_document(
        content=content,
        title="Sample Novel Chapter",
        source="demo.md"
    )
    
    print(f"Enhanced Chunks Created: {len(enhanced_chunks)}")
    for i, chunk in enumerate(enhanced_chunks):
        metadata = chunk.metadata
        print(f"\nChunk {i+1}:")
        print(f"  Size: {chunk.token_count} tokens")
        print(f"  Scene Type: {metadata.get('scene_type', 'unknown')}")
        print(f"  Content Types: {metadata.get('content_types', [])}")
        print(f"  Characters: {metadata.get('characters', [])}")
        print(f"  Locations: {metadata.get('locations', [])}")
        print(f"  Emotional Tone: {metadata.get('emotional_tone', 'unknown')}")
        print(f"  Tension Level: {metadata.get('tension_level', 0.0):.2f}")
        print(f"  Importance Score: {metadata.get('importance_score', 0.0):.2f}")
        print(f"  Dialogue Ratio: {metadata.get('dialogue_ratio', 0.0):.2f}")
        print(f"  Action Ratio: {metadata.get('action_ratio', 0.0):.2f}")
        print(f"  Narrative Function: {metadata.get('narrative_function', 'unknown')}")
        print(f"  POV Character: {metadata.get('pov_character', 'unknown')}")
        print(f"  Preview: {chunk.content[:100]}...")


def demonstrate_content_type_analysis():
    """Demonstrate content type analysis capabilities."""
    
    print("\n" + "=" * 50)
    print("CONTENT TYPE ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Create different content samples
    samples = {
        "Dialogue Scene": '''
        "I can't believe you did that," Sarah said, her voice trembling with anger.
        
        "I had no choice," John replied, avoiding her gaze. "You don't understand the pressure I was under."
        
        "Pressure?" Sarah laughed bitterly. "What about the pressure I'm under now? What about the consequences of your actions?"
        
        John finally looked at her, his eyes filled with regret. "I'm sorry, Sarah. I truly am."
        ''',
        
        "Action Scene": '''
        The explosion rocked the building, sending debris flying in all directions. Sarah dove behind the concrete barrier just as another blast shook the ground beneath her feet. Bullets whizzed past her head as she sprinted across the open courtyard. Her heart pounded as she leaped over fallen rubble, her training taking over. Behind her, she could hear the enemy forces advancing.
        ''',
        
        "Description Scene": '''
        The ancient library stretched before them like a cathedral of knowledge. Towering shelves reached toward vaulted ceilings painted with faded frescoes of mythological scenes. Dust motes danced in shafts of golden sunlight that streamed through tall, arched windows. The air was thick with the musty scent of old parchment and leather bindings, creating an atmosphere of scholarly reverence that had remained unchanged for centuries.
        ''',
        
        "Internal Monologue": '''
        Emma wondered if she was making the right decision. The letter had revealed so much, yet raised even more questions. She thought about her grandfather's final words, the urgency in his voice. Had he known what was coming? She felt a mixture of fear and determination coursing through her veins. Whatever lay ahead, she knew there was no turning back now.
        '''
    }
    
    enhanced_chunker = create_enhanced_chunker()
    
    for sample_name, sample_content in samples.items():
        print(f"\n{sample_name}:")
        print("-" * 30)
        
        # Analyze content types
        content_types = enhanced_chunker._classify_content_types(sample_content)
        dialogue_ratio = enhanced_chunker._calculate_dialogue_ratio(sample_content)
        action_ratio = enhanced_chunker._calculate_action_ratio(sample_content)
        description_ratio = enhanced_chunker._calculate_description_ratio(sample_content)
        emotional_tone = enhanced_chunker._analyze_emotional_tone(sample_content)
        tension_level = enhanced_chunker._calculate_tension_level(sample_content)
        
        print(f"Content Types: {[ct.value for ct in content_types]}")
        print(f"Dialogue Ratio: {dialogue_ratio:.3f}")
        print(f"Action Ratio: {action_ratio:.3f}")
        print(f"Description Ratio: {description_ratio:.3f}")
        print(f"Emotional Tone: {emotional_tone}")
        print(f"Tension Level: {tension_level:.3f}")
        
        # Determine scene type
        scene_type = enhanced_chunker._classify_scene_type(
            sample_content, content_types, dialogue_ratio, action_ratio
        )
        print(f"Scene Type: {scene_type.value}")
        
        # Calculate importance score
        importance = enhanced_chunker._calculate_importance_score(
            scene_type, content_types, tension_level, 2  # Assume 2 characters
        )
        print(f"Importance Score: {importance:.3f}")


def demonstrate_scene_detection():
    """Demonstrate scene detection capabilities."""
    
    print("\n" + "=" * 50)
    print("SCENE DETECTION DEMONSTRATION")
    print("=" * 50)
    
    enhanced_chunker = create_enhanced_chunker()
    content = create_sample_novel_content()
    
    # Detect scenes
    scenes = enhanced_chunker._detect_enhanced_scenes(content)
    
    print(f"Detected {len(scenes)} scenes:")
    
    for i, scene in enumerate(scenes):
        print(f"\nScene {i+1}:")
        print(f"  Length: {len(scene)} characters")
        print(f"  Word Count: {len(scene.split())} words")
        
        # Analyze scene
        scene_metadata = enhanced_chunker._analyze_scene(scene)
        
        print(f"  Scene Type: {scene_metadata.scene_type.value}")
        print(f"  Characters: {scene_metadata.characters}")
        print(f"  Locations: {scene_metadata.locations}")
        print(f"  Emotional Tone: {scene_metadata.emotional_tone}")
        print(f"  Tension Level: {scene_metadata.tension_level:.2f}")
        print(f"  Importance Score: {scene_metadata.importance_score:.2f}")
        print(f"  Narrative Function: {scene_metadata.narrative_function}")
        print(f"  Preview: {scene[:150]}...")


def save_analysis_results():
    """Save detailed analysis results to JSON file."""
    
    print("\n" + "=" * 50)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 50)
    
    enhanced_chunker = create_enhanced_chunker()
    content = create_sample_novel_content()
    
    # Create enhanced chunks
    chunks = enhanced_chunker.chunk_document(
        content=content,
        title="Sample Novel Chapter",
        source="demo.md"
    )
    
    # Prepare results for JSON serialization
    results = {
        "analysis_timestamp": "2024-01-01T00:00:00Z",
        "content_stats": {
            "total_characters": len(content),
            "total_words": len(content.split()),
            "total_chunks": len(chunks)
        },
        "chunks": []
    }
    
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata
        }
        results["chunks"].append(chunk_data)
    
    # Save to file
    output_file = Path("enhanced_chunking_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Analysis results saved to: {output_file}")
    print(f"Total chunks analyzed: {len(chunks)}")


def main():
    """Main demonstration function."""
    
    print("Starting Enhanced Scene-Level Chunking Demonstration...")
    
    try:
        # Run demonstrations
        analyze_chunking_comparison()
        demonstrate_content_type_analysis()
        demonstrate_scene_detection()
        save_analysis_results()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Benefits of Enhanced Scene Chunking:")
        print("• Scene-aware boundaries preserve narrative integrity")
        print("• Content-type optimization improves context relevance")
        print("• Character and location tracking enhances search accuracy")
        print("• Emotional tone analysis enables better content matching")
        print("• Importance scoring prioritizes critical narrative elements")
        print("• Dialogue preservation maintains character voice consistency")
        print("• Action sequence integrity improves pacing analysis")
        
        print("\nNext Steps:")
        print("• Enable enhanced chunking in ingestion configuration")
        print("• Test with your own novel content")
        print("• Monitor generation quality improvements")
        print("• Adjust chunk size parameters based on content type")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()