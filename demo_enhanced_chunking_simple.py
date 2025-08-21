#!/usr/bin/env python3
"""
Simple Demo for Enhanced Scene-Level Chunking
Demonstrates the concept without external dependencies.
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Content types for optimized chunking."""
    DIALOGUE = "dialogue"
    NARRATIVE = "narrative"
    ACTION = "action"
    DESCRIPTION = "description"
    INTERNAL_MONOLOGUE = "internal_monologue"
    TRANSITION = "transition"
    EXPOSITION = "exposition"


class SceneType(Enum):
    """Scene types for narrative structure."""
    OPENING = "opening"
    CHARACTER_INTRODUCTION = "character_introduction"
    DIALOGUE_SCENE = "dialogue_scene"
    ACTION_SEQUENCE = "action_sequence"
    EMOTIONAL_BEAT = "emotional_beat"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    TRANSITION = "transition"
    CLIMAX = "climax"
    DENOUEMENT = "denouement"


@dataclass
class SimpleChunk:
    """Simple chunk representation."""
    content: str
    index: int
    metadata: Dict[str, Any]
    token_count: int


class SimpleEnhancedChunker:
    """Simplified version of enhanced chunker for demonstration."""
    
    def __init__(self):
        """Initialize simple chunker."""
        self.scene_break_patterns = [
            r'\n\n\*\*\*\n\n',  # Explicit scene break
            r'\n\n---\n\n',     # Alternative scene break
            r'\n\n# ',          # Chapter break
            r'\n\n## ',         # Section break
        ]
        
        self.emotional_keywords = {
            'joy': ['happy', 'joyful', 'excited', 'delighted', 'cheerful', 'elated'],
            'sadness': ['sad', 'melancholy', 'depressed', 'sorrowful', 'grief', 'mourning'],
            'anger': ['angry', 'furious', 'rage', 'irritated', 'annoyed', 'livid'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'panic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'neutral': ['calm', 'peaceful', 'serene', 'composed', 'steady']
        }
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting."""
        return len(text.split())
    
    def detect_scenes(self, content: str) -> List[str]:
        """Detect scenes using explicit markers."""
        scenes = [content]
        for pattern in self.scene_break_patterns:
            new_scenes = []
            for scene in scenes:
                parts = re.split(pattern, scene)
                new_scenes.extend(parts)
            scenes = [s.strip() for s in new_scenes if s.strip()]
        return scenes
    
    def extract_characters(self, text: str) -> List[str]:
        """Extract character names from text."""
        # Look for proper names and speech attribution
        character_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper names
            r'"[^"]*",?\s+([A-Z][a-z]+)\s+(?:said|asked|replied|whispered|shouted)',
            r'([A-Z][a-z]+)\s+(?:said|asked|replied|whispered|shouted),?\s+"[^"]*"'
        ]
        
        characters = set()
        for pattern in character_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if not self.is_common_word(match):
                    characters.add(match.strip())
        
        return list(characters)
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        location_patterns = [
            r'\bin\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        locations = set()
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if not self.is_common_word(match):
                    locations.add(match.strip())
        
        return list(locations)
    
    def calculate_dialogue_ratio(self, text: str) -> float:
        """Calculate ratio of dialogue to total text."""
        if not text:
            return 0.0
        return (text.count('"') + text.count("'")) / len(text)
    
    def calculate_action_ratio(self, text: str) -> float:
        """Calculate ratio of action content."""
        action_words = [
            'ran', 'jumped', 'grabbed', 'threw', 'hit', 'kicked', 'pushed',
            'pulled', 'rushed', 'dashed', 'sprinted', 'leaped', 'dove',
            'struck', 'slammed', 'crashed', 'exploded', 'fired', 'shot'
        ]
        
        words = text.lower().split()
        action_count = sum(1 for word in words if any(action in word for action in action_words))
        
        return action_count / len(words) if words else 0.0
    
    def calculate_description_ratio(self, text: str) -> float:
        """Calculate ratio of descriptive content."""
        descriptive_words = [
            'beautiful', 'ugly', 'tall', 'short', 'wide', 'narrow', 'bright',
            'dark', 'colorful', 'pale', 'smooth', 'rough', 'soft', 'hard',
            'warm', 'cold', 'loud', 'quiet', 'sweet', 'bitter', 'fragrant'
        ]
        
        words = text.lower().split()
        desc_count = sum(1 for word in words if any(desc in word for desc in descriptive_words))
        
        return desc_count / len(words) if words else 0.0
    
    def analyze_emotional_tone(self, text: str) -> str:
        """Analyze the emotional tone of the text."""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotional_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return 'neutral'
    
    def calculate_tension_level(self, text: str) -> float:
        """Calculate tension level of the scene."""
        tension_indicators = [
            'danger', 'threat', 'fear', 'panic', 'urgent', 'crisis',
            'conflict', 'fight', 'battle', 'struggle', 'desperate',
            'intense', 'dramatic', 'climax', 'suspense', 'tension'
        ]
        
        words = text.lower().split()
        tension_count = sum(1 for word in words if any(indicator in word for indicator in tension_indicators))
        
        tension_ratio = tension_count / len(words) if words else 0.0
        return min(tension_ratio * 10, 1.0)
    
    def classify_scene_type(self, text: str, dialogue_ratio: float, action_ratio: float) -> SceneType:
        """Classify the type of scene."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['introduction', 'meet', 'first time']):
            return SceneType.CHARACTER_INTRODUCTION
        
        if dialogue_ratio > 0.4:
            return SceneType.DIALOGUE_SCENE
        
        if action_ratio > 0.3:
            return SceneType.ACTION_SEQUENCE
        
        if any(word in text_lower for word in ['conflict', 'argument', 'fight']):
            return SceneType.CONFLICT
        
        if any(word in text_lower for word in ['resolution', 'solved', 'concluded']):
            return SceneType.RESOLUTION
        
        return SceneType.DIALOGUE_SCENE if dialogue_ratio > 0.2 else SceneType.OPENING
    
    def calculate_importance_score(self, scene_type: SceneType, tension_level: float, character_count: int) -> float:
        """Calculate importance score for the scene."""
        scene_importance = {
            SceneType.CLIMAX: 1.0,
            SceneType.CONFLICT: 0.9,
            SceneType.RESOLUTION: 0.8,
            SceneType.CHARACTER_INTRODUCTION: 0.7,
            SceneType.EMOTIONAL_BEAT: 0.7,
            SceneType.DIALOGUE_SCENE: 0.6,
            SceneType.ACTION_SEQUENCE: 0.6,
            SceneType.OPENING: 0.5,
            SceneType.TRANSITION: 0.3,
        }
        
        base_score = scene_importance.get(scene_type, 0.5)
        base_score += tension_level * 0.3
        base_score += min(character_count * 0.1, 0.2)
        
        return min(base_score, 1.0)
    
    def is_common_word(self, word: str) -> bool:
        """Check if word is a common word that's not a name."""
        common_words = {
            'The', 'This', 'That', 'There', 'Then', 'When', 'Where', 'What',
            'Who', 'Why', 'How', 'But', 'And', 'Or', 'So', 'Yet', 'For',
            'After', 'Before', 'During', 'While', 'Since', 'Until', 'With'
        }
        return word in common_words
    
    def chunk_document(self, content: str, title: str, source: str) -> List[SimpleChunk]:
        """Chunk document using enhanced scene analysis."""
        scenes = self.detect_scenes(content)
        chunks = []
        
        for i, scene in enumerate(scenes):
            # Analyze scene
            characters = self.extract_characters(scene)
            locations = self.extract_locations(scene)
            dialogue_ratio = self.calculate_dialogue_ratio(scene)
            action_ratio = self.calculate_action_ratio(scene)
            description_ratio = self.calculate_description_ratio(scene)
            emotional_tone = self.analyze_emotional_tone(scene)
            tension_level = self.calculate_tension_level(scene)
            scene_type = self.classify_scene_type(scene, dialogue_ratio, action_ratio)
            importance_score = self.calculate_importance_score(scene_type, tension_level, len(characters))
            
            # Create chunk metadata
            metadata = {
                "title": title,
                "source": source,
                "chunk_method": "enhanced_scene_level",
                "scene_index": i,
                "scene_type": scene_type.value,
                "characters": characters,
                "locations": locations,
                "emotional_tone": emotional_tone,
                "tension_level": tension_level,
                "dialogue_ratio": dialogue_ratio,
                "action_ratio": action_ratio,
                "description_ratio": description_ratio,
                "importance_score": importance_score,
                "chunk_type": "complete_scene"
            }
            
            chunk = SimpleChunk(
                content=scene,
                index=i,
                metadata=metadata,
                token_count=self.count_tokens(scene)
            )
            chunks.append(chunk)
        
        return chunks


def create_sample_novel_content() -> str:
    """Create sample novel content for demonstration."""
    return '''
# Chapter 1: The Mysterious Letter

The old Victorian house stood silently against the stormy sky, its weathered shutters creaking in the wind. Rain pelted the windows with increasing intensity, creating rivulets that traced intricate patterns down the glass. Emma had always found comfort in storms, but tonight felt different—ominous, as if the very air held secrets waiting to be unveiled.

She sat in her grandfather's study, surrounded by towering bookshelves that reached toward the coffered ceiling. The leather-bound volumes seemed to watch her with ancient wisdom, their spines bearing titles in languages she couldn't read.

***

"Emma, you must read this," her grandfather had said just hours before his passing. His voice, usually strong and commanding, had been reduced to a whisper. "The letter... it explains everything."

She remembered the weight of his hand on hers, the urgency in his eyes. Now, alone in the study, she finally understood why he had been so insistent.

"I don't understand," she whispered to herself, reading the letter's heading once more.

***

Meanwhile, across town, Detective Marcus Chen was investigating a break-in at the local museum. The thieves had been selective—only one item was missing from the extensive collection of artifacts.

"Any idea what was in the box?" Chen asked the curator, Dr. Sarah Mitchell.

"According to the donation records, it contained personal papers and a few family heirlooms," she replied, adjusting her glasses nervously.

Chen made a note in his pad. Something about this case felt off. Professional thieves didn't usually target items of sentimental value unless they knew something the rest of the world didn't.

***

The explosion rocked the building, sending debris flying in all directions. Sarah dove behind the concrete barrier just as another blast shook the ground beneath her feet.

"Move! Move!" she shouted to her team, gesturing frantically toward the exit.

Bullets whizzed past her head as she sprinted across the open courtyard. Her heart pounded as she leaped over fallen rubble, her training taking over.

Behind her, she could hear the enemy forces advancing. There was no time to think, only to act.
'''


def demonstrate_enhanced_chunking():
    """Demonstrate enhanced scene chunking."""
    
    print("=" * 80)
    print("ENHANCED SCENE-LEVEL CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample content
    content = create_sample_novel_content()
    
    print(f"\nSample Content Length: {len(content)} characters")
    print(f"Word Count: {len(content.split())} words")
    
    # Create enhanced chunker
    chunker = SimpleEnhancedChunker()
    
    # Chunk the document
    chunks = chunker.chunk_document(
        content=content,
        title="Sample Novel Chapter",
        source="demo.md"
    )
    
    print(f"\nEnhanced Chunks Created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata
        print(f"\n{'='*60}")
        print(f"CHUNK {i+1}")
        print(f"{'='*60}")
        print(f"Size: {chunk.token_count} tokens")
        print(f"Scene Type: {metadata.get('scene_type', 'unknown')}")
        print(f"Characters: {metadata.get('characters', [])}")
        print(f"Locations: {metadata.get('locations', [])}")
        print(f"Emotional Tone: {metadata.get('emotional_tone', 'unknown')}")
        print(f"Tension Level: {metadata.get('tension_level', 0.0):.3f}")
        print(f"Importance Score: {metadata.get('importance_score', 0.0):.3f}")
        print(f"Dialogue Ratio: {metadata.get('dialogue_ratio', 0.0):.3f}")
        print(f"Action Ratio: {metadata.get('action_ratio', 0.0):.3f}")
        print(f"Description Ratio: {metadata.get('description_ratio', 0.0):.3f}")
        print(f"\nContent Preview:")
        print("-" * 40)
        preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
        print(preview)


def demonstrate_content_analysis():
    """Demonstrate content type analysis."""
    
    print("\n" + "=" * 80)
    print("CONTENT TYPE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    samples = {
        "Dialogue Scene": '''
        "I can't believe you did that," Sarah said, her voice trembling with anger.
        
        "I had no choice," John replied, avoiding her gaze. "You don't understand the pressure I was under."
        
        "Pressure?" Sarah laughed bitterly. "What about the pressure I'm under now?"
        ''',
        
        "Action Scene": '''
        The explosion rocked the building, sending debris flying in all directions. Sarah dove behind the concrete barrier just as another blast shook the ground beneath her feet. Bullets whizzed past her head as she sprinted across the open courtyard.
        ''',
        
        "Description Scene": '''
        The ancient library stretched before them like a cathedral of knowledge. Towering shelves reached toward vaulted ceilings painted with faded frescoes. Dust motes danced in shafts of golden sunlight that streamed through tall, arched windows.
        '''
    }
    
    chunker = SimpleEnhancedChunker()
    
    for sample_name, sample_content in samples.items():
        print(f"\n{sample_name}:")
        print("-" * 40)
        
        dialogue_ratio = chunker.calculate_dialogue_ratio(sample_content)
        action_ratio = chunker.calculate_action_ratio(sample_content)
        description_ratio = chunker.calculate_description_ratio(sample_content)
        emotional_tone = chunker.analyze_emotional_tone(sample_content)
        tension_level = chunker.calculate_tension_level(sample_content)
        scene_type = chunker.classify_scene_type(sample_content, dialogue_ratio, action_ratio)
        
        print(f"Dialogue Ratio: {dialogue_ratio:.3f}")
        print(f"Action Ratio: {action_ratio:.3f}")
        print(f"Description Ratio: {description_ratio:.3f}")
        print(f"Emotional Tone: {emotional_tone}")
        print(f"Tension Level: {tension_level:.3f}")
        print(f"Scene Type: {scene_type.value}")


def save_analysis_results():
    """Save analysis results to JSON."""
    
    print("\n" + "=" * 80)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 80)
    
    chunker = SimpleEnhancedChunker()
    content = create_sample_novel_content()
    
    chunks = chunker.chunk_document(
        content=content,
        title="Sample Novel Chapter",
        source="demo.md"
    )
    
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
            "metadata": chunk.metadata
        }
        results["chunks"].append(chunk_data)
    
    # Save to file
    with open("enhanced_chunking_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Analysis results saved to: enhanced_chunking_analysis.json")
    print(f"Total chunks analyzed: {len(chunks)}")


def main():
    """Main demonstration function."""
    
    print("Starting Enhanced Scene-Level Chunking Demonstration...")
    
    try:
        demonstrate_enhanced_chunking()
        demonstrate_content_analysis()
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
        
        print("\nNext Steps for Implementation:")
        print("• Enable enhanced chunking in ingestion configuration")
        print("• Set use_enhanced_scene_chunking=True in IngestionConfig")
        print("• Adjust chunk size parameters based on content type")
        print("• Test with your own novel content")
        print("• Monitor generation quality improvements")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()