"""
Novel-optimized knowledge graph builder for literary content.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import asyncio
import re
import spacy
from collections import defaultdict

from graphiti_core import Graphiti
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of narrative chunks."""
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    DESCRIPTION = "description"
    ACTION = "action"
    INTERNAL_MONOLOGUE = "internal_monologue"
    TRANSITION = "transition"


class EmotionalTone(Enum):
    """Emotional tones for scenes."""
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    TENSE = "tense"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    PEACEFUL = "peaceful"
    DRAMATIC = "dramatic"
    HUMOROUS = "humorous"


@dataclass
class NarrativeElement:
    """Represents a narrative element in the story."""
    name: str
    element_type: str  # character, location, object, concept
    first_mention_chunk: int
    appearances: List[int]  # chunk indices where this element appears
    relationships: List[str]  # relationships to other elements
    description: Optional[str] = None
    significance_score: float = 0.0


@dataclass
class SceneMetadata:
    """Metadata for a scene or narrative section."""
    scene_id: str
    chapter: Optional[str]
    location: Optional[str]
    characters_present: List[str]
    time_of_day: Optional[str]
    emotional_tone: Optional[EmotionalTone]
    plot_significance: float  # 0.0 to 1.0
    themes: List[str]


class NovelGraphBuilder:
    """Builds knowledge graph optimized for novel content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize novel graph builder."""
        self.config = config or self._default_config()
        self.graph_client = None  # Initialize based on your graph implementation
        self._initialized = False
        self.narrative_elements = {}  # Track all story elements
        self.chapter_boundaries = []  # Track chapter divisions
        self.scene_metadata = {}  # Track scene information
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for novel processing."""
        return {
            "max_chunk_length": 4000,  # Longer for narrative flow
            "preserve_dialogue_integrity": True,
            "preserve_scene_boundaries": True,
            "min_character_mentions": 2,  # Minimum mentions to consider as character
            "emotional_analysis": True,
            "theme_extraction": True,
            "relationship_tracking": True,
            "temporal_tracking": True
        }
    
    async def initialize(self):
        """Initialize graph client and NLP models."""
        if not self._initialized:
            # Initialize your graph client here
            # await self.graph_client.initialize()
            
            if nlp is None:
                logger.warning("spaCy model not available. Some features will be limited.")
            
            self._initialized = True
    
    async def add_novel_to_graph(
        self,
        chunks: List[DocumentChunk],
        novel_title: str,
        author: str,
        novel_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add novel chunks to knowledge graph with narrative awareness.
        
        Args:
            chunks: List of document chunks
            novel_title: Title of the novel
            author: Author name
            novel_metadata: Additional metadata (genre, publication_date, etc.)
        
        Returns:
            Processing results with narrative analytics
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"episodes_created": 0, "errors": [], "narrative_analysis": {}}
        
        logger.info(f"Processing novel '{novel_title}' by {author} with {len(chunks)} chunks")
        
        # Step 1: Analyze narrative structure
        narrative_analysis = await self._analyze_narrative_structure(chunks, novel_title)
        
        # Step 2: Extract and track narrative elements
        enriched_chunks = await self._extract_narrative_elements(chunks)
        
        # Step 3: Create narrative-aware episodes
        episodes_created = 0
        errors = []
        
        for i, chunk in enumerate(enriched_chunks):
            try:
                episode_result = await self._create_narrative_episode(
                    chunk, novel_title, author, i, narrative_analysis
                )
                
                if episode_result["success"]:
                    episodes_created += 1
                    logger.info(f"✅ Created narrative episode {i+1}/{len(chunks)}")
                else:
                    errors.append(episode_result["error"])
                
                # Gentle delay to avoid overwhelming the graph system
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.3)
                    
            except Exception as e:
                error_msg = f"Failed to create episode for chunk {i}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Step 4: Create relationship episodes
        relationship_episodes = await self._create_relationship_episodes(novel_title)
        episodes_created += relationship_episodes
        
        return {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors,
            "narrative_analysis": narrative_analysis,
            "characters_discovered": len(narrative_analysis.get("characters", {})),
            "locations_discovered": len(narrative_analysis.get("locations", {})),
            "themes_identified": len(narrative_analysis.get("themes", []))
        }
    
    async def _analyze_narrative_structure(
        self, 
        chunks: List[DocumentChunk], 
        novel_title: str
    ) -> Dict[str, Any]:
        """Analyze the overall narrative structure of the novel."""
        analysis = {
            "characters": {},
            "locations": {},
            "themes": [],
            "emotional_arc": [],
            "temporal_markers": [],
            "chapter_structure": []
        }
        
        # Track characters across all chunks
        character_tracker = defaultdict(list)
        location_tracker = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            # Detect chapter boundaries
            if self._is_chapter_boundary(chunk.content):
                chapter_info = self._extract_chapter_info(chunk.content)
                analysis["chapter_structure"].append({
                    "chunk_index": i,
                    "chapter_info": chapter_info
                })
            
            # Extract characters (improved logic)
            characters = self._extract_characters_advanced(chunk.content, i)
            for char in characters:
                character_tracker[char].append(i)
            
            # Extract locations
            locations = self._extract_locations_advanced(chunk.content, i)
            for loc in locations:
                location_tracker[loc].append(i)
            
            # Analyze emotional tone
            emotional_tone = self._analyze_emotional_tone(chunk.content)
            analysis["emotional_arc"].append({
                "chunk_index": i,
                "tone": emotional_tone,
                "intensity": self._calculate_emotional_intensity(chunk.content)
            })
        
        # Build character profiles
        for char_name, appearances in character_tracker.items():
            if len(appearances) >= self.config["min_character_mentions"]:
                analysis["characters"][char_name] = {
                    "appearances": appearances,
                    "first_mention": min(appearances),
                    "significance_score": self._calculate_character_significance(appearances, len(chunks)),
                    "relationships": self._detect_character_relationships(char_name, chunks, appearances)
                }
        
        # Build location profiles
        for loc_name, appearances in location_tracker.items():
            analysis["locations"][loc_name] = {
                "appearances": appearances,
                "first_mention": min(appearances),
                "significance_score": self._calculate_location_significance(appearances, len(chunks))
            }
        
        # Extract themes (placeholder - would use more sophisticated analysis)
        analysis["themes"] = self._extract_themes(chunks)
        
        return analysis
    
    def _is_chapter_boundary(self, content: str) -> bool:
        """Detect if content contains a chapter boundary."""
        chapter_patterns = [
            r"^Chapter \d+",
            r"^CHAPTER \d+",
            r"^Part \d+",
            r"^Book \d+",
            r"^\d+\.",
            r"^---+",
            r"^\*\*\*+"
        ]
        
        lines = content.strip().split('\n')
        first_line = lines[0] if lines else ""
        
        for pattern in chapter_patterns:
            if re.match(pattern, first_line.strip()):
                return True
        
        return False
    
    def _extract_characters_advanced(self, content: str, chunk_index: int) -> List[str]:
        """Extract character names using advanced NLP techniques."""
        characters = []
        
        if nlp:
            doc = nlp(content)
            
            # Extract named entities that are likely to be people
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Filter out obvious non-characters
                    name = ent.text.strip()
                    if self._is_likely_character_name(name):
                        characters.append(name)
        else:
            # Fallback to pattern matching
            characters = self._extract_characters_pattern_matching(content)
        
        # Also look for quoted speech (indicates character presence)
        quoted_speakers = self._extract_speakers_from_dialogue(content)
        characters.extend(quoted_speakers)
        
        return list(set(characters))
    
    def _is_likely_character_name(self, name: str) -> bool:
        """Determine if a name is likely to be a character."""
        # Filter out author names, publisher names, etc.
        exclude_patterns = [
            r"^(Mr\.|Mrs\.|Dr\.|Prof\.)",  # Titles without names
            r"^[A-Z]{2,}$",  # All caps (likely acronyms)
            r"\d",  # Contains numbers
            r"^(Chapter|Part|Book|Section)"  # Structure words
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, name):
                return False
        
        # Must be 2+ words or single word with proper capitalization
        words = name.split()
        if len(words) == 1:
            return name.istitle() and len(name) > 2
        
        return len(words) <= 4 and all(word.istitle() for word in words)
    
    def _extract_speakers_from_dialogue(self, content: str) -> List[str]:
        """Extract character names from dialogue attribution."""
        speakers = []
        
        # Look for patterns like: "Hello," said John. or John said, "Hello"
        dialogue_patterns = [
            r'"[^"]*,"\s*(?:said|asked|replied|whispered|shouted)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|asked|replied|whispered|shouted),?\s*"',
        ]
        
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, content)
            speakers.extend(matches)
        
        return speakers
    
    def _analyze_emotional_tone(self, content: str) -> EmotionalTone:
        """Analyze the emotional tone of a text chunk."""
        # Simplified emotional analysis - in practice, use sentiment analysis
        emotional_keywords = {
            EmotionalTone.JOYFUL: ["happy", "joy", "laugh", "smile", "celebrate", "delight"],
            EmotionalTone.MELANCHOLIC: ["sad", "melancholy", "sorrow", "weep", "mourn", "grief"],
            EmotionalTone.TENSE: ["tense", "anxious", "nervous", "worried", "fear", "panic"],
            EmotionalTone.ROMANTIC: ["love", "romance", "heart", "kiss", "embrace", "tender"],
            EmotionalTone.MYSTERIOUS: ["mystery", "strange", "unknown", "hidden", "secret", "enigma"],
            EmotionalTone.DRAMATIC: ["dramatic", "intense", "conflict", "confrontation", "climax"],
            EmotionalTone.HUMOROUS: ["funny", "humor", "joke", "laugh", "amusing", "witty"]
        }
        
        content_lower = content.lower()
        tone_scores = {}
        
        for tone, keywords in emotional_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                tone_scores[tone] = score
        
        if tone_scores:
            return max(tone_scores, key=tone_scores.get)
        
        return EmotionalTone.PEACEFUL  # Default
    
    def _calculate_character_significance(self, appearances: List[int], total_chunks: int) -> float:
        """Calculate character significance based on appearance frequency and distribution."""
        frequency = len(appearances) / total_chunks
        
        # Calculate distribution spread
        if len(appearances) > 1:
            spread = (max(appearances) - min(appearances)) / total_chunks
        else:
            spread = 0.0
        
        # Characters that appear frequently and throughout the story are more significant
        significance = (frequency * 0.7) + (spread * 0.3)
        
        return min(significance, 1.0)
    
    def _prepare_narrative_episode_content(
        self,
        chunk: DocumentChunk,
        novel_title: str,
        chunk_type: ChunkType,
        scene_metadata: Optional[SceneMetadata] = None
    ) -> str:
        """Prepare episode content with narrative context preservation."""
        max_length = self.config["max_chunk_length"]
        content = chunk.content
        
        if len(content) > max_length:
            # Smart truncation that preserves narrative integrity
            if chunk_type == ChunkType.DIALOGUE and self.config["preserve_dialogue_integrity"]:
                content = self._truncate_preserving_dialogue(content, max_length)
            else:
                content = self._truncate_preserving_narrative_flow(content, max_length)
        
        # Add narrative context
        context_lines = [f"[Novel: {novel_title}]"]
        
        if scene_metadata:
            if scene_metadata.chapter:
                context_lines.append(f"[Chapter: {scene_metadata.chapter}]")
            if scene_metadata.location:
                context_lines.append(f"[Location: {scene_metadata.location}]")
            if scene_metadata.characters_present:
                context_lines.append(f"[Characters: {', '.join(scene_metadata.characters_present[:3])}]")
        
        context_lines.append(f"[Type: {chunk_type.value}]")
        context = "\n".join(context_lines)
        
        return f"{context}\n\n{content}"
    
    def _truncate_preserving_dialogue(self, content: str, max_length: int) -> str:
        """Truncate content while preserving complete dialogue exchanges."""
        if len(content) <= max_length:
            return content
        
        # Find dialogue boundaries
        dialogue_pattern = r'("[^"]*")'
        dialogue_matches = list(re.finditer(dialogue_pattern, content))
        
        if not dialogue_matches:
            return self._truncate_preserving_narrative_flow(content, max_length)
        
        # Find the last complete dialogue within the limit
        for i in reversed(range(len(dialogue_matches))):
            dialogue_end = dialogue_matches[i].end()
            if dialogue_end <= max_length * 0.9:  # Leave some buffer
                # Find the end of the paragraph containing this dialogue
                remaining_content = content[dialogue_end:]
                paragraph_end = remaining_content.find('\n\n')
                
                if paragraph_end != -1 and dialogue_end + paragraph_end <= max_length:
                    return content[:dialogue_end + paragraph_end] + "\n[CONTINUED...]"
                else:
                    return content[:dialogue_end] + "\n[CONTINUED...]"
        
        # Fallback to normal truncation
        return self._truncate_preserving_narrative_flow(content, max_length)
    
    def _truncate_preserving_narrative_flow(self, content: str, max_length: int) -> str:
        """Truncate content while preserving narrative flow."""
        if len(content) <= max_length:
            return content
        
        # Try to end at paragraph boundary
        truncated = content[:max_length]
        paragraph_break = truncated.rfind('\n\n')
        
        if paragraph_break > max_length * 0.7:
            return content[:paragraph_break] + "\n[CONTINUED...]"
        
        # Try to end at sentence boundary
        sentence_endings = ['. ', '! ', '? ']
        best_end = -1
        
        for ending in sentence_endings:
            last_occurrence = truncated.rfind(ending)
            if last_occurrence > best_end:
                best_end = last_occurrence
        
        if best_end > max_length * 0.7:
            return content[:best_end + 2] + "[CONTINUED...]"
        
        # Fallback: hard truncation
        return truncated + "...[CONTINUED...]"
    
    async def _create_narrative_episode(
        self,
        chunk: DocumentChunk,
        novel_title: str,
        author: str,
        chunk_index: int,
        narrative_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a narrative-aware episode for the knowledge graph."""
        try:
            # Determine chunk type and scene metadata
            chunk_type = self._classify_chunk_type(chunk.content)
            scene_metadata = self._extract_scene_metadata(chunk, chunk_index, narrative_analysis)
            
            # Prepare episode content
            episode_content = self._prepare_narrative_episode_content(
                chunk, novel_title, chunk_type, scene_metadata
            )
            
            # Create episode ID that reflects narrative structure
            chapter = scene_metadata.chapter if scene_metadata else None
            episode_id = f"{novel_title}_{chapter or 'unknown'}_{chunk_index}_{chunk_type.value}"
            
            # Enhanced metadata for narrative context
            metadata = {
                "novel_title": novel_title,
                "author": author,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type.value,
                "original_length": len(chunk.content),
                "processed_length": len(episode_content),
                **chunk.metadata
            }
            
            if scene_metadata:
                metadata.update({
                    "chapter": scene_metadata.chapter,
                    "location": scene_metadata.location,
                    "characters_present": scene_metadata.characters_present,
                    "emotional_tone": scene_metadata.emotional_tone.value if scene_metadata.emotional_tone else None,
                    "plot_significance": scene_metadata.plot_significance,
                    "themes": scene_metadata.themes
                })
            
            # Add to graph (placeholder - implement based on your graph system)
            # await self.graph_client.add_episode(
            #     episode_id=episode_id,
            #     content=episode_content,
            #     source=f"{novel_title} by {author}",
            #     timestamp=datetime.now(timezone.utc),
            #     metadata=metadata
            # )
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _classify_chunk_type(self, content: str) -> ChunkType:
        """Classify the type of narrative content."""
        # Count dialogue markers
        dialogue_count = content.count('"')
        
        # Count action words
        action_words = ["ran", "jumped", "grabbed", "threw", "hit", "moved", "walked"]
        action_count = sum(1 for word in action_words if word.lower() in content.lower())
        
        # Count descriptive words
        descriptive_words = ["beautiful", "dark", "tall", "ancient", "mysterious", "bright"]
        descriptive_count = sum(1 for word in descriptive_words if word.lower() in content.lower())
        
        # Count internal thought indicators
        internal_indicators = ["thought", "wondered", "realized", "remembered", "felt that"]
        internal_count = sum(1 for indicator in internal_indicators if indicator.lower() in content.lower())
        
        # Classify based on predominant features
        if dialogue_count > 4:  # Significant dialogue
            return ChunkType.DIALOGUE
        elif internal_count > 0:
            return ChunkType.INTERNAL_MONOLOGUE
        elif action_count > descriptive_count:
            return ChunkType.ACTION
        elif descriptive_count > 2:
            return ChunkType.DESCRIPTION
        else:
            return ChunkType.NARRATION
    
    # Additional methods would continue here...
    # Including: _extract_themes, _create_relationship_episodes, etc.
    
    async def close(self):
        """Close graph client and clean up resources."""
        if self._initialized and self.graph_client:
            # await self.graph_client.close()
            self._initialized = False


# Factory function
def create_novel_graph_builder(config: Optional[Dict[str, Any]] = None) -> NovelGraphBuilder:
    """Create a novel-optimized graph builder instance."""
    return NovelGraphBuilder(config)