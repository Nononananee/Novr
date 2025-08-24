"""
Enhanced Chunking Strategies for Complex Narrative Content
Optimizes chunking for better success rates with real-world novel content.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, Counter

# Import from existing modules
from .chunking_strategies import NovelChunker, NovelChunk, ChunkingStrategy
from agent.error_handling_utils import robust_error_handler, ErrorSeverity
from agent.enhanced_memory_monitor import monitor_operation_memory

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enhanced content type detection."""
    DIALOGUE = "dialogue"
    NARRATIVE = "narrative" 
    DESCRIPTION = "description"
    ACTION = "action"
    INTERNAL_MONOLOGUE = "internal_monologue"
    TRANSITION = "transition"
    FLASHBACK = "flashback"
    SETTING_DESCRIPTION = "setting_description"


class ChunkPriority(Enum):
    """Priority levels for chunks."""
    CRITICAL = "critical"      # Essential plot points, character development
    HIGH = "high"             # Important dialogue, key scenes
    MEDIUM = "medium"         # Supporting narrative, descriptions
    LOW = "low"              # Background details, transitions


@dataclass
class ChunkMetadata:
    """Enhanced metadata for chunks."""
    characters: Set[str] = field(default_factory=set)
    emotions: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    time_indicators: List[str] = field(default_factory=list)
    tension_level: float = 0.0
    pov_character: Optional[str] = None
    dialogue_ratio: float = 0.0
    action_ratio: float = 0.0
    description_ratio: float = 0.0


@dataclass
class EnhancedChunk:
    """Enhanced chunk with comprehensive metadata."""
    id: str
    content: str
    content_type: ContentType
    priority: ChunkPriority
    metadata: ChunkMetadata
    token_count: int
    quality_score: float
    coherence_score: float
    dependencies: List[str] = field(default_factory=list)
    semantic_embedding: Optional[List[float]] = None


class AdaptiveChunkingStrategy:
    """Adaptive chunking that selects optimal strategy based on content analysis."""
    
    def __init__(self):
        self.base_chunker = NovelChunker()
        self.chunk_history = []
        self.strategy_performance = defaultdict(float)
        
        # Content analysis patterns
        self.dialogue_patterns = [
            r'"[^"]*"',  # Quoted dialogue
            r"'[^']*'",  # Single quoted dialogue
            r'\b(said|asked|replied|whispered|shouted|exclaimed|murmured)\b'
        ]
        
        self.action_patterns = [
            r'\b(ran|walked|grabbed|threw|jumped|hit|kicked|pushed|pulled)\b',
            r'\b(suddenly|quickly|slowly|carefully|frantically)\b',
            r'\b\w+(ed|ing)\b'  # Past tense and present participle
        ]
        
        self.emotion_patterns = [
            r'\b(angry|sad|happy|excited|nervous|afraid|confident|worried)\b',
            r'\b(smiled|frowned|laughed|cried|sighed|gasped)\b'
        ]
        
        logger.info("Adaptive chunking strategy initialized")
    
    @robust_error_handler("adaptive_chunking", ErrorSeverity.HIGH, max_retries=2)
    async def chunk_content(self, 
                          content: str, 
                          max_chunk_size: int = 800,
                          target_overlap: float = 0.1,
                          preserve_context: bool = True) -> List[EnhancedChunk]:
        """
        Adaptively chunk content using optimal strategy.
        
        Args:
            content: Content to chunk
            max_chunk_size: Maximum tokens per chunk
            target_overlap: Overlap ratio between chunks
            preserve_context: Whether to preserve narrative context
            
        Returns:
            List of enhanced chunks
        """
        async with await monitor_operation_memory("adaptive_chunking") as profiler:
            # Analyze content to determine optimal strategy
            content_analysis = await self._analyze_content(content)
            
            # Select optimal chunking strategy
            optimal_strategy = self._select_chunking_strategy(content_analysis)
            
            # Perform chunking with selected strategy
            chunks = await self._chunk_with_strategy(
                content, 
                optimal_strategy, 
                max_chunk_size,
                content_analysis
            )
            
            # Post-process chunks for optimization
            optimized_chunks = await self._optimize_chunks(
                chunks, 
                target_overlap, 
                preserve_context
            )
            
            # Record performance for adaptive learning
            self._record_chunking_performance(optimal_strategy, optimized_chunks)
            
            return optimized_chunks
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics to determine optimal chunking strategy."""
        analysis = {
            "total_length": len(content),
            "word_count": len(content.split()),
            "paragraph_count": len(content.split('\n\n')),
            "sentence_count": len(content.split('.')),
            "dialogue_ratio": 0.0,
            "action_ratio": 0.0,
            "description_ratio": 0.0,
            "character_count": 0,
            "time_indicators": [],
            "complexity_score": 0.0,
            "narrative_type": "unknown"
        }
        
        # Calculate dialogue ratio
        dialogue_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                           for pattern in self.dialogue_patterns)
        analysis["dialogue_ratio"] = dialogue_count / max(analysis["sentence_count"], 1)
        
        # Calculate action ratio
        action_count = sum(len(re.findall(pattern, content, re.IGNORECASE))
                          for pattern in self.action_patterns)
        analysis["action_ratio"] = action_count / max(analysis["word_count"], 1)
        
        # Detect characters (proper nouns)
        characters = set(re.findall(r'\b[A-Z][a-z]+\b', content))
        common_words = {'The', 'A', 'An', 'This', 'That', 'He', 'She', 'It', 'They', 'We', 'I'}
        characters = characters - common_words
        analysis["character_count"] = len(characters)
        analysis["characters"] = characters
        
        # Detect time indicators
        time_patterns = [
            r'\b(morning|afternoon|evening|night|dawn|dusk)\b',
            r'\b(yesterday|today|tomorrow|later|earlier|soon)\b',
            r'\b(hours?|minutes?|days?|weeks?|months?|years?) (ago|later)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["time_indicators"].extend(matches)
        
        # Calculate complexity score
        complexity_factors = [
            analysis["character_count"] / 10,  # More characters = more complex
            analysis["dialogue_ratio"],
            len(analysis["time_indicators"]) / 5,
            analysis["paragraph_count"] / 10
        ]
        analysis["complexity_score"] = sum(complexity_factors) / len(complexity_factors)
        
        # Determine narrative type
        if analysis["dialogue_ratio"] > 0.5:
            analysis["narrative_type"] = "dialogue_heavy"
        elif analysis["action_ratio"] > 0.3:
            analysis["narrative_type"] = "action_oriented"
        elif analysis["character_count"] > 5:
            analysis["narrative_type"] = "character_driven"
        else:
            analysis["narrative_type"] = "descriptive"
        
        return analysis
    
    def _select_chunking_strategy(self, content_analysis: Dict[str, Any]) -> str:
        """Select optimal chunking strategy based on content analysis."""
        
        # Strategy selection based on content characteristics
        if content_analysis["dialogue_ratio"] > 0.4:
            return "dialogue_preserving"
        elif content_analysis["action_ratio"] > 0.3:
            return "action_oriented"
        elif content_analysis["character_count"] > 5:
            return "character_focused"
        elif content_analysis["complexity_score"] > 0.7:
            return "scene_aware"
        else:
            return "semantic_chunking"
    
    async def _chunk_with_strategy(self, 
                                 content: str, 
                                 strategy: str, 
                                 max_chunk_size: int,
                                 content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Chunk content using the selected strategy."""
        
        if strategy == "dialogue_preserving":
            return await self._dialogue_preserving_chunking(content, max_chunk_size, content_analysis)
        elif strategy == "action_oriented":
            return await self._action_oriented_chunking(content, max_chunk_size, content_analysis)
        elif strategy == "character_focused":
            return await self._character_focused_chunking(content, max_chunk_size, content_analysis)
        elif strategy == "scene_aware":
            return await self._scene_aware_chunking(content, max_chunk_size, content_analysis)
        else:
            return await self._semantic_chunking(content, max_chunk_size, content_analysis)
    
    async def _dialogue_preserving_chunking(self, 
                                          content: str, 
                                          max_chunk_size: int,
                                          content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Chunking that preserves dialogue integrity."""
        chunks = []
        
        # Split content into dialogue and non-dialogue sections
        dialogue_sections = []
        narrative_sections = []
        
        # Find dialogue blocks
        dialogue_pattern = r'("[^"]*"[^"]*?(?:said|asked|replied|whispered)[^.]*\.)'
        dialogue_matches = list(re.finditer(dialogue_pattern, content, re.DOTALL))
        
        current_pos = 0
        for match in dialogue_matches:
            # Add narrative before dialogue
            if match.start() > current_pos:
                narrative_text = content[current_pos:match.start()].strip()
                if narrative_text:
                    narrative_sections.append(narrative_text)
            
            # Add dialogue
            dialogue_text = match.group().strip()
            dialogue_sections.append(dialogue_text)
            current_pos = match.end()
        
        # Add remaining narrative
        if current_pos < len(content):
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                narrative_sections.append(remaining_text)
        
        # Create chunks preserving dialogue integrity
        chunk_id = 0
        
        # Process dialogue sections
        for dialogue in dialogue_sections:
            if len(dialogue.split()) <= max_chunk_size:
                chunk = await self._create_enhanced_chunk(
                    chunk_id, dialogue, ContentType.DIALOGUE, content_analysis
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large dialogue blocks carefully
                sub_chunks = await self._split_large_dialogue(dialogue, max_chunk_size, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        # Process narrative sections
        for narrative in narrative_sections:
            if len(narrative.split()) <= max_chunk_size:
                chunk = await self._create_enhanced_chunk(
                    chunk_id, narrative, ContentType.NARRATIVE, content_analysis
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large narrative blocks
                sub_chunks = await self._split_large_narrative(narrative, max_chunk_size, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return chunks
    
    async def _action_oriented_chunking(self, 
                                      content: str, 
                                      max_chunk_size: int,
                                      content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Chunking optimized for action sequences."""
        chunks = []
        
        # Split content by action beats
        action_beats = await self._detect_action_beats(content)
        
        chunk_id = 0
        for beat in action_beats:
            if len(beat["content"].split()) <= max_chunk_size:
                chunk = await self._create_enhanced_chunk(
                    chunk_id, beat["content"], ContentType.ACTION, content_analysis
                )
                chunk.metadata.tension_level = beat.get("tension_level", 0.5)
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large action sequences carefully
                sub_chunks = await self._split_action_sequence(beat["content"], max_chunk_size, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return chunks
    
    async def _character_focused_chunking(self, 
                                        content: str, 
                                        max_chunk_size: int,
                                        content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Chunking that groups content by character presence."""
        chunks = []
        
        # Identify character scenes
        characters = content_analysis.get("characters", set())
        character_scenes = await self._identify_character_scenes(content, characters)
        
        chunk_id = 0
        for scene in character_scenes:
            if len(scene["content"].split()) <= max_chunk_size:
                chunk = await self._create_enhanced_chunk(
                    chunk_id, scene["content"], ContentType.NARRATIVE, content_analysis
                )
                chunk.metadata.characters = scene.get("characters", set())
                chunk.metadata.pov_character = scene.get("pov_character")
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large character scenes
                sub_chunks = await self._split_character_scene(scene, max_chunk_size, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return chunks
    
    async def _scene_aware_chunking(self, 
                                  content: str, 
                                  max_chunk_size: int,
                                  content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Advanced chunking that respects scene boundaries."""
        chunks = []
        
        # Detect scene breaks
        scene_breaks = await self._detect_scene_boundaries(content)
        
        chunk_id = 0
        current_pos = 0
        
        for break_pos in scene_breaks + [len(content)]:
            scene_content = content[current_pos:break_pos].strip()
            
            if scene_content and len(scene_content.split()) > 10:  # Minimum scene length
                if len(scene_content.split()) <= max_chunk_size:
                    chunk = await self._create_enhanced_chunk(
                        chunk_id, scene_content, ContentType.NARRATIVE, content_analysis
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    # Split large scenes intelligently
                    sub_chunks = await self._split_large_scene(scene_content, max_chunk_size, chunk_id)
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
            
            current_pos = break_pos
        
        return chunks
    
    async def _semantic_chunking(self, 
                               content: str, 
                               max_chunk_size: int,
                               content_analysis: Dict[str, Any]) -> List[EnhancedChunk]:
        """Semantic chunking for descriptive content."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk.split()) <= max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk = await self._create_enhanced_chunk(
                        chunk_id, current_chunk, ContentType.DESCRIPTION, content_analysis
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with current paragraph
                if len(paragraph.split()) <= max_chunk_size:
                    current_chunk = paragraph
                else:
                    # Split large paragraph
                    sub_chunks = await self._split_large_paragraph(paragraph, max_chunk_size, chunk_id)
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk:
            chunk = await self._create_enhanced_chunk(
                chunk_id, current_chunk, ContentType.DESCRIPTION, content_analysis
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_enhanced_chunk(self, 
                                   chunk_id: int, 
                                   content: str, 
                                   content_type: ContentType,
                                   content_analysis: Dict[str, Any]) -> EnhancedChunk:
        """Create an enhanced chunk with comprehensive metadata."""
        
        # Extract metadata from content
        metadata = ChunkMetadata()
        
        # Extract characters
        characters = set(re.findall(r'\b[A-Z][a-z]+\b', content))
        common_words = {'The', 'A', 'An', 'This', 'That', 'He', 'She', 'It', 'They', 'We', 'I'}
        metadata.characters = characters - common_words
        
        # Extract emotions
        emotions = set()
        for pattern in self.emotion_patterns:
            emotions.update(re.findall(pattern, content, re.IGNORECASE))
        metadata.emotions = emotions
        
        # Calculate content ratios
        total_words = len(content.split())
        if total_words > 0:
            dialogue_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                               for pattern in self.dialogue_patterns)
            action_count = sum(len(re.findall(pattern, content, re.IGNORECASE))
                             for pattern in self.action_patterns)
            
            metadata.dialogue_ratio = dialogue_count / total_words
            metadata.action_ratio = action_count / total_words
            metadata.description_ratio = 1.0 - metadata.dialogue_ratio - metadata.action_ratio
        
        # Calculate quality and coherence scores
        quality_score = await self._calculate_chunk_quality(content, content_type)
        coherence_score = await self._calculate_chunk_coherence(content)
        
        # Determine priority
        priority = self._determine_chunk_priority(content, metadata, content_type)
        
        return EnhancedChunk(
            id=f"enhanced_chunk_{chunk_id}",
            content=content,
            content_type=content_type,
            priority=priority,
            metadata=metadata,
            token_count=len(content.split()),
            quality_score=quality_score,
            coherence_score=coherence_score
        )
    
    async def _calculate_chunk_quality(self, content: str, content_type: ContentType) -> float:
        """Calculate quality score for a chunk."""
        quality_score = 0.8  # Base quality
        
        # Length factor
        word_count = len(content.split())
        if 50 <= word_count <= 500:
            quality_score += 0.1
        
        # Content type specific bonuses
        if content_type == ContentType.DIALOGUE:
            if '"' in content or "'" in content:
                quality_score += 0.05
        elif content_type == ContentType.ACTION:
            action_words = len(re.findall(r'\b(ran|walked|grabbed|threw|jumped)\b', content, re.IGNORECASE))
            if action_words > 0:
                quality_score += min(0.1, action_words * 0.02)
        
        # Specificity bonus
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', content))
        quality_score += min(0.1, proper_nouns * 0.01)
        
        return min(1.0, quality_score)
    
    async def _calculate_chunk_coherence(self, content: str) -> float:
        """Calculate coherence score for a chunk."""
        sentences = content.split('.')
        if len(sentences) < 2:
            return 0.9
        
        coherence_score = 0.9
        
        # Check for connecting words
        connecting_words = ['then', 'after', 'before', 'while', 'when', 'as', 'but', 'however']
        has_connections = any(word in content.lower() for word in connecting_words)
        
        if has_connections:
            coherence_score += 0.05
        
        # Check for proper flow
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            if 5 <= avg_length <= 25:  # Good sentence length range
                coherence_score += 0.05
        
        return min(1.0, coherence_score)
    
    def _determine_chunk_priority(self, 
                                content: str, 
                                metadata: ChunkMetadata, 
                                content_type: ContentType) -> ChunkPriority:
        """Determine priority level for a chunk."""
        
        # High priority indicators
        high_priority_words = ['important', 'crucial', 'critical', 'key', 'main', 'central']
        has_high_priority = any(word in content.lower() for word in high_priority_words)
        
        # Character importance
        has_many_characters = len(metadata.characters) > 2
        
        # Dialogue importance
        has_dialogue = metadata.dialogue_ratio > 0.3
        
        # Determine priority
        if has_high_priority or (has_dialogue and has_many_characters):
            return ChunkPriority.CRITICAL
        elif has_dialogue or content_type == ContentType.ACTION:
            return ChunkPriority.HIGH
        elif len(metadata.characters) > 0:
            return ChunkPriority.MEDIUM
        else:
            return ChunkPriority.LOW
    
    async def _optimize_chunks(self, 
                             chunks: List[EnhancedChunk], 
                             target_overlap: float,
                             preserve_context: bool) -> List[EnhancedChunk]:
        """Post-process chunks for optimization."""
        
        if not preserve_context or target_overlap <= 0:
            return chunks
        
        # Add overlaps between related chunks
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            current_chunk = chunk
            
            # Add context from previous chunk if beneficial
            if i > 0 and target_overlap > 0:
                prev_chunk = chunks[i-1]
                overlap_content = await self._create_overlap_content(prev_chunk, current_chunk, target_overlap)
                
                if overlap_content:
                    current_chunk.content = overlap_content + "\n\n" + current_chunk.content
                    current_chunk.token_count = len(current_chunk.content.split())
            
            optimized_chunks.append(current_chunk)
        
        return optimized_chunks
    
    async def _create_overlap_content(self, 
                                    prev_chunk: EnhancedChunk, 
                                    current_chunk: EnhancedChunk, 
                                    target_overlap: float) -> str:
        """Create appropriate overlap content between chunks."""
        
        # Calculate overlap size
        target_words = int(current_chunk.token_count * target_overlap)
        
        # Extract last part of previous chunk
        prev_words = prev_chunk.content.split()
        if len(prev_words) > target_words:
            overlap_words = prev_words[-target_words:]
            return " ".join(overlap_words)
        
        return ""
    
    def _record_chunking_performance(self, strategy: str, chunks: List[EnhancedChunk]):
        """Record performance for adaptive learning."""
        
        # Calculate performance metrics
        avg_quality = sum(chunk.quality_score for chunk in chunks) / len(chunks) if chunks else 0
        avg_coherence = sum(chunk.coherence_score for chunk in chunks) / len(chunks) if chunks else 0
        
        performance_score = (avg_quality + avg_coherence) / 2
        
        # Update strategy performance
        self.strategy_performance[strategy] = (
            self.strategy_performance[strategy] * 0.8 + performance_score * 0.2
        )
        
        logger.debug(f"Strategy {strategy} performance: {performance_score:.3f}")
    
    # Helper methods for specific chunking strategies
    async def _detect_action_beats(self, content: str) -> List[Dict[str, Any]]:
        """Detect action sequences in content."""
        beats = []
        
        # Simple action detection based on action verbs
        action_sentences = []
        sentences = content.split('.')
        
        for sentence in sentences:
            action_count = sum(len(re.findall(pattern, sentence, re.IGNORECASE))
                             for pattern in self.action_patterns)
            if action_count > 0:
                action_sentences.append({
                    "content": sentence.strip() + '.',
                    "tension_level": min(1.0, action_count / 10)
                })
        
        # Group consecutive action sentences
        current_beat = {"content": "", "tension_level": 0.0}
        
        for sentence in action_sentences:
            if current_beat["content"]:
                current_beat["content"] += " " + sentence["content"]
                current_beat["tension_level"] = max(current_beat["tension_level"], sentence["tension_level"])
            else:
                current_beat = sentence.copy()
            
            # End beat if tension drops or gets too long
            if len(current_beat["content"].split()) > 200:
                beats.append(current_beat)
                current_beat = {"content": "", "tension_level": 0.0}
        
        if current_beat["content"]:
            beats.append(current_beat)
        
        return beats if beats else [{"content": content, "tension_level": 0.5}]
    
    async def _identify_character_scenes(self, content: str, characters: Set[str]) -> List[Dict[str, Any]]:
        """Identify scenes based on character presence."""
        scenes = []
        
        # Split content into potential scenes
        paragraphs = content.split('\n\n')
        
        current_scene = {"content": "", "characters": set(), "pov_character": None}
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Find characters in this paragraph
            paragraph_characters = set()
            for character in characters:
                if character.lower() in paragraph.lower():
                    paragraph_characters.add(character)
            
            # Determine if this continues the current scene or starts a new one
            if not current_scene["content"] or paragraph_characters.intersection(current_scene["characters"]):
                # Continue current scene
                current_scene["content"] += "\n\n" + paragraph if current_scene["content"] else paragraph
                current_scene["characters"].update(paragraph_characters)
            else:
                # Start new scene
                if current_scene["content"]:
                    scenes.append(current_scene)
                
                current_scene = {
                    "content": paragraph,
                    "characters": paragraph_characters,
                    "pov_character": None
                }
        
        if current_scene["content"]:
            scenes.append(current_scene)
        
        return scenes if scenes else [{"content": content, "characters": characters, "pov_character": None}]
    
    async def _detect_scene_boundaries(self, content: str) -> List[int]:
        """Detect scene boundaries in content."""
        boundaries = []
        
        # Scene break indicators
        scene_break_patterns = [
            r'\n\n\*\*\*\n\n',  # Asterisk breaks
            r'\n\n---\n\n',     # Dash breaks
            r'\n\n\s*\n\n',     # Extra whitespace
            r'Chapter \d+',      # Chapter breaks
            r'Later that day',   # Time transitions
            r'The next morning', # Time transitions
        ]
        
        for pattern in scene_break_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                boundaries.append(match.start())
        
        # Sort and remove duplicates
        boundaries = sorted(set(boundaries))
        
        # Add implicit scene breaks based on content analysis
        paragraphs = content.split('\n\n')
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Look for significant topic changes
            if i > 0 and len(paragraph.split()) > 20:
                # Simple heuristic: new scene if many new words appear
                prev_words = set(paragraphs[i-1].lower().split())
                curr_words = set(paragraph.lower().split())
                
                overlap = len(prev_words.intersection(curr_words))
                if overlap < len(curr_words) * 0.3:  # Less than 30% overlap
                    boundaries.append(current_pos)
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return sorted(set(boundaries))
    
    # Splitting methods for large content
    async def _split_large_dialogue(self, dialogue: str, max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large dialogue blocks intelligently."""
        # Implementation for splitting dialogue
        chunks = []
        sentences = dialogue.split('.')
        
        current_chunk = ""
        chunk_id = start_id
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            potential_chunk = current_chunk + sentence + '.' if current_chunk else sentence + '.'
            
            if len(potential_chunk.split()) <= max_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunk = EnhancedChunk(
                        id=f"dialogue_chunk_{chunk_id}",
                        content=current_chunk,
                        content_type=ContentType.DIALOGUE,
                        priority=ChunkPriority.HIGH,
                        metadata=ChunkMetadata(),
                        token_count=len(current_chunk.split()),
                        quality_score=0.8,
                        coherence_score=0.9
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                current_chunk = sentence + '.'
        
        if current_chunk:
            chunk = EnhancedChunk(
                id=f"dialogue_chunk_{chunk_id}",
                content=current_chunk,
                content_type=ContentType.DIALOGUE,
                priority=ChunkPriority.HIGH,
                metadata=ChunkMetadata(),
                token_count=len(current_chunk.split()),
                quality_score=0.8,
                coherence_score=0.9
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _split_large_narrative(self, narrative: str, max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large narrative blocks."""
        # Similar implementation for narrative content
        return await self._split_large_dialogue(narrative, max_size, start_id)  # Simplified
    
    async def _split_action_sequence(self, action: str, max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large action sequences."""
        # Similar implementation for action content
        return await self._split_large_dialogue(action, max_size, start_id)  # Simplified
    
    async def _split_character_scene(self, scene: Dict[str, Any], max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large character scenes."""
        # Implementation for character scene splitting
        return await self._split_large_dialogue(scene["content"], max_size, start_id)  # Simplified
    
    async def _split_large_scene(self, scene: str, max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large scenes."""
        # Implementation for scene splitting
        return await self._split_large_dialogue(scene, max_size, start_id)  # Simplified
    
    async def _split_large_paragraph(self, paragraph: str, max_size: int, start_id: int) -> List[EnhancedChunk]:
        """Split large paragraphs."""
        # Implementation for paragraph splitting
        return await self._split_large_dialogue(paragraph, max_size, start_id)  # Simplified


# Global adaptive chunker instance
adaptive_chunker = AdaptiveChunkingStrategy()


# Convenience functions
async def chunk_novel_content(content: str, 
                            max_chunk_size: int = 800,
                            strategy: Optional[str] = None) -> List[EnhancedChunk]:
    """
    Chunk novel content with adaptive strategy selection.
    
    Args:
        content: Content to chunk
        max_chunk_size: Maximum tokens per chunk
        strategy: Optional specific strategy to use
        
    Returns:
        List of enhanced chunks
    """
    if strategy:
        # Use specific strategy if provided
        content_analysis = await adaptive_chunker._analyze_content(content)
        return await adaptive_chunker._chunk_with_strategy(content, strategy, max_chunk_size, content_analysis)
    else:
        # Use adaptive strategy selection
        return await adaptive_chunker.chunk_content(content, max_chunk_size)


async def analyze_chunking_performance(chunks: List[EnhancedChunk]) -> Dict[str, Any]:
    """
    Analyze performance of chunking results.
    
    Args:
        chunks: List of chunks to analyze
        
    Returns:
        Performance analysis
    """
    if not chunks:
        return {"error": "No chunks provided"}
    
    total_chunks = len(chunks)
    avg_quality = sum(chunk.quality_score for chunk in chunks) / total_chunks
    avg_coherence = sum(chunk.coherence_score for chunk in chunks) / total_chunks
    avg_token_count = sum(chunk.token_count for chunk in chunks) / total_chunks
    
    content_type_distribution = Counter(chunk.content_type.value for chunk in chunks)
    priority_distribution = Counter(chunk.priority.value for chunk in chunks)
    
    return {
        "total_chunks": total_chunks,
        "avg_quality_score": avg_quality,
        "avg_coherence_score": avg_coherence,
        "avg_token_count": avg_token_count,
        "content_type_distribution": dict(content_type_distribution),
        "priority_distribution": dict(priority_distribution),
        "overall_performance": (avg_quality + avg_coherence) / 2
    }
