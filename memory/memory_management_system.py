# src/rag/memory_manager.py
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import json

class MemoryPriority(Enum):
    CRITICAL = "critical"      # Core plot, main characters
    HIGH = "high"             # Important events, major character development
    MEDIUM = "medium"         # Supporting details, minor characters
    LOW = "low"              # Atmosphere, minor descriptions
    CONTEXTUAL = "contextual" # Scene-specific details

class MemoryType(Enum):
    CHARACTER_STATE = "character_state"
    PLOT_POINT = "plot_point"
    WORLD_ELEMENT = "world_element"
    RELATIONSHIP = "relationship"
    TIMELINE_EVENT = "timeline_event"
    TONE_SHIFT = "tone_shift"
    FORESHADOWING = "foreshadowing"

@dataclass
class MemoryChunk:
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    chapter_range: Tuple[int, int]  # (start_chapter, end_chapter)
    word_range: Tuple[int, int]     # (start_word, end_word)
    characters_involved: List[str]
    plot_threads: List[str]
    embedding: Optional[List[float]] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    relevance_score: float = 1.0
    dependencies: List[str] = field(default_factory=list)  # Other chunk IDs
    tags: List[str] = field(default_factory=list)

@dataclass
class NovelMemoryState:
    """Current state of the novel for memory management"""
    current_chapter: int
    current_word_count: int
    active_characters: List[str]
    active_plot_threads: List[str]
    current_pov_character: Optional[str]
    scene_location: Optional[str]
    emotional_tone: Dict[str, float]
    recent_events: List[str]  # Last 5-10 major events

class NovelMemoryManager:
    def __init__(self, 
                 max_context_tokens: int = 32000,
                 chunk_size: int = 1000,
                 overlap_size: int = 200,
                 critical_memory_ratio: float = 0.4,
                 vectorstore_client = None):
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.critical_memory_ratio = critical_memory_ratio
        self.vectorstore = vectorstore_client
        
        # Memory storage
        self.memory_chunks: Dict[str, MemoryChunk] = {}
        self.character_memories: Dict[str, List[str]] = {}  # character_id -> chunk_ids
        self.plot_thread_memories: Dict[str, List[str]] = {}  # thread_id -> chunk_ids
        self.chapter_index: Dict[int, List[str]] = {}  # chapter -> chunk_ids
        
        # Current context
        self.active_context: List[str] = []  # Currently loaded chunk IDs
        self.novel_state: Optional[NovelMemoryState] = None

    async def chunk_novel_content(self, content: str, metadata: Dict) -> List[MemoryChunk]:
        """Intelligently chunk novel content based on narrative structure"""
        chunks = []
        
        # Split by scenes/chapters first
        scenes = self._detect_scene_breaks(content)
        
        for i, scene in enumerate(scenes):
            # Further chunk if scene is too long
            if len(scene.split()) > self.chunk_size:
                sub_chunks = self._chunk_by_size(scene)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk = await self._create_memory_chunk(
                        content=sub_chunk,
                        scene_index=i,
                        sub_index=j,
                        metadata=metadata
                    )
                    chunks.append(chunk)
            else:
                chunk = await self._create_memory_chunk(
                    content=scene,
                    scene_index=i,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        return chunks

    async def _create_memory_chunk(self, content: str, scene_index: int, 
                                 sub_index: int = 0, metadata: Dict = None) -> MemoryChunk:
        """Create a memory chunk with intelligent classification"""
        
        # Analyze content for classification
        analysis = await self._analyze_chunk_content(content)
        
        chunk = MemoryChunk(
            id=f"chunk_{scene_index}_{sub_index}",
            content=content,
            memory_type=analysis['type'],
            priority=analysis['priority'],
            chapter_range=analysis['chapter_range'],
            word_range=analysis['word_range'],
            characters_involved=analysis['characters'],
            plot_threads=analysis['plot_threads'],
            tags=analysis['tags']
        )
        
        # Generate embedding
        chunk.embedding = await self._generate_embedding(content)
        
        return chunk

    def _detect_scene_breaks(self, content: str) -> List[str]:
        """Detect natural scene breaks in content"""
        # Look for scene break indicators
        scene_indicators = [
            "\n\n***\n\n",  # Explicit scene break
            "\n\n---\n\n",  # Alternative scene break
            "Chapter ",      # Chapter break
            "\n\nTime:",     # Time jump
            "\n\nLocation:", # Location change
        ]
        
        scenes = []
        current_scene = []
        lines = content.split('\n')
        
        for line in lines:
            # Check for scene break
            is_break = any(indicator in line for indicator in scene_indicators)
            
            if is_break and current_scene:
                scenes.append('\n'.join(current_scene))
                current_scene = [line]
            else:
                current_scene.append(line)
        
        # Add final scene
        if current_scene:
            scenes.append('\n'.join(current_scene))
            
        return scenes

    async def _analyze_chunk_content(self, content: str) -> Dict:
        """Analyze chunk content to determine type, priority, and metadata"""
        # This would use NLP/LLM analysis in real implementation
        analysis = {
            'type': MemoryType.PLOT_POINT,  # Default
            'priority': MemoryPriority.MEDIUM,  # Default
            'chapter_range': (1, 1),  # Would be calculated
            'word_range': (0, len(content.split())),
            'characters': self._extract_characters(content),
            'plot_threads': self._extract_plot_threads(content),
            'tags': self._extract_tags(content)
        }
        
        # Determine priority based on content analysis
        if any(keyword in content.lower() for keyword in ['death', 'betrayal', 'revelation', 'climax']):
            analysis['priority'] = MemoryPriority.CRITICAL
        elif any(keyword in content.lower() for keyword in ['conflict', 'romance', 'discovery']):
            analysis['priority'] = MemoryPriority.HIGH
            
        return analysis

    async def build_context_for_generation(self, 
                                         current_position: int,
                                         generation_type: str = "continuation",
                                         target_characters: List[str] = None,
                                         plot_threads: List[str] = None) -> str:
        """Build optimal context for novel generation"""
        
        # Get relevant memory chunks
        relevant_chunks = await self._retrieve_relevant_memories(
            current_position=current_position,
            characters=target_characters,
            plot_threads=plot_threads,
            generation_type=generation_type
        )
        
        # Prioritize and fit within token limit
        context_chunks = self._prioritize_and_fit_context(relevant_chunks)
        
        # Build structured context
        context = self._build_structured_context(context_chunks)
        
        return context

    async def _retrieve_relevant_memories(self, 
                                        current_position: int,
                                        characters: List[str] = None,
                                        plot_threads: List[str] = None,
                                        generation_type: str = "continuation") -> List[MemoryChunk]:
        """Retrieve relevant memories using multiple strategies"""
        
        relevant_chunks = []
        
        # 1. Recency-based retrieval (recent chapters)
        recent_chunks = self._get_recent_chunks(current_position, window=3)
        relevant_chunks.extend(recent_chunks)
        
        # 2. Character-based retrieval
        if characters:
            for character in characters:
                char_chunks = self._get_character_chunks(character, limit=5)
                relevant_chunks.extend(char_chunks)
        
        # 3. Plot thread-based retrieval
        if plot_threads:
            for thread in plot_threads:
                thread_chunks = self._get_plot_thread_chunks(thread, limit=3)
                relevant_chunks.extend(thread_chunks)
        
        # 4. Critical memory retrieval
        critical_chunks = self._get_critical_memories(limit=10)
        relevant_chunks.extend(critical_chunks)
        
        # 5. Semantic similarity retrieval (if we have a query)
        if generation_type != "continuation":
            similar_chunks = await self._get_semantically_similar_chunks(
                query=generation_type, 
                limit=5
            )
            relevant_chunks.extend(similar_chunks)
        
        # Remove duplicates and update access patterns
        unique_chunks = list({chunk.id: chunk for chunk in relevant_chunks}.values())
        
        for chunk in unique_chunks:
            chunk.last_accessed = datetime.now()
            chunk.access_count += 1
        
        return unique_chunks

    def _prioritize_and_fit_context(self, chunks: List[MemoryChunk]) -> List[MemoryChunk]:
        """Prioritize chunks and fit within token limit"""
        
        # Sort by priority and relevance
        priority_order = {
            MemoryPriority.CRITICAL: 5,
            MemoryPriority.HIGH: 4,
            MemoryPriority.MEDIUM: 3,
            MemoryPriority.LOW: 2,
            MemoryPriority.CONTEXTUAL: 1
        }
        
        chunks_sorted = sorted(
            chunks, 
            key=lambda c: (
                priority_order[c.priority], 
                c.relevance_score, 
                -c.access_count  # More accessed = more important
            ),
            reverse=True
        )
        
        # Fit within token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks_sorted:
            chunk_tokens = len(chunk.content.split()) * 1.3  # Rough token estimation
            
            if total_tokens + chunk_tokens <= self.max_context_tokens * 0.8:  # Reserve 20% for generation
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            elif chunk.priority == MemoryPriority.CRITICAL:
                # Always include critical memories, even if we exceed limit slightly
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
        
        return selected_chunks

    def _build_structured_context(self, chunks: List[MemoryChunk]) -> str:
        """Build structured context from memory chunks"""
        
        context_parts = []
        
        # Group chunks by type
        chunk_groups = {}
        for chunk in chunks:
            chunk_type = chunk.memory_type
            if chunk_type not in chunk_groups:
                chunk_groups[chunk_type] = []
            chunk_groups[chunk_type].append(chunk)
        
        # Build context sections
        if MemoryType.CHARACTER_STATE in chunk_groups:
            context_parts.append("=== CHARACTER STATES ===")
            for chunk in chunk_groups[MemoryType.CHARACTER_STATE]:
                context_parts.append(f"Characters: {', '.join(chunk.characters_involved)}")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        if MemoryType.PLOT_POINT in chunk_groups:
            context_parts.append("=== PLOT DEVELOPMENTS ===")
            for chunk in chunk_groups[MemoryType.PLOT_POINT]:
                context_parts.append(f"Plot Threads: {', '.join(chunk.plot_threads)}")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        if MemoryType.RELATIONSHIP in chunk_groups:
            context_parts.append("=== CHARACTER RELATIONSHIPS ===")
            for chunk in chunk_groups[MemoryType.RELATIONSHIP]:
                context_parts.append(chunk.content)
                context_parts.append("")
        
        # Add recent context last (most relevant for continuation)
        recent_chunks = [c for c in chunks if c.memory_type not in chunk_groups or len(chunk_groups) == 0]
        if recent_chunks:
            context_parts.append("=== RECENT CONTEXT ===")
            for chunk in sorted(recent_chunks, key=lambda c: c.word_range[0]):
                context_parts.append(chunk.content)
                context_parts.append("")
        
        return "\n".join(context_parts)

    async def update_memory_after_generation(self, 
                                           new_content: str, 
                                           generation_metadata: Dict):
        """Update memory system after new content generation"""
        
        # Chunk new content
        new_chunks = await self.chunk_novel_content(new_content, generation_metadata)
        
        # Store new chunks
        for chunk in new_chunks:
            await self.store_memory_chunk(chunk)
        
        # Update indexes
        await self._update_indexes(new_chunks)
        
        # Prune old memories if needed
        await self._prune_memories()

    async def store_memory_chunk(self, chunk: MemoryChunk):
        """Store memory chunk in vector database and local indexes"""
        
        # Store in vector database
        if self.vectorstore:
            await self.vectorstore.add_embedding(
                id=chunk.id,
                embedding=chunk.embedding,
                metadata={
                    'content': chunk.content,
                    'type': chunk.memory_type.value,
                    'priority': chunk.priority.value,
                    'characters': chunk.characters_involved,
                    'plot_threads': chunk.plot_threads,
                    'chapter_range': chunk.chapter_range,
                    'tags': chunk.tags
                }
            )
        
        # Store in local memory
        self.memory_chunks[chunk.id] = chunk
        
        # Update character index
        for character in chunk.characters_involved:
            if character not in self.character_memories:
                self.character_memories[character] = []
            self.character_memories[character].append(chunk.id)
        
        # Update plot thread index
        for thread in chunk.plot_threads:
            if thread not in self.plot_thread_memories:
                self.plot_thread_memories[thread] = []
            self.plot_thread_memories[thread].append(chunk.id)
    
    def get_memory_statistics(self) -> Dict:
        """Get memory system statistics"""
        return {
            'total_chunks': len(self.memory_chunks),
            'total_characters': len(self.character_memories),
            'total_plot_threads': len(self.plot_thread_memories),
            'priority_distribution': {
                priority.value: len([c for c in self.memory_chunks.values() 
                                   if c.priority == priority])
                for priority in MemoryPriority
            },
            'memory_usage': {
                'active_context_chunks': len(self.active_context),
                'max_context_tokens': self.max_context_tokens
            }
        }

# Helper functions implementation would go here...
