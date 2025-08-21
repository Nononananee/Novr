"""
Enhanced Long-Term Memory System for Creative RAG
Provides persistent, hierarchical memory with intelligent retrieval and archival
"""

import asyncio
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import sqlite3
import aiosqlite
from pathlib import Path

class MemoryTier(Enum):
    """Memory storage tiers based on access patterns and importance"""
    HOT = "hot"           # Frequently accessed, in-memory
    WARM = "warm"         # Occasionally accessed, fast disk storage
    COLD = "cold"         # Rarely accessed, compressed storage
    ARCHIVED = "archived" # Historical data, slow retrieval

class AccessPattern(Enum):
    """Memory access patterns for optimization"""
    SEQUENTIAL = "sequential"     # Chapter-by-chapter reading
    RANDOM = "random"            # Character/plot-based queries
    TEMPORAL = "temporal"        # Timeline-based access
    CONTEXTUAL = "contextual"    # Context-building for generation

@dataclass
class MemoryMetrics:
    """Metrics for memory chunk performance"""
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    retrieval_latency_ms: float = 0.0
    relevance_score: float = 0.0
    compression_ratio: float = 1.0
    storage_tier: MemoryTier = MemoryTier.HOT
    access_pattern: AccessPattern = AccessPattern.RANDOM

@dataclass
class LongTermMemoryChunk:
    """Enhanced memory chunk with long-term storage capabilities"""
    id: str
    content: str
    content_hash: str
    embedding: Optional[List[float]] = None
    compressed_content: Optional[bytes] = None
    
    # Metadata
    chapter_range: Tuple[int, int] = (0, 0)
    word_range: Tuple[int, int] = (0, 0)
    characters_involved: List[str] = field(default_factory=list)
    plot_threads: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Long-term memory specific
    importance_score: float = 0.5
    narrative_role: str = "background"  # "climax", "setup", "development", "resolution"
    emotional_weight: float = 0.0
    conflict_level: float = 0.0
    
    # Performance metrics
    metrics: MemoryMetrics = field(default_factory=MemoryMetrics)
    
    # Relationships
    related_chunks: Set[str] = field(default_factory=set)
    causal_predecessors: Set[str] = field(default_factory=set)
    causal_successors: Set[str] = field(default_factory=set)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class LongTermMemoryManager:
    """Advanced long-term memory management with intelligent tiering"""
    
    def __init__(self, 
                 storage_path: str = "memory_storage",
                 max_hot_memory_mb: int = 512,
                 max_warm_memory_mb: int = 2048,
                 compression_threshold: float = 0.7):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Memory tiers
        self.hot_memory: Dict[str, LongTermMemoryChunk] = {}  # In-memory
        self.warm_memory_db = self.storage_path / "warm_memory.db"
        self.cold_storage_path = self.storage_path / "cold_storage"
        self.cold_storage_path.mkdir(exist_ok=True)
        
        # Configuration
        self.max_hot_memory_mb = max_hot_memory_mb
        self.max_warm_memory_mb = max_warm_memory_mb
        self.compression_threshold = compression_threshold
        
        # Indices for fast retrieval
        self.character_index: Dict[str, Set[str]] = {}
        self.plot_thread_index: Dict[str, Set[str]] = {}
        self.chapter_index: Dict[int, Set[str]] = {}
        self.temporal_index: Dict[str, Set[str]] = {}  # date -> chunk_ids
        self.semantic_clusters: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self.access_stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tier_promotions': 0,
            'tier_demotions': 0,
            'compression_saves_mb': 0.0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize the long-term memory system"""
        await self._initialize_warm_storage()
        await self._load_indices()
        await self._start_background_tasks()
        
    async def _initialize_warm_storage(self):
        """Initialize SQLite database for warm storage"""
        async with aiosqlite.connect(self.warm_memory_db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    content_hash TEXT,
                    embedding BLOB,
                    compressed_content BLOB,
                    metadata TEXT,
                    metrics TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON memory_chunks(content_hash)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at ON memory_chunks(updated_at DESC)
            """)
            
            await db.commit()
    
    async def store_memory_chunk(self, chunk: LongTermMemoryChunk) -> bool:
        """Store memory chunk with intelligent tiering"""
        try:
            # Generate content hash for deduplication
            chunk.content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            
            # Check for duplicates
            if await self._is_duplicate(chunk):
                return False
            
            # Determine initial storage tier
            tier = self._determine_storage_tier(chunk)
            chunk.metrics.storage_tier = tier
            
            # Store in appropriate tier
            if tier == MemoryTier.HOT:
                await self._store_in_hot_memory(chunk)
            elif tier == MemoryTier.WARM:
                await self._store_in_warm_memory(chunk)
            else:
                await self._store_in_cold_storage(chunk)
            
            # Update indices
            await self._update_indices(chunk)
            
            # Trigger memory management if needed
            await self._manage_memory_pressure()
            
            return True
            
        except Exception as e:
            print(f"Error storing memory chunk {chunk.id}: {e}")
            return False
    
    async def retrieve_memory_chunk(self, chunk_id: str) -> Optional[LongTermMemoryChunk]:
        """Retrieve memory chunk with tier promotion"""
        start_time = datetime.now()
        
        try:
            # Check hot memory first
            if chunk_id in self.hot_memory:
                chunk = self.hot_memory[chunk_id]
                self.access_stats['cache_hits'] += 1
            else:
                # Check warm memory
                chunk = await self._retrieve_from_warm_memory(chunk_id)
                if not chunk:
                    # Check cold storage
                    chunk = await self._retrieve_from_cold_storage(chunk_id)
                
                if chunk:
                    self.access_stats['cache_misses'] += 1
                    # Consider promoting to hot memory
                    await self._consider_tier_promotion(chunk)
                else:
                    return None
            
            # Update access metrics
            chunk.metrics.access_count += 1
            chunk.metrics.last_accessed = datetime.now()
            
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            chunk.metrics.retrieval_latency_ms = retrieval_time
            
            self.access_stats['total_retrievals'] += 1
            
            return chunk
            
        except Exception as e:
            print(f"Error retrieving memory chunk {chunk_id}: {e}")
            return None
    
    async def semantic_search(self, 
                            query_embedding: List[float], 
                            limit: int = 10,
                            min_similarity: float = 0.7,
                            tier_preference: Optional[MemoryTier] = None) -> List[Tuple[LongTermMemoryChunk, float]]:
        """Semantic search across all memory tiers"""
        
        results = []
        
        # Search hot memory first (fastest)
        for chunk in self.hot_memory.values():
            if chunk.embedding:
                similarity = self._calculate_cosine_similarity(query_embedding, chunk.embedding)
                if similarity >= min_similarity:
                    results.append((chunk, similarity))
        
        # Search warm memory if needed
        if len(results) < limit:
            warm_results = await self._semantic_search_warm_memory(
                query_embedding, limit - len(results), min_similarity
            )
            results.extend(warm_results)
        
        # Search cold storage if still needed
        if len(results) < limit and (not tier_preference or tier_preference == MemoryTier.COLD):
            cold_results = await self._semantic_search_cold_storage(
                query_embedding, limit - len(results), min_similarity
            )
            results.extend(cold_results)
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def contextual_retrieval(self, 
                                 context: Dict[str, Any],
                                 max_chunks: int = 20,
                                 relevance_threshold: float = 0.6) -> List[LongTermMemoryChunk]:
        """Retrieve contextually relevant memory chunks"""
        
        relevant_chunks = []
        
        # Character-based retrieval
        if 'characters' in context:
            for character in context['characters']:
                if character in self.character_index:
                    for chunk_id in self.character_index[character]:
                        chunk = await self.retrieve_memory_chunk(chunk_id)
                        if chunk and chunk.metrics.relevance_score >= relevance_threshold:
                            relevant_chunks.append(chunk)
        
        # Plot thread retrieval
        if 'plot_threads' in context:
            for thread in context['plot_threads']:
                if thread in self.plot_thread_index:
                    for chunk_id in self.plot_thread_index[thread]:
                        chunk = await self.retrieve_memory_chunk(chunk_id)
                        if chunk and chunk.metrics.relevance_score >= relevance_threshold:
                            relevant_chunks.append(chunk)
        
        # Chapter-based retrieval
        if 'chapter_range' in context:
            start_chapter, end_chapter = context['chapter_range']
            for chapter in range(start_chapter, end_chapter + 1):
                if chapter in self.chapter_index:
                    for chunk_id in self.chapter_index[chapter]:
                        chunk = await self.retrieve_memory_chunk(chunk_id)
                        if chunk and chunk.metrics.relevance_score >= relevance_threshold:
                            relevant_chunks.append(chunk)
        
        # Remove duplicates and sort by relevance
        unique_chunks = {chunk.id: chunk for chunk in relevant_chunks}
        sorted_chunks = sorted(
            unique_chunks.values(), 
            key=lambda x: (x.importance_score, x.metrics.relevance_score), 
            reverse=True
        )
        
        return sorted_chunks[:max_chunks]
    
    async def _store_in_hot_memory(self, chunk: LongTermMemoryChunk):
        """Store chunk in hot (in-memory) storage"""
        self.hot_memory[chunk.id] = chunk
    
    async def _store_in_warm_memory(self, chunk: LongTermMemoryChunk):
        """Store chunk in warm (SQLite) storage"""
        async with aiosqlite.connect(self.warm_memory_db) as db:
            # Serialize complex data
            metadata = json.dumps({
                'chapter_range': chunk.chapter_range,
                'word_range': chunk.word_range,
                'characters_involved': chunk.characters_involved,
                'plot_threads': chunk.plot_threads,
                'locations': chunk.locations,
                'tags': chunk.tags,
                'importance_score': chunk.importance_score,
                'narrative_role': chunk.narrative_role,
                'emotional_weight': chunk.emotional_weight,
                'conflict_level': chunk.conflict_level,
                'related_chunks': list(chunk.related_chunks),
                'causal_predecessors': list(chunk.causal_predecessors),
                'causal_successors': list(chunk.causal_successors)
            })
            
            metrics = json.dumps({
                'access_count': chunk.metrics.access_count,
                'last_accessed': chunk.metrics.last_accessed.isoformat(),
                'retrieval_latency_ms': chunk.metrics.retrieval_latency_ms,
                'relevance_score': chunk.metrics.relevance_score,
                'compression_ratio': chunk.metrics.compression_ratio,
                'storage_tier': chunk.metrics.storage_tier.value,
                'access_pattern': chunk.metrics.access_pattern.value
            })
            
            # Compress content if beneficial
            compressed_content = None
            if len(chunk.content) > 1000:  # Only compress larger content
                import gzip
                compressed = gzip.compress(chunk.content.encode())
                if len(compressed) < len(chunk.content) * self.compression_threshold:
                    compressed_content = compressed
                    chunk.metrics.compression_ratio = len(compressed) / len(chunk.content)
            
            embedding_blob = None
            if chunk.embedding:
                embedding_blob = pickle.dumps(chunk.embedding)
            
            await db.execute("""
                INSERT OR REPLACE INTO memory_chunks 
                (id, content, content_hash, embedding, compressed_content, metadata, metrics, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.content if not compressed_content else None,
                chunk.content_hash,
                embedding_blob,
                compressed_content,
                metadata,
                metrics,
                chunk.created_at.isoformat(),
                chunk.updated_at.isoformat()
            ))
            
            await db.commit()
    
    async def _store_in_cold_storage(self, chunk: LongTermMemoryChunk):
        """Store chunk in cold (file-based) storage"""
        chunk_file = self.cold_storage_path / f"{chunk.id}.pkl"
        
        # Compress the entire chunk
        import gzip
        import pickle
        
        chunk_data = pickle.dumps(chunk)
        compressed_data = gzip.compress(chunk_data)
        
        with open(chunk_file, 'wb') as f:
            f.write(compressed_data)
        
        chunk.metrics.compression_ratio = len(compressed_data) / len(chunk_data)
        chunk.metrics.storage_tier = MemoryTier.COLD
    
    def _determine_storage_tier(self, chunk: LongTermMemoryChunk) -> MemoryTier:
        """Determine appropriate storage tier for new chunk"""
        
        # Critical chunks go to hot memory
        if chunk.importance_score >= 0.9 or chunk.narrative_role in ['climax', 'resolution']:
            return MemoryTier.HOT
        
        # Important chunks go to warm memory
        if chunk.importance_score >= 0.7 or chunk.emotional_weight >= 0.8:
            return MemoryTier.WARM
        
        # Everything else starts in cold storage
        return MemoryTier.COLD
    
    async def _manage_memory_pressure(self):
        """Manage memory pressure by moving chunks between tiers"""
        
        # Check hot memory size
        hot_memory_size = sum(len(chunk.content.encode()) for chunk in self.hot_memory.values())
        hot_memory_mb = hot_memory_size / (1024 * 1024)
        
        if hot_memory_mb > self.max_hot_memory_mb:
            # Move least recently used chunks to warm storage
            chunks_by_access = sorted(
                self.hot_memory.values(),
                key=lambda x: (x.metrics.last_accessed, x.importance_score)
            )
            
            for chunk in chunks_by_access:
                if hot_memory_mb <= self.max_hot_memory_mb * 0.8:  # Target 80% capacity
                    break
                
                await self._demote_to_warm_memory(chunk)
                del self.hot_memory[chunk.id]
                
                chunk_size_mb = len(chunk.content.encode()) / (1024 * 1024)
                hot_memory_mb -= chunk_size_mb
                self.access_stats['tier_demotions'] += 1
    
    async def _consider_tier_promotion(self, chunk: LongTermMemoryChunk):
        """Consider promoting chunk to higher tier based on access patterns"""
        
        # Promote frequently accessed chunks
        if (chunk.metrics.access_count >= 5 and 
            chunk.metrics.storage_tier != MemoryTier.HOT and
            len(self.hot_memory) < self.max_hot_memory_mb * 1024 * 1024 / 2000):  # Rough estimate
            
            await self._promote_to_hot_memory(chunk)
            self.access_stats['tier_promotions'] += 1
    
    async def _promote_to_hot_memory(self, chunk: LongTermMemoryChunk):
        """Promote chunk to hot memory"""
        self.hot_memory[chunk.id] = chunk
        chunk.metrics.storage_tier = MemoryTier.HOT
        
        # Remove from lower tiers if present
        if chunk.metrics.storage_tier == MemoryTier.WARM:
            await self._remove_from_warm_memory(chunk.id)
        elif chunk.metrics.storage_tier == MemoryTier.COLD:
            await self._remove_from_cold_storage(chunk.id)
    
    async def _demote_to_warm_memory(self, chunk: LongTermMemoryChunk):
        """Demote chunk to warm memory"""
        await self._store_in_warm_memory(chunk)
        chunk.metrics.storage_tier = MemoryTier.WARM
    
    async def _update_indices(self, chunk: LongTermMemoryChunk):
        """Update all indices with new chunk"""
        
        # Character index
        for character in chunk.characters_involved:
            if character not in self.character_index:
                self.character_index[character] = set()
            self.character_index[character].add(chunk.id)
        
        # Plot thread index
        for thread in chunk.plot_threads:
            if thread not in self.plot_thread_index:
                self.plot_thread_index[thread] = set()
            self.plot_thread_index[thread].add(chunk.id)
        
        # Chapter index
        for chapter in range(chunk.chapter_range[0], chunk.chapter_range[1] + 1):
            if chapter not in self.chapter_index:
                self.chapter_index[chapter] = set()
            self.chapter_index[chapter].add(chunk.id)
        
        # Temporal index (by creation date)
        date_key = chunk.created_at.strftime('%Y-%m-%d')
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = set()
        self.temporal_index[date_key].add(chunk.id)
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _is_duplicate(self, chunk: LongTermMemoryChunk) -> bool:
        """Check if chunk is a duplicate based on content hash"""
        
        # Check hot memory
        for existing_chunk in self.hot_memory.values():
            if existing_chunk.content_hash == chunk.content_hash:
                return True
        
        # Check warm memory
        async with aiosqlite.connect(self.warm_memory_db) as db:
            cursor = await db.execute(
                "SELECT id FROM memory_chunks WHERE content_hash = ?",
                (chunk.content_hash,)
            )
            result = await cursor.fetchone()
            if result:
                return True
        
        return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        # Count chunks in each tier
        hot_count = len(self.hot_memory)
        
        async with aiosqlite.connect(self.warm_memory_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM memory_chunks")
            warm_count = (await cursor.fetchone())[0]
        
        cold_count = len(list(self.cold_storage_path.glob("*.pkl")))
        
        # Calculate memory usage
        hot_memory_mb = sum(len(chunk.content.encode()) for chunk in self.hot_memory.values()) / (1024 * 1024)
        
        return {
            'tier_distribution': {
                'hot': hot_count,
                'warm': warm_count,
                'cold': cold_count,
                'total': hot_count + warm_count + cold_count
            },
            'memory_usage': {
                'hot_memory_mb': hot_memory_mb,
                'max_hot_memory_mb': self.max_hot_memory_mb,
                'utilization_percent': (hot_memory_mb / self.max_hot_memory_mb) * 100
            },
            'performance': self.access_stats,
            'indices': {
                'characters_tracked': len(self.character_index),
                'plot_threads_tracked': len(self.plot_thread_index),
                'chapters_indexed': len(self.chapter_index),
                'temporal_entries': len(self.temporal_index)
            }
        }
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Memory cleanup task
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)
        
        # Index optimization task
        optimize_task = asyncio.create_task(self._periodic_index_optimization())
        self._background_tasks.add(optimize_task)
        optimize_task.add_done_callback(self._background_tasks.discard)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old, unused memory chunks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_date = datetime.now() - timedelta(days=30)
                chunks_cleaned = 0
                
                # Clean cold storage
                for chunk_file in self.cold_storage_path.glob("*.pkl"):
                    if chunk_file.stat().st_mtime < cutoff_date.timestamp():
                        # Load chunk to check access patterns
                        try:
                            import gzip
                            import pickle
                            
                            with open(chunk_file, 'rb') as f:
                                chunk_data = gzip.decompress(f.read())
                                chunk = pickle.loads(chunk_data)
                            
                            # Remove if rarely accessed and low importance
                            if (chunk.metrics.access_count < 2 and 
                                chunk.importance_score < 0.3):
                                chunk_file.unlink()
                                chunks_cleaned += 1
                                
                        except Exception as e:
                            print(f"Error during cleanup of {chunk_file}: {e}")
                
                if chunks_cleaned > 0:
                    print(f"Cleaned up {chunks_cleaned} old memory chunks")
                    
            except Exception as e:
                print(f"Error in periodic cleanup: {e}")
    
    async def _periodic_index_optimization(self):
        """Periodic optimization of memory indices"""
        while True:
            try:
                await asyncio.sleep(7200)  # Run every 2 hours
                
                # Rebuild indices to remove stale references
                await self._rebuild_indices()
                
                # Optimize warm storage database
                async with aiosqlite.connect(self.warm_memory_db) as db:
                    await db.execute("VACUUM")
                    await db.execute("ANALYZE")
                    await db.commit()
                
            except Exception as e:
                print(f"Error in periodic index optimization: {e}")
    
    async def _rebuild_indices(self):
        """Rebuild all indices from scratch"""
        
        # Clear existing indices
        self.character_index.clear()
        self.plot_thread_index.clear()
        self.chapter_index.clear()
        self.temporal_index.clear()
        
        # Rebuild from hot memory
        for chunk in self.hot_memory.values():
            await self._update_indices(chunk)
        
        # Rebuild from warm memory
        async with aiosqlite.connect(self.warm_memory_db) as db:
            cursor = await db.execute("SELECT id, metadata FROM memory_chunks")
            async for row in cursor:
                chunk_id, metadata_json = row
                try:
                    metadata = json.loads(metadata_json)
                    
                    # Update character index
                    for character in metadata.get('characters_involved', []):
                        if character not in self.character_index:
                            self.character_index[character] = set()
                        self.character_index[character].add(chunk_id)
                    
                    # Update plot thread index
                    for thread in metadata.get('plot_threads', []):
                        if thread not in self.plot_thread_index:
                            self.plot_thread_index[thread] = set()
                        self.plot_thread_index[thread].add(chunk_id)
                    
                    # Update chapter index
                    chapter_range = metadata.get('chapter_range', [0, 0])
                    for chapter in range(chapter_range[0], chapter_range[1] + 1):
                        if chapter not in self.chapter_index:
                            self.chapter_index[chapter] = set()
                        self.chapter_index[chapter].add(chunk_id)
                        
                except Exception as e:
                    print(f"Error rebuilding index for chunk {chunk_id}: {e}")
    
    async def close(self):
        """Clean shutdown of the memory system"""
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save any pending data
        await self._save_indices()
    
    async def _save_indices(self):
        """Save indices to disk for faster startup"""
        indices_file = self.storage_path / "indices.json"
        
        indices_data = {
            'character_index': {k: list(v) for k, v in self.character_index.items()},
            'plot_thread_index': {k: list(v) for k, v in self.plot_thread_index.items()},
            'chapter_index': {str(k): list(v) for k, v in self.chapter_index.items()},
            'temporal_index': {k: list(v) for k, v in self.temporal_index.items()},
            'access_stats': self.access_stats
        }
        
        with open(indices_file, 'w') as f:
            json.dump(indices_data, f, indent=2)
    
    async def _load_indices(self):
        """Load indices from disk"""
        indices_file = self.storage_path / "indices.json"
        
        if indices_file.exists():
            try:
                with open(indices_file, 'r') as f:
                    indices_data = json.load(f)
                
                self.character_index = {k: set(v) for k, v in indices_data.get('character_index', {}).items()}
                self.plot_thread_index = {k: set(v) for k, v in indices_data.get('plot_thread_index', {}).items()}
                self.chapter_index = {int(k): set(v) for k, v in indices_data.get('chapter_index', {}).items()}
                self.temporal_index = {k: set(v) for k, v in indices_data.get('temporal_index', {}).items()}
                self.access_stats.update(indices_data.get('access_stats', {}))
                
            except Exception as e:
                print(f"Error loading indices: {e}")
