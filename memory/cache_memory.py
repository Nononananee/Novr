"""
High-Performance Cache Memory System for Creative RAG
Provides multi-level caching with intelligent prefetching and eviction policies
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import weakref
from collections import OrderedDict, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

class CacheLevel(Enum):
    """Cache levels with different performance characteristics"""
    L1 = "l1"  # Ultra-fast, small capacity (recent queries)
    L2 = "l2"  # Fast, medium capacity (frequent patterns)
    L3 = "l3"  # Moderate, large capacity (contextual data)

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    SEMANTIC = "semantic"  # Semantic similarity-based

class PrefetchStrategy(Enum):
    """Prefetching strategies"""
    SEQUENTIAL = "sequential"     # Next chapters/scenes
    CONTEXTUAL = "contextual"     # Related characters/plots
    PREDICTIVE = "predictive"     # ML-based prediction
    NONE = "none"                # No prefetching

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    semantic_hash: Optional[str] = None
    related_keys: Set[str] = field(default_factory=set)
    access_pattern: str = "random"
    importance_score: float = 0.5

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_requests: int = 0
    average_latency_ms: float = 0.0
    memory_usage_bytes: int = 0
    hit_ratio: float = 0.0

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                self.stats.hits += 1
                self._update_hit_ratio()
                return entry.value
            
            self.stats.misses += 1
            self._update_hit_ratio()
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None, 
            importance_score: float = 0.5) -> bool:
        """Put item in cache"""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if item is too large
            if size_bytes > self.max_memory_bytes:
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Evict items if necessary
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + size_bytes > self.max_memory_bytes):
                if not self._evict_lru():
                    return False
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                importance_score=importance_score
            )
            
            self.cache[key] = entry
            self.current_memory += size_bytes
            return True
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory -= entry.size_bytes
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Find LRU item (considering importance)
        lru_key = None
        lru_score = float('inf')
        
        for key, entry in self.cache.items():
            # Score based on recency and importance
            recency_score = time.time() - entry.last_accessed
            importance_penalty = 1.0 / (entry.importance_score + 0.1)
            combined_score = recency_score * importance_penalty
            
            if combined_score < lru_score:
                lru_score = combined_score
                lru_key = key
        
        if lru_key:
            self._remove_entry(lru_key)
            self.stats.evictions += 1
            return True
        
        return False
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                # Rough estimate
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _update_hit_ratio(self):
        """Update hit ratio statistics"""
        total = self.stats.hits + self.stats.misses
        if total > 0:
            self.stats.hit_ratio = self.stats.hits / total
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.memory_usage_bytes = self.current_memory
            return self.stats

class SemanticCache:
    """Semantic similarity-based cache for embeddings and search results"""
    
    def __init__(self, max_entries: int = 1000, similarity_threshold: float = 0.85):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.entries: Dict[str, CacheEntry] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get_similar(self, query_embedding: List[float], threshold: Optional[float] = None) -> Optional[Tuple[str, Any, float]]:
        """Get cached result for similar query"""
        threshold = threshold or self.similarity_threshold
        
        with self.lock:
            best_match = None
            best_similarity = 0.0
            
            for key, embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold and similarity > best_similarity:
                    if key in self.entries:
                        entry = self.entries[key]
                        
                        # Check TTL
                        if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                            self._remove_entry(key)
                            continue
                        
                        best_match = (key, entry.value, similarity)
                        best_similarity = similarity
            
            if best_match:
                # Update access stats
                entry = self.entries[best_match[0]]
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            
            self._update_hit_ratio()
            return best_match
    
    def put(self, key: str, query_embedding: List[float], value: Any, 
            ttl_seconds: Optional[float] = None) -> bool:
        """Cache result with embedding"""
        with self.lock:
            # Evict if necessary
            while len(self.entries) >= self.max_entries:
                self._evict_lru()
            
            # Store entry and embedding
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=self._calculate_size(value),
                ttl_seconds=ttl_seconds
            )
            
            self.entries[key] = entry
            self.embeddings[key] = query_embedding.copy()
            return True
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        self.entries.pop(key, None)
        self.embeddings.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.entries:
            return
        
        lru_key = min(self.entries.keys(), 
                     key=lambda k: self.entries[k].last_accessed)
        self._remove_entry(lru_key)
        self.stats.evictions += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value"""
        try:
            return len(str(value).encode('utf-8'))
        except:
            return 1024
    
    def _update_hit_ratio(self):
        """Update hit ratio"""
        total = self.stats.hits + self.stats.misses
        if total > 0:
            self.stats.hit_ratio = self.stats.hits / total

class MultiLevelCacheManager:
    """Multi-level cache manager with intelligent prefetching"""
    
    def __init__(self, 
                 l1_size: int = 100,
                 l2_size: int = 500, 
                 l3_size: int = 2000,
                 l1_memory_mb: int = 50,
                 l2_memory_mb: int = 200,
                 l3_memory_mb: int = 500,
                 prefetch_strategy: PrefetchStrategy = PrefetchStrategy.CONTEXTUAL):
        
        # Cache levels
        self.l1_cache = LRUCache(l1_size, l1_memory_mb)  # Hot data
        self.l2_cache = LRUCache(l2_size, l2_memory_mb)  # Warm data
        self.l3_cache = LRUCache(l3_size, l3_memory_mb)  # Cold data
        
        # Semantic cache for embeddings
        self.semantic_cache = SemanticCache(max_entries=1000)
        
        # Configuration
        self.prefetch_strategy = prefetch_strategy
        
        # Prefetching
        self.prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_stats = {'requests': 0, 'hits': 0, 'misses': 0}
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.related_keys: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize cache manager"""
        # Start background prefetching
        prefetch_task = asyncio.create_task(self._prefetch_worker())
        self._background_tasks.add(prefetch_task)
        prefetch_task.add_done_callback(self._background_tasks.discard)
        
        # Start pattern analysis
        pattern_task = asyncio.create_task(self._pattern_analyzer())
        self._background_tasks.add(pattern_task)
        pattern_task.add_done_callback(self._background_tasks.discard)
    
    async def get(self, key: str, fetch_func: Optional[callable] = None) -> Optional[Any]:
        """Get item from multi-level cache"""
        start_time = time.time()
        
        # Try L1 cache first
        result = self.l1_cache.get(key)
        if result is not None:
            await self._record_access(key, CacheLevel.L1)
            await self._trigger_prefetch(key)
            return result
        
        # Try L2 cache
        result = self.l2_cache.get(key)
        if result is not None:
            # Promote to L1
            self.l1_cache.put(key, result, importance_score=0.7)
            await self._record_access(key, CacheLevel.L2)
            await self._trigger_prefetch(key)
            return result
        
        # Try L3 cache
        result = self.l3_cache.get(key)
        if result is not None:
            # Promote to L2
            self.l2_cache.put(key, result, importance_score=0.5)
            await self._record_access(key, CacheLevel.L3)
            await self._trigger_prefetch(key)
            return result
        
        # Cache miss - fetch if function provided
        if fetch_func:
            try:
                result = await fetch_func(key)
                if result is not None:
                    await self.put(key, result)
                    await self._record_access(key, None)  # Cache miss
                    return result
            except Exception as e:
                print(f"Error fetching {key}: {e}")
        
        return None
    
    async def put(self, key: str, value: Any, level: Optional[CacheLevel] = None, 
                  ttl_seconds: Optional[float] = None, importance_score: float = 0.5):
        """Put item in appropriate cache level"""
        
        # Determine cache level if not specified
        if level is None:
            level = self._determine_cache_level(key, value, importance_score)
        
        # Store in appropriate level
        if level == CacheLevel.L1:
            self.l1_cache.put(key, value, ttl_seconds, importance_score)
        elif level == CacheLevel.L2:
            self.l2_cache.put(key, value, ttl_seconds, importance_score)
        else:
            self.l3_cache.put(key, value, ttl_seconds, importance_score)
    
    async def get_semantic(self, query_embedding: List[float], 
                          fetch_func: Optional[callable] = None,
                          threshold: float = 0.85) -> Optional[Tuple[Any, float]]:
        """Get semantically similar cached result"""
        
        # Check semantic cache
        result = self.semantic_cache.get_similar(query_embedding, threshold)
        if result:
            key, value, similarity = result
            await self._record_access(f"semantic_{key}", CacheLevel.L1)
            return value, similarity
        
        # Cache miss - fetch if function provided
        if fetch_func:
            try:
                result = await fetch_func(query_embedding)
                if result is not None:
                    # Cache the result
                    cache_key = f"semantic_{hashlib.md5(str(query_embedding).encode()).hexdigest()[:8]}"
                    self.semantic_cache.put(cache_key, query_embedding, result)
                    return result, 1.0
            except Exception as e:
                print(f"Error in semantic fetch: {e}")
        
        return None
    
    def _determine_cache_level(self, key: str, value: Any, importance_score: float) -> CacheLevel:
        """Determine appropriate cache level for item"""
        
        # High importance items go to L1
        if importance_score >= 0.8:
            return CacheLevel.L1
        
        # Medium importance to L2
        if importance_score >= 0.5:
            return CacheLevel.L2
        
        # Everything else to L3
        return CacheLevel.L3
    
    async def _record_access(self, key: str, level: Optional[CacheLevel]):
        """Record access pattern for analytics"""
        current_time = time.time()
        
        # Record access time
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff_time]
        
        # Update related keys based on temporal proximity
        await self._update_related_keys(key, current_time)
    
    async def _update_related_keys(self, key: str, access_time: float):
        """Update related keys based on access patterns"""
        
        # Find keys accessed within a short time window
        time_window = 60  # 1 minute
        
        for other_key, access_times in self.access_patterns.items():
            if other_key == key:
                continue
            
            # Check if accessed recently
            recent_accesses = [t for t in access_times if abs(t - access_time) <= time_window]
            if recent_accesses:
                self.related_keys[key].add(other_key)
                self.related_keys[other_key].add(key)
    
    async def _trigger_prefetch(self, key: str):
        """Trigger prefetching based on access patterns"""
        
        if self.prefetch_strategy == PrefetchStrategy.NONE:
            return
        
        # Get related keys to prefetch
        prefetch_candidates = set()
        
        if self.prefetch_strategy == PrefetchStrategy.CONTEXTUAL:
            # Prefetch related keys
            prefetch_candidates.update(self.related_keys.get(key, set()))
        
        elif self.prefetch_strategy == PrefetchStrategy.SEQUENTIAL:
            # Prefetch sequential keys (e.g., next chapters)
            if key.startswith('chapter_'):
                try:
                    chapter_num = int(key.split('_')[1])
                    prefetch_candidates.add(f'chapter_{chapter_num + 1}')
                    prefetch_candidates.add(f'chapter_{chapter_num + 2}')
                except:
                    pass
        
        elif self.prefetch_strategy == PrefetchStrategy.PREDICTIVE:
            # ML-based prediction (simplified)
            prefetch_candidates.update(await self._predict_next_accesses(key))
        
        # Queue prefetch requests
        for candidate in prefetch_candidates:
            if not self.prefetch_queue.full():
                try:
                    await self.prefetch_queue.put_nowait(candidate)
                except asyncio.QueueFull:
                    break
    
    async def _predict_next_accesses(self, key: str) -> Set[str]:
        """Predict next likely accesses (simplified ML approach)"""
        
        # Simple pattern-based prediction
        predictions = set()
        
        # Analyze access frequency patterns
        access_times = self.access_patterns.get(key, [])
        if len(access_times) >= 3:
            # Find frequently co-accessed keys
            for other_key, other_times in self.access_patterns.items():
                if other_key == key:
                    continue
                
                # Calculate correlation
                correlation = self._calculate_access_correlation(access_times, other_times)
                if correlation > 0.5:
                    predictions.add(other_key)
        
        return predictions
    
    def _calculate_access_correlation(self, times1: List[float], times2: List[float]) -> float:
        """Calculate correlation between access patterns"""
        
        if not times1 or not times2:
            return 0.0
        
        # Simple correlation based on temporal proximity
        correlations = []
        
        for t1 in times1:
            min_distance = min(abs(t1 - t2) for t2 in times2)
            # Correlation decreases with distance
            correlation = max(0, 1 - min_distance / 3600)  # 1 hour window
            correlations.append(correlation)
        
        return sum(correlations) / len(correlations) if correlations else 0.0
    
    async def _prefetch_worker(self):
        """Background worker for prefetching"""
        
        while True:
            try:
                # Get prefetch request
                key = await self.prefetch_queue.get()
                
                # Check if already cached
                if (self.l1_cache.get(key) is not None or 
                    self.l2_cache.get(key) is not None or 
                    self.l3_cache.get(key) is not None):
                    self.prefetch_stats['hits'] += 1
                    continue
                
                # Simulate prefetch (would call actual fetch function)
                # In real implementation, this would fetch from long-term memory
                await asyncio.sleep(0.1)  # Simulate fetch time
                
                self.prefetch_stats['requests'] += 1
                self.prefetch_stats['misses'] += 1
                
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                await asyncio.sleep(1)
    
    async def _pattern_analyzer(self):
        """Background pattern analysis"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze access patterns
                await self._analyze_access_patterns()
                
                # Optimize cache levels
                await self._optimize_cache_levels()
                
            except Exception as e:
                print(f"Error in pattern analyzer: {e}")
    
    async def _analyze_access_patterns(self):
        """Analyze and optimize access patterns"""
        
        current_time = time.time()
        
        # Identify hot keys (frequently accessed)
        hot_keys = []
        for key, access_times in self.access_patterns.items():
            recent_accesses = [t for t in access_times if current_time - t <= 3600]  # Last hour
            if len(recent_accesses) >= 5:  # 5+ accesses in last hour
                hot_keys.append(key)
        
        # Promote hot keys to L1 cache
        for key in hot_keys:
            value = (self.l2_cache.get(key) or self.l3_cache.get(key))
            if value is not None:
                self.l1_cache.put(key, value, importance_score=0.9)
    
    async def _optimize_cache_levels(self):
        """Optimize cache level assignments"""
        
        # Move rarely accessed items from L1 to lower levels
        current_time = time.time()
        
        for key in list(self.l1_cache.cache.keys()):
            access_times = self.access_patterns.get(key, [])
            recent_accesses = [t for t in access_times if current_time - t <= 1800]  # Last 30 minutes
            
            if len(recent_accesses) == 0:  # No recent access
                value = self.l1_cache.get(key)
                if value is not None:
                    # Move to L2
                    self.l2_cache.put(key, value, importance_score=0.3)
                    # Remove from L1 (will happen automatically due to LRU)
    
    async def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        
        # Remove from all levels
        self.l1_cache._remove_entry(key)
        self.l2_cache._remove_entry(key)
        self.l3_cache._remove_entry(key)
        
        # Remove from semantic cache if present
        self.semantic_cache._remove_entry(key)
        
        # Clean up access patterns
        if key in self.access_patterns:
            del self.access_patterns[key]
        
        # Clean up related keys
        for related_set in self.related_keys.values():
            related_set.discard(key)
        if key in self.related_keys:
            del self.related_keys[key]
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern"""
        
        keys_to_invalidate = []
        
        # Find matching keys in all caches
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            for key in cache.cache.keys():
                if pattern in key:
                    keys_to_invalidate.append(key)
        
        # Invalidate all matching keys
        for key in keys_to_invalidate:
            await self.invalidate(key)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        semantic_stats = self.semantic_cache.stats
        
        total_hits = l1_stats.hits + l2_stats.hits + l3_stats.hits + semantic_stats.hits
        total_misses = l1_stats.misses + l2_stats.misses + l3_stats.misses + semantic_stats.misses
        total_requests = total_hits + total_misses
        
        return {
            'overall': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'overall_hit_ratio': total_hits / total_requests if total_requests > 0 else 0.0,
                'total_memory_mb': (l1_stats.memory_usage_bytes + 
                                   l2_stats.memory_usage_bytes + 
                                   l3_stats.memory_usage_bytes) / (1024 * 1024)
            },
            'l1_cache': {
                'hits': l1_stats.hits,
                'misses': l1_stats.misses,
                'hit_ratio': l1_stats.hit_ratio,
                'memory_usage_mb': l1_stats.memory_usage_bytes / (1024 * 1024),
                'entries': len(self.l1_cache.cache)
            },
            'l2_cache': {
                'hits': l2_stats.hits,
                'misses': l2_stats.misses,
                'hit_ratio': l2_stats.hit_ratio,
                'memory_usage_mb': l2_stats.memory_usage_bytes / (1024 * 1024),
                'entries': len(self.l2_cache.cache)
            },
            'l3_cache': {
                'hits': l3_stats.hits,
                'misses': l3_stats.misses,
                'hit_ratio': l3_stats.hit_ratio,
                'memory_usage_mb': l3_stats.memory_usage_bytes / (1024 * 1024),
                'entries': len(self.l3_cache.cache)
            },
            'semantic_cache': {
                'hits': semantic_stats.hits,
                'misses': semantic_stats.misses,
                'hit_ratio': semantic_stats.hit_ratio,
                'entries': len(self.semantic_cache.entries)
            },
            'prefetch': self.prefetch_stats,
            'patterns': {
                'tracked_keys': len(self.access_patterns),
                'related_key_pairs': sum(len(related) for related in self.related_keys.values()) // 2
            }
        }
    
    async def clear_all(self):
        """Clear all caches"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        self.semantic_cache.entries.clear()
        self.semantic_cache.embeddings.clear()
        self.access_patterns.clear()
        self.related_keys.clear()
    
    async def close(self):
        """Clean shutdown"""
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.prefetch_executor.shutdown(wait=True)
