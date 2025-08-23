"""
Memory optimization for novel content processing and storage.
"""

import asyncio
import logging
import weakref
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import time
import pickle
import hashlib
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for novel content."""
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate size after initialization."""
        self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of cached data."""
        try:
            return len(pickle.dumps(self.data))
        except:
            # Fallback for non-picklable objects
            return len(str(self.data).encode('utf-8'))
    
    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class NovelContentCache:
    """Intelligent cache for novel content with memory optimization."""
    
    def __init__(self, max_size_mb: int = 100, max_entries: int = 1000):
        """
        Initialize novel content cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.cache_lock = asyncio.Lock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"Novel content cache initialized: {max_size_mb}MB, {max_entries} entries")
    
    def _generate_key(self, novel_id: str, content_type: str, **kwargs) -> str:
        """Generate cache key for content."""
        key_data = f"{novel_id}:{content_type}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, novel_id: str, content_type: str, **kwargs) -> Optional[Any]:
        """Get content from cache."""
        key = self._generate_key(novel_id, content_type, **kwargs)
        
        async with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access()
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.data
            else:
                self.misses += 1
                return None
    
    async def set(self, novel_id: str, content_type: str, data: Any, **kwargs):
        """Set content in cache."""
        key = self._generate_key(novel_id, content_type, **kwargs)
        entry = CacheEntry(data=data, timestamp=time.time())
        
        async with self.cache_lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_size_bytes -= old_entry.size_bytes
            
            # Check if we need to evict entries
            await self._evict_if_needed(entry.size_bytes)
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes
    
    async def _evict_if_needed(self, new_entry_size: int):
        """Evict entries if cache is full."""
        # Check size limit
        while (self.current_size_bytes + new_entry_size > self.max_size_bytes or 
               len(self.cache) >= self.max_entries):
            
            if not self.cache:
                break
            
            # Evict least recently used entry
            key, entry = self.cache.popitem(last=False)
            self.current_size_bytes -= entry.size_bytes
            self.evictions += 1
    
    async def invalidate(self, novel_id: str, content_type: str = None):
        """Invalidate cache entries for a novel."""
        async with self.cache_lock:
            keys_to_remove = []
            
            for key in self.cache.keys():
                if content_type:
                    cache_key = self._generate_key(novel_id, content_type)
                    if key.startswith(cache_key[:16]):  # Partial match
                        keys_to_remove.append(key)
                else:
                    if key.startswith(novel_id):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self.cache.pop(key)
                self.current_size_bytes -= entry.size_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }


class NovelMemoryManager:
    """Memory manager for novel processing operations."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.content_cache = NovelContentCache(max_size_mb=100)
        self.processing_cache = NovelContentCache(max_size_mb=50)
        self.weak_references: Dict[str, weakref.ref] = {}
        self.memory_stats = defaultdict(int)
        
        # Memory optimization settings
        self.chunk_size = 1000  # Process in chunks
        self.batch_size = 10    # Batch operations
        
        logger.info("Novel memory manager initialized")
    
    async def get_novel_content(self, novel_id: str, content_type: str, **kwargs) -> Optional[Any]:
        """Get novel content with caching."""
        # Try cache first
        cached_content = await self.content_cache.get(novel_id, content_type, **kwargs)
        if cached_content is not None:
            return cached_content
        
        # Load from database
        content = await self._load_content_from_db(novel_id, content_type, **kwargs)
        if content is not None:
            # Cache for future use
            await self.content_cache.set(novel_id, content_type, content, **kwargs)
        
        return content
    
    async def _load_content_from_db(self, novel_id: str, content_type: str, **kwargs) -> Optional[Any]:
        """Load content from database."""
        try:
            if content_type == "chapters":
                from .db_utils import get_novel_chapters
                return await get_novel_chapters(novel_id, **kwargs)
            
            elif content_type == "characters":
                from .db_utils import list_characters
                return await list_characters(novel_id)
            
            elif content_type == "character_arc":
                from .db_utils import get_character_arc
                character_id = kwargs.get("character_id")
                if character_id:
                    return await get_character_arc(character_id)
            
            elif content_type == "novel_search":
                from .db_utils import search_novel_content
                return await search_novel_content(novel_id, **kwargs)
            
            else:
                logger.warning(f"Unknown content type: {content_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {content_type} for novel {novel_id}: {e}")
            return None
    
    async def process_novel_in_chunks(
        self, 
        novel_id: str, 
        processing_func: callable, 
        chunk_size: int = None
    ) -> List[Any]:
        """Process novel content in memory-efficient chunks."""
        chunk_size = chunk_size or self.chunk_size
        
        # Get chapters for chunked processing
        chapters = await self.get_novel_content(novel_id, "chapters")
        if not chapters:
            return []
        
        results = []
        
        # Process in chunks
        for i in range(0, len(chapters), chunk_size):
            chunk = chapters[i:i + chunk_size]
            
            try:
                chunk_results = await processing_func(chunk)
                results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])
                
                # Allow garbage collection between chunks
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                continue
        
        return results
    
    async def batch_character_analysis(
        self, 
        novel_id: str, 
        analysis_func: callable,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """Perform character analysis in batches."""
        batch_size = batch_size or self.batch_size
        
        # Get characters
        characters = await self.get_novel_content(novel_id, "characters")
        if not characters:
            return {}
        
        results = {}
        
        # Process in batches
        for i in range(0, len(characters), batch_size):
            batch = characters[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for character in batch:
                task = analysis_func(character)
                batch_tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for character, result in zip(batch, batch_results):
                    if not isinstance(result, Exception):
                        results[character["id"]] = result
                    else:
                        logger.error(f"Error analyzing character {character['name']}: {result}")
                
            except Exception as e:
                logger.error(f"Error in batch character analysis: {e}")
        
        return results
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across the system."""
        optimization_results = {
            "cache_optimizations": [],
            "memory_freed_mb": 0,
            "performance_improvements": []
        }
        
        # Optimize content cache
        content_stats = self.content_cache.get_stats()
        if content_stats["utilization"] > 0.9:
            # Cache is nearly full, increase eviction
            old_size = self.content_cache.max_size_bytes
            self.content_cache.max_size_bytes = int(old_size * 1.2)  # Increase by 20%
            optimization_results["cache_optimizations"].append("Increased content cache size")
        
        # Optimize processing cache
        processing_stats = self.processing_cache.get_stats()
        if processing_stats["hit_rate"] < 0.3:
            # Low hit rate, adjust cache strategy
            await self.processing_cache.invalidate("")  # Clear all
            optimization_results["cache_optimizations"].append("Cleared low-efficiency processing cache")
        
        # Clean up weak references
        dead_refs = []
        for key, ref in self.weak_references.items():
            if ref() is None:
                dead_refs.append(key)
        
        for key in dead_refs:
            del self.weak_references[key]
        
        if dead_refs:
            optimization_results["memory_freed_mb"] += len(dead_refs) * 0.1  # Estimate
            optimization_results["performance_improvements"].append(f"Cleaned {len(dead_refs)} dead references")
        
        # Adjust chunk sizes based on performance
        if self.memory_stats["processing_time"] > 5000:  # 5 seconds
            self.chunk_size = max(500, self.chunk_size - 100)
            optimization_results["performance_improvements"].append("Reduced chunk size for better performance")
        
        return optimization_results
    
    async def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        content_stats = self.content_cache.get_stats()
        processing_stats = self.processing_cache.get_stats()
        
        return {
            "content_cache": content_stats,
            "processing_cache": processing_stats,
            "weak_references": len(self.weak_references),
            "chunk_size": self.chunk_size,
            "batch_size": self.batch_size,
            "memory_stats": dict(self.memory_stats),
            "optimization_recommendations": await self._get_optimization_recommendations()
        }
    
    async def _get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        content_stats = self.content_cache.get_stats()
        processing_stats = self.processing_cache.get_stats()
        
        if content_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing content cache size for better hit rate")
        
        if content_stats["utilization"] > 0.95:
            recommendations.append("Content cache is nearly full, consider increasing size or reducing retention")
        
        if processing_stats["evictions"] > processing_stats["hits"]:
            recommendations.append("Processing cache is evicting too frequently, consider optimization")
        
        if self.memory_stats.get("processing_time", 0) > 3000:
            recommendations.append("Processing time is high, consider reducing chunk size")
        
        if len(self.weak_references) > 1000:
            recommendations.append("High number of weak references, consider cleanup")
        
        return recommendations
    
    async def preload_novel_data(self, novel_id: str, priority_content: List[str] = None):
        """Preload novel data into cache for better performance."""
        priority_content = priority_content or ["chapters", "characters"]
        
        preload_tasks = []
        
        for content_type in priority_content:
            task = self.get_novel_content(novel_id, content_type)
            preload_tasks.append(task)
        
        try:
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            logger.info(f"Preloaded {len(priority_content)} content types for novel {novel_id}")
        except Exception as e:
            logger.error(f"Error preloading novel data: {e}")
    
    async def cleanup_novel_data(self, novel_id: str):
        """Clean up cached data for a specific novel."""
        await self.content_cache.invalidate(novel_id)
        await self.processing_cache.invalidate(novel_id)
        
        # Clean up weak references
        keys_to_remove = [key for key in self.weak_references.keys() if key.startswith(novel_id)]
        for key in keys_to_remove:
            del self.weak_references[key]
        
        logger.info(f"Cleaned up cached data for novel {novel_id}")


# Global memory manager instance
novel_memory_manager = NovelMemoryManager()


# Decorator for memory-optimized operations
def memory_optimized(chunk_processing: bool = False):
    """Decorator for memory-optimized novel operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            novel_id = kwargs.get('novel_id') or (args[1] if len(args) > 1 else None)
            
            if novel_id and chunk_processing:
                # Use chunked processing
                return await novel_memory_manager.process_novel_in_chunks(
                    novel_id, 
                    lambda chunk: func(*args, **kwargs)
                )
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator