"""
Caching module untuk API performance improvement.
"""

import json
import hashlib
import asyncio
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
import logging

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

from .api_config import APIConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """Async cache manager dengan Redis fallback ke memory cache."""
    
    def __init__(self):
        self.redis_client: Optional[Any] = None  # Type hint untuk Redis client
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.is_redis_connected = False
        
    async def initialize(self):
        """Initialize cache manager."""
        if APIConfig.ENABLE_CACHING and REDIS_AVAILABLE:
            try:
                self.redis_client = aioredis.from_url(
                    APIConfig.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                await self.redis_client.ping()
                self.is_redis_connected = True
                logger.info("✅ Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed, using memory cache: {e}")
                self.redis_client = None
                self.is_redis_connected = False
        else:
            logger.info("📝 Using memory cache (Redis disabled or unavailable)")
    
    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_redis_connected = False
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key dari data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value dari cache."""
        try:
            if self.is_redis_connected:
                # Redis cache
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Memory cache
                cached_item = self.memory_cache.get(key)
                if cached_item:
                    # Check expiration
                    if datetime.now() < cached_item["expires_at"]:
                        return cached_item["data"]
                    else:
                        # Expired, remove from cache
                        del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value ke cache."""
        try:
            ttl = ttl or APIConfig.CACHE_TTL
            
            if self.is_redis_connected:
                # Redis cache
                serialized_value = json.dumps(value)
                await self.redis_client.setex(key, ttl, serialized_value)
                return True
            else:
                # Memory cache
                expires_at = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[key] = {
                    "data": value,
                    "expires_at": expires_at
                }
                
                # Clean expired entries (simple cleanup)
                await self._cleanup_expired_memory_cache()
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value dari cache."""
        try:
            if self.is_redis_connected:
                await self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self, pattern: str = None) -> bool:
        """Clear cache berdasarkan pattern."""
        try:
            if self.is_redis_connected:
                if pattern:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    await self.redis_client.flushall()
            else:
                if pattern:
                    # Simple pattern matching untuk memory cache
                    keys_to_delete = [
                        key for key in self.memory_cache.keys() 
                        if pattern.replace("*", "") in key
                    ]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                else:
                    self.memory_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def _cleanup_expired_memory_cache(self):
        """Clean up expired entries dari memory cache."""
        try:
            now = datetime.now()
            expired_keys = [
                key for key, item in self.memory_cache.items()
                if now >= item["expires_at"]
            ]
            for key in expired_keys:
                del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Memory cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.is_redis_connected:
                info = await self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "memory_usage": info.get("used_memory_human", "N/A"),
                    "keys": await self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            else:
                return {
                    "type": "memory",
                    "connected": True,
                    "keys": len(self.memory_cache),
                    "memory_usage": f"{len(str(self.memory_cache))} bytes (approx)"
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {
                "type": "unknown",
                "connected": False,
                "error": str(e)
            }


# Global cache manager instance
cache_manager = CacheManager()


def cache_key_for_chat(session_id: str, message: str) -> str:
    """Generate cache key untuk chat response."""
    return cache_manager._generate_key("chat", {"session_id": session_id, "message": message})


def cache_key_for_search(search_type: str, query: str, limit: int = 10) -> str:
    """Generate cache key untuk search results."""
    return cache_manager._generate_key("search", {
        "type": search_type,
        "query": query,
        "limit": limit
    })


def cache_key_for_generation(content: str, generation_type: str, max_tokens: int) -> str:
    """Generate cache key untuk generated content."""
    return cache_manager._generate_key("generation", {
        "content": content,
        "type": generation_type,
        "max_tokens": max_tokens
    })


async def cached_operation(
    cache_key: str,
    operation_func,
    ttl: Optional[int] = None,
    force_refresh: bool = False
) -> Any:
    """
    Execute operation dengan caching.
    
    Args:
        cache_key: Cache key
        operation_func: Async function untuk execute jika cache miss
        ttl: Time to live untuk cache entry
        force_refresh: Force refresh cache
    
    Returns:
        Operation result
    """
    if not force_refresh:
        # Try to get dari cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_result
    
    # Cache miss atau force refresh, execute operation
    logger.debug(f"Cache miss for key: {cache_key}")
    result = await operation_func()
    
    # Store ke cache
    await cache_manager.set(cache_key, result, ttl)
    
    return result


class CacheStats:
    """Track cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
    
    def record_hit(self):
        """Record cache hit."""
        self.hits += 1
    
    def record_miss(self):
        """Record cache miss.""" 
        self.misses += 1
    
    def record_error(self):
        """Record cache error."""
        self.errors += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": self.get_hit_rate(),
            "total_requests": self.hits + self.misses
        }
    
    def reset(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0


# Global cache stats tracker
cache_stats = CacheStats()
