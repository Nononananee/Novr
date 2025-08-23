"""
Document embedding generation for vector search - Improved Version.
"""

import os
import asyncio
import logging
import hashlib
import random
import threading
import inspect
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Protocol
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from copy import deepcopy
import json
from contextlib import asynccontextmanager
import weakref

from openai import RateLimitError, APIError
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Load environment variables early
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configuration with validation
def _get_env_int(key: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
    """Get and validate integer environment variable."""
    try:
        value = int(os.getenv(key, str(default)))
        return max(min_val, min(max_val, value))
    except ValueError:
        logger.warning(f"Invalid {key}, using default {default}")
        return default

DEFAULT_BATCH_SIZE = _get_env_int("EMBEDDING_BATCH_SIZE", 16, 1, 100)
DEFAULT_MAX_CONCURRENCY = _get_env_int("EMBEDDING_MAX_CONCURRENCY", 5, 1, 20)

# Provider interface
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    async def create_embeddings(self, model: str, inputs: List[str]) -> List[List[float]]:
        """Create embeddings for input texts."""
        ...

# Thread-safe singleton pattern for global resources
class _GlobalResources:
    """Thread-safe singleton for global embedding resources."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._embedding_client = None
        self._embedding_model = None
        self._tiktoken_encoding = None
        self._client_lock = threading.RLock()
        self._initialized = True
    
    def get_tiktoken_encoding(self) -> Optional[tiktoken.Encoding]:
        """Thread-safe tiktoken encoding initialization."""
        if self._tiktoken_encoding is None:
            with self._client_lock:
                if self._tiktoken_encoding is None:
                    try:
                        model_name = os.getenv("LLM_CHOICE", "gpt-4")
                        if "gpt-4" in model_name:
                            self._tiktoken_encoding = tiktoken.encoding_for_model("gpt-4")
                        elif "gpt-3.5" in model_name:
                            self._tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                        else:
                            self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                    except Exception as e:
                        logger.warning(f"Failed to initialize tiktoken: {e}")
                        self._tiktoken_encoding = None
        return self._tiktoken_encoding
    
    def get_embedding_client(self):
        """Thread-safe embedding client initialization."""
        if self._embedding_client is None:
            with self._client_lock:
                if self._embedding_client is None:
                    try:
                        from ..agent.providers import get_embedding_client
                        self._embedding_client = get_embedding_client()
                    except ImportError as e:
                        logger.error(f"Failed to import embedding client: {e}")
                        raise ImportError("Cannot import embedding client from agent.providers") from e
        return self._embedding_client
    
    def get_embedding_model(self) -> str:
        """Thread-safe embedding model initialization."""
        if self._embedding_model is None:
            with self._client_lock:
                if self._embedding_model is None:
                    try:
                        from ..agent.providers import get_embedding_model
                        self._embedding_model = get_embedding_model()
                    except ImportError as e:
                        logger.error(f"Failed to import embedding model: {e}")
                        raise ImportError("Cannot import embedding model from agent.providers") from e
        return self._embedding_model

# Global instance
_global_resources = _GlobalResources()

async def _call_provider_create_embeddings(client, model: str, inputs: List[str]) -> List[List[float]]:
    """
    Robust call to provider embedding method with better error handling.
    """
    if not inputs:
        return []
    
    create_fn = None
    
    # Try different provider patterns
    if hasattr(client, "create_embeddings"):
        create_fn = client.create_embeddings
    elif hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
        create_fn = client.embeddings.create
    else:
        # Search for common method names
        for method_name in ("embeddings_create", "embeddings_create_async", "embed"):
            if hasattr(client, method_name):
                create_fn = getattr(client, method_name)
                break
    
    if create_fn is None:
        raise ValueError("Embedding client does not expose a known create method")
    
    try:
        # Call function (async or sync)
        if inspect.iscoroutinefunction(create_fn):
            raw = await create_fn(model=model, input=inputs)
        else:
            raw = await asyncio.to_thread(lambda: create_fn(model=model, input=inputs))
        
        # Extract embeddings with multiple fallback patterns
        return _extract_embeddings(raw)
        
    except (RateLimitError, APIError) as e:
        logger.error(f"API error in embedding call: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in embedding call: {e}")
        raise RuntimeError(f"Embedding API call failed: {e}") from e

def _extract_embeddings(raw_response) -> List[List[float]]:
    """Extract embeddings from various provider response formats."""
    
    # OpenAI-style response
    if isinstance(raw_response, dict) and "data" in raw_response:
        try:
            return [item["embedding"] for item in raw_response["data"]]
        except (KeyError, TypeError) as e:
            logger.debug(f"Failed OpenAI-style extraction: {e}")
    
    # SDK object response
    if hasattr(raw_response, "data"):
        try:
            return [getattr(item, "embedding") for item in raw_response.data]
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed SDK object extraction: {e}")
    
    # Direct list response
    if isinstance(raw_response, list) and raw_response:
        if isinstance(raw_response[0], list) and isinstance(raw_response[0][0], (int, float)):
            return raw_response
    
    # JSON string response
    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict) and "data" in parsed:
                return [d["embedding"] for d in parsed["data"]]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed JSON string extraction: {e}")
    
    # Search for embedding fields in nested structures
    if isinstance(raw_response, list):
        embeddings = []
        for item in raw_response:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])
        if embeddings:
            return embeddings
    
    raise ValueError(f"Could not extract embeddings from response type: {type(raw_response)}")

@dataclass
class EmbeddingMetrics:
    """Thread-safe metrics for embedding generation."""
    api_calls: int = 0
    failures: int = 0
    retries: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def avg_latency(self) -> float:
        """Calculate average latency per API call."""
        return self.total_latency / max(1, self.api_calls) if self.api_calls > 0 else 0.0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total) if total > 0 else 0.0

class ThreadSafeEmbeddingCache:
    """Improved thread-safe cache with better memory management."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """Initialize cache with validation."""
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
            
        self._cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # Use weak references for cleanup
        self._cleanup_refs = weakref.WeakSet()
    
    def _normalize_text_for_key(self, text: str) -> str:
        """Enhanced text normalization for cache keys."""
        # Handle empty/whitespace
        if not text or not text.strip():
            return ""
        
        # Normalize unicode
        import unicodedata
        normalized = unicodedata.normalize('NFKC', text.strip())
        
        # Collapse whitespace but preserve structure
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common variations
        normalized = normalized.lower()
        
        return normalized
    
    def _make_key(self, text: str, model: str) -> str:
        """Generate robust cache key."""
        norm_text = self._normalize_text_for_key(text)
        
        # Include text length to avoid collisions
        content = f"{model}:{len(text)}:{norm_text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache with TTL check."""
        key = self._make_key(text, model)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            embedding, created_at = self._cache[key]
            
            # Check TTL using UTC
            if datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc) > self.ttl:
                self._evict_key(key)
                return None
            
            # Update access time
            self._access_times[key] = datetime.now(timezone.utc)
            
            # Return shallow copy (embeddings are typically not modified)
            return embedding[:]
    
    def put(self, text: str, model: str, embedding: List[float]):
        """Store embedding with LRU eviction."""
        if not embedding:
            logger.warning("Attempted to cache empty embedding")
            return
            
        key = self._make_key(text, model)
        
        with self._lock:
            # Evict if needed
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            now = datetime.now(timezone.utc)
            # Store shallow copy
            self._cache[key] = (embedding[:], now)
            self._access_times[key] = now
    
    def _evict_key(self, key: str):
        """Remove specific key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._access_times:
            return
            
        oldest_key = min(self._access_times.keys(), 
                        key=lambda k: self._access_times[k])
        self._evict_key(oldest_key)
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now(timezone.utc)
        expired_keys = []
        
        with self._lock:
            for key, (_, created_at) in self._cache.items():
                if now - created_at.replace(tzinfo=timezone.utc) > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._evict_key(key)

class EmbeddingGenerator:
    """Enhanced embedding generator with better error handling and resource management."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
        cache_max_size: int = 1000
    ):
        """Initialize with validation."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative") 
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
            
        self._model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        
        # Initialize semaphore lazily to avoid event loop issues
        self._semaphore = None
        self._max_concurrency = max_concurrency
        
        # Cache with better configuration
        if use_cache:
            self._cache = ThreadSafeEmbeddingCache(
                max_size=cache_max_size, 
                ttl_hours=cache_ttl_hours
            )
        else:
            self._cache = None
        
        # Thread-safe metrics
        self.metrics = EmbeddingMetrics()
        self._metrics_lock = threading.RLock()
        
        # Model configurations with validation
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191}, 
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
            "nomic-embed-text": {"dimensions": 768, "max_tokens": 2048}
        }
        
        # Cleanup flag
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
    
    async def close(self):
        """Cleanup resources."""
        self._closed = True
        if self._cache:
            self._cache.clear()
    
    def _ensure_semaphore(self):
        """Lazy semaphore initialization."""
        if self._semaphore is None:
            try:
                self._semaphore = asyncio.Semaphore(self._max_concurrency)
            except RuntimeError:
                # No running event loop, will be created when needed
                pass
    
    @property
    def model(self) -> str:
        """Get embedding model with caching."""
        if self._model is None:
            self._model = _global_resources.get_embedding_model()
        return self._model
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration with fallback."""
        model_key = self.model
        if model_key not in self.model_configs:
            logger.warning(f"Unknown model {model_key}, using default config")
            return {"dimensions": 1536, "max_tokens": 8191}
        return self.model_configs[model_key]
    
    def _count_tokens(self, text: str) -> int:
        """Accurate token counting with fallback."""
        encoding = _global_resources.get_tiktoken_encoding()
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                logger.debug(f"Token counting failed: {e}")
        
        # Fallback estimation (conservative)
        return len(text) // 3
    
    def _truncate_text(self, text: str) -> str:
        """Smart text truncation with sentence boundary preservation."""
        if not text:
            return text
            
        max_tokens = self.config["max_tokens"]
        encoding = _global_resources.get_tiktoken_encoding()
        
        if encoding:
            try:
                tokens = encoding.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                
                # Truncate by tokens
                truncated_tokens = tokens[:max_tokens]
                truncated_text = encoding.decode(truncated_tokens)
                
                # Try to end at sentence boundary
                sentences = re.split(r'[.!?]+', truncated_text)
                if len(sentences) > 1:
                    # Keep all complete sentences except the last (potentially incomplete)
                    complete_text = '. '.join(sentences[:-1])
                    if complete_text:
                        truncated_text = complete_text + '.'
                
                logger.debug(f"Truncated from {len(tokens)} to {len(encoding.encode(truncated_text))} tokens")
                return truncated_text
                
            except Exception as e:
                logger.warning(f"Token-based truncation failed: {e}")
        
        # Fallback to character-based truncation
        max_chars = max_tokens * 3  # Conservative estimate
        if len(text) <= max_chars:
            return text
        
        # Truncate by sentences first
        truncated = text[:max_chars]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_chars * 0.7:  # If we can keep most of the text
            truncated = truncated[:last_sentence + 1]
        else:
            # Truncate by words
            words = truncated.split()
            if len(words) > 1:
                truncated = ' '.join(words[:-1])
        
        logger.debug(f"Character-based truncation: {len(text)} -> {len(truncated)} chars")
        return truncated
    
    async def _call_embedding_api(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Enhanced API call with better error handling."""
        if self._closed:
            raise RuntimeError("EmbeddingGenerator is closed")
            
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        if not texts:
            return []
        
        # Validate and truncate texts
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"Expected str, got {type(text)}")
            processed_texts.append(self._truncate_text(text))
        
        # Lazy semaphore initialization
        self._ensure_semaphore()
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        
        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                start_time = asyncio.get_event_loop().time()
                
                try:
                    client = _global_resources.get_embedding_client()
                    embeddings = await _call_provider_create_embeddings(
                        client, self.model, processed_texts
                    )
                    
                    if not embeddings:
                        raise ValueError("Provider returned empty embeddings")
                    
                    if len(embeddings) != len(processed_texts):
                        raise ValueError(f"Expected {len(processed_texts)} embeddings, got {len(embeddings)}")
                    
                    # Update metrics atomically
                    latency = asyncio.get_event_loop().time() - start_time
                    with self._metrics_lock:
                        self.metrics.api_calls += 1
                        self.metrics.total_latency += latency
                    
                    return embeddings[0] if is_single else embeddings
                    
                except (RateLimitError, APIError) as e:
                    with self._metrics_lock:
                        if attempt < self.max_retries:
                            self.metrics.retries += 1
                        else:
                            self.metrics.failures += 1
                    
                    if attempt == self.max_retries:
                        logger.error(f"Final API attempt failed: {e}")
                        raise
                    
                    # Smart retry delay
                    if "rate limit" in str(e).lower():
                        delay = min(self.retry_delay * (2 ** attempt), 60)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        logger.warning(f"Rate limit, retrying in {total_delay:.2f}s")
                    else:
                        total_delay = self.retry_delay * (attempt + 1)
                        logger.warning(f"API error, retrying in {total_delay:.2f}s: {e}")
                    
                    await asyncio.sleep(total_delay)
                    
                except Exception as e:
                    with self._metrics_lock:
                        if attempt < self.max_retries:
                            self.metrics.retries += 1
                        else:
                            self.metrics.failures += 1
                    
                    if attempt == self.max_retries:
                        logger.error(f"Embedding generation failed after {self.max_retries} retries: {e}")
                        raise RuntimeError(f"Embedding generation failed: {e}") from e
                    
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate single embedding with enhanced error handling."""
        if not text or not text.strip():
            return [0.0] * self.config["dimensions"]
        
        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached is not None:
                with self._metrics_lock:
                    self.metrics.cache_hits += 1
                return cached
            
            with self._metrics_lock:
                self.metrics.cache_misses += 1
        
        # Generate embedding
        try:
            embedding = await self._call_embedding_api(text)
            
            # Validate embedding
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Invalid embedding format")
            
            if len(embedding) != self.config["dimensions"]:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)} vs {self.config['dimensions']}")
            
            # Cache result
            if self._cache:
                self._cache.put(text, self.model, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text (len={len(text)}): {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch with smart caching and fallback."""
        if not texts:
            return []
        
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []
        
        # Check cache and handle empty texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = [0.0] * self.config["dimensions"]
                continue
            
            if self._cache:
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    results[i] = cached
                    with self._metrics_lock:
                        self.metrics.cache_hits += 1
                    continue
                
                with self._metrics_lock:
                    self.metrics.cache_misses += 1
            
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            try:
                embeddings = await self._call_embedding_api(uncached_texts)
                
                # Fill results and cache
                for idx, embedding in zip(uncached_indices, embeddings):
                    if embedding and len(embedding) > 0:
                        results[idx] = embedding
                        if self._cache:
                            self._cache.put(texts[idx], self.model, embedding)
                    else:
                        logger.warning(f"Empty embedding for text at index {idx}")
                        results[idx] = [0.0] * self.config["dimensions"]
                        
            except Exception as e:
                logger.error(f"Batch embedding failed, trying individual fallback: {e}")
                
                # Individual fallback with error isolation
                for idx in uncached_indices:
                    try:
                        embedding = await self.generate_embedding(texts[idx])
                        results[idx] = embedding
                    except Exception as individual_error:
                        logger.error(f"Individual embedding failed for index {idx}: {individual_error}")
                        # Use zero vector as last resort
                        results[idx] = [0.0] * self.config["dimensions"]
        
        # Ensure all results are filled
        for i, result in enumerate(results):
            if result is None:
                logger.error(f"Missing result for index {i}")
                results[i] = [0.0] * self.config["dimensions"]
        
        return results
    
    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DocumentChunk]:
        """Embed chunks with enhanced progress tracking and error recovery."""
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using model {self.model}")
        
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        successful_batches = 0
        failed_batches = 0
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[batch_idx:batch_idx + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            current_batch = (batch_idx // self.batch_size) + 1
            
            try:
                embeddings = await self.generate_embeddings_batch(batch_texts)
                
                # Create embedded chunks
                for chunk, embedding in zip(batch_chunks, embeddings):
                    # Create new chunk with embedding
                    new_chunk = DocumentChunk(
                        content=chunk.content,
                        start_idx=chunk.start_idx,
                        end_idx=chunk.end_idx,
                        metadata=chunk.metadata.copy(),
                        embedding=embedding
                    )
                    
                    # Update metadata
                    new_chunk.metadata.update({
                        "embedding_model": self.model,
                        "embedding_generated_at": datetime.now(timezone.utc).isoformat(),
                        "embedding_dimension": len(embedding),
                        "embedding_batch": current_batch
                    })
                    
                    embedded_chunks.append(new_chunk)
                
                successful_batches += 1
                logger.debug(f"Successfully processed batch {current_batch}/{total_batches}")
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"Failed to process batch {current_batch}/{total_batches}: {e}")
                
                # Add chunks with error markers instead of zero embeddings
                for chunk in batch_chunks:
                    new_chunk = DocumentChunk(
                        content=chunk.content,
                        start_idx=chunk.start_idx,
                        end_idx=chunk.end_idx,
                        metadata=chunk.metadata.copy(),
                        embedding=None  # Explicitly None for failed embeddings
                    )
                    
                    new_chunk.metadata.update({
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now(timezone.utc).isoformat(),
                        "embedding_failed": True,
                        "embedding_batch": current_batch
                    })
                    
                    embedded_chunks.append(new_chunk)
            
            # Progress callback
            if progress_callback:
                try:
                    progress_callback(current_batch, total_batches)
                except Exception as cb_error:
                    logger.warning(f"Progress callback failed: {cb_error}")
        
        # Summary logging
        logger.info(f"Embedding complete: {len(embedded_chunks)} chunks processed")
        logger.info(f"Batches - Success: {successful_batches}, Failed: {failed_batches}")
        logger.info(f"Metrics - API calls: {self.metrics.api_calls}, "
                   f"Cache hits: {self.metrics.cache_hits}, "
                   f"Hit rate: {self.metrics.hit_rate():.2%}, "
                   f"Avg latency: {self.metrics.avg_latency():.3f}s")
        
        return embedded_chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query with validation."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        return await self.generate_embedding(query.strip())
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for this model."""
        return self.config["dimensions"]
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Get current metrics (thread-safe copy)."""
        with self._metrics_lock:
            return EmbeddingMetrics(
                api_calls=self.metrics.api_calls,
                failures=self.metrics.failures,
                retries=self.metrics.retries,
                total_latency=self.metrics.total_latency,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses
            )
    
    def clear_cache(self):
        """Clear embedding cache and cleanup expired entries."""
        if self._cache:
            self._cache.clear()
    
    def cleanup_cache(self):
        """Remove expired cache entries."""
        if self._cache:
            self._cache.cleanup_expired()


@asynccontextmanager
async def embedding_context(
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingGenerator:
    """Async context manager for embedding generator with proper cleanup."""
    embedder = None
    try:
        embedder = EmbeddingGenerator(
            model=model,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs
        )
        yield embedder
    finally:
        if embedder:
            await embedder.close()


def create_embedder(
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingGenerator:
    """
    Create embedding generator with validation.
    
    Args:
        model: Embedding model to use
        batch_size: Batch size for processing
        use_cache: Whether to use caching
        **kwargs: Additional arguments for EmbeddingGenerator
    
    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(
        model=model,
        batch_size=batch_size,
        use_cache=use_cache,
        **kwargs
    )


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class EmbeddingBatchProcessor:
    """Advanced batch processor for large-scale embedding generation."""
    
    def __init__(self, embedder: EmbeddingGenerator, max_concurrent_batches: int = 3):
        self.embedder = embedder
        self.max_concurrent_batches = max_concurrent_batches
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_large_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[List[float]]:
        """Process very large batches with concurrent sub-batches."""
        if not texts:
            return []
        
        batch_size = self.embedder.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        async def process_sub_batch(start_idx: int, batch_num: int) -> Tuple[int, List[List[float]]]:
            """Process a single sub-batch."""
            async with self._semaphore:
                end_idx = min(start_idx + batch_size, len(texts))
                sub_batch = texts[start_idx:end_idx]
                
                if progress_callback:
                    try:
                        progress_callback(batch_num, total_batches, f"Processing batch {batch_num}")
                    except Exception:
                        pass
                
                embeddings = await self.embedder.generate_embeddings_batch(sub_batch)
                return start_idx, embeddings
        
        # Create tasks for all sub-batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            task = asyncio.create_task(process_sub_batch(i, batch_num))
            tasks.append(task)
        
        # Collect results in order
        results = [None] * len(texts)
        completed_batches = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                start_idx, embeddings = await task
                end_idx = min(start_idx + len(embeddings), len(texts))
                results[start_idx:end_idx] = embeddings
                
                completed_batches += 1
                if progress_callback:
                    try:
                        progress_callback(
                            completed_batches, 
                            total_batches, 
                            f"Completed {completed_batches}/{total_batches} batches"
                        )
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fill with zero embeddings for failed batches
                start_idx = tasks.index(task) * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                zero_embedding = [0.0] * self.embedder.get_embedding_dimension()
                for i in range(start_idx, end_idx):
                    if results[i] is None:
                        results[i] = zero_embedding
        
        # Ensure no None values remain
        zero_embedding = [0.0] * self.embedder.get_embedding_dimension()
        for i, result in enumerate(results):
            if result is None:
                results[i] = zero_embedding
        
        return results


# Example usage with improved error handling
async def main():
    """Enhanced example usage demonstrating best practices."""
    from .chunker import ChunkingConfig, create_chunker, ChunkingStrategy
    
    # Configuration
    config = ChunkingConfig(
        chunk_size=200, 
        use_semantic_splitting=False,
        strategy=ChunkingStrategy.NARRATIVE_FLOW
    )
    
    sample_text = """
    Chapter 1: The Discovery
    
    Elena stepped into the ancient library, her footsteps echoing in the vast silence.
    Dust motes danced in the shafts of sunlight that pierced through the tall windows.
    She had been searching for this place for months, following cryptic clues left
    by her grandmother.
    
    "There," she whispered, spotting the ornate tome on the highest shelf.
    
    As her fingers touched the leather binding, the book began to glow with
    an otherworldly light. Elena gasped, stepping back as ancient symbols
    appeared on the cover, pulsing with magical energy.
    
    "You have found it at last," came a voice from behind her.
    """
    
    # Use context manager for proper resource management
    async with embedding_context(
        batch_size=8, 
        use_cache=True,
        max_concurrency=3
    ) as embedder:
        
        # Create chunker
        chunker = create_chunker(config)
        
        # Chunk document
        chunks = chunker.chunk_document(
            content=sample_text,
            title="The Enchanted Library",
            source="chapter1.md",
            metadata={"genre": "fantasy", "chapter": 1}
        )
        
        print(f"Created {len(chunks)} chunks")
        
        # Progress tracking
        def progress_callback(current: int, total: int):
            percentage = (current / total) * 100
            print(f"Progress: {current}/{total} ({percentage:.1f}%)")
        
        try:
            # Generate embeddings
            embedded_chunks = await embedder.embed_chunks(chunks, progress_callback)
            
            # Display results
            successful_embeddings = 0
            for i, chunk in enumerate(embedded_chunks):
                has_embedding = chunk.embedding is not None
                embedding_dim = len(chunk.embedding) if has_embedding else 0
                
                if has_embedding:
                    successful_embeddings += 1
                
                print(f"Chunk {i}: {len(chunk.content)} chars")
                print(f"  Embedding: {'✓' if has_embedding else '✗'} (dim: {embedding_dim})")
                print(f"  Metadata keys: {list(chunk.metadata.keys())}")
                
                if chunk.metadata.get('embedding_failed'):
                    print(f"  Error: {chunk.metadata.get('embedding_error', 'Unknown')}")
            
            print(f"\nSuccessful embeddings: {successful_embeddings}/{len(embedded_chunks)}")
            
            # Test query embedding
            if successful_embeddings > 0:
                query_embedding = await embedder.embed_query("magical library discovery")
                print(f"Query embedding dimension: {len(query_embedding)}")
            
            # Show final metrics
            metrics = embedder.get_metrics()
            print(f"\nFinal Metrics:")
            print(f"  API calls: {metrics.api_calls}")
            print(f"  Cache hit rate: {metrics.hit_rate():.2%}")
            print(f"  Average latency: {metrics.avg_latency():.3f}s")
            print(f"  Failures: {metrics.failures}")
            print(f"  Retries: {metrics.retries}")
            
        except Exception as e:
            logger.error(f"Embedding process failed: {e}")
            raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())