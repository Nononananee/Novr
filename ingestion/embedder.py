"""
Document embedding generation for vector search.
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
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from copy import deepcopy
import json

from openai import RateLimitError, APIError
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import flexible providers with lazy loading
_embedding_client = None
_embedding_model = None
_tiktoken_encoding = None

def _get_tiktoken_encoding():
    """Lazy initialization of tiktoken encoding."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        try:
            model_name = os.getenv("LLM_CHOICE", "gpt-4")
            if "gpt-4" in model_name:
                _tiktoken_encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_name:
                _tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}")
            _tiktoken_encoding = None
    return _tiktoken_encoding

def _get_embedding_client():
    """Lazy initialization of embedding client."""
    global _embedding_client
    if _embedding_client is None:
        try:
            from ..agent.providers import get_embedding_client
            _embedding_client = get_embedding_client()
        except ImportError:
            # For direct execution or testing
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from agent.providers import get_embedding_client
            _embedding_client = get_embedding_client()
    return _embedding_client

def _get_embedding_model():
    """Lazy initialization of embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from ..agent.providers import get_embedding_model
            _embedding_model = get_embedding_model()
        except ImportError:
            # For direct execution or testing
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from agent.providers import get_embedding_model
            _embedding_model = get_embedding_model()
    return _embedding_model

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get default batch size from environment or use conservative default
DEFAULT_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
DEFAULT_MAX_CONCURRENCY = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "5"))  # Reduced for safety


async def _call_provider_create_embeddings(client, model: str, inputs: List[str]) -> List[List[float]]:
    """
    Robust call to provider embedding method.
    Supports:
      - client.embeddings.create(model=..., input=...) (OpenAI-style async or sync)
      - client.create_embeddings(model, inputs) adapter
    Returns list of embeddings (list of list[float]).
    """
    # Prefer provider adapter method if exists
    create_fn = None
    if hasattr(client, "create_embeddings"):
        create_fn = client.create_embeddings  # expected async
    else:
        # try OpenAI-style nested API (client.embeddings.create)
        try:
            create_fn = client.embeddings.create
        except Exception:
            # fallback to attribute search
            for name in ("embeddings_create", "embeddings_create_async"):
                if hasattr(client, name):
                    create_fn = getattr(client, name)
                    break

    if create_fn is None:
        raise RuntimeError("Embedding client does not expose a known create method")

    # call sync function in thread if not coroutinefunction
    if inspect.iscoroutinefunction(create_fn):
        raw = await create_fn(model=model, input=inputs)
    else:
        # sync — run in thread
        raw = await asyncio.to_thread(lambda: create_fn(model=model, input=inputs))

    # Extract embeddings robustly
    # Common shapes: OpenAI -> {"data":[{"embedding":[..]}, ...]}
    if isinstance(raw, dict) and "data" in raw:
        try:
            return [item.get("embedding") or item["embedding"] for item in raw["data"]]
        except Exception:
            # try other patterns
            pass

    # If raw has attribute .data (SDK objects)
    if hasattr(raw, "data"):
        try:
            return [getattr(item, "embedding", None) or item["embedding"] for item in raw.data]
        except Exception:
            pass

    # If provider already returned list of lists
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, float)):
        return raw

    # If raw is JSON string
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "data" in parsed:
                return [d.get("embedding") for d in parsed["data"]]
        except Exception:
            pass

    # Last resort - try to find embeddings anywhere
    # Flatten search for "embedding" fields in dicts
    if isinstance(raw, list):
        out = []
        for elem in raw:
            if isinstance(elem, dict) and "embedding" in elem:
                out.append(elem["embedding"])
        if out:
            return out

    raise RuntimeError("Could not extract embeddings from provider response")


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding generation."""
    api_calls: int = 0
    failures: int = 0
    retries: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def avg_latency(self) -> float:
        """Calculate average latency per API call."""
        return self.total_latency / max(1, self.api_calls)


class ThreadSafeEmbeddingCache:
    """Thread-safe cache for embeddings with TTL and model-aware keys."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_hours: Time-to-live in hours
        """
        self._cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _normalize_text_for_key(self, text: str) -> str:
        """Normalize text for consistent cache keys."""
        t = text.strip()
        t = re.sub(r'\s+', ' ', t)  # Collapse whitespace
        return t
    
    def _make_key(self, text: str, model: str) -> str:
        """Generate cache key with model information."""
        norm = self._normalize_text_for_key(text)
        content_hash = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        return f"{model}:{content_hash}"
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._make_key(text, model)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            embedding, created_at = self._cache[key]
            
            # Check TTL
            if datetime.now() - created_at > self.ttl:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            # Return deep copy to prevent mutation
            return deepcopy(embedding)
    
    def put(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        key = self._make_key(text, model)
        
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._access_times.keys(), 
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            now = datetime.now()
            # Store deep copy to prevent mutation
            self._cache[key] = (deepcopy(embedding), now)
            self._access_times[key] = now
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class EmbeddingGenerator:
    """Generates embeddings for document chunks with improved error handling and caching."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        use_cache: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model to use (lazy loaded if None)
            batch_size: Number of texts to process in parallel
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            max_concurrency: Maximum concurrent API calls
            use_cache: Whether to use caching
            cache_ttl_hours: Cache TTL in hours
        """
        self._model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrency)
        
        # Cache
        if use_cache:
            self._cache = ThreadSafeEmbeddingCache(ttl_hours=cache_ttl_hours)
        else:
            self._cache = None
        
        # Metrics with thread-safe updates
        self.metrics = EmbeddingMetrics()
        self._metrics_lock = threading.RLock()
        
        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
            "nomic-embed-text": {"dimensions": 768, "max_tokens": 2048}
        }
    
    @property
    def model(self) -> str:
        """Get the embedding model (lazy loaded)."""
        if self._model is None:
            self._model = _get_embedding_model()
        return self._model
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        if self.model not in self.model_configs:
            logger.warning(f"Unknown model {self.model}, using default config")
            return {"dimensions": 1536, "max_tokens": 8191}
        return self.model_configs[self.model]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken."""
        encoding = _get_tiktoken_encoding()
        if encoding:
            return len(encoding.encode(text))
        else:
            # Fallback estimation
            return len(text) // 4
    
    def _truncate_text(self, text: str) -> str:
        """Safely truncate text to fit model limits using accurate token counting."""
        max_tokens = self.config["max_tokens"]
        
        # Use tiktoken for accurate token counting
        encoding = _get_tiktoken_encoding()
        if encoding:
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate by tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            
            # Try to end at sentence boundary
            sentences = truncated_text.split('. ')
            if len(sentences) > 1:
                # Remove last incomplete sentence
                truncated_text = '. '.join(sentences[:-1]) + '.'
            
            logger.debug(f"Truncated text from {len(tokens)} to {len(encoding.encode(truncated_text))} tokens")
            return truncated_text
        else:
            # Fallback to character-based truncation
            max_chars = max_tokens * 3  # Conservative estimate
        
            if len(text) <= max_chars:
                return text
        
            # Truncate by words to avoid breaking mid-word
            words = text[:max_chars].split()
            if words:
                words = words[:-1]
        
            truncated = ' '.join(words)
            logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
            return truncated
    
    async def _call_embedding_api(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Call embedding API with concurrency control and retries.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Truncate texts
        processed_texts = [self._truncate_text(text) for text in texts]
        
        async with self._semaphore:  # Concurrency control
            for attempt in range(self.max_retries):
                start_time = asyncio.get_event_loop().time()
                
                try:
                    client = _get_embedding_client()
                    embeddings = await _call_provider_create_embeddings(client, self.model, processed_texts)
                    
                    # Update metrics thread-safely
                    with self._metrics_lock:
                        self.metrics.api_calls += 1
                        self.metrics.total_latency += asyncio.get_event_loop().time() - start_time
                    
                    return embeddings[0] if is_single else embeddings
                    
                except (RateLimitError, Exception) as e:
                    with self._metrics_lock:
                        self.metrics.retries += 1
                    
                    # Handle rate limits specially
                    is_rate_limit = (
                        isinstance(e, RateLimitError) or
                        "rate limit" in str(e).lower() or
                        "too many requests" in str(e).lower()
                    )
                    
                    if attempt == self.max_retries - 1:
                        with self._metrics_lock:
                            self.metrics.failures += 1
                        raise
                    
                    if is_rate_limit:
                        # Exponential backoff with jitter for rate limits
                        delay = self.retry_delay * (2 ** attempt)
                        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                        total_delay = min(delay + jitter, 60)  # Cap at 60 seconds
                        
                        logger.warning(f"Rate limit hit, retrying in {total_delay:.2f}s")
                        await asyncio.sleep(total_delay)
                    else:
                        # Regular API error
                        logger.error(f"API error: {e}")
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
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
        embedding = await self._call_embedding_api(text)
        
        # Cache result
        if self._cache:
            self._cache.put(text, self.model, embedding)
        
        return embedding
    
    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with intelligent caching.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append([0.0] * self.config["dimensions"])
                continue
            
            if self._cache:
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    results.append(cached)
                    with self._metrics_lock:
                        self.metrics.cache_hits += 1
                    continue
                with self._metrics_lock:
                    self.metrics.cache_misses += 1
            
            # Track texts that need embedding
            results.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                embeddings = await self._call_embedding_api(uncached_texts)
                
                # Fill in results and cache
                for idx, embedding in zip(uncached_indices, embeddings):
                    results[idx] = embedding
                    if self._cache:
                        self._cache.put(texts[idx], self.model, embedding)
                        
            except Exception as e:
                logger.error(f"Batch embedding failed, falling back to individual: {e}")
                # Fallback to individual processing
                for idx in uncached_indices:
                    try:
                        embedding = await self.generate_embedding(texts[idx])
                        results[idx] = embedding
                    except Exception as individual_error:
                        logger.error(f"Individual embedding failed: {individual_error}")
                        results[idx] = [0.0] * self.config["dimensions"]
        
        return results
    
    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates
        
        Returns:
            New list of chunks with embeddings added
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            try:
                # Generate embeddings for this batch
                embeddings = await self.generate_embeddings_batch(batch_texts)
                
                # Create new chunks with embeddings
                for chunk, embedding in zip(batch_chunks, embeddings):
                    # Update chunk metadata and add embedding
                    chunk.metadata.update({
                        "embedding_model": self.model,
                        "embedding_generated_at": datetime.now().isoformat(),
                        "embedding_dimension": len(embedding)
                    })
                    
                    # Set embedding directly on the chunk
                    chunk.embedding = embedding
                    embedded_chunks.append(chunk)
                
                # Progress update
                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)
                
                logger.debug(f"Processed batch {current_batch}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                
                # Add chunks with zero embeddings as fallback
                for chunk in batch_chunks:
                    chunk.metadata.update({
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now().isoformat(),
                        "embedding_dimension": self.config["dimensions"]
                    })
                    chunk.embedding = [0.0] * self.config["dimensions"]
                    embedded_chunks.append(chunk)
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        logger.info(f"Metrics - API calls: {self.metrics.api_calls}, "
                   f"Cache hits: {self.metrics.cache_hits}, "
                   f"Avg latency: {self.metrics.avg_latency():.3f}s")
        
        return embedded_chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query
        
        Returns:
            Query embedding
        """
        return await self.generate_embedding(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Get current metrics."""
        return self.metrics
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()


# Factory function
def create_embedder(
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingGenerator:
    """
    Create embedding generator.
    
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


# Example usage
async def main():
    """Example usage of the improved embedder."""
    from .chunker import ChunkingConfig, create_chunker, ChunkingStrategy
    
    # Create chunker and embedder
    config = ChunkingConfig(
        chunk_size=200, 
        use_semantic_splitting=False,
        strategy=ChunkingStrategy.NARRATIVE_FLOW
    )
    chunker = create_chunker(config)
    embedder = create_embedder(batch_size=8, use_cache=True)
    
    sample_novel_text = """
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
    
    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_novel_text,
        title="The Enchanted Library",
        source="chapter1.md",
        metadata={"genre": "fantasy", "chapter": 1}
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings with progress tracking
    def progress_callback(current, total):
        print(f"Processing batch {current}/{total}")
    
    embedded_chunks = await embedder.embed_chunks(chunks, progress_callback)
    
    # Display results
    for i, chunk in enumerate(embedded_chunks):
        embedding_dim = len(chunk.embedding) if chunk.has_embedding else 0
        print(f"Chunk {i}: {len(chunk.content)} chars, embedding dim: {embedding_dim}")
        print(f"  Characters: {chunk.metadata.get('characters', [])}")
        print(f"  Locations: {chunk.metadata.get('locations', [])}")
    
    # Test query embedding
    query_embedding = await embedder.embed_query("magical library discovery")
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Show metrics
    metrics = embedder.get_metrics()
    print(f"Metrics: API calls={metrics.api_calls}, Cache hits={metrics.cache_hits}, "
          f"Avg latency={metrics.avg_latency():.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
