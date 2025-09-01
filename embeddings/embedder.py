import numpy as np
import logging
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
try:
    # Tests may patch AsyncOpenAI on this module; provide a lightweight symbol
    from openai import AsyncOpenAI  # type: ignore
except Exception:
    class AsyncOpenAI:  # fallback stub for test patching
        pass
import torch

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = "intfloat/e5-large-v2", device: Optional[str] = None):
        """
        Initialize text embedder with sentence-transformers
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on (auto-detected if None)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Input text
            normalize: Whether to normalize embeddings
            
        Returns:
            Normalized embedding vector
        """
        return self.embed_texts([text], normalize=normalize)[0]
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Embed multiple texts in batches
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        
        try:
            # Add query prefix for e5 models (improves retrieval performance)
            if "e5" in self.model_name.lower():
                processed_texts = [f"query: {text}" for text in texts]
            else:
                processed_texts = texts
            
            # Generate embeddings
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_chunks(self, chunks: List[dict], text_key: str = "text", batch_size: int = 32) -> List[dict]:
        """
        Embed text chunks and add embeddings to chunk metadata
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key containing text in chunk dict
            batch_size: Batch size for processing
            
        Returns:
            Chunks with added 'embedding' key
        """
        if not chunks:
            return np.array([])

        # Extract texts
        texts = [chunk.get(text_key, "") for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        # Attach embeddings to chunks as lists for compatibility
        for chunk, embedding in zip(chunks, embeddings):
            try:
                chunk["embedding"] = embedding.tolist()
            except Exception:
                chunk["embedding"] = list(embedding)

        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def find_most_similar(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Convenience functions
def embed_text(text: str, model_name: str = "intfloat/e5-large-v2") -> np.ndarray:
    """Convenience function to embed single text"""
    embedder = TextEmbedder(model_name)
    return embedder.embed_text(text)

def embed_texts(texts: List[str], model_name: str = "intfloat/e5-large-v2", batch_size: int = 32) -> np.ndarray:
    """Convenience function to embed multiple texts"""
    embedder = TextEmbedder(model_name)
    return embedder.embed_texts(texts, batch_size=batch_size)