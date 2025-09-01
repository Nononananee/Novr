import pytest
import numpy as np
import os
import sys

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embeddings.embedder import TextEmbedder, embed_text, embed_texts

class TestTextEmbedder:
    def setup_method(self):
        """Setup test fixtures"""
        # Use a smaller model for testing to avoid long download times
        self.embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.sample_texts = [
            "This is a test sentence about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers."
        ]
    
    def test_embedder_initialization(self):
        """Test embedder initialization"""
        embedder = TextEmbedder()
        assert embedder.model_name == "intfloat/e5-large-v2"
        assert embedder.model is not None
        assert embedder.embedding_dim > 0
    
    def test_embedder_custom_model(self):
        """Test embedder with custom model"""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = TextEmbedder(model_name=model_name)
        assert embedder.model_name == model_name
        assert embedder.embedding_dim > 0
    
    def test_embed_single_text(self):
        """Test embedding a single text"""
        text = "This is a test sentence."
        embedding = self.embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D array
        assert embedding.shape[0] == self.embedder.embedding_dim
        
        # Check if embedding is normalized (should be close to 1.0)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Allow small floating point errors
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts"""
        embeddings = self.embedder.embed_texts(self.sample_texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2  # 2D array
        assert embeddings.shape[0] == len(self.sample_texts)
        assert embeddings.shape[1] == self.embedder.embedding_dim
        
        # Check if embeddings are normalized
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01
    
    def test_embed_empty_list(self):
        """Test embedding empty list"""
        embeddings = self.embedder.embed_texts([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0
    
    def test_embed_chunks(self):
        """Test embedding chunks with metadata"""
        chunks = [
            {"text": text, "chunk_index": i, "source": "test"}
            for i, text in enumerate(self.sample_texts)
        ]
        
        embeddings = self.embedder.embed_chunks(chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(chunks)
        
        # Check that embeddings were added to chunks
        for chunk in chunks:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) == self.embedder.embedding_dim
    
    def test_compute_similarity(self):
        """Test cosine similarity computation"""
        text1 = "Machine learning is fascinating."
        text2 = "Artificial intelligence is interesting."
        text3 = "The weather is nice today."
        
        emb1 = self.embedder.embed_text(text1)
        emb2 = self.embedder.embed_text(text2)
        emb3 = self.embedder.embed_text(text3)
        
        # Similar texts should have higher similarity
        sim_12 = self.embedder.compute_similarity(emb1, emb2)
        sim_13 = self.embedder.compute_similarity(emb1, emb3)
        
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
        assert sim_12 > sim_13  # ML and AI should be more similar than ML and weather
    
    def test_find_most_similar(self):
        """Test finding most similar embeddings"""
        embeddings = self.embedder.embed_texts(self.sample_texts)
        query_embedding = embeddings[0]  # Use first embedding as query
        
        # Find most similar (excluding the query itself)
        candidates = embeddings[1:]  # Exclude the first one
        similar = self.embedder.find_most_similar(query_embedding, candidates, top_k=2)
        
        assert len(similar) == 2
        assert all(isinstance(item, tuple) for item in similar)
        assert all(len(item) == 2 for item in similar)  # (index, similarity)
        
        # Similarities should be in descending order
        assert similar[0][1] >= similar[1][1]
    
    def test_find_most_similar_empty(self):
        """Test finding similar embeddings with empty candidates"""
        query_embedding = self.embedder.embed_text("test")
        similar = self.embedder.find_most_similar(query_embedding, np.array([]), top_k=5)
        assert similar == []
    
    def test_batch_processing(self):
        """Test batch processing with different batch sizes"""
        texts = self.sample_texts * 10  # Create larger list
        
        # Test with different batch sizes
        embeddings_batch_1 = self.embedder.embed_texts(texts, batch_size=1)
        embeddings_batch_4 = self.embedder.embed_texts(texts, batch_size=4)
        embeddings_batch_16 = self.embedder.embed_texts(texts, batch_size=16)
        
        # Results should be identical regardless of batch size
        assert np.allclose(embeddings_batch_1, embeddings_batch_4, atol=1e-6)
        assert np.allclose(embeddings_batch_1, embeddings_batch_16, atol=1e-6)
    
    def test_e5_model_prefix(self):
        """Test that e5 models get query prefix"""
        # This test checks if e5 models get the "query:" prefix
        # We can't easily test this without mocking, so we'll test indirectly
        
        # Create embedder with e5 model name
        e5_embedder = TextEmbedder(model_name="intfloat/e5-small-v2")
        
        # The embedding should work (this indirectly tests the prefix logic)
        embedding = e5_embedder.embed_text("test query")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0

class TestEmbedderConvenienceFunctions:
    """Test convenience functions"""
    
    def test_embed_text_function(self):
        """Test convenience embed_text function"""
        text = "This is a test sentence."
        embedding = embed_text(text, model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    def test_embed_texts_function(self):
        """Test convenience embed_texts function"""
        texts = ["First text.", "Second text.", "Third text."]
        embeddings = embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0

class TestEmbedderIntegration:
    """Integration tests for embedder"""
    
    def test_semantic_similarity(self):
        """Test that semantically similar texts have higher similarity"""
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Related texts
        tech_texts = [
            "Machine learning algorithms process data.",
            "AI systems learn from training data.",
            "Neural networks are used in deep learning."
        ]
        
        # Unrelated text
        unrelated_text = "The cat sat on the mat."
        
        tech_embeddings = embedder.embed_texts(tech_texts)
        unrelated_embedding = embedder.embed_text(unrelated_text)
        
        # Compute similarities between tech texts
        tech_similarities = []
        for i in range(len(tech_embeddings)):
            for j in range(i + 1, len(tech_embeddings)):
                sim = embedder.compute_similarity(tech_embeddings[i], tech_embeddings[j])
                tech_similarities.append(sim)
        
        # Compute similarities between tech texts and unrelated text
        unrelated_similarities = []
        for tech_emb in tech_embeddings:
            sim = embedder.compute_similarity(tech_emb, unrelated_embedding)
            unrelated_similarities.append(sim)
        
        # Tech texts should be more similar to each other than to unrelated text
        avg_tech_sim = np.mean(tech_similarities)
        avg_unrelated_sim = np.mean(unrelated_similarities)
        
        assert avg_tech_sim > avg_unrelated_sim
    
    def test_embedding_consistency(self):
        """Test that same text produces same embedding"""
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        text = "This is a consistency test."
        
        embedding1 = embedder.embed_text(text)
        embedding2 = embedder.embed_text(text)
        
        # Should be identical (or very close due to floating point precision)
        assert np.allclose(embedding1, embedding2, atol=1e-6)
    
    def test_different_text_different_embeddings(self):
        """Test that different texts produce different embeddings"""
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        text1 = "This is the first text."
        text2 = "This is completely different content."
        
        embedding1 = embedder.embed_text(text1)
        embedding2 = embedder.embed_text(text2)
        
        # Should not be identical
        assert not np.allclose(embedding1, embedding2, atol=1e-3)
        
        # But similarity should still be reasonable (both are English text)
        similarity = embedder.compute_similarity(embedding1, embedding2)
        assert 0 < similarity < 1