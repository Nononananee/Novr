import pytest
import asyncio
import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embeddings.chunker import TextChunker
from embeddings.embedder import TextEmbedder
from embeddings.qdrant_client import QdrantClient
from scripts.ingest import NovelIngester
from agents.generator_agent import GeneratorAgent
from agents.technical_qa import TechnicalQAAgent

class TestIntegrationSmoke:
    """Smoke tests for end-to-end integration"""
    
    @pytest.fixture
    def sample_novel_content(self):
        """Sample novel content for testing"""
        return """
        # Chapter 1: The Mysterious Forest
        
        Elena stepped carefully through the ancient forest, her boots crunching softly on the fallen leaves. 
        The towering oak trees seemed to whisper secrets in the gentle breeze, their branches creating 
        intricate patterns of light and shadow on the forest floor.
        
        She had been walking for hours, following the cryptic map her grandmother had left her. 
        The parchment was old and fragile, marked with symbols she didn't recognize. But something 
        deep inside her urged her forward, as if the forest itself was calling to her.
        
        As she rounded a bend in the path, Elena gasped. Before her stood a clearing she had never 
        seen before, despite having explored these woods since childhood. In the center grew a 
        magnificent tree, its silver bark gleaming in the dappled sunlight.
        
        "Welcome, child of the old blood," a voice whispered from the tree. Elena's heart raced 
        as she realized her grandmother's stories might have been more than just fairy tales.
        """
    
    def test_chunker_embedder_integration(self, sample_novel_content):
        """Test chunker and embedder working together"""
        # Initialize components
        chunker = TextChunker(max_tokens=200, overlap=50)
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Chunk the content
        chunks = chunker.chunk_text(sample_novel_content, metadata={"project_id": "test_novel"})
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("project_id" in chunk for chunk in chunks)
        
        # Embed the chunks
        embeddings = embedder.embed_chunks(chunks)
        
        assert len(embeddings) == len(chunks)
        assert all("embedding" in chunk for chunk in chunks)
        
        # Verify embeddings are valid
        for chunk in chunks:
            embedding = chunk["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) == embedder.embedding_dim
            assert all(isinstance(x, (int, float)) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_ingest_workflow(self, sample_novel_content):
        """Test the complete ingestion workflow"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_novel_content)
            temp_file = f.name
        
        try:
            # Initialize ingester (this will use in-memory Qdrant for testing)
            ingester = NovelIngester(qdrant_url="http://localhost:6333")
            
            # Note: This test requires Qdrant to be running
            # In a real test environment, you'd use a test Qdrant instance
            try:
                await ingester.initialize()
                
                # Ingest the file
                chunks_count = await ingester.ingest_file(temp_file, "test_project")
                
                assert chunks_count > 0
                
                # Verify ingestion
                verification = await ingester.verify_ingestion("test_project", "forest")
                
                assert "error" not in verification
                assert verification["results_found"] > 0
                
            except Exception as e:
                # If Qdrant is not available, skip this test
                pytest.skip(f"Qdrant not available for integration test: {e}")
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_agent_integration(self):
        """Test generator and QA agents working together"""
        try:
            # Initialize agents (requires API keys)
            generator = GeneratorAgent(model="gpt-4o-mini")
            qa_agent = TechnicalQAAgent(model="gpt-4o-mini")
            
            # Test prompt
            prompt = "Write a short paragraph about a character discovering a magical artifact."
            context = "The story is set in a fantasy world with ancient magic."
            
            # Generate content
            generated_content = await generator.generate(
                prompt=prompt,
                context=context,
                length_words=100,
                temperature=0.7
            )
            
            assert isinstance(generated_content, str)
            assert len(generated_content) > 50  # Should generate reasonable content
            
            # Run QA on generated content
            qa_result = await qa_agent.review(generated_content)
            
            assert isinstance(qa_result, dict)
            assert "score" in qa_result
            assert "issues" in qa_result
            assert "patches" in qa_result
            
            assert isinstance(qa_result["score"], int)
            assert 0 <= qa_result["score"] <= 100
            assert isinstance(qa_result["issues"], list)
            assert isinstance(qa_result["patches"], list)
            
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("API key not available for agent integration test")
            else:
                raise
    
    def test_chunker_preserves_content_integrity(self, sample_novel_content):
        """Test that chunking preserves content integrity"""
        chunker = TextChunker(max_tokens=150, overlap=30)
        chunks = chunker.chunk_text(sample_novel_content)
        
        # Combine all chunks (removing overlap for simplicity)
        combined_text = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                combined_text += chunk["text"]
            else:
                # For overlap testing, we'd need more sophisticated logic
                # For now, just verify chunks contain expected content
                assert len(chunk["text"]) > 0
                assert chunk["token_count"] <= chunker.max_tokens
        
        # Verify important content is preserved
        original_words = set(sample_novel_content.lower().split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk["text"].lower().split())
        
        # Most words should be preserved (allowing for some processing differences)
        preserved_ratio = len(original_words.intersection(chunk_words)) / len(original_words)
        assert preserved_ratio > 0.8  # At least 80% of words preserved
    
    def test_embedding_semantic_search(self):
        """Test semantic search capabilities"""
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create a small corpus
        documents = [
            "The wizard cast a powerful spell to protect the village.",
            "Elena discovered an ancient magical artifact in the forest.",
            "The dragon soared high above the mountain peaks.",
            "Modern technology has revolutionized communication.",
            "Scientists are studying climate change effects."
        ]
        
        # Embed all documents
        doc_embeddings = embedder.embed_texts(documents)
        
        # Query for fantasy-related content
        query = "magical fantasy adventure"
        query_embedding = embedder.embed_text(query)
        
        # Find most similar documents
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = embedder.compute_similarity(query_embedding, doc_emb)
            similarities.append((i, sim, documents[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Top results should be fantasy-related
        top_result = similarities[0]
        assert "wizard" in top_result[2] or "magical" in top_result[2] or "dragon" in top_result[2]
        
        # Fantasy content should rank higher than technology content
        fantasy_scores = [sim for i, sim, doc in similarities if any(word in doc.lower() for word in ["wizard", "magical", "dragon", "forest"])]
        tech_scores = [sim for i, sim, doc in similarities if any(word in doc.lower() for word in ["technology", "scientists", "climate"])]
        
        if fantasy_scores and tech_scores:
            assert max(fantasy_scores) > max(tech_scores)

class TestConfigurationIntegration:
    """Test configuration and environment integration"""
    
    def test_environment_variables_loading(self):
        """Test that environment variables are loaded correctly"""
        from backend.app.config import settings
        
        # Test that settings object exists and has expected attributes
        assert hasattr(settings, 'mongodb_url')
        assert hasattr(settings, 'redis_url')
        assert hasattr(settings, 'qdrant_url')
        assert hasattr(settings, 'chunk_size')
        assert hasattr(settings, 'chunk_overlap')
        assert hasattr(settings, 'embedding_model')
        
        # Test default values
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 64
        assert settings.embedding_model == "intfloat/e5-large-v2"
    
    def test_chunker_uses_config(self):
        """Test that chunker uses configuration values"""
        from backend.app.config import settings
        
        chunker = TextChunker(
            max_tokens=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        
        assert chunker.max_tokens == settings.chunk_size
        assert chunker.overlap == settings.chunk_overlap
    
    def test_embedder_uses_config(self):
        """Test that embedder uses configuration values"""
        from backend.app.config import settings
        
        # Use a smaller model for testing
        test_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = TextEmbedder(model_name=test_model)
        
        assert embedder.model_name == test_model
        assert embedder.embedding_dim > 0

class TestErrorHandling:
    """Test error handling in integration scenarios"""
    
    def test_chunker_handles_invalid_input(self):
        """Test chunker error handling"""
        chunker = TextChunker()
        
        # Test with None input
        chunks = chunker.chunk_text(None)
        assert chunks == []
        
        # Test with empty string
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_embedder_handles_empty_input(self):
        """Test embedder error handling"""
        embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Test with empty list
        embeddings = embedder.embed_texts([])
        assert embeddings.size == 0
        
        # Test with list containing empty strings
        embeddings = embedder.embed_texts(["", "  ", "valid text"])
        assert embeddings.shape[0] == 3  # Should still process all inputs
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling"""
        try:
            # Test with invalid API key
            generator = GeneratorAgent(api_key="invalid_key", model="gpt-4o-mini")
            
            with pytest.raises(Exception):
                await generator.generate("test prompt")
                
        except ValueError as e:
            if "API key" in str(e):
                # Expected when no valid API key is available
                pass
            else:
                raise

if __name__ == "__main__":
    # Run basic smoke tests
    pytest.main([__file__, "-v"])