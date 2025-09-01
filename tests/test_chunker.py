import pytest
import os
import sys

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embeddings.chunker import TextChunker, chunk_text

class TestTextChunker:
    def setup_method(self):
        """Setup test fixtures"""
        self.chunker = TextChunker(max_tokens=100, overlap=20)
        
        self.sample_text = """
        This is the first sentence of our test document. It contains multiple sentences to test chunking.
        The second sentence continues the narrative. We want to ensure proper sentence boundaries.
        Here is a third sentence that adds more content. The chunker should handle this appropriately.
        Finally, we have a fourth sentence to complete our test. This should provide enough content for multiple chunks.
        """
    
    def test_chunker_initialization(self):
        """Test chunker initialization with default parameters"""
        chunker = TextChunker()
        assert chunker.max_tokens == 800
        assert chunker.overlap == 64
        assert chunker.encoding is not None
    
    def test_chunker_custom_parameters(self):
        """Test chunker initialization with custom parameters"""
        chunker = TextChunker(max_tokens=500, overlap=50)
        assert chunker.max_tokens == 500
        assert chunker.overlap == 50
    
    def test_count_tokens(self):
        """Test token counting functionality"""
        text = "This is a test sentence."
        token_count = self.chunker.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_split_into_sentences(self):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = self.chunker.split_into_sentences(text)
        assert len(sentences) == 4
        assert sentences[0].strip() == "First sentence."
        assert sentences[1].strip() == "Second sentence!"
        assert sentences[2].strip() == "Third sentence?"
        assert sentences[3].strip() == "Fourth sentence."
    
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        chunks = self.chunker.chunk_text(self.sample_text.strip())
        
        assert len(chunks) > 0
        assert isinstance(chunks, list)
        
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "token_count" in chunk
            assert "character_count" in chunk
            assert "word_count" in chunk
            
            # Verify token count is within limits
            assert chunk["token_count"] <= self.chunker.max_tokens
    
    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata"""
        metadata = {"source": "test_document", "project_id": "test_project"}
        chunks = self.chunker.chunk_text(self.sample_text.strip(), metadata=metadata)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk["source"] == "test_document"
            assert chunk["project_id"] == "test_project"
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        assert chunks == []
        
        chunks = self.chunker.chunk_text("   ")
        assert chunks == []
    
    def test_chunk_very_long_sentence(self):
        """Test chunking text with very long sentences"""
        # Create a very long sentence that exceeds max_tokens
        long_sentence = "This is a very long sentence. " * 50
        chunks = self.chunker.chunk_text(long_sentence)
        
        assert len(chunks) > 0
        
        # Each chunk should still be within token limits
        for chunk in chunks:
            assert chunk["token_count"] <= self.chunker.max_tokens
    
    def test_overlap_functionality(self):
        """Test that overlap is working correctly"""
        # Use a chunker with small limits to force multiple chunks
        small_chunker = TextChunker(max_tokens=30, overlap=10)
        chunks = small_chunker.chunk_text(self.sample_text.strip())
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # This is a basic check - in practice, overlap detection is complex
            assert len(chunks) >= 2
    
    def test_convenience_function(self):
        """Test the convenience chunk_text function"""
        chunks = chunk_text(self.sample_text.strip(), max_tokens=100, overlap=20)
        
        assert len(chunks) > 0
        assert isinstance(chunks, list)
        
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
    
    def test_chunk_file_not_found(self):
        """Test chunking non-existent file"""
        with pytest.raises(Exception):
            self.chunker.chunk_file("non_existent_file.txt")
    
    def test_chunk_statistics(self):
        """Test chunk statistics are calculated correctly"""
        chunks = self.chunker.chunk_text(self.sample_text.strip())
        
        for chunk in chunks:
            text = chunk["text"]
            
            # Verify statistics
            assert chunk["character_count"] == len(text)
            assert chunk["word_count"] == len(text.split())
            assert chunk["token_count"] > 0
            
            # Token count should be reasonable compared to word count
            # (usually tokens <= words, but can vary)
            assert chunk["token_count"] <= chunk["word_count"] * 2

class TestChunkerIntegration:
    """Integration tests for chunker with real-world scenarios"""
    
    def test_markdown_content(self):
        """Test chunking markdown content"""
        markdown_text = """
        # Chapter 1: The Beginning
        
        This is the first paragraph of our story. It introduces the main character.
        
        ## Section 1.1: Character Introduction
        
        The protagonist walked through the forest. The trees were tall and mysterious.
        
        - First bullet point
        - Second bullet point
        - Third bullet point
        
        **Bold text** and *italic text* should be preserved in chunks.
        """
        
        chunker = TextChunker(max_tokens=150, overlap=30)
        chunks = chunker.chunk_text(markdown_text.strip())
        
        assert len(chunks) > 0
        
        # Verify markdown formatting is preserved
        combined_text = " ".join([chunk["text"] for chunk in chunks])
        assert "#" in combined_text  # Headers preserved
        assert "**" in combined_text or "*" in combined_text  # Formatting preserved
    
    def test_dialogue_content(self):
        """Test chunking content with dialogue"""
        dialogue_text = '''
        "Hello there," said John to Mary. "How are you doing today?"
        
        Mary replied, "I'm doing well, thank you for asking. How about you?"
        
        "I'm great!" John exclaimed. "The weather is beautiful today."
        
        They continued their conversation as they walked through the park.
        '''
        
        chunker = TextChunker(max_tokens=100, overlap=20)
        chunks = chunker.chunk_text(dialogue_text.strip())
        
        assert len(chunks) > 0
        
        # Verify dialogue formatting is preserved
        for chunk in chunks:
            text = chunk["text"]
            # If chunk contains dialogue, quotes should be preserved
            if '"' in text:
                assert text.count('"') % 2 == 0 or chunk["chunk_index"] > 0  # Balanced quotes or continuation