import re
import tiktoken
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, encoding_name: str = "cl100k_base", max_tokens: int = 800, overlap: int = 64):
        """
        Initialize text chunker with tiktoken
        
        Args:
            encoding_name: Tiktoken encoding name
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap = overlap
        logger.info(f"Initialized chunker: max_tokens={max_tokens}, overlap={overlap}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Basic sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, chunk_size: int = None) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments
        
        Args:
            text: Input text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # If chunk_size is provided, treat it as max_tokens for this call
        max_tokens = chunk_size if chunk_size is not None else self.max_tokens

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max_tokens, split it further
            if sentence_tokens > max_tokens:
                # If we have accumulated sentences, save them first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = self.count_tokens(word + " ")

                    if word_tokens + word_token_count <= max_tokens:
                        word_chunk.append(word)
                        word_tokens += word_token_count
                    else:
                        if word_chunk:
                            chunk_text = " ".join(word_chunk)
                            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))

                        word_chunk = [word]
                        word_tokens = word_token_count
                
                # Add remaining words
                if word_chunk:
                    chunk_text = " ".join(word_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                continue
            
            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences or self.overlap <= 0:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take sentences from the end until we reach overlap limit
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, text: str, chunk_index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create chunk dictionary with metadata"""
        chunk = {
            "text": text,
            "chunk_index": chunk_index,
            "token_count": self.count_tokens(text),
            "character_count": len(text),
            "word_count": len(text.split())
        }
        
        if metadata:
            chunk.update(metadata)
        
        return chunk
    
    def chunk_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            file_metadata = {"source_file": file_path}
            if metadata:
                file_metadata.update(metadata)
            
            return self.chunk_text(text, file_metadata)
            
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            raise

# Convenience function
def chunk_text(text: str, max_tokens: int = 800, overlap: int = 64, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Convenience function to chunk text"""
    chunker = TextChunker(max_tokens=max_tokens, overlap=overlap)
    return chunker.chunk_text(text, metadata)