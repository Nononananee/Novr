"""
Memory Optimizer for Large Document Processing
Addresses memory spikes during large document processing with streaming and batch optimization.
"""

import asyncio
import logging
import gc
import time
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    initial_memory_mb: float
    current_memory_mb: float
    peak_memory_mb: float
    memory_increase_mb: float
    gc_collections: int


@dataclass
class ProcessingConfig:
    """Configuration for memory-optimized processing."""
    max_memory_mb: int = 500
    batch_size: int = 10
    gc_threshold: int = 100
    streaming_enabled: bool = True
    memory_check_interval: int = 50


class MemoryOptimizer:
    """Memory optimizer for large document processing."""
    
    def __init__(self, config: ProcessingConfig = None):
        """Initialize memory optimizer."""
        self.config = config or ProcessingConfig()
        self.metrics = MemoryMetrics(0, 0, 0, 0, 0)
        self.processing_count = 0
        self.start_time = time.time()
        
        # Initialize memory tracking
        self._update_memory_metrics()
        self.metrics.initial_memory_mb = self.metrics.current_memory_mb
    
    def _update_memory_metrics(self):
        """Update current memory metrics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            current_mb = memory_info.rss / 1024 / 1024
            
            self.metrics.current_memory_mb = current_mb
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, current_mb)
            self.metrics.memory_increase_mb = current_mb - self.metrics.initial_memory_mb
            
        except Exception as e:
            logger.warning(f"Could not update memory metrics: {e}")
    
    def _check_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limits."""
        self._update_memory_metrics()
        return self.metrics.current_memory_mb > self.config.max_memory_mb
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        collected = gc.collect()
        self.metrics.gc_collections += 1
        logger.info(f"Forced garbage collection, freed {collected} objects")
    
    async def process_large_document_streaming(
        self,
        document_content: str,
        chunk_processor: callable,
        progress_callback: Optional[callable] = None
    ) -> List[Any]:
        """
        Process large document with streaming to avoid memory spikes.
        
        Args:
            document_content: Large document content
            chunk_processor: Function to process each chunk
            progress_callback: Optional progress callback
        
        Returns:
            List of processed chunks
        """
        logger.info(f"Starting streaming processing of document ({len(document_content)} chars)")
        
        # Split document into manageable sections
        sections = self._split_document_for_streaming(document_content)
        total_sections = len(sections)
        
        processed_chunks = []
        
        for i, section in enumerate(sections):
            try:
                # Check memory pressure
                if self._check_memory_pressure():
                    logger.warning(f"Memory pressure detected at section {i+1}/{total_sections}")
                    self._force_garbage_collection()
                    
                    # Wait a bit for memory to stabilize
                    await asyncio.sleep(0.1)
                
                # Process section
                section_chunks = await chunk_processor(section, i)
                processed_chunks.extend(section_chunks)
                
                self.processing_count += 1
                
                # Periodic garbage collection
                if self.processing_count % self.config.gc_threshold == 0:
                    self._force_garbage_collection()
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total_sections)
                
                # Memory check interval
                if i % self.config.memory_check_interval == 0:
                    self._update_memory_metrics()
                    logger.debug(f"Memory usage: {self.metrics.current_memory_mb:.2f}MB")
                
            except Exception as e:
                logger.error(f"Error processing section {i+1}: {e}")
                # Continue with next section instead of failing completely
                continue
        
        # Final memory check
        self._update_memory_metrics()
        
        logger.info(f"Streaming processing completed: {len(processed_chunks)} chunks, "
                   f"memory increase: {self.metrics.memory_increase_mb:.2f}MB")
        
        return processed_chunks
    
    def _split_document_for_streaming(self, content: str) -> List[str]:
        """Split document into sections for streaming processing."""
        
        # Target section size (in characters)
        target_section_size = 50000  # 50KB sections
        
        # Try to split on natural boundaries
        sections = []
        current_section = ""
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_section) + len(paragraph) > target_section_size and current_section:
                sections.append(current_section.strip())
                current_section = paragraph
            else:
                current_section += "\n\n" + paragraph if current_section else paragraph
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        logger.info(f"Split document into {len(sections)} sections for streaming")
        return sections
    
    async def batch_process_with_memory_management(
        self,
        items: List[Any],
        processor: callable,
        progress_callback: Optional[callable] = None
    ) -> List[Any]:
        """
        Process items in batches with memory management.
        
        Args:
            items: List of items to process
            processor: Function to process each batch
            progress_callback: Optional progress callback
        
        Returns:
            List of processed results
        """
        logger.info(f"Starting batch processing of {len(items)} items")
        
        results = []
        total_batches = (len(items) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            try:
                # Check memory before processing batch
                if self._check_memory_pressure():
                    logger.warning(f"Memory pressure before batch {batch_num}/{total_batches}")
                    self._force_garbage_collection()
                    await asyncio.sleep(0.1)
                
                # Process batch
                batch_results = await processor(batch)
                results.extend(batch_results)
                
                # Progress callback
                if progress_callback:
                    progress_callback(batch_num, total_batches)
                
                # Periodic memory management
                if batch_num % 5 == 0:  # Every 5 batches
                    self._force_garbage_collection()
                    self._update_memory_metrics()
                    
                    logger.debug(f"Batch {batch_num}/{total_batches} completed, "
                               f"memory: {self.metrics.current_memory_mb:.2f}MB")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch
                continue
        
        logger.info(f"Batch processing completed: {len(results)} results, "
                   f"memory increase: {self.metrics.memory_increase_mb:.2f}MB")
        
        return results
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        
        self._update_memory_metrics()
        
        return {
            "memory_metrics": {
                "initial_memory_mb": self.metrics.initial_memory_mb,
                "current_memory_mb": self.metrics.current_memory_mb,
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "memory_increase_mb": self.metrics.memory_increase_mb,
                "gc_collections": self.metrics.gc_collections
            },
            "processing_stats": {
                "items_processed": self.processing_count,
                "processing_time_seconds": time.time() - self.start_time,
                "items_per_second": self.processing_count / (time.time() - self.start_time) if time.time() > self.start_time else 0
            },
            "configuration": {
                "max_memory_mb": self.config.max_memory_mb,
                "batch_size": self.config.batch_size,
                "gc_threshold": self.config.gc_threshold,
                "streaming_enabled": self.config.streaming_enabled
            }
        }


class LargeDocumentProcessor:
    """Specialized processor for large documents with memory optimization."""
    
    def __init__(self, memory_optimizer: MemoryOptimizer = None):
        """Initialize large document processor."""
        self.optimizer = memory_optimizer or MemoryOptimizer()
    
    async def process_large_document(
        self,
        file_path: str,
        chunker: callable,
        embedder: callable,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process large document with memory optimization.
        
        Args:
            file_path: Path to large document
            chunker: Chunking function
            embedder: Embedding function
            progress_callback: Progress callback
        
        Returns:
            Processing results
        """
        try:
            # Check file size
            file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
            logger.info(f"Processing large document: {file_path} ({file_size:.2f}MB)")
            
            # Read document in chunks if very large
            if file_size > 50:  # > 50MB
                content = await self._read_large_file_streaming(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Process with streaming
            async def chunk_processor(section: str, section_index: int) -> List[Any]:
                """Process a document section."""
                try:
                    # Chunk the section
                    chunks = chunker(section, f"section_{section_index}")
                    
                    # Generate embeddings in smaller batches
                    embedded_chunks = await self.optimizer.batch_process_with_memory_management(
                        items=chunks,
                        processor=lambda batch: embedder(batch)
                    )
                    
                    return embedded_chunks
                    
                except Exception as e:
                    logger.error(f"Error processing section {section_index}: {e}")
                    return []
            
            # Process document with streaming
            all_chunks = await self.optimizer.process_large_document_streaming(
                document_content=content,
                chunk_processor=chunk_processor,
                progress_callback=progress_callback
            )
            
            # Generate report
            memory_report = self.optimizer.get_memory_report()
            
            return {
                "success": True,
                "file_path": file_path,
                "file_size_mb": file_size,
                "chunks_created": len(all_chunks),
                "memory_report": memory_report,
                "processing_time_seconds": time.time() - self.optimizer.start_time
            }
            
        except Exception as e:
            logger.error(f"Large document processing failed: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "memory_report": self.optimizer.get_memory_report()
            }
    
    async def _read_large_file_streaming(self, file_path: str) -> str:
        """Read large file with streaming to avoid memory spikes."""
        
        content_parts = []
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content_parts.append(chunk)
                
                # Check memory pressure
                if self.optimizer._check_memory_pressure():
                    self.optimizer._force_garbage_collection()
                    await asyncio.sleep(0.01)  # Brief pause
        
        return ''.join(content_parts)


# Factory functions
def create_memory_optimizer(max_memory_mb: int = 500, batch_size: int = 10) -> MemoryOptimizer:
    """Create memory optimizer with custom configuration."""
    config = ProcessingConfig(
        max_memory_mb=max_memory_mb,
        batch_size=batch_size,
        streaming_enabled=True
    )
    return MemoryOptimizer(config)


def create_large_document_processor(max_memory_mb: int = 500) -> LargeDocumentProcessor:
    """Create large document processor with memory optimization."""
    optimizer = create_memory_optimizer(max_memory_mb)
    return LargeDocumentProcessor(optimizer)


# Example usage
async def main():
    """Example usage of memory optimizer."""
    
    # Create optimizer
    optimizer = create_memory_optimizer(max_memory_mb=200)
    
    # Simulate large document processing
    large_content = "This is test content. " * 10000  # ~200KB
    
    async def mock_chunk_processor(section: str, section_index: int) -> List[Dict]:
        """Mock chunk processor."""
        # Simulate chunking
        chunks = [
            {"content": section[:1000], "index": section_index * 10 + i}
            for i in range(min(10, len(section) // 1000 + 1))
        ]
        return chunks
    
    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} sections processed")
    
    # Process with streaming
    results = await optimizer.process_large_document_streaming(
        document_content=large_content,
        chunk_processor=mock_chunk_processor,
        progress_callback=progress_callback
    )
    
    # Get memory report
    report = optimizer.get_memory_report()
    print(f"Processing completed: {len(results)} chunks")
    print(f"Memory usage: {report['memory_metrics']['memory_increase_mb']:.2f}MB increase")
    print(f"GC collections: {report['memory_metrics']['gc_collections']}")


if __name__ == "__main__":
    asyncio.run(main())