"""
Main ingestion script for processing markdown documents into vector DB and knowledge graph.
Improved version with better error handling, performance, and security.
"""

import os
import asyncio
import logging
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import argparse
import hashlib
from contextlib import asynccontextmanager

import asyncpg
import numpy as np
from dotenv import load_dotenv

from .chunker import DocumentChunk  # Keep for data structure

# Import agent utilities with better error handling
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult

from .embedder import create_embedder
from .graph_builder import create_graph_builder

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass


class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector DB and knowledge graph."""
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        backup_before_clean: bool = True
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing markdown documents
            clean_before_ingest: Whether to clean existing data before ingestion
            backup_before_clean: Whether to backup data before cleaning
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.backup_before_clean = backup_before_clean
        
        # Initialize components with lazy loading to avoid circular imports
        self._chunker = None
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()
        
        self._initialized = False
        self._processed_files: Set[str] = set()  # Track processed files to avoid duplicates
    
    @property
    def chunker(self):
        """Lazy load chunker to avoid circular imports."""
        if self._chunker is None:
            # Local import to avoid circular dependency
            from ..memory.chunking_strategies import NovelChunker
            self._chunker = NovelChunker()
        return self._chunker
    
    def _calculate_character_positions(self, text: str, chunks: List[Any]) -> List[tuple]:
        """
        Calculate character positions for chunks by matching content.
        
        Args:
            text: Original text
            chunks: List of chunks
            
        Returns:
            List of (start_char, end_char) tuples
        """
        positions = []
        search_start = 0
        
        for chunk in chunks:
            # Find the chunk content in the original text
            chunk_start = text.find(chunk.content, search_start)
            if chunk_start == -1:
                # If exact match not found, use approximate position
                logger.warning(f"Could not find exact position for chunk: {chunk.content[:50]}...")
                chunk_start = search_start
                chunk_end = search_start + len(chunk.content)
            else:
                chunk_end = chunk_start + len(chunk.content)
            
            positions.append((chunk_start, chunk_end))
            search_start = chunk_end
        
        return positions
    
    def _convert_novel_chunks_to_document_chunks(
        self, 
        novel_chunks: List[Any], 
        doc_title: str, 
        doc_source: str,
        original_text: str,
        extracted_entities: Optional[Dict[str, List[str]]] = None
    ) -> List[DocumentChunk]:
        """
        Converts chunks from NovelChunker to the DocumentChunk format.
        
        Args:
            novel_chunks: Chunks from NovelChunker
            doc_title: Document title
            doc_source: Document source
            original_text: Original document text for position calculation
            extracted_entities: Pre-extracted entities to include
        """
        # Calculate character positions
        positions = self._calculate_character_positions(original_text, novel_chunks)
        
        document_chunks = []
        for i, (n_chunk, (start_char, end_char)) in enumerate(zip(novel_chunks, positions)):
            # Prepare metadata with entities if available
            metadata = {
                "strategy_used": n_chunk.strategy_used.value,
                "chunk_type": n_chunk.chunk_type,
                "importance_score": n_chunk.importance_score,
                "narrative_elements": n_chunk.narrative_elements,
                "title": doc_title,
                "source": doc_source,
                "char_start": start_char,
                "char_end": end_char
            }
            
            # Add pre-extracted entities if available
            if extracted_entities:
                metadata["entities"] = extracted_entities
            
            doc_chunk = DocumentChunk(
                content=n_chunk.content,
                index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks

    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing ingestion pipeline...")
        
        try:
            # Initialize database connections
            await initialize_database()
            await initialize_graph()
            await self.graph_builder.initialize()
            
            self._initialized = True
            logger.info("Ingestion pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            try:
                await self.graph_builder.close()
                await close_graph()
                await close_database()
                self._initialized = False
                logger.info("Pipeline connections closed")
            except Exception as e:
                logger.error(f"Error closing pipeline: {e}")
    
    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()
        
        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases_safe()
        
        # Find all markdown files (with deduplication)
        markdown_files = self._find_markdown_files_deduplicated()
        
        if not markdown_files:
            logger.warning(f"No markdown files found in {self.documents_folder}")
            return []
        
        logger.info(f"Found {len(markdown_files)} unique markdown files to process")
        
        results = []
        
        for i, file_path in enumerate(markdown_files):
            try:
                logger.info(f"Processing file {i+1}/{len(markdown_files)}: {file_path}")
                
                result = await self._ingest_single_document(file_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(markdown_files))
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))
        
        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors")
        
        return results
    
    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document with comprehensive error handling.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Read document
            document_content = self._read_document_safe(file_path)
            document_title = self._extract_title(document_content, file_path)
            document_source = os.path.relpath(file_path, self.documents_folder)
            
            # Extract metadata from content
            document_metadata = self._extract_document_metadata_safe(document_content, file_path)
            
            logger.info(f"Processing document: {document_title}")
            
            # Pre-extract entities if enabled (before chunking for better context)
            extracted_entities = None
            if self.config.extract_entities:
                try:
                    extracted_entities = await self._extract_entities_from_text(document_content)
                    logger.info(f"Pre-extracted {sum(len(v) for v in extracted_entities.values())} entities")
                except Exception as e:
                    error_msg = f"Entity extraction failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Chunk the document using the advanced NovelChunker
            try:
                # Use extracted entities for better chunking context
                chunking_context = {
                    'characters': extracted_entities.get('people', []) if extracted_entities else [],
                    'chapter': document_metadata.get('chapter', 1),
                    'generation_type': 'ingestion',
                    'entities': extracted_entities or {}
                }
                
                novel_chunks = self.chunker.adaptive_chunking(
                    text=document_content,
                    context=chunking_context
                )
                
                if not novel_chunks:
                    raise ChunkingError("No chunks created by chunker")
                
                # Convert chunks to the format expected by the rest of the pipeline
                chunks = self._convert_novel_chunks_to_document_chunks(
                    novel_chunks, 
                    document_title, 
                    document_source, 
                    document_content,
                    extracted_entities
                )
                
                logger.info(f"Created {len(chunks)} chunks using advanced chunker")
                
            except Exception as e:
                raise ChunkingError(f"Chunking failed: {str(e)}")
            
            # Generate embeddings with error handling
            try:
                embedded_chunks = await self.embedder.embed_chunks(chunks)
                logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            except Exception as e:
                raise EmbeddingError(f"Embedding generation failed: {str(e)}")
            
            # Save to PostgreSQL with transaction safety
            try:
                document_id = await self._save_to_postgres_safe(
                    document_title,
                    document_source,
                    document_content,
                    embedded_chunks,
                    document_metadata
                )
                logger.info(f"Saved document to PostgreSQL with ID: {document_id}")
            except Exception as e:
                raise DatabaseError(f"PostgreSQL save failed: {str(e)}")
            
            # Add to knowledge graph (if enabled)
            relationships_created = 0
            
            if not self.config.skip_graph_building:
                try:
                    logger.info("Building knowledge graph relationships...")
                    graph_result = await self.graph_builder.add_document_to_graph(
                        chunks=embedded_chunks,
                        document_title=document_title,
                        document_source=document_source,
                        document_metadata=document_metadata
                    )
                    
                    relationships_created = graph_result.get("episodes_created", 0)
                    graph_errors = graph_result.get("errors", [])
                    errors.extend(graph_errors)
                    
                    logger.info(f"Added {relationships_created} episodes to knowledge graph")
                    
                except Exception as e:
                    error_msg = f"Knowledge graph building failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            else:
                logger.info("Skipping knowledge graph building (skip_graph_building=True)")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                document_id=str(document_id),
                title=document_title,
                chunks_created=len(chunks),
                entities_extracted=sum(len(v) for v in (extracted_entities or {}).values()),
                relationships_created=relationships_created,
                processing_time_ms=processing_time,
                errors=errors
            )
            
        except (ChunkingError, EmbeddingError, DatabaseError) as e:
            # These are our custom errors with good context
            logger.error(f"Processing failed for {file_path}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=processing_time,
                errors=[str(e)]
            )
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=processing_time,
                errors=[f"Unexpected error: {str(e)}"]
            )
    
    async def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from full text before chunking.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary of entity types and their values
        """
        # Create a temporary chunk for entity extraction
        temp_chunk = DocumentChunk(
            content=text,
            index=0,
            start_char=0,
            end_char=len(text),
            metadata={}
        )
        
        # Use graph builder's entity extraction
        chunks_with_entities = await self.graph_builder.extract_entities_from_chunks([temp_chunk])
        
        if chunks_with_entities and chunks_with_entities[0].metadata.get("entities"):
            return chunks_with_entities[0].metadata["entities"]
        
        return {}
    
    def _find_markdown_files_deduplicated(self) -> List[str]:
        """Find all markdown files in the documents folder with deduplication."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []
        
        patterns = ["*.md", "*.markdown", "*.txt"]
        files_set = set()  # Use set for automatic deduplication
        
        for pattern in patterns:
            pattern_files = glob.glob(
                os.path.join(self.documents_folder, "**", pattern), 
                recursive=True
            )
            files_set.update(pattern_files)
        
        # Convert back to sorted list
        return sorted(list(files_set))
    
    def _read_document_safe(self, file_path: str) -> str:
        """Read document content from file with better error handling."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                raise
        
        raise UnicodeDecodeError(f"Could not decode {file_path} with any of the tried encodings")
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                if title:  # Make sure it's not just whitespace
                    return title
        
        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _extract_document_metadata_safe(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content with robust error handling."""
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()
        }
        
        # Try to extract YAML frontmatter with better error handling
        if content.startswith('---'):
            try:
                # Find the end of frontmatter
                end_marker_pos = content.find('\n---\n', 4)
                if end_marker_pos == -1:
                    # Try alternative end marker
                    end_marker_pos = content.find('\n...\n', 4)
                
                if end_marker_pos != -1:
                    frontmatter = content[4:end_marker_pos]
                    
                    # Try to import and parse YAML
                    try:
                        import yaml
                        yaml_metadata = yaml.safe_load(frontmatter)
                        if isinstance(yaml_metadata, dict):
                            # Sanitize values to ensure they're JSON serializable
                            for key, value in yaml_metadata.items():
                                if isinstance(value, (str, int, float, bool, list, dict)):
                                    metadata[key] = value
                                else:
                                    metadata[key] = str(value)
                            logger.debug(f"Extracted YAML metadata: {list(yaml_metadata.keys())}")
                    except ImportError:
                        logger.warning("PyYAML not installed, skipping frontmatter extraction")
                    except yaml.YAMLError as e:
                        logger.warning(f"Failed to parse YAML frontmatter: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error parsing frontmatter: {e}")
            except Exception as e:
                logger.warning(f"Error processing frontmatter in {file_path}: {e}")
        
        # Extract basic content statistics
        lines = content.split('\n')
        metadata.update({
            'line_count': len(lines),
            'word_count': len(content.split()),
            'character_count': len(content),
            'paragraph_count': len([line for line in lines if line.strip()])
        })
        
        return metadata
    
    @staticmethod
    def _format_vector_for_postgres(embedding: List[float]) -> str:
        """
        Safely format embedding vector for PostgreSQL.
        
        Args:
            embedding: List of float values
            
        Returns:
            PostgreSQL vector format string
        """
        if not embedding:
            return None
        
        try:
            # Ensure all values are finite
            clean_embedding = []
            for val in embedding:
                if np.isfinite(val):
                    clean_embedding.append(val)
                else:
                    logger.warning(f"Non-finite value in embedding: {val}, replacing with 0.0")
                    clean_embedding.append(0.0)
            
            # Format as PostgreSQL vector: [1.0,2.0,3.0]
            return '[' + ','.join(f'{val:.8f}' for val in clean_embedding) + ']'
        except Exception as e:
            logger.error(f"Error formatting vector: {e}")
            return None
    
    async def _save_to_postgres_safe(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL with batch operations and error handling."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Insert document with parameterized query
                    document_result = await conn.fetchrow(
                        """
                        INSERT INTO documents (title, source, content, metadata)
                        VALUES ($1, $2, $3, $4::jsonb)
                        RETURNING id::text
                        """,
                        title,
                        source,
                        content,
                        json.dumps(metadata, ensure_ascii=False)
                    )
                    
                    document_id = document_result["id"]
                    
                    # Prepare chunk data for batch insertion
                    chunk_records = []
                    for chunk in chunks:
                        # Format embedding safely
                        embedding_data = None
                        if hasattr(chunk, 'embedding') and chunk.embedding:
                            embedding_data = self._format_vector_for_postgres(chunk.embedding)
                        
                        chunk_records.append((
                            document_id,  # document_id
                            chunk.content,  # content
                            embedding_data,  # embedding
                            chunk.index,  # chunk_index
                            json.dumps(chunk.metadata, ensure_ascii=False),  # metadata
                            chunk.token_count  # token_count
                        ))
                    
                    # Batch insert chunks
                    if chunk_records:
                        await conn.executemany(
                            """
                            INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                            VALUES ($1::uuid, $2, $3::vector, $4, $5::jsonb, $6)
                            """,
                            chunk_records
                        )
                        
                        logger.debug(f"Batch inserted {len(chunk_records)} chunks")
                    
                    return document_id
                    
                except Exception as e:
                    logger.error(f"Database error saving document '{title}': {e}")
                    raise DatabaseError(f"Failed to save to PostgreSQL: {str(e)}")
    
    async def _backup_databases(self) -> str:
        """Create backup of databases before cleaning."""
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backups/backup_{backup_timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"Creating backup in {backup_dir}...")
        
        try:
            # Backup PostgreSQL data
            async with db_pool.acquire() as conn:
                # Export documents
                docs = await conn.fetch("SELECT * FROM documents")
                with open(f"{backup_dir}/documents.json", 'w') as f:
                    json.dump([dict(row) for row in docs], f, indent=2, default=str)
                
                # Export chunks (without embeddings to save space)
                chunks = await conn.fetch("SELECT id, document_id, content, chunk_index, metadata, token_count FROM chunks")
                with open(f"{backup_dir}/chunks.json", 'w') as f:
                    json.dump([dict(row) for row in chunks], f, indent=2, default=str)
                
                logger.info(f"Backed up {len(docs)} documents and {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
        
        return backup_dir
    
    async def _clean_databases_safe(self):
        """Clean existing data from databases with optional backup."""
        if self.backup_before_clean:
            try:
                backup_dir = await self._backup_databases()
                logger.info(f"Backup completed: {backup_dir}")
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                response = input("Backup failed. Continue with cleaning? (y/N): ")
                if response.lower() != 'y':
                    raise Exception("Cleaning aborted due to backup failure")
        
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL with proper order (foreign key constraints)
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions") 
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")
        
        # Clean knowledge graph
        await self.graph_builder.clear_graph()
        logger.info("Cleaned knowledge graph")


async def main():
    """Main function for running ingestion with improved argument handling."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector DB and knowledge graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--documents", "-d", default="documents", 
                       help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", 
                       help="Clean existing data before ingestion")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup when cleaning (dangerous)")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Chunk size for splitting documents")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                       help="Chunk overlap size")
    parser.add_argument("--no-semantic", action="store_true", 
                       help="Disable semantic chunking")
    parser.add_argument("--no-entities", action="store_true", 
                       help="Disable entity extraction")
    parser.add_argument("--fast", "-f", action="store_true", 
                       help="Fast mode: skip knowledge graph building")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingestion.log')
        ]
    )
    
    # Create ingestion configuration with novel-specific settings
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=not args.no_entities,
        skip_graph_building=args.fast,
        # Enable novel-specific enhanced chunking
        use_enhanced_scene_chunking=True,
        dialogue_chunk_size=800,
        narrative_chunk_size=1200,
        action_chunk_size=600,
        description_chunk_size=1000,
        preserve_dialogue_integrity=True,
        preserve_emotional_beats=True,
        preserve_action_sequences=True
    )
    
    # Create and run pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        backup_before_clean=not args.no_backup
    )
    
    def progress_callback(current: int, total: int):
        percentage = (current / total) * 100
        print(f"Progress: {current}/{total} documents processed ({percentage:.1f}%)")
    
    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total entities extracted: {sum(r.entities_extracted for r in results)}")
        print(f"Total graph episodes: {sum(r.relationships_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per document: {total_time/len(results):.2f} seconds")
        print()
        
        # Print individual results
        successful = 0
        for result in results:
            status = "✅" if not result.errors else "❌"
            print(f"{status} {result.title}: {result.chunks_created} chunks, {result.entities_extracted} entities")
            
            if not result.errors:
                successful += 1
            else:
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"   Error: {error}")
                if len(result.errors) > 2:
                    print(f"   ... and {len(result.errors) - 2} more errors")
        
        print(f"\n✅ Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"\n❌ Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()
        print("🔒 Pipeline connections closed")


if __name__ == "__main__":
    asyncio.run(main())