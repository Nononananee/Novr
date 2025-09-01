#!/usr/bin/env python3
"""
Ingestion script for novel worldbook and reference materials
"""
import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embeddings.chunker import TextChunker
from embeddings.embedder import TextEmbedder
from embeddings.qdrant_client import QdrantClient, create_novel_chunks_collection
from backend.app.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NovelIngester:
    def __init__(self, qdrant_url: str = None):
        """Initialize the ingester with required components"""
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.chunker = TextChunker(
            max_tokens=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        self.embedder = TextEmbedder(model_name=settings.embedding_model)
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        
        logger.info(f"Initialized ingester with Qdrant URL: {self.qdrant_url}")
    
    async def initialize(self):
        """Initialize all components"""
        await self.qdrant_client.initialize()
        await create_novel_chunks_collection(self.qdrant_client)
    
    def read_file(self, file_path: str) -> str:
        """Read text content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read {len(content)} characters from {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def read_directory(self, dir_path: str, extensions: List[str] = None) -> Dict[str, str]:
        """Read all text files from directory"""
        if extensions is None:
            extensions = ['.txt', '.md', '.markdown']
        
        files_content = {}
        dir_path = Path(dir_path)
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    content = self.read_file(str(file_path))
                    files_content[str(file_path)] = content
                except Exception as e:
                    logger.warning(f"Skipped file {file_path}: {e}")
        
        logger.info(f"Read {len(files_content)} files from {dir_path}")
        return files_content
    
    async def ingest_text(
        self, 
        text: str, 
        project_id: str, 
        source_name: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Ingest text content into vector database
        
        Args:
            text: Text content to ingest
            project_id: Project identifier
            source_name: Name/path of source file
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        try:
            logger.info(f"Ingesting text from {source_name} for project {project_id}")
            
            # Prepare metadata
            base_metadata = {
                "project_id": project_id,
                "source": source_name,
                "ingested_at": str(asyncio.get_event_loop().time())
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Chunk the text
            chunks = self.chunker.chunk_text(text, base_metadata)
            
            if not chunks:
                logger.warning(f"No chunks created from {source_name}")
                return 0
            
            # Embed the chunks
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            # Add unique IDs to chunks
            for i, chunk in enumerate(embedded_chunks):
                chunk["id"] = f"{project_id}_{source_name}_{i}"
            
            # Upsert to Qdrant
            self.qdrant_client.upsert_points(
                collection_name="novel_chunks",
                points=embedded_chunks
            )
            
            logger.info(f"Successfully ingested {len(embedded_chunks)} chunks from {source_name}")
            return len(embedded_chunks)
            
        except Exception as e:
            logger.error(f"Failed to ingest text from {source_name}: {e}")
            raise
    
    async def ingest_file(self, file_path: str, project_id: str, metadata: Dict[str, Any] = None) -> int:
        """Ingest a single file"""
        content = self.read_file(file_path)
        source_name = Path(file_path).name
        
        return await self.ingest_text(content, project_id, source_name, metadata)
    
    async def ingest_directory(self, dir_path: str, project_id: str, extensions: List[str] = None) -> int:
        """Ingest all files from a directory"""
        files_content = self.read_directory(dir_path, extensions)
        
        total_chunks = 0
        for file_path, content in files_content.items():
            source_name = Path(file_path).name
            chunks_count = await self.ingest_text(content, project_id, source_name)
            total_chunks += chunks_count
        
        logger.info(f"Total chunks ingested from directory: {total_chunks}")
        return total_chunks
    
    async def verify_ingestion(self, project_id: str, sample_query: str = "story") -> Dict[str, Any]:
        """Verify ingestion by performing a test search"""
        try:
            results = await self.qdrant_client.search_text(
                collection_name="novel_chunks",
                query_text=sample_query,
                top_k=3,
                filter_conditions={"project_id": project_id}
            )
            
            verification = {
                "project_id": project_id,
                "query": sample_query,
                "results_found": len(results),
                "sample_results": []
            }
            
            for result in results:
                verification["sample_results"].append({
                    "score": result.score,
                    "source": result.payload.get("source", "unknown"),
                    "text_preview": result.payload.get("text", "")[:100] + "..."
                })
            
            logger.info(f"Verification complete: {len(results)} results found for '{sample_query}'")
            return verification
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"error": str(e)}

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest novel reference materials into vector database")
    parser.add_argument("--file", type=str, help="Path to file to ingest")
    parser.add_argument("--directory", type=str, help="Path to directory to ingest")
    parser.add_argument("--project", type=str, required=True, help="Project ID")
    parser.add_argument("--qdrant-url", type=str, help="Qdrant server URL")
    parser.add_argument("--verify", action="store_true", help="Verify ingestion with test search")
    parser.add_argument("--extensions", nargs="+", default=[".txt", ".md", ".markdown"], 
                       help="File extensions to process")
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.error("Either --file or --directory must be specified")
    
    try:
        # Initialize ingester
        ingester = NovelIngester(qdrant_url=args.qdrant_url)
        await ingester.initialize()
        
        total_chunks = 0
        
        # Ingest file or directory
        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return 1
            
            total_chunks = await ingester.ingest_file(args.file, args.project)
            
        elif args.directory:
            if not os.path.exists(args.directory):
                logger.error(f"Directory not found: {args.directory}")
                return 1
            
            total_chunks = await ingester.ingest_directory(
                args.directory, 
                args.project, 
                args.extensions
            )
        
        logger.info(f"Ingestion completed successfully. Total chunks: {total_chunks}")
        
        # Verify ingestion if requested
        if args.verify:
            verification = await ingester.verify_ingestion(args.project)
            print("\nVerification Results:")
            print(f"Project: {verification.get('project_id')}")
            print(f"Results found: {verification.get('results_found')}")
            
            for i, result in enumerate(verification.get('sample_results', []), 1):
                print(f"\nResult {i}:")
                print(f"  Score: {result['score']:.3f}")
                print(f"  Source: {result['source']}")
                print(f"  Preview: {result['text_preview']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))