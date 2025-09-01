import logging
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import uuid

from embeddings.embedder import TextEmbedder

logger = logging.getLogger(__name__)

class QdrantClient:
    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        """
        Initialize Qdrant client
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication (if required)
        """
        self.url = url
        self.client = QdrantClientBase(url=url, api_key=api_key)
        self.embedder = None
        
        logger.info(f"Initialized Qdrant client: {url}")
    
    async def initialize(self):
        """Initialize embedder and check connection"""
        try:
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections")
            
            # Initialize embedder
            self.embedder = TextEmbedder()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 1024, 
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ):
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric
            recreate: Whether to recreate if exists
        """
        try:
            if recreate:
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass  # Collection might not exist
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            
            logger.info(f"Created collection: {collection_name} (size={vector_size}, distance={distance})")
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def ensure_collection(self, collection_name: str, vector_size: int = 1024):
        """Ensure collection exists, create if not"""
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size)
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")
    
    def upsert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]], 
        vector_key: str = "embedding",
        id_key: str = "id"
    ):
        """
        Upsert points to collection
        
        Args:
            collection_name: Name of the collection
            points: List of point dictionaries with vectors and metadata
            vector_key: Key containing vector in point dict
            id_key: Key containing ID in point dict
        """
        try:
            qdrant_points = []
            
            for point in points:
                # Generate ID if not provided
                point_id = point.get(id_key, str(uuid.uuid4()))
                
                # Extract vector
                vector = point.get(vector_key)
                if vector is None:
                    logger.warning(f"Point {point_id} missing vector key '{vector_key}'")
                    continue
                
                # Convert to numpy array if needed
                if isinstance(vector, list):
                    vector = np.array(vector)
                
                # Create payload (all data except vector and id)
                payload = {k: v for k, v in point.items() if k not in [vector_key, id_key]}
                
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                        payload=payload
                    )
                )
            
            # Upsert points
            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            
            logger.info(f"Upserted {len(qdrant_points)} points to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to upsert points to {collection_name}: {e}")
            raise
    
    def search(
        self, 
        collection_name: str, 
        query_vector: Union[np.ndarray, List[float]], 
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[models.ScoredPoint]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter_conditions: Filter conditions
            score_threshold: Minimum similarity score
            
        Returns:
            List of scored points
        """
        try:
            # Convert query vector to list if needed
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Build filter
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold
            )
            
            logger.info(f"Found {len(results)} results in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in {collection_name}: {e}")
            raise
    
    async def search_text(
        self, 
        collection_name: str, 
        query_text: str, 
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[models.ScoredPoint]:
        """
        Search using text query (will be embedded)
        
        Args:
            collection_name: Name of the collection
            query_text: Query text to embed and search
            top_k: Number of results to return
            filter_conditions: Filter conditions
            score_threshold: Minimum similarity score
            
        Returns:
            List of scored points
        """
        if not self.embedder:
            await self.initialize()
        
        # Embed query text
        query_vector = self.embedder.embed_text(query_text)
        
        # Search with embedded vector
        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions,
            score_threshold=score_threshold
        )
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise
    
    def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points by IDs"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete points from {collection_name}: {e}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete entire collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

# Convenience function
async def create_novel_chunks_collection(client: QdrantClient, recreate: bool = False):
    """Create the novel_chunks collection with standard settings"""
    collection_name = "novel_chunks"
    
    if recreate or not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vector_size=1024,  # e5-large-v2 dimension
            distance=Distance.COSINE,
            recreate=recreate
        )
        logger.info(f"Created {collection_name} collection")
    else:
        logger.info(f"{collection_name} collection already exists")