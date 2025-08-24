"""Search endpoints for the API."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends

from ..api_exceptions import APIBaseException, ServiceUnavailableError
from ..api_cache import cached_operation, cache_key_for_search
from ..api_config import APIConfig
from ..api_utils import RequestValidator, SearchOperations
from ...models import SearchRequest, SearchResponse
from ...monitoring.advanced_system_monitor import monitor_operation, ComponentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/vector")
@monitor_operation("vector_search", ComponentType.API_LAYER)
async def search_vector(request: SearchRequest):
    """Enhanced vector search endpoint dengan caching."""
    try:
        # Validate request
        validated_data = RequestValidator.validate_search_request(request.dict())
        
        # Generate cache key
        cache_key = cache_key_for_search("vector", validated_data["query"], validated_data["limit"])
        
        async def execute_search():
            results, query_time = await SearchOperations.execute_vector_search(
                validated_data["query"], 
                validated_data["limit"]
            )
            return {
                "results": results,
                "total_results": len(results),
                "search_type": "vector",
                "query_time_ms": query_time
            }
        
        # Execute dengan caching
        result = await cached_operation(
            cache_key=cache_key,
            operation_func=execute_search,
            ttl=APIConfig.CACHE_TTL
        )
        
        return SearchResponse(**result)
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise ServiceUnavailableError("vector_search", f"Vector search error: {e}")


@router.post("/graph")
@monitor_operation("graph_search", ComponentType.API_LAYER)
async def search_graph(request: SearchRequest):
    """Enhanced graph search endpoint dengan caching."""
    try:
        # Validate request
        validated_data = RequestValidator.validate_search_request(request.dict())
        
        # Generate cache key
        cache_key = cache_key_for_search("graph", validated_data["query"], 0)
        
        async def execute_search():
            results, query_time = await SearchOperations.execute_graph_search(
                validated_data["query"]
            )
            return {
                "graph_results": results,
                "total_results": len(results),
                "search_type": "graph",
                "query_time_ms": query_time
            }
        
        # Execute dengan caching
        result = await cached_operation(
            cache_key=cache_key,
            operation_func=execute_search,
            ttl=APIConfig.CACHE_TTL
        )
        
        return SearchResponse(**result)
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise ServiceUnavailableError("graph_search", f"Graph search error: {e}")


@router.post("/hybrid")
@monitor_operation("hybrid_search", ComponentType.API_LAYER)
async def search_hybrid(request: SearchRequest):
    """Enhanced hybrid search endpoint dengan caching."""
    try:
        # Validate request
        validated_data = RequestValidator.validate_search_request(request.dict())
        
        # Generate cache key
        cache_key = cache_key_for_search("hybrid", validated_data["query"], validated_data["limit"])
        
        async def execute_search():
            results, query_time = await SearchOperations.execute_hybrid_search(
                validated_data["query"], 
                validated_data["limit"]
            )
            return {
                "results": results,
                "total_results": len(results),
                "search_type": "hybrid",
                "query_time_ms": query_time
            }
        
        # Execute dengan caching
        result = await cached_operation(
            cache_key=cache_key,
            operation_func=execute_search,
            ttl=APIConfig.CACHE_TTL
        )
        
        return SearchResponse(**result)
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise ServiceUnavailableError("hybrid_search", f"Hybrid search error: {e}")