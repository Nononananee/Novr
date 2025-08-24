"""
Database Connection Pool Optimizer
Addresses concurrent access bottlenecks and connection pool management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)


@dataclass
class PoolConfiguration:
    """Database pool configuration."""
    min_size: int = 5
    max_size: int = 25  # Increased from 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300
    command_timeout: float = 60
    server_settings: Dict[str, str] = None


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_wait_time_ms: float
    max_wait_time_ms: float


class OptimizedDatabasePool:
    """Optimized database connection pool with monitoring."""
    
    def __init__(self, database_url: str, config: PoolConfiguration = None):
        """Initialize optimized database pool."""
        self.database_url = database_url
        self.config = config or PoolConfiguration()
        self.pool: Optional[Pool] = None
        
        # Metrics tracking
        self.metrics = ConnectionMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        self.wait_times = []
        self.request_count = 0
        
        # Connection management
        self.connection_semaphore = asyncio.Semaphore(self.config.max_size)
        self.active_connections = set()
    
    async def initialize(self):
        """Initialize connection pool with optimized settings."""
        
        if self.pool:
            return
        
        try:
            # Optimized server settings
            server_settings = self.config.server_settings or {
                'application_name': 'creative_rag_system',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3',
                'statement_timeout': '30000',  # 30 seconds
                'idle_in_transaction_session_timeout': '60000'  # 1 minute
            }
            
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                command_timeout=self.config.command_timeout,
                server_settings=server_settings
            )
            
            logger.info(f"Database pool initialized: {self.config.min_size}-{self.config.max_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire_connection(self, timeout: float = 30.0):
        """Acquire connection with monitoring and timeout."""
        
        start_time = time.time()
        connection_id = f"conn_{time.time()}"
        
        try:
            # Wait for connection with timeout
            async with asyncio.timeout(timeout):
                async with self.connection_semaphore:
                    if not self.pool:
                        await self.initialize()
                    
                    async with self.pool.acquire() as connection:
                        # Track connection
                        self.active_connections.add(connection_id)
                        wait_time = (time.time() - start_time) * 1000
                        
                        self._update_connection_metrics(wait_time)
                        
                        try:
                            yield connection
                            self.metrics.successful_requests += 1
                            
                        except Exception as e:
                            self.metrics.failed_requests += 1
                            logger.error(f"Database operation failed: {e}")
                            raise
                        
                        finally:
                            self.active_connections.discard(connection_id)
        
        except asyncio.TimeoutError:
            self.metrics.failed_requests += 1
            logger.error(f"Database connection timeout after {timeout}s")
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Database connection error: {e}")
            raise
    
    def _update_connection_metrics(self, wait_time_ms: float):
        """Update connection metrics."""
        
        self.request_count += 1
        self.metrics.total_requests += 1
        
        # Track wait times
        self.wait_times.append(wait_time_ms)
        if len(self.wait_times) > 1000:  # Keep last 1000 measurements
            self.wait_times = self.wait_times[-1000:]
        
        # Update metrics
        self.metrics.avg_wait_time_ms = sum(self.wait_times) / len(self.wait_times)
        self.metrics.max_wait_time_ms = max(self.wait_times)
        self.metrics.active_connections = len(self.active_connections)
        
        # Alert on high wait times
        if wait_time_ms > 1000:  # > 1 second
            logger.warning(f"High database wait time: {wait_time_ms:.2f}ms")
    
    async def execute_with_retry(
        self,
        query: str,
        *args,
        max_retries: int = 3,
        base_delay: float = 0.1
    ) -> Any:
        """Execute query with retry logic for connection issues."""
        
        for attempt in range(max_retries):
            try:
                async with self.acquire_connection() as conn:
                    return await conn.fetch(query, *args)
                    
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError,
                    asyncio.TimeoutError) as e:
                
                if attempt == max_retries - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Database retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Don't retry for other types of errors
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def batch_execute(
        self,
        operations: List[tuple],  # List of (query, args) tuples
        batch_size: int = 10
    ) -> List[Any]:
        """Execute multiple operations in batches."""
        
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            async with self.acquire_connection() as conn:
                async with conn.transaction():
                    batch_results = []
                    
                    for query, args in batch:
                        try:
                            result = await conn.fetch(query, *args)
                            batch_results.append(result)
                        except Exception as e:
                            logger.error(f"Batch operation failed: {e}")
                            batch_results.append(None)
                    
                    results.extend(batch_results)
            
            # Brief pause between batches
            if i + batch_size < len(operations):
                await asyncio.sleep(0.01)
        
        return results
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        
        pool_stats = {}
        
        if self.pool:
            # Get actual pool statistics
            pool_stats = {
                "size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "idle_size": self.pool.get_idle_size(),
            }
        
        return {
            "pool_configuration": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_queries": self.config.max_queries,
                "command_timeout": self.config.command_timeout
            },
            "pool_statistics": pool_stats,
            "connection_metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests),
                "avg_wait_time_ms": self.metrics.avg_wait_time_ms,
                "max_wait_time_ms": self.metrics.max_wait_time_ms,
                "active_connections": len(self.active_connections)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        
        try:
            start_time = time.time()
            
            async with self.acquire_connection(timeout=5.0) as conn:
                await conn.fetchval("SELECT 1")
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "pool_size": self.pool.get_size() if self.pool else 0,
                "active_connections": len(self.active_connections)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_size": self.pool.get_size() if self.pool else 0,
                "active_connections": len(self.active_connections)
            }


class ConcurrencyManager:
    """Manage concurrent operations to prevent bottlenecks."""
    
    def __init__(self, max_concurrent_operations: int = 15):
        """Initialize concurrency manager."""
        self.max_concurrent_operations = max_concurrent_operations
        self.operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        self.active_operations = {}
        self.operation_queue = asyncio.Queue(maxsize=100)
        
        # Performance tracking
        self.queue_wait_times = []
        self.operation_times = []
    
    @asynccontextmanager
    async def managed_operation(self, operation_name: str, priority: int = 5):
        """Context manager for managed concurrent operations."""
        
        operation_id = f"{operation_name}_{time.time()}"
        queue_start = time.time()
        
        try:
            # Queue management for high priority operations
            if priority >= 8:  # High priority
                async with self.operation_semaphore:
                    queue_wait = (time.time() - queue_start) * 1000
                    self.queue_wait_times.append(queue_wait)
                    
                    operation_start = time.time()
                    self.active_operations[operation_id] = {
                        "name": operation_name,
                        "start_time": operation_start,
                        "priority": priority
                    }
                    
                    try:
                        yield operation_id
                    finally:
                        operation_time = (time.time() - operation_start) * 1000
                        self.operation_times.append(operation_time)
                        self.active_operations.pop(operation_id, None)
            else:
                # Regular priority - use queue
                await self.operation_queue.put((operation_id, operation_name, priority))
                
                async with self.operation_semaphore:
                    # Process from queue
                    queue_wait = (time.time() - queue_start) * 1000
                    self.queue_wait_times.append(queue_wait)
                    
                    operation_start = time.time()
                    self.active_operations[operation_id] = {
                        "name": operation_name,
                        "start_time": operation_start,
                        "priority": priority
                    }
                    
                    try:
                        yield operation_id
                    finally:
                        operation_time = (time.time() - operation_start) * 1000
                        self.operation_times.append(operation_time)
                        self.active_operations.pop(operation_id, None)
                        
                        # Remove from queue
                        try:
                            self.operation_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
        
        except Exception as e:
            self.active_operations.pop(operation_id, None)
            raise
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get concurrency statistics."""
        
        return {
            "configuration": {
                "max_concurrent_operations": self.max_concurrent_operations,
                "queue_size": self.operation_queue.qsize(),
                "queue_max_size": self.operation_queue.maxsize
            },
            "current_state": {
                "active_operations": len(self.active_operations),
                "queued_operations": self.operation_queue.qsize(),
                "available_slots": self.operation_semaphore._value
            },
            "performance": {
                "avg_queue_wait_ms": sum(self.queue_wait_times) / len(self.queue_wait_times) if self.queue_wait_times else 0,
                "max_queue_wait_ms": max(self.queue_wait_times) if self.queue_wait_times else 0,
                "avg_operation_time_ms": sum(self.operation_times) / len(self.operation_times) if self.operation_times else 0,
                "max_operation_time_ms": max(self.operation_times) if self.operation_times else 0
            },
            "active_operations_details": [
                {
                    "name": op["name"],
                    "duration_seconds": time.time() - op["start_time"],
                    "priority": op["priority"]
                }
                for op in self.active_operations.values()
            ]
        }


# Integration with existing db_utils
class OptimizedDBUtils:
    """Optimized database utilities wrapper."""
    
    def __init__(self, database_url: str):
        """Initialize optimized database utilities."""
        self.database_url = database_url
        self.pool = OptimizedDatabasePool(database_url)
        self.concurrency_manager = ConcurrencyManager()
        
    async def initialize(self):
        """Initialize optimized database utilities."""
        await self.pool.initialize()
    
    async def close(self):
        """Close database connections."""
        await self.pool.close()
    
    async def optimized_vector_search(
        self,
        embedding: List[float],
        limit: int = 10,
        timeout: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Optimized vector search with concurrency management."""
        
        async with self.concurrency_manager.managed_operation("vector_search", priority=7):
            async with self.pool.acquire_connection(timeout=timeout) as conn:
                # Convert embedding to PostgreSQL vector format
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                result = await conn.fetch(
                    "SELECT * FROM match_chunks($1::vector, $2)",
                    embedding_str,
                    limit
                )
                
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "content": row["content"],
                        "similarity": row["similarity"],
                        "metadata": row["metadata"],
                        "document_title": row["document_title"],
                        "document_source": row["document_source"]
                    }
                    for row in result
                ]
    
    async def optimized_hybrid_search(
        self,
        embedding: List[float],
        query_text: str,
        limit: int = 10,
        text_weight: float = 0.3,
        timeout: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Optimized hybrid search with concurrency management."""
        
        async with self.concurrency_manager.managed_operation("hybrid_search", priority=8):
            async with self.pool.acquire_connection(timeout=timeout) as conn:
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                result = await conn.fetch(
                    "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
                    embedding_str,
                    query_text,
                    limit,
                    text_weight
                )
                
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "content": row["content"],
                        "combined_score": row["combined_score"],
                        "vector_similarity": row["vector_similarity"],
                        "text_similarity": row["text_similarity"],
                        "metadata": row["metadata"],
                        "document_title": row["document_title"],
                        "document_source": row["document_source"]
                    }
                    for row in result
                ]
    
    async def batch_insert_chunks(
        self,
        chunks_data: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> int:
        """Optimized batch insert for chunks."""
        
        total_inserted = 0
        
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i + batch_size]
            
            async with self.concurrency_manager.managed_operation("batch_insert", priority=6):
                async with self.pool.acquire_connection() as conn:
                    async with conn.transaction():
                        for chunk_data in batch:
                            await conn.execute(
                                """
                                INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                                VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                                """,
                                chunk_data["document_id"],
                                chunk_data["content"],
                                chunk_data["embedding"],
                                chunk_data["chunk_index"],
                                chunk_data["metadata"],
                                chunk_data["token_count"]
                            )
                        
                        total_inserted += len(batch)
            
            # Brief pause between batches
            if i + batch_size < len(chunks_data):
                await asyncio.sleep(0.01)
        
        return total_inserted
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        pool_stats = self.pool.get_pool_statistics()
        concurrency_stats = self.concurrency_manager.get_concurrency_stats()
        
        return {
            "database_pool": pool_stats,
            "concurrency_management": concurrency_stats,
            "optimization_recommendations": self._get_optimization_recommendations(pool_stats, concurrency_stats)
        }
    
    def _get_optimization_recommendations(
        self,
        pool_stats: Dict[str, Any],
        concurrency_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Pool size recommendations
        success_rate = pool_stats["connection_metrics"]["success_rate"]
        if success_rate < 0.95:
            recommendations.append("Consider increasing database pool size")
        
        avg_wait_time = pool_stats["connection_metrics"]["avg_wait_time_ms"]
        if avg_wait_time > 100:
            recommendations.append("High connection wait times - consider pool optimization")
        
        # Concurrency recommendations
        active_ops = concurrency_stats["current_state"]["active_operations"]
        max_ops = concurrency_stats["configuration"]["max_concurrent_operations"]
        
        if active_ops / max_ops > 0.8:
            recommendations.append("High concurrency usage - monitor for bottlenecks")
        
        queue_size = concurrency_stats["current_state"]["queued_operations"]
        if queue_size > 10:
            recommendations.append("High operation queue - consider increasing concurrency limits")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations


# Factory functions
def create_optimized_db_pool(database_url: str, max_connections: int = 25) -> OptimizedDatabasePool:
    """Create optimized database pool."""
    config = PoolConfiguration(max_size=max_connections)
    return OptimizedDatabasePool(database_url, config)


def create_optimized_db_utils(database_url: str) -> OptimizedDBUtils:
    """Create optimized database utilities."""
    return OptimizedDBUtils(database_url)


# Example usage
async def main():
    """Example usage of database optimizer."""
    
    # This would use your actual DATABASE_URL
    db_utils = create_optimized_db_utils("postgresql://test:test@localhost:5432/test")
    
    try:
        await db_utils.initialize()
        
        # Test concurrent operations
        async def test_operation(op_id: int):
            # Simulate database operation
            await asyncio.sleep(0.01)
            return f"result_{op_id}"
        
        # Run concurrent operations
        tasks = [test_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Get optimization report
        report = db_utils.get_optimization_report()
        print(f"Optimization test completed: {len(results)} operations")
        print(f"Success rate: {report['database_pool']['connection_metrics']['success_rate']:.2%}")
        print(f"Recommendations: {report['optimization_recommendations']}")
        
    finally:
        await db_utils.close()


if __name__ == "__main__":
    asyncio.run(main())