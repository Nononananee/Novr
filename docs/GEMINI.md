# Gemini Development Guidelines
## Panduan Pengembangan Sistem RAG + Knowledge Graph

### 🎯 **PRINSIP UTAMA**

#### **DO's (Yang HARUS Dilakukan)**

##### **📚 Sebelum Coding**
- **SELALU** baca dan pahami file terkait sebelum melakukan perubahan
- **WAJIB** review struktur modul yang ada terlebih dahulu
- **HARUS** memahami konteks sistem RAG + Knowledge Graph
- **SELALU** cek dependencies dan environment variables
- **MANDATORY** understand PostgreSQL + Neo4j integration

##### **🏗️ Arsitektur & Modularitas**
- **MODULAR**: Setiap modul maksimal 200-300 baris kode
- **SINGLE RESPONSIBILITY**: Satu modul = satu tanggung jawab
- **REFACTOR AGGRESSIVELY**: Jika modul >300 baris, pecah segera
- **DEPENDENCY INJECTION**: Gunakan untuk testability
- **ASYNC/AWAIT**: Konsisten untuk operasi I/O
- **SEPARATION OF CONCERNS**: Vector search ≠ Graph search ≠ Hybrid search

##### **🧪 Testing & Quality**
- **TEST-DRIVEN**: Tulis test sebelum implementasi
- **COVERAGE**: Minimal 80% test coverage
- **INTEGRATION TESTS**: Untuk RAG pipeline dan graph operations
- **UNIT TESTS**: Untuk setiap function/method
- **DEBUG FIRST**: Selalu debug sebelum commit
- **DATABASE TESTS**: Mock PostgreSQL dan Neo4j untuk testing

##### **📝 Documentation**
- **DOCSTRINGS**: Setiap function/class harus ada
- **TYPE HINTS**: Wajib untuk semua parameter dan return
- **INLINE COMMENTS**: Untuk logic kompleks
- **CHANGELOG**: Update setiap perubahan signifikan
- **API DOCS**: FastAPI auto-documentation harus lengkap

##### **🔄 Development Workflow**
```bash
1. Read relevant files (agent/, ingestion/, memory/)
2. Understand RAG + Graph context
3. Write tests (vector + graph + hybrid)
4. Implement feature
5. Debug thoroughly (check DB connections)
6. Run all tests (including integration)
7. Review code (focus on async patterns)
8. Update documentation
9. Commit with detailed message
```

##### **🗃️ Database & Graph Guidelines**
- **VECTOR OPERATIONS**: Always use pgvector efficiently
- **GRAPH OPERATIONS**: Leverage Neo4j + Graphiti properly
- **HYBRID SEARCH**: Combine vector + graph intelligently
- **CONNECTION POOLING**: Always use async connection pools
- **TRANSACTION MANAGEMENT**: Proper rollback on failures
- **MEMORY MANAGEMENT**: Efficient chunking and caching

---

### ❌ **DON'Ts (Yang TIDAK Boleh Dilakukan)**

##### **🚫 Code Quality**
- **JANGAN** buat modul monolitik >300 baris
- **JANGAN** hardcode values (gunakan config)
- **JANGAN** skip testing
- **JANGAN** commit tanpa review
- **JANGAN** ignore type hints
- **JANGAN** copy-paste code (refactor instead)

##### **🚫 Architecture**
- **JANGAN** tight coupling antar modul
- **JANGAN** circular imports
- **JANGAN** global state yang mutable
- **JANGAN** blocking operations di async context
- **JANGAN** mix sync/async patterns

##### **🚫 Database & Integration**
- **JANGAN** block async operations dengan sync calls
- **JANGAN** ignore connection pool limits
- **JANGAN** hardcode database queries
- **JANGAN** mix vector dan graph operations tanpa strategy
- **JANGAN** ignore transaction boundaries
- **JANGAN** skip connection error handling

---

### 🛠️ **TECHNICAL STANDARDS**

#### **Code Structure**
```python
# Template untuk setiap modul
"""
Module: [nama_modul]
Purpose: [tujuan spesifik]
Dependencies: [list dependencies]
"""

from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModuleName:
    """
    Brief description of the class.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    
    Attributes:
        attr1: Description of attribute
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2
    
    async def method_name(self, input_param: str) -> Dict[str, Any]:
        """
        Description of what this method does.
        
        Args:
            input_param: Description
            
        Returns:
            Dict containing result
            
        Raises:
            ValueError: When input is invalid
        """
        try:
            # Implementation here
            result = await self._process_input(input_param)
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Error in method_name: {e}")
            raise
```

#### **Database Operations (PostgreSQL + pgvector)**
```python
# Selalu gunakan connection pooling
# Selalu handle exceptions
# Selalu log operations
# Gunakan transactions untuk operasi kompleks

async def vector_search_operation(pool: asyncpg.Pool, query_vector: List[float]) -> Dict[str, Any]:
    async with pool.acquire() as conn:
        try:
            # Vector similarity search dengan pgvector
            result = await conn.fetch("""
                SELECT id, content, metadata, 
                       embedding <-> $1 as distance
                FROM documents 
                ORDER BY embedding <-> $1 
                LIMIT 10
            """, query_vector)
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise

async def hybrid_operation(pool: asyncpg.Pool, graph_client, query: str) -> Dict[str, Any]:
    """Combine vector search + graph traversal"""
    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                # 1. Vector search untuk initial results
                vector_results = await vector_search_operation(pool, query_vector)
                
                # 2. Graph traversal untuk related entities
                graph_results = await graph_client.search_related_entities(
                    entities=extract_entities(vector_results)
                )
                
                # 3. Combine dan rank results
                combined_results = merge_vector_graph_results(
                    vector_results, graph_results
                )
                
                return {"success": True, "data": combined_results}
            except Exception as e:
                logger.error(f"Hybrid search error: {e}")
                raise
```

#### **Graph Operations (Neo4j + Graphiti)**
```python
# Selalu gunakan Graphiti client
# Handle temporal relationships
# Maintain graph consistency

async def graph_entity_operation(graph_client: GraphitiClient, entities: List[str]) -> Dict[str, Any]:
    try:
        # Search related entities dengan temporal context
        related_entities = await graph_client.search_related_entities(
            entities=entities,
            temporal_context=True,
            max_depth=3
        )
        
        # Build relationship graph
        relationship_graph = await graph_client.build_relationship_graph(
            entities=entities + related_entities
        )
        
        return {
            "entities": related_entities,
            "relationships": relationship_graph,
            "temporal_data": await graph_client.get_temporal_context(entities)
        }
    except Exception as e:
        logger.error(f"Graph operation error: {e}")
        raise

async def update_graph_knowledge(graph_client: GraphitiClient, new_content: str) -> bool:
    """Update knowledge graph dengan content baru"""
    try:
        # Extract entities dan relationships
        entities = await extract_entities_from_content(new_content)
        relationships = await extract_relationships_from_content(new_content)
        
        # Add to graph dengan temporal context
        await graph_client.add_entities_and_relationships(
            entities=entities,
            relationships=relationships,
            timestamp=datetime.now(),
            source_content=new_content
        )
        
        return True
    except Exception as e:
        logger.error(f"Graph update error: {e}")
        return False
```

#### **RAG API Endpoints**
```python
# Selalu validate input
# Selalu handle errors gracefully
# Selalu return consistent format
# Gunakan proper HTTP status codes
# Support streaming responses

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, deps: AgentDependencies = Depends()) -> ChatResponse:
    """
    RAG-powered chat endpoint dengan vector + graph search.
    
    Args:
        request: Chat request dengan query dan session
        deps: Database dan graph dependencies
        
    Returns:
        Chat response dengan context dan sources
    """
    try:
        # Validate request
        if not request.message:
            raise HTTPException(400, "Message is required")
        
        # RAG pipeline: vector + graph search
        context = await build_rag_context(
            query=request.message,
            session_id=request.session_id,
            db_pool=deps.db_pool,
            graph_client=deps.graph_client
        )
        
        # Generate response dengan context
        response = await rag_agent.run(
            user_prompt=request.message,
            deps=deps,
            context=context
        )
        
        return ChatResponse(
            message=response.data,
            sources=context.sources,
            session_id=request.session_id,
            metadata=response.metadata
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")

@app.post("/api/search/hybrid")
async def hybrid_search_endpoint(
    request: SearchRequest, 
    deps: AgentDependencies = Depends()
) -> SearchResponse:
    """
    Hybrid search: vector similarity + graph traversal.
    """
    try:
        # Vector search
        vector_results = await vector_search_tool.run(
            VectorSearchInput(query=request.query, limit=request.limit),
            deps
        )
        
        # Graph search untuk related entities
        graph_results = await graph_search_tool.run(
            GraphSearchInput(
                entities=extract_entities(request.query),
                max_depth=2
            ),
            deps
        )
        
        # Combine results
        combined_results = merge_search_results(vector_results, graph_results)
        
        return SearchResponse(
            results=combined_results,
            total=len(combined_results),
            query=request.query
        )
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")
```

---

### 🧠 **RAG SYSTEM ARCHITECTURE**

#### **Memory Management**
- **Integrated Memory System**: Combine short-term + long-term memory
- **Chunking Strategies**: Semantic chunking untuk narrative content
- **Context Building**: Hierarchical context dari multiple sources
- **Consistency Management**: Maintain narrative consistency across sessions

#### **Pipeline Components**
```python
# RAG Pipeline Architecture
class RAGPipeline:
    def __init__(self):
        self.vector_store = PostgreSQLVectorStore()  # pgvector
        self.graph_store = Neo4jGraphStore()         # Neo4j + Graphiti
        self.memory_manager = IntegratedMemorySystem()
        self.context_builder = EnhancedContextBuilder()
        
    async def process_query(self, query: str, session_id: str) -> RAGResponse:
        # 1. Vector similarity search
        vector_results = await self.vector_store.similarity_search(query)
        
        # 2. Graph traversal untuk related entities
        entities = extract_entities(query)
        graph_results = await self.graph_store.traverse_relationships(entities)
        
        # 3. Memory retrieval dari session
        memory_context = await self.memory_manager.get_relevant_memory(
            session_id, query
        )
        
        # 4. Build comprehensive context
        context = await self.context_builder.build_context(
            vector_results, graph_results, memory_context
        )
        
        # 5. Generate response dengan context
        response = await self.generate_with_context(query, context)
        
        # 6. Update memory dengan new information
        await self.memory_manager.update_memory(session_id, query, response)
        
        return response
```

#### **Context Building Strategy**
- **Hierarchical Context**: Layer information by relevance dan recency
- **Multi-Source Integration**: Combine vector + graph + memory sources
- **Temporal Awareness**: Consider time-based relationships
- **Consistency Validation**: Check for contradictions dalam context

---

### 🔧 **DEVELOPMENT COMMANDS**

#### **Pre-Development Checklist**
```bash
# 1. Understand RAG system components
python -c "import agent; help(agent)"
python -c "import ingestion; help(ingestion)"
python -c "import memory; help(memory)"

# 2. Check database connections
python -c "from agent.db_utils import test_connection; import asyncio; asyncio.run(test_connection())"
python -c "from agent.graph_utils import test_graph_connection; import asyncio; asyncio.run(test_graph_connection())"

# 3. Check current tests
pytest tests/ -v
pytest tests/agent/ -v  # Agent tests
pytest tests/ingestion/ -v  # Ingestion tests

# 4. Review recent changes
git log --oneline -10

# 5. Check RAG dependencies
pip list | grep -E "(fastapi|pydantic|asyncpg|graphiti|pgvector)"

# 6. Verify environment variables
python -c "import os; print('DB:', bool(os.getenv('DATABASE_URL'))); print('Neo4j:', bool(os.getenv('NEO4J_URI')))"
```

#### **RAG Development Workflow**
```bash
# 1. Create feature branch
git checkout -b feature/rag-enhancement

# 2. Write tests first (vector + graph + hybrid)
# Edit tests/test_new_rag_feature.py
# Include tests untuk vector search, graph operations, dan hybrid search

# 3. Run tests (should fail initially)
pytest tests/test_new_rag_feature.py -v

# 4. Implement RAG feature
# Edit relevant modules: agent/, ingestion/, memory/
# Ensure proper async/await patterns
# Handle database connections properly

# 5. Debug dengan database connections
python -m pdb your_module.py
# Test PostgreSQL dan Neo4j connections
# Verify vector embeddings dan graph relationships

# 6. Run comprehensive tests
pytest tests/ --cov=agent --cov=ingestion --cov=memory
pytest tests/agent/ -v  # Agent-specific tests
pytest tests/ingestion/ -v  # Ingestion pipeline tests

# 7. Check code quality
black agent/ ingestion/ memory/
ruff check agent/ ingestion/ memory/
mypy agent/ ingestion/ memory/

# 8. Manual RAG testing
python -m agent.api  # Test RAG API endpoints
python cli.py        # Test RAG CLI interactions
# Test vector search: /api/search/vector
# Test graph search: /api/search/graph  
# Test hybrid search: /api/search/hybrid
# Test chat dengan RAG: /api/chat

# 9. Integration testing
python -m pytest tests/integration/ -v
# Test full RAG pipeline
# Test database integrations
# Test memory consistency

# 10. Update documentation
# Edit relevant .md files
# Update API documentation
# Document new RAG capabilities

# 11. Commit dengan detailed RAG context
git add .
git commit -m "feat(rag): detailed description of RAG enhancement

- Vector search improvements: [details]
- Graph integration changes: [details]  
- Memory system updates: [details]
- API endpoint modifications: [details]
- Database schema changes: [details]
- Test coverage: [percentage]
- Breaking changes: [if any]"
```

#### **RAG Post-Development Review**
```bash
# 1. RAG-specific code review checklist
echo "✓ Modular design (<300 lines per module)"
echo "✓ Comprehensive tests (>80% coverage)"
echo "✓ Vector + Graph + Hybrid search tests"
echo "✓ Database connection handling"
echo "✓ Async/await consistency"
echo "✓ Type hints everywhere"
echo "✓ Proper error handling"
echo "✓ Memory management efficiency"
echo "✓ Documentation updated"
echo "✓ No hardcoded database queries"
echo "✓ Graph relationship consistency"

# 2. RAG Integration testing
python -m pytest tests/integration/ -v
python -m pytest tests/agent/ -v
python -m pytest tests/ingestion/ -v

# 3. Database performance check
python -c "
import asyncio
from agent.db_utils import test_connection
from agent.graph_utils import test_graph_connection
asyncio.run(test_connection())
asyncio.run(test_graph_connection())
"

# 4. RAG pipeline performance
python -m cProfile -o rag_profile.stats -c "
import asyncio
from agent.api import app
# Profile RAG operations
"

# 5. Memory usage untuk RAG operations
python -m memory_profiler -c "
# Test memory usage untuk vector search
# Test memory usage untuk graph operations
# Test memory usage untuk hybrid search
"

# 6. API endpoint testing
curl -X POST http://localhost:8058/api/chat -H "Content-Type: application/json" -d '{"message": "test query", "session_id": "test"}'
curl -X POST http://localhost:8058/api/search/hybrid -H "Content-Type: application/json" -d '{"query": "test search", "limit": 10}'
```

---

### 📊 **MONITORING & DEBUGGING**

#### **Logging Standards**
```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Usage examples
logger.info("Starting generation process")
logger.warning("Unusual pattern detected in input")
logger.error("Generation failed", exc_info=True)
logger.debug("Intermediate result: %s", result)
```

#### **Performance Monitoring**
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper
```

---

### 🎯 **SPECIFIC INSTRUCTIONS FOR AI**

#### **When Reading This File**
1. **ALWAYS** refer to this file before making any changes to RAG system
2. **MANDATORY** to follow the DO's and avoid DON'Ts for RAG development
3. **REQUIRED** to understand PostgreSQL + Neo4j integration patterns
4. **ESSENTIAL** to follow the RAG development workflow
5. **CRITICAL** to test vector + graph + hybrid search functionality

#### **RAG Code Review Prompts**
- "Is this module under 300 lines?"
- "Are there comprehensive tests for vector + graph operations?"
- "Is database connection handling proper?"
- "Are there proper type hints for all RAG functions?"
- "Is error handling robust for database failures?"
- "Is async/await used consistently?"
- "Is documentation complete for RAG endpoints?"
- "Are vector embeddings handled efficiently?"
- "Is graph traversal optimized?"
- "Is memory management efficient?"

#### **RAG System Review Prompts**
- "Does vector search work with pgvector properly?"
- "Are graph relationships maintained consistently?"
- "Is hybrid search combining results intelligently?"
- "Are database transactions handled properly?"
- "Is the memory system integrated correctly?"
- "Are API endpoints following FastAPI best practices?"
- "Is the ingestion pipeline working efficiently?"
- "Are context building strategies effective?"

---

### 🚀 **FINAL REMINDERS**

> **"RAG is not just search—it's intelligent context building."**

> **"Vector similarity + Graph relationships = Powerful hybrid intelligence."**

> **"Every database operation should be async, efficient, and error-handled."**

> **"Refactor ruthlessly, test comprehensively, document thoroughly."**

> **"PostgreSQL + Neo4j integration requires careful connection management."**

> **"Memory systems should enhance, not replace, intelligent retrieval."**

---

**Last Updated**: [Current Date]  
**Version**: 1.0  
**Maintainer**: Development Team  
**Review Cycle**: Monthly or after major changes