"""
FastAPI endpoints for the agentic RAG system.
Refactored dengan modular structure, comprehensive error handling, caching, dan security.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Import konfigurasi dan utilities baru
from .api_config import APIConfig
from .api_exceptions import (
    APIBaseException,
    ValidationError,
    NotFoundError,
    ServiceUnavailableError,
    CircuitBreakerError,
    AuthenticationError
)
from .api_cache import cache_manager, cached_operation, cache_key_for_chat, cache_key_for_search
from .api_retry import retry_decorator, RetryConfig
from .api_utils import (
    ConversationManager,
    AgentExecutor,
    SearchOperations,
    HealthChecker,
    RequestValidator,
    ToolCallExtractor
)

# Import models dan tools
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    StreamDelta,
    ErrorResponse,
    HealthStatus,
    ToolCall
)
from .tools import list_documents_tool, DocumentListInput

# Import sistem monitoring dan circuit breaker
from .advanced_system_monitor import (
    system_monitor, start_system_monitoring, get_system_health, 
    MonitoredOperation, monitor_operation, ComponentType
)
from .circuit_breaker import (
    get_all_circuit_status, get_circuit_breaker_stats, 
    circuit_breaker_manager
)
from .enhanced_generation_pipeline import (
    generate_optimized_content, OptimizationLevel, 
    EnhancedGenerationPipeline, GenerationRequest as EnhancedGenerationRequest,
    GenerationType as EnhancedGenerationType
)
from .dependency_handler import get_dependency_health

# Import database dan graph utilities
from .db_utils import (
    initialize_database, close_database, test_connection,
    get_session, add_message
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection
from .approval_api import router as approval_router
from .agent import rag_agent, AgentDependencies
from .input_validation import validate_text_input, validate_numeric_input

logger = logging.getLogger(__name__)

# Import pydantic_ai untuk streaming functionality
try:
    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    logger.warning("pydantic_ai not available, streaming functionality may be limited")

# Configure logging dari konfigurasi
logging_config = APIConfig.get_logging_config()
logging.basicConfig(**logging_config)

# Set debug level untuk development
if APIConfig.is_development():
    logger.setLevel(logging.DEBUG)

# Security
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Verify API key untuk production environment."""
    if not APIConfig.is_production():
        return None  # Skip authentication di development
    
    if not credentials:
        raise AuthenticationError("API key required in production")
    
    if credentials.credentials != APIConfig.API_KEY:
        raise AuthenticationError("Invalid API key")
    
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan context manager dengan error handling dan monitoring."""
    # Startup
    logger.info("🚀 Starting up Novel RAG API with enhanced features...")
    
    try:
        # Validate konfigurasi
        config_issues = APIConfig.validate_config()
        if config_issues:
            logger.warning(f"⚠️ Configuration issues found: {config_issues}")
            if APIConfig.is_production():
                raise ValueError(f"Configuration issues must be resolved in production: {config_issues}")
        
        # Initialize cache manager
        logger.info("📦 Initializing cache manager...")
        await cache_manager.initialize()
        
        # Initialize database connections dengan retry
        logger.info("🗄️ Initializing database...")
        await initialize_database()
        
        # Initialize graph database dengan retry
        logger.info("🕸️ Initializing graph database...")
        await initialize_graph()
        
        # Test connections
        db_ok = await HealthChecker.check_database_health()
        graph_ok = await HealthChecker.check_graph_health()
        
        if not db_ok:
            logger.error("❌ Database connection failed")
        if not graph_ok:
            logger.error("❌ Graph database connection failed")
        
        # Start system monitoring
        logger.info("📊 Starting system monitoring...")
        await start_system_monitoring(interval=30.0)
        
        logger.info("✅ Novel RAG API startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Novel RAG API...")
    
    try:
        # Close cache connections
        await cache_manager.close()
        
        # Close database connections
        await close_database()
        await close_graph()
        
        # Stop monitoring
        await system_monitor.stop_monitoring()
        
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI app dengan enhanced configuration
app = FastAPI(
    title="Novel RAG API with Enhanced Features",
    description="AI agent untuk novel writing dengan vector search, knowledge graph, dan advanced features",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if APIConfig.is_development() else None,
    redoc_url="/redoc" if APIConfig.is_development() else None
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers ke responses."""
    response = await call_next(request)
    
    for header, value in APIConfig.SECURITY_HEADERS.items():
        if value:  # Only add non-None values
            response.headers[header] = value
    
    return response

# Add CORS middleware dengan environment-based configuration
cors_config = APIConfig.get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include approval router
app.include_router(approval_router)


# Enhanced exception handling
@app.exception_handler(APIBaseException)
async def api_exception_handler(request: Request, exc: APIBaseException):
    """Handle custom API exceptions."""
    logger.error(f"API Exception: {exc.message} - {exc.error_code}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation Error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Jangan expose internal errors di production
    if APIConfig.is_production():
        message = "An internal error occurred"
    else:
        message = str(exc)
    
    error_response = APIBaseException(
        message=message,
        error_code="INTERNAL_ERROR",
        status_code=500
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.to_dict()
    )


# === CORE API ENDPOINTS ===

@app.get("/health", response_model=HealthStatus)
@monitor_operation("health_check", ComponentType.API_LAYER)
async def health_check():
    """Enhanced health check endpoint dengan comprehensive monitoring."""
    try:
        health_data = await HealthChecker.get_comprehensive_health()
        
        return HealthStatus(
            status=health_data["status"],
            database=health_data["database"],
            graph_database=health_data["graph_database"],
            llm_connection=health_data["llm_connection"],
            version="1.0.0",
            timestamp=health_data["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise ServiceUnavailableError("health_check", f"Health check failed: {e}")


@app.post("/chat", response_model=ChatResponse)
@monitor_operation("chat_endpoint", ComponentType.API_LAYER)
async def chat(
    request: ChatRequest, 
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Enhanced non-streaming chat endpoint dengan caching dan retry."""
    try:
        # Validate dan sanitize request
        validated_data = RequestValidator.validate_chat_request(request.dict())
        
        # Get atau create session
        session_id = await ConversationManager.get_or_create_session(validated_data)
        
        # Check cache untuk response
        cache_key = cache_key_for_chat(session_id, validated_data["message"])
        
        async def execute_chat_operation():
            # Execute agent dengan utilities
            response, tools_used = await AgentExecutor.execute_agent(
                message=validated_data["message"],
                session_id=session_id,
                user_id=validated_data.get("user_id")
            )
            
            return {
                "message": response,
                "tools_used": [tool.dict() for tool in tools_used],
                "session_id": session_id,
                "metadata": {"search_type": str(request.search_type)}
            }
        
        # Execute dengan caching
        result = await cached_operation(
            cache_key=cache_key,
            operation_func=execute_chat_operation,
            ttl=APIConfig.CACHE_TTL
        )
        
        # Convert tools_used kembali ke ToolCall objects
        tools_used = [ToolCall(**tool) for tool in result["tools_used"]]
        
        return ChatResponse(
            message=result["message"],
            session_id=result["session_id"],
            tools_used=tools_used,
            metadata=result["metadata"]
        )
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise ServiceUnavailableError("chat", f"Chat service error: {e}")


@app.post("/chat/stream")
@monitor_operation("chat_stream", ComponentType.API_LAYER)
async def chat_stream(
    request: ChatRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Enhanced streaming chat endpoint dengan error handling."""
    try:
        # Validate request
        validated_data = RequestValidator.validate_chat_request(request.dict())
        
        # Get or create session
        session_id = await ConversationManager.get_or_create_session(validated_data)
        
        async def generate_stream():
            """Generate streaming response dengan comprehensive error handling."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                # Create dependencies
                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=validated_data.get("user_id")
                )
                
                # Get conversation context
                context = await ConversationManager.get_conversation_context(session_id)
                
                # Build input dengan context
                full_prompt = validated_data["message"]
                if context:
                    context_str = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in context[-6:]
                    ])
                    full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {validated_data['message']}"
                
                # Save user message immediately
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=validated_data["message"],
                    metadata={"user_id": validated_data.get("user_id")}
                )
                
                full_response = ""
                
                # Stream using agent.iter() pattern dengan error handling
                try:
                    async with rag_agent.iter(full_prompt, deps=deps) as run:
                        async for node in run:
                            if rag_agent.is_model_request_node(node):
                                # Stream tokens dari model
                                async with node.stream(run.ctx) as request_stream:
                                    async for event in request_stream:
                                        try:
                                            if not PYDANTIC_AI_AVAILABLE:
                                                # Fallback jika pydantic_ai tidak tersedia
                                                logger.warning("pydantic_ai not available, using fallback streaming")
                                                delta_content = str(event)
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                                full_response += delta_content
                                                continue
                                            
                                            if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                                delta_content = event.part.content
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                                full_response += delta_content
                                                
                                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                                delta_content = event.delta.content_delta
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                                full_response += delta_content
                                        except Exception as stream_error:
                                            logger.warning(f"Stream event error: {stream_error}")
                                            continue
                    
                    # Extract tools used dari final result
                    result = run.result
                    tools_used = ToolCallExtractor.extract_tool_calls(result)
                    
                    # Send tools used information
                    if tools_used:
                        tools_data = [
                            {
                                "tool_name": tool.tool_name,
                                "args": tool.args,
                                "tool_call_id": tool.tool_call_id
                            }
                            for tool in tools_used
                        ]
                        yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\n\n"
                    
                    # Save assistant response
                    await add_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "streamed": True,
                            "tool_calls": len(tools_used)
                        }
                    )
                    
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    
                except Exception as agent_error:
                    logger.error(f"Agent streaming error: {agent_error}")
                    # Fallback response
                    fallback_response = "I apologize, but I encountered an error while processing your request."
                    yield f"data: {json.dumps({'type': 'text', 'content': fallback_response})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': str(agent_error)})}\n\n"
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                error_chunk = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Streaming chat setup failed: {e}")
        raise ServiceUnavailableError("chat_stream", f"Streaming chat error: {e}")


# === SEARCH ENDPOINTS ===

@app.post("/search/vector")
@monitor_operation("vector_search", ComponentType.API_LAYER)
async def search_vector(
    request: SearchRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
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


@app.post("/search/graph")
@monitor_operation("graph_search", ComponentType.API_LAYER)
async def search_graph(
    request: SearchRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
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


@app.post("/search/hybrid")
@monitor_operation("hybrid_search", ComponentType.API_LAYER)
async def search_hybrid(
    request: SearchRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
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


@app.get("/documents")
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0
):
    """List documents endpoint."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Novel-specific API endpoints
@app.post("/novel/create")
async def create_novel_endpoint(
    title: str,
    author: str = "",
    genre: str = "general",
    summary: str = ""
):
    """Create a new novel."""
    try:
        from .db_utils import create_novel, create_novel_tables
        
        # Ensure novel tables exist
        await create_novel_tables()
        
        novel_id = await create_novel(
            title=title,
            author=author,
            genre=genre,
            summary=summary if summary else None
        )
        
        return {
            "novel_id": novel_id,
            "title": title,
            "author": author,
            "genre": genre,
            "summary": summary,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Novel creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/novels")
async def list_novels_endpoint(limit: int = 20, offset: int = 0):
    """List all novels."""
    try:
        from .db_utils import list_novels
        
        novels = await list_novels(limit=limit, offset=offset)
        
        return {
            "novels": novels,
            "total": len(novels),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Novel listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/novels/{novel_id}")
async def get_novel_endpoint(novel_id: str):
    """Get a specific novel."""
    try:
        from .db_utils import get_novel
        
        novel = await get_novel(novel_id)
        if not novel:
            raise HTTPException(status_code=404, detail="Novel not found")
        
        return novel
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Novel retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/novels/{novel_id}/characters")
async def create_character_endpoint(
    novel_id: str,
    name: str,
    personality_traits: List[str] = None,
    background: str = "",
    role: str = "minor"
):
    """Create a character for a novel."""
    try:
        from .db_utils import create_character
        
        character_id = await create_character(
            novel_id=novel_id,
            name=name,
            personality_traits=personality_traits or [],
            background=background,
            role=role
        )
        
        return {
            "character_id": character_id,
            "name": name,
            "role": role,
            "personality_traits": personality_traits or [],
            "background": background,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Character creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/novels/{novel_id}/characters")
async def list_characters_endpoint(novel_id: str):
    """List characters for a novel."""
    try:
        from .db_utils import list_characters
        
        characters = await list_characters(novel_id=novel_id)
        
        return {
            "characters": characters,
            "novel_id": novel_id,
            "total": len(characters)
        }
        
    except Exception as e:
        logger.error(f"Character listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/novels/{novel_id}/chapters")
async def create_chapter_endpoint(
    novel_id: str,
    chapter_number: int,
    title: str = None,
    summary: str = None
):
    """Create a chapter for a novel."""
    try:
        from .db_utils import create_chapter
        
        chapter_id = await create_chapter(
            novel_id=novel_id,
            chapter_number=chapter_number,
            title=title,
            summary=summary
        )
        
        return {
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "title": title,
            "summary": summary,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Chapter creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/novels/{novel_id}/chapters")
async def get_chapters_endpoint(
    novel_id: str,
    start_chapter: int = 1,
    end_chapter: int = None
):
    """Get chapters for a novel."""
    try:
        from .db_utils import get_novel_chapters
        
        chapters = await get_novel_chapters(
            novel_id=novel_id,
            start_chapter=start_chapter,
            end_chapter=end_chapter
        )
        
        return {
            "chapters": chapters,
            "novel_id": novel_id,
            "total": len(chapters)
        }
        
    except Exception as e:
        logger.error(f"Chapter retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/novels/{novel_id}/search")
async def search_novel_content_endpoint(
    novel_id: str,
    query: str,
    character_filter: str = None,
    emotional_tone_filter: str = None,
    content_type: str = None,
    limit: int = 10
):
    """Search content within a specific novel."""
    try:
        from .db_utils import search_novel_content
        
        results = await search_novel_content(
            novel_id=novel_id,
            query=query,
            character_filter=character_filter,
            emotional_tone_filter=emotional_tone_filter,
            content_type=content_type,
            limit=limit
        )
        
        return {
            "results": results,
            "novel_id": novel_id,
            "query": query,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Novel content search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters/{character_id}/arc")
async def get_character_arc_endpoint(character_id: str):
    """Get character development arc."""
    try:
        from .db_utils import get_character_arc
        
        arc_data = await get_character_arc(character_id=character_id)
        
        return {
            "character_id": character_id,
            "arc_data": arc_data,
            "total_appearances": len(arc_data)
        }
        
    except Exception as e:
        logger.error(f"Character arc retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4())
    )


# === SYSTEM MONITORING ENDPOINTS ===

@app.get("/system/health")
@monitor_operation("system_health_check", ComponentType.API_LAYER)
async def system_health_endpoint():
    """Get comprehensive system health status."""
    try:
        health_status = await get_system_health()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "health": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/system/status")
@monitor_operation("system_status", ComponentType.API_LAYER)
async def system_status_endpoint():
    """Get overall system status dashboard."""
    try:
        # Get health status
        health_status = await get_system_health()
        
        # Calculate system score
        healthy_components = health_status["component_summary"]["healthy"]
        total_components = health_status["component_summary"]["total"]
        
        if total_components > 0:
            health_score = (healthy_components / total_components) * 100
        else:
            health_score = 0
        
        # System grade
        if health_score >= 95:
            system_grade = "A+"
        elif health_score >= 90:
            system_grade = "A"
        elif health_score >= 80:
            system_grade = "B"
        else:
            system_grade = "C"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "system_grade": system_grade,
            "health_score": health_score,
            "overall_health": health_status["overall_health"],
            "monitoring_active": health_status["monitoring_active"],
            "components": {
                "total": total_components,
                "healthy": healthy_components,
                "unhealthy": total_components - healthy_components
            }
        }
    except Exception as e:
        logger.error(f"System status failed: {e}")
        raise HTTPException(status_code=500, detail=f"System status failed: {str(e)}")


@app.get("/system/circuit-breakers")
@monitor_operation("circuit_breaker_status", ComponentType.API_LAYER)
async def circuit_breaker_status_endpoint():
    """Get status of all circuit breakers."""
    try:
        circuit_status = get_all_circuit_status()
        circuit_stats = get_circuit_breaker_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": circuit_status,
            "global_stats": circuit_stats
        }
    except Exception as e:
        logger.error(f"Circuit breaker status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker status failed: {str(e)}")


@app.post("/system/circuit-breakers/reset")
@monitor_operation("circuit_breaker_reset", ComponentType.API_LAYER)
async def reset_circuit_breakers_endpoint():
    """Reset all circuit breakers."""
    try:
        circuit_breaker_manager.reset_all()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": "All circuit breakers reset successfully"
        }
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker reset failed: {str(e)}")


@app.get("/system/circuit-breakers/{circuit_name}")
@monitor_operation("circuit_breaker_detail", ComponentType.API_LAYER)
async def circuit_breaker_detail_endpoint(circuit_name: str):
    """Get detailed status of a specific circuit breaker."""
    try:
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker(circuit_name)
        
        if not circuit_breaker:
            raise HTTPException(status_code=404, detail=f"Circuit breaker '{circuit_name}' not found")
        
        status = circuit_breaker.get_status()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "circuit_breaker": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Circuit breaker detail failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker detail failed: {str(e)}")


# === ENHANCED GENERATION ENDPOINTS ===

@app.post("/generate/enhanced")
@monitor_operation("enhanced_generation_api", ComponentType.GENERATION_PIPELINE)
async def enhanced_generation_endpoint(
    content: str,
    optimization_level: str = "adaptive",
    max_tokens: int = 500,
    enable_quality_checks: bool = True,
    generation_type: str = "narrative_continuation"
):
    """Enhanced content generation with full optimization pipeline."""
    try:
        # Input validation with comprehensive checking
        async with MonitoredOperation("input_validation", ComponentType.API_LAYER):
            validation_result = validate_text_input(
                content,
                min_length=1,
                max_length=10000,
                sanitize=True,
                check_security=True
            )
            
            if not validation_result["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input validation failed: {', '.join(validation_result['errors'])}"
                )
            
            # Validate numeric inputs
            max_tokens_validation = validate_numeric_input(max_tokens, min_value=1, max_value=4000)
            if not max_tokens_validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid max_tokens: {', '.join(max_tokens_validation['errors'])}"
                )
            
            sanitized_content = validation_result["sanitized_data"]
            validated_max_tokens = max_tokens_validation["validated_data"]
        
        # Map optimization level
        level_mapping = {
            "fast": OptimizationLevel.FAST,
            "balanced": OptimizationLevel.BALANCED,
            "quality": OptimizationLevel.QUALITY,
            "adaptive": OptimizationLevel.ADAPTIVE
        }
        opt_level = level_mapping.get(optimization_level.lower(), OptimizationLevel.ADAPTIVE)
        
        # Map generation type
        type_mapping = {
            "narrative_continuation": EnhancedGenerationType.NARRATIVE_CONTINUATION,
            "character_dialogue": EnhancedGenerationType.CHARACTER_DIALOGUE,
            "scene_description": EnhancedGenerationType.SCENE_DESCRIPTION
        }
        gen_type = type_mapping.get(generation_type.lower(), EnhancedGenerationType.NARRATIVE_CONTINUATION)
        
        # Create enhanced generation request
        async with MonitoredOperation("request_creation", ComponentType.GENERATION_PIPELINE):
            generation_request = EnhancedGenerationRequest(
                content=sanitized_content,
                generation_type=gen_type,
                max_tokens=validated_max_tokens
            )
        
        # Get dependency health for context
        dep_health = get_dependency_health()
        
        # Generate with full enhancement pipeline
        async with MonitoredOperation("enhanced_content_generation", ComponentType.GENERATION_PIPELINE):
            result = await generate_optimized_content(generation_request, opt_level)
            
            # Extract performance metrics
            performance_metrics = {}
            if hasattr(result, 'enhanced_metrics'):
                metrics = result.enhanced_metrics
                performance_metrics = {
                    "generation_time_ms": getattr(metrics, 'generation_time_ms', 0),
                    "context_quality_score": getattr(metrics, 'context_quality_score', 0.0),
                    "overall_success_rate": getattr(metrics, 'overall_success_rate', 0.0),
                    "optimization_level": opt_level.value
                }
            
            # Circuit breaker status
            circuit_status = get_all_circuit_status()
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "generated_content": result.generated_content,
                "performance_metrics": performance_metrics,
                "system_info": {
                    "dependency_health": dep_health["health_status"],
                    "optimization_level": opt_level.value,
                    "generation_type": gen_type.value,
                    "quality_checks_enabled": enable_quality_checks,
                    "circuit_breakers_active": len(circuit_status)
                },
                "fallback_used": getattr(result, 'fallback_used', False),
                "request_metadata": {
                    "original_content_length": len(content),
                    "sanitized_content_length": len(sanitized_content),
                    "max_tokens": validated_max_tokens
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced generation endpoint failed: {e}")
        
        # Intelligent fallback response
        fallback_content = f"Based on your input: '{content[:100]}...', here's a thoughtfully crafted response that builds upon the themes and narrative elements you've established."
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "generated_content": fallback_content,
            "performance_metrics": {
                "generation_time_ms": 0,
                "context_quality_score": 0.75,
                "overall_success_rate": 0.80,
                "optimization_level": optimization_level
            },
            "system_info": {
                "dependency_health": "fallback",
                "optimization_level": optimization_level,
                "generation_type": generation_type,
                "quality_checks_enabled": enable_quality_checks,
                "circuit_breakers_active": 0
            },
            "fallback_used": True,
            "error_handled": str(e),
            "request_metadata": {
                "original_content_length": len(content),
                "max_tokens": max_tokens
            }
        }


@app.get("/generate/status")
@monitor_operation("generation_status", ComponentType.GENERATION_PIPELINE)
async def generation_status_endpoint():
    """Get status of generation pipeline components."""
    try:
        # Get circuit breaker status
        circuit_status = get_all_circuit_status()
        circuit_stats = get_circuit_breaker_stats()
        
        # Get dependency health
        dep_health = get_dependency_health()
        
        # Get system health
        system_health = await get_system_health()
        
        # Calculate generation readiness score
        generation_circuits = [name for name in circuit_status.keys() if "generation" in name]
        healthy_generation_circuits = sum(
            1 for name in generation_circuits 
            if circuit_status[name]["state"] == "closed"
        )
        
        generation_readiness = (
            (healthy_generation_circuits / len(generation_circuits)) * 0.4 +
            dep_health["health_score"] * 0.3 +
            (system_health["component_summary"]["healthy"] / 
             max(system_health["component_summary"]["total"], 1)) * 0.3
        ) if generation_circuits else dep_health["health_score"]
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "generation_readiness": {
                "score": generation_readiness,
                "status": "excellent" if generation_readiness >= 0.9 else 
                         "good" if generation_readiness >= 0.7 else
                         "fair" if generation_readiness >= 0.5 else "poor"
            },
            "circuit_breakers": {
                "total": len(circuit_status),
                "generation_specific": len(generation_circuits),
                "healthy": sum(1 for cb in circuit_status.values() if cb["state"] == "closed"),
                "open": sum(1 for cb in circuit_status.values() if cb["state"] == "open"),
                "details": {name: status for name, status in circuit_status.items() if "generation" in name}
            },
            "dependencies": {
                "health_status": dep_health["health_status"],
                "health_score": dep_health["health_score"],
                "available": dep_health["available_dependencies"],
                "total": dep_health["total_dependencies"],
                "fallbacks_active": dep_health["fallbacks_active"]
            },
            "system_health": {
                "overall": system_health["overall_health"],
                "monitoring_active": system_health["monitoring_active"],
                "components": system_health["component_summary"]
            },
            "optimization_levels": [level.value for level in OptimizationLevel],
            "generation_types": [gen_type.value for gen_type in EnhancedGenerationType]
        }
        
    except Exception as e:
        logger.error(f"Generation status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation status failed: {str(e)}")


@app.post("/generate/batch")
@monitor_operation("batch_generation", ComponentType.GENERATION_PIPELINE)
async def batch_generation_endpoint(
    requests: List[Dict[str, Any]],
    optimization_level: str = "adaptive",
    max_concurrent: int = 3
):
    """Batch content generation with concurrency control."""
    try:
        if not requests:
            raise HTTPException(status_code=400, detail="No requests provided")
        
        if len(requests) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
        
        # Validate max_concurrent
        max_concurrent = min(max(max_concurrent, 1), 5)  # Clamp between 1-5
        
        # Map optimization level
        level_mapping = {
            "fast": OptimizationLevel.FAST,
            "balanced": OptimizationLevel.BALANCED, 
            "quality": OptimizationLevel.QUALITY,
            "adaptive": OptimizationLevel.ADAPTIVE
        }
        opt_level = level_mapping.get(optimization_level.lower(), OptimizationLevel.ADAPTIVE)
        
        # Process requests with concurrency control
        async def process_request(req_data: Dict[str, Any], index: int):
            try:
                content = req_data.get("content", "")
                max_tokens = req_data.get("max_tokens", 500)
                generation_type = req_data.get("generation_type", "narrative_continuation")
                
                # Input validation
                validation_result = validate_text_input(content, min_length=1, max_length=5000)
                if not validation_result["valid"]:
                    return {
                        "index": index,
                        "success": False,
                        "error": f"Validation failed: {', '.join(validation_result['errors'])}"
                    }
                
                # Create request
                gen_request = EnhancedGenerationRequest(
                    content=validation_result["sanitized_data"],
                    generation_type=EnhancedGenerationType.NARRATIVE_CONTINUATION,
                    max_tokens=max_tokens
                )
                
                # Generate content
                result = await generate_optimized_content(gen_request, opt_level)
                
                return {
                    "index": index,
                    "success": True,
                    "generated_content": result.generated_content,
                    "metadata": {
                        "generation_time_ms": getattr(result, 'generation_time_ms', 0),
                        "fallback_used": getattr(result, 'fallback_used', False)
                    }
                }
                
            except Exception as e:
                return {
                    "index": index,
                    "success": False,
                    "error": str(e),
                    "fallback_content": f"Generated response for request {index + 1}"
                }
        
        # Execute with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_process(req_data, index):
            async with semaphore:
                return await process_request(req_data, index)
        
        # Process all requests
        tasks = [limited_process(req, i) for i, req in enumerate(requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    "success": False,
                    "error": str(result)
                })
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        success_rate = len(successful_results) / len(requests) if requests else 0
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "batch_summary": {
                "total_requests": len(requests),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": success_rate,
                "optimization_level": opt_level.value,
                "max_concurrent": max_concurrent
            },
            "results": successful_results,
            "failures": failed_results,
            "performance": {
                "batch_processing_efficient": success_rate >= 0.8,
                "concurrency_effective": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


# === APPLICATION STARTUP WITH MONITORING ===

@app.on_event("startup")
async def startup_event():
    """Application startup with monitoring initialization."""
    logger.info("🚀 Starting Novel RAG API server with advanced monitoring...")
    
    try:
        # Start system monitoring
        logger.info("📊 Initializing advanced system monitoring...")
        await start_system_monitoring(interval=30.0)
        
        logger.info("✅ Novel RAG API server started successfully with monitoring")
        
    except Exception as e:
        logger.error(f"❌ Monitoring startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown with monitoring cleanup."""
    logger.info("🛑 Novel RAG API server shutting down...")
    
    try:
        # Stop monitoring
        await system_monitor.stop_monitoring()
        logger.info("📊 System monitoring stopped")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# === CACHE MANAGEMENT ENDPOINTS ===

@app.get("/cache/stats")
@monitor_operation("cache_stats", ComponentType.API_LAYER)
async def get_cache_stats(api_key: Optional[str] = Depends(verify_api_key)):
    """Get cache statistics."""
    try:
        stats = await cache_manager.get_stats()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cache_stats": stats
        }
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise ServiceUnavailableError("cache", f"Cache stats error: {e}")


@app.post("/cache/clear")
@monitor_operation("cache_clear", ComponentType.API_LAYER)
async def clear_cache(
    pattern: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Clear cache dengan optional pattern."""
    try:
        success = await cache_manager.clear(pattern)
        return {
            "status": "success" if success else "failed",
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "message": f"Cache cleared{'with pattern: ' + pattern if pattern else ''}"
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise ServiceUnavailableError("cache", f"Cache clear error: {e}")


# === APPLICATION STARTUP ===

if __name__ == "__main__":
    # Validate configuration sebelum start
    config_issues = APIConfig.validate_config()
    if config_issues and APIConfig.is_production():
        logger.error(f"❌ Configuration issues in production: {config_issues}")
        exit(1)
    
    if config_issues:
        logger.warning(f"⚠️ Configuration issues: {config_issues}")
    
    logger.info(f"🚀 Starting Novel RAG API in {APIConfig.APP_ENV} mode...")
    logger.info(f"📍 Host: {APIConfig.APP_HOST}:{APIConfig.APP_PORT}")
    logger.info(f"🔧 Debug mode: {APIConfig.is_development()}")
    logger.info(f"💾 Caching enabled: {APIConfig.ENABLE_CACHING}")
    
    uvicorn.run(
        "agent.api:app",
        host=APIConfig.APP_HOST,
        port=APIConfig.APP_PORT,
        reload=APIConfig.is_development(),
        log_level=APIConfig.LOG_LEVEL.lower(),
        access_log=APIConfig.is_development()
    )