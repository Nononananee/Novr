"""
Main FastAPI application with modular endpoints.
Refactored from the large api.py file for better maintainability.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Import konfigurasi dan utilities
from .api_config import APIConfig
from .api_exceptions import (
    APIBaseException,
    ValidationError,
    AuthenticationError
)
from .api_cache import cache_manager
from .api_utils import HealthChecker

# Import models
from ..models import HealthStatus, ErrorResponse

# Import sistem monitoring
from ..monitoring.advanced_system_monitor import (
    system_monitor, start_system_monitoring, monitor_operation, ComponentType
)

# Import database dan graph utilities
from ..core.db_utils import initialize_database, close_database, get_session
from ..core.graph_utils import initialize_graph, close_graph
from ..core.tools import list_documents_tool, DocumentListInput

# Import endpoint routers
from .endpoints.chat import router as chat_router
from .endpoints.search import router as search_router
from .endpoints.novels import router as novels_router
from .endpoints.system import router as system_router
from .approval_api import router as approval_router

logger = logging.getLogger(__name__)

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

# Include all routers
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(novels_router)
app.include_router(system_router)
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
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


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


@app.get("/characters/{character_id}/arc")
async def get_character_arc_endpoint(character_id: str):
    """Get character development arc."""
    try:
        from ..core.db_utils import get_character_arc
        
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
async def global_exception_handler_final(request: Request, exc: Exception):
    """Final global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4())
    )


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
        "agent.api.main:app",
        host=APIConfig.APP_HOST,
        port=APIConfig.APP_PORT,
        reload=APIConfig.is_development(),
        log_level=APIConfig.LOG_LEVEL.lower(),
        access_log=APIConfig.is_development()
    )