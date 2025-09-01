from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.app.config import settings
from backend.app.routes import generate, job
from backend.app.db.mongodb_client import init_mongodb

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Novel AI System",
    description="Multi-agent novel generation system with CrewAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generate.router, prefix="/generate", tags=["generation"])
app.include_router(job.router, prefix="/job", tags=["jobs"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Novel AI System...")
    
    # Initialize MongoDB
    try:
        await init_mongodb()
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise
    
    logger.info("Novel AI System started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Novel AI System API",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "mongodb": settings.mongodb_url,
            "redis": settings.redis_url,
            "qdrant": settings.qdrant_url
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )