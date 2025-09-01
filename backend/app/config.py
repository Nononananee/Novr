import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database Configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "novel_ai")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # Neo4j Configuration
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "novelai123")
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    qdrant_key: Optional[str] = os.getenv("QDRANT_KEY")
    
    # Application Settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_revision_rounds: int = int(os.getenv("MAX_REVISION_ROUNDS", "2"))
    
    # QA Thresholds
    structural_qa_threshold: int = int(os.getenv("STRUCTURAL_QA_THRESHOLD", "70"))
    character_qa_threshold: int = int(os.getenv("CHARACTER_QA_THRESHOLD", "75"))
    style_qa_threshold: int = int(os.getenv("STYLE_QA_THRESHOLD", "70"))
    technical_qa_threshold: int = int(os.getenv("TECHNICAL_QA_THRESHOLD", "80"))
    
    # Embedding Settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "1024"))
    
    # Generation Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))
    
    # Phase 2 Settings
    enable_parallel_qa: bool = os.getenv("ENABLE_PARALLEL_QA", "true").lower() == "true"
    qa_timeout_seconds: int = int(os.getenv("QA_TIMEOUT_SECONDS", "30"))
    context_retrieval_timeout: int = int(os.getenv("CONTEXT_RETRIEVAL_TIMEOUT", "5"))
    
    class Config:
        env_file = ".env"

settings = Settings()