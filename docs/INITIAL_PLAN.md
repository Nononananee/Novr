# Creative RAG System for Novel Generation - Project Plan

## Project Overview

This project builds a sophisticated AI system for creative novel generation that combines RAG (Retrieval Augmented Generation) with knowledge graph capabilities and a human-in-the-loop approval workflow. The system uses PostgreSQL with pgvector for semantic search and Neo4j with Graphiti for temporal knowledge graphs to maintain narrative consistency and character development.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                             │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   FastAPI       │        │   Streaming SSE    │     │
│  │   Endpoints     │        │   Responses        │     │
│  └────────┬────────┘        └────────────────────┘     │
│           │                                              │
├───────────┴──────────────────────────────────────────────┤
│                 Generation Layer                         │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │  Generation     │        │   Context Builder  │     │
│  │   Pipeline      │◄──────►│   & Validators     │     │
│  └────────┬────────┘        └────────────────────┘     │
│           │                                              │
├───────────┴──────────────────────────────────────────────┤
│                    Agent Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │  Pydantic AI    │        │   Agent Tools      │     │
│  │    Agent        │◄──────►│  - Vector Search   │     │
│  └────────┬────────┘        │  - Graph Search    │     │
│           │                 │  - Doc Retrieval   │     │
│           │                 └────────────────────┘     │
├───────────┴──────────────────────────────────────────────┤
│                  Memory Layer                            │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │  Integrated     │        │   Cache Memory     │     │
│  │  Memory System  │◄──────►│   & Consistency    │     │
│  └────────┬────────┘        └────────────────────┘     │
│           │                                              │
├───────────┴──────────────────────────────────────────────┤
│                  Storage Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   PostgreSQL    │        │      Neo4j         │     │
│  │   + pgvector    │        │   (via Graphiti)   │     │
│  └─────────────────┘        └────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent System (`/agent`)
- **agent.py**: Main Pydantic AI agent with system prompts and configuration
- **tools.py**: All agent tools for RAG and knowledge graph operations
- **prompts.py**: System prompts controlling agent tool selection behavior
- **api.py**: FastAPI endpoints with streaming support and tool usage extraction
- **db_utils.py**: PostgreSQL database utilities and connection management
- **graph_utils.py**: Neo4j/Graphiti utilities with OpenAI-compatible client configuration
- **models.py**: Pydantic models for data validation including ToolCall tracking
- **providers.py**: Flexible LLM provider abstraction supporting multiple backends

### 2. Novel Generation System (`/agent`)
- **enhanced_context_builder.py**: Builds intelligent, hierarchical context for generation
- **generation_pipeline.py**: Orchestrates content generation with validation
- **consistency_validators.py**: Ensures narrative consistency and character authenticity
- **approval_api.py**: Human-in-the-loop review and approval process

### 3. Memory System (`/memory`)
- **integrated_memory_system.py**: Manages all memory components
- **cache_memory.py**: Short-term memory for recent context
- **long_term_memory.py**: Persistent storage for narrative elements
- **consistency_manager.py**: Ensures narrative and character consistency
- **chunking_strategies.py**: Specialized chunking for narrative content
- **memory_helpers.py**: Utility functions for memory operations

### 4. Ingestion System (`/ingestion`)
- **ingest.py**: Main ingestion script to process markdown files
- **chunker.py**: Semantic chunking implementation
- **embedder.py**: Document embedding generation
- **graph_builder.py**: Knowledge graph construction from documents

### 5. Database Schema (`/sql`)
- **schema.sql**: PostgreSQL schema with pgvector and approval tables

### 6. UI Templates (`/templates`)
- **approval_flow.html**: Human-in-the-loop approval interface

### 7. Tests (`/tests`)
- Comprehensive unit and integration tests
- Mocked external dependencies
- Test fixtures and utilities

### 8. CLI Interface (`/cli.py`)
- Interactive command-line interface for the agent
- Real-time streaming with Server-Sent Events
- Tool usage visibility showing agent reasoning
- Session management and conversation context

## Technical Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **Pydantic AI**: Agent framework
- **FastAPI**: API framework
- **PostgreSQL + pgvector**: Vector database
- **Neo4j + Graphiti**: Knowledge graph
- **Flexible LLM Providers**: OpenAI, Ollama, OpenRouter, Gemini

### Key Libraries
- **asyncpg**: PostgreSQL async driver
- **httpx**: Async HTTP client
- **python-dotenv**: Environment management
- **pytest + pytest-asyncio**: Testing
- **black + ruff**: Code formatting/linting
- **numpy + scipy**: Numerical operations
- **scikit-learn**: Machine learning utilities

## Design Principles

### 1. Modularity
- Clear separation of concerns
- Reusable components
- Clean dependency injection

### 2. Type Safety
- Comprehensive type hints
- Pydantic models for validation
- Dataclasses for dependencies

### 3. Async-First
- All database operations async
- Concurrent processing where applicable
- Proper resource management

### 4. Error Handling
- Graceful degradation
- Comprehensive logging
- User-friendly error messages

### 5. Testing
- Unit tests for all components
- Integration tests for workflows
- Mocked external dependencies

## Key Features

### 1. Novel Generation Pipeline
- Enhanced context building for narrative coherence
- Multiple generation strategies for different content types
- Quality assessment and validation
- Human-in-the-loop approval workflow

### 2. Consistency Validation
- Fact checking against established narrative
- Character behavior consistency
- Dialogue style authenticity
- Trope detection and avoidance
- Timeline consistency

### 3. Memory Management
- Integrated memory system for narrative elements
- Cache memory for recent context
- Long-term memory for persistent narrative
- Consistency management across chapters

### 4. Hybrid Search
- Vector similarity search for semantic queries
- Knowledge graph traversal for relationship queries
- Combined results with intelligent ranking

### 5. Human-in-the-Loop Workflow
- Proposal creation and validation
- Web-based approval interface
- Risk assessment and confidence scoring
- Neo4j push functionality for approved content

### 6. API Capabilities
- Streaming responses (SSE)
- Session management
- Approval workflow endpoints

## Implementation Strategy

### Phase 1: Foundation
1. Set up project structure
2. Configure PostgreSQL and Neo4j
3. Implement database utilities
4. Create base models

### Phase 2: Core Agent
1. Build Pydantic AI agent
2. Implement RAG tools
3. Implement knowledge graph tools
4. Create prompts and configurations

### Phase 3: Memory System
1. Build integrated memory controller
2. Implement cache memory
3. Create long-term memory storage
4. Add consistency management

### Phase 4: Generation Pipeline
1. Create enhanced context builder
2. Implement generation strategies
3. Add quality assessment
4. Create approval workflow integration

### Phase 5: Approval Workflow
1. Build proposal models and database tables
2. Implement approval API endpoints
3. Create validation results storage
4. Build approval UI template

### Phase 6: Testing & Documentation
1. Write comprehensive tests
2. Create detailed README
3. Generate API documentation
4. Add usage examples

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/creative_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration  
LLM_PROVIDER=openai  # openai, ollama, openrouter, gemini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
INGESTION_LLM_CHOICE=gpt-4.1-nano

# Application
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

### Database Schema
- **documents**: Store document metadata
- **chunks**: Store document chunks with embeddings
- **sessions**: Manage conversation sessions
- **messages**: Store conversation history
- **proposals**: Store content proposals for approval
- **validation_results**: Store validation outcomes

## Security Considerations
- Environment-based configuration
- No hardcoded credentials
- Input validation at all layers
- SQL injection prevention
- Rate limiting on API

## Performance Optimizations
- Connection pooling for databases
- Embedding caching
- Batch processing for ingestion
- Indexed vector searches
- Async operations throughout

## Monitoring & Logging
- Structured logging with context
- Performance metrics
- Error tracking
- Usage analytics

## Future Enhancements
- Enhanced narrative-aware chunking
- Multi-embedding strategy for different content types
- Character canonicalization and relationship extraction
- Advanced validation for narrative consistency
- Writer dashboard with visualization
- Genre-specific modules for different writing styles
- Export and publishing tools