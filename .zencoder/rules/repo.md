---
description: Repository Information Overview
alwaysApply: true
---

# Creative RAG System for Novel Generation

## Summary
A sophisticated AI system for creative novel generation that combines RAG (Retrieval Augmented Generation) with knowledge graph capabilities and a human-in-the-loop approval workflow. The system uses PostgreSQL with pgvector for semantic search and Neo4j with Graphiti for temporal knowledge graphs to maintain narrative consistency and character development.

## Structure
- **agent/**: AI agent implementation, API, generation pipeline, and consistency validators
- **ingestion/**: Document processing pipeline for narrative-aware chunking and embedding
- **memory/**: Integrated memory system for maintaining narrative consistency
- **sql/**: Database schema definitions including proposal tables for approval workflow
- **templates/**: HTML templates for human-in-the-loop approval UI
- **tests/**: Comprehensive test suite for all components

## Language & Runtime
**Language**: Python
**Version**: 3.11 or higher
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- fastapi==0.104.1
- pydantic==2.5.0
- pydantic-ai==0.0.13
- asyncpg==0.29.0
- psycopg2-binary==2.9.9
- graphiti-core==0.3.0
- openai==1.45.0
- anthropic==0.34.0
- numpy==1.24.3
- scipy==1.11.4
- scikit-learn==1.3.2

**Development Dependencies**:
- pytest==7.4.3
- pytest-asyncio==0.21.1
- pytest-cov==4.1.0
- black==23.11.0
- ruff==0.1.6
- mypy==1.7.1

## Build & Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Set up required tables in Postgres
# Execute the SQL in sql/schema.sql

# Set up Neo4j
# Configure environment variables in .env file
```

## Main Entry Points
**API Server**:
```bash
python -m agent.api
# Server will be available at http://localhost:8058
```

**CLI Client**:
```bash
python cli.py
# Connect to a different URL
python cli.py --url http://localhost:8058
```

**Document Ingestion**:
```bash
# Basic ingestion with semantic chunking
python -m ingestion.ingest

# Clean existing data and re-ingest everything
python -m ingestion.ingest --clean
```

**Approval Workflow UI**:
```
# Access the human-in-the-loop approval interface
http://localhost:8058/approval/ui/review
```

## Generation Pipeline
The system includes a sophisticated generation pipeline for creative content:

**Components**:
- **EnhancedContextBuilder**: Builds intelligent, hierarchical context for generation
- **AdvancedGenerationPipeline**: Orchestrates content generation with validation
- **ConsistencyValidators**: Ensures narrative consistency and character authenticity
- **ApprovalWorkflow**: Human-in-the-loop review and approval process

**Generation Types**:
- Narrative Continuation
- Character Dialogue
- Scene Description
- Chapter Opening
- Conflict/Resolution Scenes
- Character Introduction
- World Building

## Database Configuration
**PostgreSQL**: Required with pgvector extension for vector embeddings
- Schema defined in `sql/schema.sql`
- Includes tables for proposals and validation results
- Connection configured via `DATABASE_URL` environment variable

**Neo4j**: Required for knowledge graph functionality
- Connection configured via `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` environment variables

## Testing
**Framework**: pytest
**Test Location**: tests/ directory
**Configuration**: tests/conftest.py
**Run Command**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test categories
pytest tests/agent/
pytest tests/ingestion/
```

## Environment Variables
Required environment variables in `.env` file:
- `DATABASE_URL`: PostgreSQL connection string
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection details
- `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_CHOICE`: LLM configuration
- `EMBEDDING_PROVIDER`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`: Embedding configuration
- `APP_ENV`, `LOG_LEVEL`, `APP_PORT`: Application configuration---
description: Repository Information Overview
alwaysApply: true
---

# Agentic RAG with Knowledge Graph Information

## Summary
An AI agent system that combines traditional RAG (vector search) with knowledge graph capabilities to analyze and provide insights about big tech companies and their AI initiatives. The system uses PostgreSQL with pgvector for semantic search and Neo4j with Graphiti for temporal knowledge graphs.

## Structure
- **agent/**: AI agent implementation, API, and tools
- **ingestion/**: Document processing pipeline for chunking and embedding
- **memory/**: Memory system for caching and consistency management
- **sql/**: Database schema definitions
- **templates/**: HTML templates for UI components
- **tests/**: Comprehensive test suite

## Language & Runtime
**Language**: Python
**Version**: 3.11 or higher
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- fastapi==0.104.1
- pydantic==2.5.0
- pydantic-ai==0.0.13
- asyncpg==0.29.0
- psycopg2-binary==2.9.9
- graphiti-core==0.3.0
- openai==1.45.0
- anthropic==0.34.0

**Development Dependencies**:
- pytest==7.4.3
- pytest-asyncio==0.21.1
- pytest-cov==4.1.0
- black==23.11.0
- ruff==0.1.6
- mypy==1.7.1

## Build & Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Set up required tables in Postgres
# Execute the SQL in sql/schema.sql

# Set up Neo4j
# Configure environment variables in .env file
```

## Main Entry Points
**API Server**:
```bash
python -m agent.api
# Server will be available at http://localhost:8058
```

**CLI Client**:
```bash
python cli.py
# Connect to a different URL
python cli.py --url http://localhost:8058
```

**Document Ingestion**:
```bash
# Basic ingestion with semantic chunking
python -m ingestion.ingest

# Clean existing data and re-ingest everything
python -m ingestion.ingest --clean
```

## Database Configuration
**PostgreSQL**: Required with pgvector extension for vector embeddings
- Schema defined in `sql/schema.sql`
- Connection configured via `DATABASE_URL` environment variable

**Neo4j**: Required for knowledge graph functionality
- Connection configured via `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` environment variables

## Testing
**Framework**: pytest
**Test Location**: tests/ directory
**Configuration**: tests/conftest.py
**Run Command**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test categories
pytest tests/agent/
pytest tests/ingestion/
```

## Environment Variables
Required environment variables in `.env` file:
- `DATABASE_URL`: PostgreSQL connection string
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection details
- `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_CHOICE`: LLM configuration
- `EMBEDDING_PROVIDER`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`: Embedding configuration
- `APP_ENV`, `LOG_LEVEL`, `APP_PORT`: Application configuration