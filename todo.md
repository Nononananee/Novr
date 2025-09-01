# Novel AI System - Phase 1 Implementation Plan

## Project Overview
Multi-agent novel generation system using CrewAI with FastAPI backend, vector database context retrieval, and quality assurance agents.

## Architecture Components

### ✅ Infrastructure (Docker Compose)
- [x] Qdrant vector database (port 6333)
- [x] Redis job queue (port 6379) 
- [x] MongoDB document storage (port 27017)
- [x] FastAPI backend service (port 8000)
- [x] Background worker service
- [x] Health checks and service dependencies

### ✅ Backend API (FastAPI)
- [x] **main.py** - FastAPI application entry point with CORS and startup events
- [x] **config.py** - Centralized configuration with environment variables
- [x] **routes/generate.py** - POST /generate/chapter endpoint (returns job_id, 202 status)
- [x] **routes/job.py** - GET /job/{job_id} endpoint for status tracking
- [x] **schemas/requests.py** - Pydantic models for API requests
- [x] **schemas/responses.py** - Pydantic models for API responses
- [x] **db/mongodb_client.py** - MongoDB client with job and chapter collections
- [x] **services/orchestrator.py** - Job queue management with Redis + RQ

### ✅ Job Processing (Workers)
- [x] **worker.py** - RQ worker process that consumes Redis queue
- [x] **tasks/generate_task.py** - Main job execution with full generation flow:
  - Context retrieval from Qdrant
  - Content generation with Generator Agent
  - Technical QA review and revision loop (max 2 iterations)
  - Chapter version persistence to MongoDB
  - Job status updates throughout process

### ✅ Embedding System
- [x] **chunker.py** - Text chunking with tiktoken (max_tokens=800, overlap=64)
- [x] **embedder.py** - E5-large-v2 embeddings with sentence-transformers
- [x] **qdrant_client.py** - Vector database operations (collection: novel_chunks, dim=1024)

### ✅ Multi-Agent System
- [x] **generator_agent.py** - OpenAI GPT-4o-mini wrapper for novel writing
  - System prompts for creative writing
  - Context integration and consistency checking
  - Revision capabilities with feedback
- [x] **technical_qa.py** - Grammar and formatting QA agent
  - JSON schema output with scores, issues, and patches
  - Automatic patch application
  - Fallback analysis for error handling

### ✅ Utilities and Scripts
- [x] **scripts/ingest.py** - Worldbook and reference material ingestion
  - File and directory processing
  - Chunking and embedding pipeline
  - Qdrant collection management
  - Verification with test searches

### ✅ Testing Suite
- [x] **test_chunker.py** - Unit tests for text chunking functionality
- [x] **test_embedder.py** - Unit tests for embedding generation
- [x] **test_orchestrator.py** - Unit tests for job orchestration with mocks

### ✅ Sample Data
- [x] **samples/worldbook.md** - Complete fantasy world guide (Aethermoor)
- [x] **samples/character_notes.md** - Character profiles and relationships
- [x] **.env.example** - Environment variable template
- [x] **README.md** - Comprehensive setup and usage documentation

## Data Flow Architecture

```
1. POST /generate/chapter
   ↓
2. Create job document in MongoDB
   ↓  
3. Enqueue job in Redis queue
   ↓
4. Worker picks up job
   ↓
5. Retrieve context from Qdrant
   ↓
6. Generate content with Generator Agent
   ↓
7. Review with Technical QA Agent
   ↓
8. Apply revisions if needed (max 2 rounds)
   ↓
9. Save chapter version to MongoDB
   ↓
10. Update job status to success
```

## Database Schemas

### MongoDB Collections

**jobs**
```javascript
{
  job_id: "uuid",
  user_id: "string", 
  project_id: "string",
  chapter_id: "string",
  payload: {}, // original request
  state: "queued|running|success|failed",
  progress: 0.0-1.0,
  result: {}, // final output
  error: "string",
  created_at: Date,
  updated_at: Date
}
```

**chapter_versions**
```javascript
{
  project_id: "string",
  chapter_id: "string", 
  version_number: 1,
  content: "markdown text",
  qa_score: 85,
  revision_count: 1,
  word_count: 1200,
  character_count: 6500,
  created_at: Date
}
```

### Qdrant Collection

**novel_chunks**
- Vector dimension: 1024 (e5-large-v2)
- Distance metric: Cosine
- Payload: {text, project_id, source, chunk_index, token_count}

## API Endpoints

### POST /generate/chapter
**Request:**
```json
{
  "user_id": "user123",
  "project_id": "fantasy_novel_1", 
  "chapter_id": "chapter_1",
  "prompt": "Write opening chapter where protagonist discovers magical abilities",
  "settings": {
    "length_words": 1200,
    "max_revision_rounds": 2,
    "temperature": 0.7,
    "tone": "mysterious"
  }
}
```

**Response (202):**
```json
{
  "status": "queued",
  "code": 202,
  "data": {
    "job_id": "job_123456789"
  }
}
```

### GET /job/{job_id}
**Response:**
```json
{
  "job_id": "job_123456789",
  "status": "success",
  "progress": 1.0,
  "result": {
    "version_id": "version_abc123",
    "chapter_content": "# Chapter 1\n\nThe forest was alive...",
    "qa_score": 85,
    "revision_count": 1,
    "word_count": 1247
  }
}
```

## Development Workflow

### 1. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with API keys:
# OPENAI_API_KEY=your_key_here
# MONGODB_URL=mongodb://localhost:27017
# REDIS_URL=redis://localhost:6379
# QDRANT_URL=http://localhost:6333
```

### 2. Infrastructure Startup
```bash
cd infra
docker-compose up -d

# Verify services are healthy
docker-compose ps
```

### 3. Dependencies Installation
```bash
pip install -r requirements.txt
```

### 4. Sample Data Ingestion
```bash
python scripts/ingest.py --file samples/worldbook.md --project demo --verify
python scripts/ingest.py --file samples/character_notes.md --project demo --verify
```

### 5. Service Startup
```bash
# Terminal 1: Backend API
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Background Worker  
python workers/worker.py
```

### 6. Testing
```bash
# Run unit tests
pytest tests/

# Test API endpoint
curl -X POST http://localhost:8000/generate/chapter \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "project_id": "demo", 
    "prompt": "Write the opening chapter where Lyra discovers her Storm Crystal",
    "settings": {"length_words": 1000}
  }'

# Check job status
curl http://localhost:8000/job/{job_id}
```

## Phase 1 Acceptance Criteria

### ✅ Infrastructure Requirements
- [x] Docker compose with Qdrant, Redis, MongoDB services
- [x] All containers start and pass health checks
- [x] Services accessible on specified ports

### ✅ Ingestion Pipeline
- [x] Text chunking with tiktoken (800 tokens, 64 overlap)
- [x] E5-large-v2 embeddings generation
- [x] Qdrant collection creation and data ingestion
- [x] Search functionality returns relevant results

### ✅ API Functionality
- [x] POST /generate/chapter returns job_id with 202 status
- [x] GET /job/{job_id} returns current job status
- [x] Proper error handling and validation
- [x] OpenAPI documentation available at /docs

### ✅ Job Processing
- [x] Worker consumes Redis queue jobs
- [x] Context retrieval from Qdrant vector database
- [x] Content generation with OpenAI GPT-4o-mini
- [x] Technical QA review with JSON output
- [x] Revision loop (max 2 iterations based on QA score)
- [x] Chapter version persistence to MongoDB
- [x] Job status updates throughout process

### ✅ Quality Assurance
- [x] Technical QA agent returns structured JSON
- [x] QA scoring system (0-100)
- [x] Issue identification and patch suggestions
- [x] Automatic revision when score below threshold
- [x] Fallback handling for QA failures

### ✅ Testing and Documentation
- [x] Unit tests for core components (chunker, embedder, orchestrator)
- [x] Integration test scenarios with mocks
- [x] Comprehensive README with setup instructions
- [x] Sample data for testing (worldbook, characters)
- [x] Environment configuration template

## Monitoring and Logging

### Implemented Logging
- [x] Structured logging with timestamps and levels
- [x] Job lifecycle tracking (queued → running → success/failed)
- [x] Performance metrics (token usage, processing time)
- [x] Error tracking with stack traces
- [x] QA score logging and revision tracking

### Available Metrics
- Job processing duration
- QA scores distribution  
- Revision frequency
- Token usage per generation
- Queue depth and worker status

## Known Limitations

1. **Single QA Agent**: Only technical QA implemented (grammar/formatting)
2. **Basic Context Retrieval**: Simple vector similarity search
3. **Limited Error Recovery**: Basic retry logic only
4. **No User Authentication**: Open API endpoints
5. **Memory Usage**: Large models require significant RAM

## Next Phase Roadmap

### Phase 2 Enhancements
- [ ] Structural QA (plot consistency)
- [ ] Character QA (character consistency) 
- [ ] Style QA (writing style analysis)
- [ ] Neo4j character relationship graphs
- [ ] Advanced context retrieval strategies
- [ ] User authentication and authorization
- [ ] Web UI for job management
- [ ] Advanced monitoring dashboard

## Deployment Notes

### Production Considerations
- Set strong passwords for MongoDB and Redis
- Configure proper CORS origins
- Use environment-specific API keys
- Set up log aggregation (ELK stack)
- Configure resource limits in docker-compose
- Implement backup strategies for MongoDB
- Set up SSL/TLS for external access

### Scaling Options
- Horizontal worker scaling (multiple worker containers)
- Qdrant cluster for large datasets
- MongoDB replica sets for high availability
- Redis Cluster for queue reliability
- Load balancer for FastAPI instances

---

**Status**: ✅ Phase 1 Complete - Ready for End-to-End Testing

All Phase 1 requirements have been implemented and are ready for testing. The system provides a complete novel generation pipeline with multi-agent orchestration, quality assurance, and persistent storage.