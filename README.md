# Novel AI System - Phase 1

A multi-agent novel generation system using CrewAI, with FastAPI backend, vector database context retrieval, and quality assurance agents.

## Architecture

- **Backend**: FastAPI with async job processing
- **Agents**: CrewAI-based Generator and Technical QA agents
- **Vector DB**: Qdrant for context retrieval
- **Storage**: MongoDB for jobs and chapter versions
- **Queue**: Redis + RQ for background processing
- **Embeddings**: E5-large-v2 with tiktoken chunking

## Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start Infrastructure**
   ```bash
   cd infra
   docker-compose up -d
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ingest Sample Data**
   ```bash
   python scripts/ingest.py --file samples/worldbook.md --project demo
   ```

5. **Start Services**
   ```bash
   # Terminal 1: Backend API
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   
   # Terminal 2: Worker
   python workers/worker.py
   ```

6. **Test Generation**
   ```bash
   curl -X POST http://localhost:8000/generate/chapter \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user1",
       "project_id": "demo",
       "chapter_id": "ch1",
       "prompt": "Write the opening chapter where the protagonist discovers their magical abilities",
       "settings": {"length_words": 1200, "max_revision_rounds": 2}
     }'
   ```

## API Endpoints

- `POST /generate/chapter` - Queue chapter generation job
- `GET /job/{job_id}` - Check job status
- `GET /health` - Health check

## Project Structure

```
novel-ai-system/
├── backend/           # FastAPI application
├── workers/           # Background job processors
├── agents/            # CrewAI agents
├── embeddings/        # Vector processing
├── scripts/           # Utility scripts
├── infra/             # Docker infrastructure
└── tests/             # Test suite
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black . && isort .
```

## Phase 1 Features

✅ Docker infrastructure (Qdrant, Redis, MongoDB)
✅ Text chunking and embedding ingestion
✅ FastAPI backend with job queue
✅ Generator agent with context retrieval
✅ Technical QA agent with JSON output
✅ Revision loop (max 2 iterations)
✅ MongoDB persistence for jobs and chapters
✅ Basic logging and metrics

## Next Phase

- Structural QA (plot consistency)
- Character QA (character consistency)
- Style QA (writing style analysis)
- Neo4j character relationship graphs
- Advanced context retrieval strategies