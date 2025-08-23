"""
Database utilities for PostgreSQL connection and operations.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from uuid import UUID
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """Create connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection


# Global database pool instance
db_pool = DatabasePool()


async def initialize_database():
    """Initialize database connection pool."""
    await db_pool.initialize()


async def close_database():
    """Close database connection pool."""
    await db_pool.close()


# Session Management Functions
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
) -> str:
    """
    Create a new session.
    
    Args:
        user_id: Optional user identifier
        metadata: Optional session metadata
        timeout_minutes: Session timeout in minutes
    
    Returns:
        Session ID
    """
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(metadata or {}),
            expires_at
        )
        
        return result["id"]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get session by ID.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Session data or None if not found/expired
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id,
                metadata,
                created_at,
                updated_at,
                expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        
        return None


async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Update session metadata.
    
    Args:
        session_id: Session UUID
        metadata: New metadata to merge
    
    Returns:
        True if updated, False if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = metadata || $2::jsonb
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
            json.dumps(metadata)
        )
        
        return result.split()[-1] != "0"


# Message Management Functions
async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a message to a session.
    
    Args:
        session_id: Session UUID
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Optional message metadata
    
    Returns:
        Message ID
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get messages for a session.
    
    Args:
        session_id: Session UUID
        limit: Maximum number of messages to return
    
    Returns:
        List of messages ordered by creation time
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                id::text,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Document Management Functions
async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document by ID.
    
    Args:
        document_id: Document UUID
    
    Returns:
        Document data or None if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    List documents with optional filtering.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        metadata_filter: Optional metadata filter
    
    Returns:
        List of documents
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        conditions = []
        
        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]


# Vector Search Functions
async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search.
    
    Args:
        embedding: Query embedding vector
        limit: Maximum number of results
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
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
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search (vector + keyword).
    
    Args:
        embedding: Query embedding vector
        query_text: Query text for keyword search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
    
    Returns:
        List of matching chunks ordered by combined score (best first)
    """
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
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
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


# Chunk Management Functions
async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document.
    
    Args:
        document_id: Document UUID
    
    Returns:
        List of chunks ordered by chunk index
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"])
            }
            for row in results
        ]


# Utility Functions
async def execute_query(query: str, *params) -> List[Dict[str, Any]]:
    """
    Execute a custom query.
    
    Args:
        query: SQL query
        *params: Query parameters
    
    Returns:
        Query results
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]


async def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# Proposal Management Functions
async def create_proposal(
    kind: str,
    payload: Dict[str, Any],
    source_doc: Optional[str] = None,
    suggested_by: Optional[str] = None,
    confidence: Optional[float] = None
) -> str:
    """Create a new proposal."""
    try:
        async with db_pool.acquire() as conn:
            proposal_id = await conn.fetchval(
                """
                INSERT INTO proposals (kind, payload, source_doc, suggested_by, confidence)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id::text
                """,
                kind, json.dumps(payload), source_doc, suggested_by, confidence
            )
            logger.info(f"Created proposal {proposal_id} of kind {kind}")
            return proposal_id
    except Exception as e:
        logger.error(f"Failed to create proposal: {e}")
        raise


async def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    """Get proposal by ID."""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT p.*, 
                       COALESCE(ps.validation_count, 0) as validation_count,
                       ps.avg_validation_score,
                       ps.min_validation_score,
                       ps.risk_level
                FROM proposals p
                LEFT JOIN proposal_summaries ps ON p.id = ps.id
                WHERE p.id = $1::uuid
                """,
                proposal_id
            )
            
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('payload'):
                    result['payload'] = json.loads(result['payload'])
                if result.get('neo4j_tx'):
                    result['neo4j_tx'] = json.loads(result['neo4j_tx'])
                if result.get('errors'):
                    result['errors'] = json.loads(result['errors'])
                return result
            return None
    except Exception as e:
        logger.error(f"Failed to get proposal {proposal_id}: {e}")
        raise


async def list_proposals(
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """List proposals with optional filtering."""
    try:
        conditions = []
        params = []
        param_count = 0
        
        if status:
            param_count += 1
            conditions.append(f"p.status = ${param_count}")
            params.append(status)
            
        if kind:
            param_count += 1
            conditions.append(f"p.kind = ${param_count}")
            params.append(kind)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        query = f"""
        SELECT p.id::text as id, p.kind, p.status, p.confidence, p.created_at, 
               p.processed_at, p.suggested_by, p.processed_by,
               COALESCE(ps.validation_count, 0) as validation_count,
               ps.avg_validation_score,
               ps.min_validation_score,
               ps.risk_level
        FROM proposals p
        LEFT JOIN proposal_summaries ps ON p.id = ps.id
        {where_clause}
        ORDER BY p.created_at DESC
        LIMIT {limit_param} OFFSET {offset_param}
        """
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list proposals: {e}")
        raise


async def update_proposal_status(
    proposal_id: str,
    status: str,
    processed_by: Optional[str] = None,
    neo4j_tx: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    rejection_reason: Optional[str] = None
) -> bool:
    """Update proposal status."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proposals 
                SET status = $1, 
                    processed_at = CURRENT_TIMESTAMP,
                    processed_by = $2,
                    neo4j_tx = $3,
                    errors = $4,
                    rejection_reason = $5
                WHERE id = $6::uuid
                """,
                status, processed_by, 
                json.dumps(neo4j_tx) if neo4j_tx else None,
                json.dumps(errors) if errors else None,
                rejection_reason, proposal_id
            )
            logger.info(f"Updated proposal {proposal_id} status to {status}")
            return True
    except Exception as e:
        logger.error(f"Failed to update proposal {proposal_id}: {e}")
        raise


async def store_validation_result(
    proposal_id: str,
    validator_name: str,
    score: float,
    violations: List[Dict[str, Any]],
    suggestions: List[str]
) -> str:
    """Store validation result."""
    try:
        async with db_pool.acquire() as conn:
            result_id = await conn.fetchval(
                """
                INSERT INTO validation_results 
                (proposal_id, validator_name, score, violations, suggestions)
                VALUES ($1::uuid, $2, $3, $4, $5)
                RETURNING id::text
                """,
                proposal_id, validator_name, score,
                json.dumps(violations), json.dumps(suggestions)
            )
            logger.info(f"Stored validation result {result_id} for proposal {proposal_id}")
            return result_id
    except Exception as e:
        logger.error(f"Failed to store validation result: {e}")
        raise


async def get_validation_results(proposal_id: str) -> List[Dict[str, Any]]:
    """Get validation results for a proposal."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT validator_name, score, violations, suggestions, created_at
                FROM validation_results
                WHERE proposal_id = $1::uuid
                ORDER BY created_at DESC
                """,
                proposal_id
            )
            results = []
            for row in rows:
                result = dict(row)
                # Parse JSON fields
                if result.get('violations'):
                    result['violations'] = json.loads(result['violations'])
                if result.get('suggestions'):
                    result['suggestions'] = json.loads(result['suggestions'])
                results.append(result)
            return results
    except Exception as e:
        logger.error(f"Failed to get validation results for {proposal_id}: {e}")
        raise

# Emotional Memory System Functions

async def create_emotion_analysis_run(
    method: str,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """Creates a new record for an emotion analysis run."""
    async with db_pool.acquire() as conn:
        run_id = await conn.fetchval(
            """
            INSERT INTO emotion_analysis_runs (method, model_name, model_version, params, status)
            VALUES ($1, $2, $3, $4, 'running')
            RETURNING id::text
            """,
            method, model_name, model_version, json.dumps(params or {})
        )
        return run_id

async def save_character_emotions(
    run_id: str,
    emotions_data: List[Dict[str, Any]]
):
    """
    Saves a batch of character emotion extractions within a single transaction.
    
    Args:
        run_id: The ID of the analysis run.
        emotions_data: A list of dictionaries, where each dict contains the data
                       for a row in the character_emotions table.
    """
    if not emotions_data:
        return

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            # The `copy_records_to_table` is highly efficient for bulk inserts.
            # Column names in the table must match the keys in the dictionaries.
            
            # Prepare data by adding run_id and serializing JSON fields
            for record in emotions_data:
                record['run_id'] = run_id
                record['emotion_vector'] = json.dumps(record.get('emotion_vector', {}))
                record['top_emotions'] = record.get('top_emotions', [])

            columns = [
                'run_id', 'scene_id', 'chunk_id', 'character_name', 'emotion_vector',
                'dominant_emotion', 'intensity', 'emotion_category', 'trigger_event',
                'related_character', 'source_type', 'confidence_score', 'method',
                'model_name', 'model_version', 'prompt_hash', 'span_start', 'span_end',
                'sentence_index', 'intra_chunk_order', 'intensity_calibrated', 'top_emotions'
            ]
            
            # We need to ensure every record has all keys for copy_records_to_table
            data_to_insert = []
            for record in emotions_data:
                data_to_insert.append(
                    tuple(record.get(col) for col in columns)
                )

            await conn.copy_records_to_table(
                'character_emotions',
                records=data_to_insert,
                columns=columns,
                timeout=120
            )

async def update_emotion_analysis_run_status(
    run_id: str,
    status: str,
    error: Optional[str] = None
):
    """Updates the status and completion time of an emotion analysis run."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE emotion_analysis_runs
            SET status = $1, error = $2, finished_at = now()
            WHERE id = $3::uuid
            """,
            status, error, run_id
        )


# Novel-specific Database Functions

async def create_novel_tables():
    """
    Create tables specific for novel data.
    """
    async with db_pool.acquire() as conn:
        # Table untuk novel
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS novels (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                author TEXT,
                genre TEXT DEFAULT 'general',
                summary TEXT,
                total_word_count INTEGER,
                chapter_count INTEGER,
                main_characters TEXT[] DEFAULT '{}',
                main_themes TEXT[] DEFAULT '{}',
                setting_overview TEXT,
                target_audience TEXT,
                completion_status TEXT DEFAULT 'in_progress',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Table untuk chapter
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                chapter_number INTEGER NOT NULL,
                title TEXT,
                summary TEXT,
                word_count INTEGER,
                scenes TEXT[] DEFAULT '{}',
                main_characters TEXT[] DEFAULT '{}',
                plot_threads TEXT[] DEFAULT '{}',
                emotional_arc JSONB DEFAULT '{}',
                significance TEXT DEFAULT 'moderate',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(novel_id, chapter_number)
            )
        """)
        
        # Table untuk karakter
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                personality_traits TEXT[] DEFAULT '{}',
                background TEXT DEFAULT '',
                motivations TEXT[] DEFAULT '{}',
                relationships JSONB DEFAULT '{}',
                emotional_state JSONB DEFAULT '{}',
                development_arc TEXT,
                role TEXT DEFAULT 'minor',
                first_appearance TEXT,
                dialogue_patterns TEXT[] DEFAULT '{}',
                physical_description TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(novel_id, name)
            )
        """)
        
        # Table untuk locations
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                location_type TEXT DEFAULT 'general',
                atmosphere TEXT,
                significance TEXT DEFAULT 'background',
                first_appearance TEXT,
                associated_characters TEXT[] DEFAULT '{}',
                associated_events TEXT[] DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(novel_id, name)
            )
        """)
        
        # Table untuk scenes
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                chapter_id UUID REFERENCES chapters(id) ON DELETE CASCADE,
                title TEXT,
                content TEXT NOT NULL,
                setting TEXT,
                characters_present TEXT[] DEFAULT '{}',
                plot_points TEXT[] DEFAULT '{}',
                emotional_tone TEXT,
                conflict_level FLOAT DEFAULT 0.5 CHECK (conflict_level >= 0.0 AND conflict_level <= 1.0),
                purpose TEXT DEFAULT '',
                chunk_type TEXT DEFAULT 'narration',
                significance TEXT DEFAULT 'moderate',
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Table untuk plot threads
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS plot_threads (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                significance TEXT DEFAULT 'moderate',
                involved_characters TEXT[] DEFAULT '{}',
                key_events TEXT[] DEFAULT '{}',
                resolution TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(novel_id, name)
            )
        """)
        
        # Table untuk emotional arcs
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS emotional_arcs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                novel_id UUID REFERENCES novels(id) ON DELETE CASCADE,
                entity_id UUID NOT NULL,
                entity_type TEXT NOT NULL CHECK (entity_type IN ('character', 'scene', 'chapter')),
                emotional_progression JSONB DEFAULT '[]',
                dominant_emotions TEXT[] DEFAULT '{}',
                emotional_intensity FLOAT DEFAULT 0.5 CHECK (emotional_intensity >= 0.0 AND emotional_intensity <= 1.0),
                turning_points JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(entity_id, entity_type)
            )
        """)
        
        logger.info("Novel-specific tables created successfully")


# Novel CRUD Functions

async def create_novel(
    title: str,
    author: str = "",
    genre: str = "general",
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Create a new novel."""
    async with db_pool.acquire() as conn:
        novel_id = await conn.fetchval(
            """
            INSERT INTO novels (title, author, genre, summary, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id::text
            """,
            title, author, genre, summary, json.dumps(metadata or {})
        )
        logger.info(f"Created novel {novel_id}: {title}")
        return novel_id


async def get_novel(novel_id: str) -> Optional[Dict[str, Any]]:
    """Get novel by ID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                author,
                genre,
                summary,
                total_word_count,
                chapter_count,
                main_characters,
                main_themes,
                setting_overview,
                target_audience,
                completion_status,
                metadata,
                created_at,
                updated_at
            FROM novels
            WHERE id = $1::uuid
            """,
            novel_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "author": result["author"],
                "genre": result["genre"],
                "summary": result["summary"],
                "total_word_count": result["total_word_count"],
                "chapter_count": result["chapter_count"],
                "main_characters": result["main_characters"],
                "main_themes": result["main_themes"],
                "setting_overview": result["setting_overview"],
                "target_audience": result["target_audience"],
                "completion_status": result["completion_status"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        return None


async def list_novels(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List all novels."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                id::text,
                title,
                author,
                genre,
                completion_status,
                chapter_count,
                total_word_count,
                created_at,
                updated_at
            FROM novels
            ORDER BY updated_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset
        )
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "author": row["author"],
                "genre": row["genre"],
                "completion_status": row["completion_status"],
                "chapter_count": row["chapter_count"],
                "total_word_count": row["total_word_count"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            }
            for row in results
        ]


# Character Functions

async def create_character(
    novel_id: str,
    name: str,
    personality_traits: List[str] = None,
    background: str = "",
    role: str = "minor",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Create a new character."""
    async with db_pool.acquire() as conn:
        character_id = await conn.fetchval(
            """
            INSERT INTO characters (novel_id, name, personality_traits, background, role)
            VALUES ($1::uuid, $2, $3, $4, $5)
            RETURNING id::text
            """,
            novel_id, name, personality_traits or [], background, role
        )
        logger.info(f"Created character {character_id}: {name}")
        return character_id


async def get_character(character_id: str) -> Optional[Dict[str, Any]]:
    """Get character by ID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                novel_id::text,
                name,
                personality_traits,
                background,
                motivations,
                relationships,
                emotional_state,
                development_arc,
                role,
                first_appearance,
                dialogue_patterns,
                physical_description,
                created_at,
                updated_at
            FROM characters
            WHERE id = $1::uuid
            """,
            character_id
        )
        
        if result:
            return {
                "id": result["id"],
                "novel_id": result["novel_id"],
                "name": result["name"],
                "personality_traits": result["personality_traits"],
                "background": result["background"],
                "motivations": result["motivations"],
                "relationships": json.loads(result["relationships"]),
                "emotional_state": json.loads(result["emotional_state"]) if result["emotional_state"] else None,
                "development_arc": result["development_arc"],
                "role": result["role"],
                "first_appearance": result["first_appearance"],
                "dialogue_patterns": result["dialogue_patterns"],
                "physical_description": result["physical_description"],
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        return None


async def list_characters(novel_id: str) -> List[Dict[str, Any]]:
    """List all characters for a novel."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                id::text,
                name,
                role,
                personality_traits,
                background,
                created_at
            FROM characters
            WHERE novel_id = $1::uuid
            ORDER BY role, name
            """,
            novel_id
        )
        
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "role": row["role"],
                "personality_traits": row["personality_traits"],
                "background": row["background"],
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Chapter Functions

async def create_chapter(
    novel_id: str,
    chapter_number: int,
    title: Optional[str] = None,
    summary: Optional[str] = None
) -> str:
    """Create a new chapter."""
    async with db_pool.acquire() as conn:
        chapter_id = await conn.fetchval(
            """
            INSERT INTO chapters (novel_id, chapter_number, title, summary)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            novel_id, chapter_number, title, summary
        )
        logger.info(f"Created chapter {chapter_id}: Chapter {chapter_number}")
        return chapter_id


async def get_novel_chapters(
    novel_id: str, 
    start_chapter: int = 1, 
    end_chapter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get chapters of a novel with optional range."""
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                id::text,
                chapter_number,
                title,
                summary,
                word_count,
                main_characters,
                plot_threads,
                emotional_arc,
                significance,
                created_at,
                updated_at
            FROM chapters 
            WHERE novel_id = $1::uuid 
            AND chapter_number BETWEEN $2 AND $3
            ORDER BY chapter_number
        """
        
        end_chapter = end_chapter or 9999
        results = await conn.fetch(query, novel_id, start_chapter, end_chapter)
        
        return [
            {
                "id": row["id"],
                "chapter_number": row["chapter_number"],
                "title": row["title"],
                "summary": row["summary"],
                "word_count": row["word_count"],
                "main_characters": row["main_characters"],
                "plot_threads": row["plot_threads"],
                "emotional_arc": json.loads(row["emotional_arc"]) if row["emotional_arc"] else None,
                "significance": row["significance"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            }
            for row in results
        ]


# Scene Functions

async def create_scene(
    novel_id: str,
    chapter_id: str,
    content: str,
    title: Optional[str] = None,
    characters_present: List[str] = None,
    emotional_tone: Optional[str] = None
) -> str:
    """Create a new scene."""
    async with db_pool.acquire() as conn:
        scene_id = await conn.fetchval(
            """
            INSERT INTO scenes (novel_id, chapter_id, title, content, characters_present, emotional_tone)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
            RETURNING id::text
            """,
            novel_id, chapter_id, title, content, characters_present or [], emotional_tone
        )
        logger.info(f"Created scene {scene_id}")
        return scene_id


async def get_chapter_scenes(chapter_id: str) -> List[Dict[str, Any]]:
    """Get all scenes for a chapter."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                id::text,
                title,
                content,
                setting,
                characters_present,
                plot_points,
                emotional_tone,
                conflict_level,
                purpose,
                chunk_type,
                significance,
                word_count,
                created_at
            FROM scenes
            WHERE chapter_id = $1::uuid
            ORDER BY created_at
            """,
            chapter_id
        )
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "setting": row["setting"],
                "characters_present": row["characters_present"],
                "plot_points": row["plot_points"],
                "emotional_tone": row["emotional_tone"],
                "conflict_level": row["conflict_level"],
                "purpose": row["purpose"],
                "chunk_type": row["chunk_type"],
                "significance": row["significance"],
                "word_count": row["word_count"],
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Character Development Functions

async def get_character_arc(character_id: str) -> List[Dict[str, Any]]:
    """Get the development arc of a character across chapters."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                c.name,
                ch.chapter_number,
                ch.title as chapter_title,
                s.title as scene_title,
                s.content,
                s.emotional_tone,
                s.created_at
            FROM characters c
            JOIN scenes s ON c.name = ANY(s.characters_present)
            JOIN chapters ch ON s.chapter_id = ch.id
            WHERE c.id = $1::uuid
            ORDER BY ch.chapter_number, s.created_at
            """,
            character_id
        )
        
        return [
            {
                "character_name": row["name"],
                "chapter_number": row["chapter_number"],
                "chapter_title": row["chapter_title"],
                "scene_title": row["scene_title"],
                "content": row["content"],
                "emotional_tone": row["emotional_tone"],
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Search Functions with Novel Context

async def search_novel_content(
    novel_id: str,
    query: str,
    content_type: Optional[str] = None,
    character_filter: Optional[str] = None,
    emotional_tone_filter: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search novel content with filters."""
    async with db_pool.acquire() as conn:
        conditions = ["s.novel_id = $1::uuid"]
        params = [novel_id]
        param_count = 1
        
        # Add text search
        if query:
            param_count += 1
            conditions.append(f"s.content ILIKE $%{param_count}")
            params.append(f"%{query}%")
        
        # Add content type filter
        if content_type:
            param_count += 1
            conditions.append(f"s.chunk_type = ${param_count}")
            params.append(content_type)
        
        # Add character filter
        if character_filter:
            param_count += 1
            conditions.append(f"${param_count} = ANY(s.characters_present)")
            params.append(character_filter)
        
        # Add emotional tone filter
        if emotional_tone_filter:
            param_count += 1
            conditions.append(f"s.emotional_tone = ${param_count}")
            params.append(emotional_tone_filter)
        
        where_clause = " AND ".join(conditions)
        param_count += 1
        
        query_sql = f"""
            SELECT 
                s.id::text as scene_id,
                s.title,
                s.content,
                s.characters_present,
                s.emotional_tone,
                s.chunk_type,
                ch.chapter_number,
                ch.title as chapter_title
            FROM scenes s
            JOIN chapters ch ON s.chapter_id = ch.id
            WHERE {where_clause}
            ORDER BY ch.chapter_number, s.created_at
            LIMIT ${param_count}
        """
        
        params.append(limit)
        results = await conn.fetch(query_sql, *params)
        
        return [
            {
                "scene_id": row["scene_id"],
                "title": row["title"],
                "content": row["content"],
                "characters_present": row["characters_present"],
                "emotional_tone": row["emotional_tone"],
                "chunk_type": row["chunk_type"],
                "chapter_number": row["chapter_number"],
                "chapter_title": row["chapter_title"]
            }
            for row in results
        ]