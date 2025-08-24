"""Novel-specific database functions."""

import json
import logging
from typing import Dict, Any, Optional, List

from .connection import db_pool

logger = logging.getLogger(__name__)


async def create_novel_tables():
    """Create tables specific for novel data."""
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
        
        logger.info("Novel-specific tables created successfully")


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