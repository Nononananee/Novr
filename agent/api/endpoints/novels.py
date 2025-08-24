"""Novel-specific endpoints for the API."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from ...monitoring.advanced_system_monitor import monitor_operation, ComponentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/novels", tags=["novels"])


@router.post("")
@monitor_operation("create_novel", ComponentType.API_LAYER)
async def create_novel_endpoint(
    title: str,
    author: str = "",
    genre: str = "general",
    summary: str = ""
):
    """Create a new novel."""
    try:
        from ...core.db_utils import create_novel, create_novel_tables
        
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


@router.get("")
@monitor_operation("list_novels", ComponentType.API_LAYER)
async def list_novels_endpoint(limit: int = 20, offset: int = 0):
    """List all novels."""
    try:
        from ...core.db_utils import list_novels
        
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


@router.get("/{novel_id}")
@monitor_operation("get_novel", ComponentType.API_LAYER)
async def get_novel_endpoint(novel_id: str):
    """Get a specific novel."""
    try:
        from ...core.db_utils import get_novel
        
        novel = await get_novel(novel_id)
        if not novel:
            raise HTTPException(status_code=404, detail="Novel not found")
        
        return novel
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Novel retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{novel_id}/characters")
@monitor_operation("create_character", ComponentType.API_LAYER)
async def create_character_endpoint(
    novel_id: str,
    name: str,
    personality_traits: List[str] = None,
    background: str = "",
    role: str = "minor"
):
    """Create a character for a novel."""
    try:
        from ...core.db_utils import create_character
        
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


@router.get("/{novel_id}/characters")
@monitor_operation("list_characters", ComponentType.API_LAYER)
async def list_characters_endpoint(novel_id: str):
    """List characters for a novel."""
    try:
        from ...core.db_utils import list_characters
        
        characters = await list_characters(novel_id=novel_id)
        
        return {
            "characters": characters,
            "novel_id": novel_id,
            "total": len(characters)
        }
        
    except Exception as e:
        logger.error(f"Character listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{novel_id}/chapters")
@monitor_operation("create_chapter", ComponentType.API_LAYER)
async def create_chapter_endpoint(
    novel_id: str,
    chapter_number: int,
    title: str = None,
    summary: str = None
):
    """Create a chapter for a novel."""
    try:
        from ...core.db_utils import create_chapter
        
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


@router.get("/{novel_id}/chapters")
@monitor_operation("get_chapters", ComponentType.API_LAYER)
async def get_chapters_endpoint(
    novel_id: str,
    start_chapter: int = 1,
    end_chapter: int = None
):
    """Get chapters for a novel."""
    try:
        from ...core.db_utils import get_novel_chapters
        
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


@router.get("/{novel_id}/search")
@monitor_operation("search_novel_content", ComponentType.API_LAYER)
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
        from ...core.db_utils import search_novel_content
        
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