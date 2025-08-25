"""
Enhanced Scene-Level Chunker used in tests.
Implements analysis helpers and chunking that produces DocumentChunk objects with rich metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

from .chunker import DocumentChunk, ChunkingConfig


class ContentType:
    DIALOGUE = "dialogue"
    ACTION = "action"
    DESCRIPTION = "description"
    NARRATIVE = "narrative"


class SceneType:
    DIALOGUE_SCENE = "dialogue_scene"
    ACTION_SEQUENCE = "action_sequence"
    EXPLORATION = "exploration"
    TRANSITION = "transition"
    CLIMAX = "climax"
    CHARACTER_INTRODUCTION = "character_introduction"
    CONFLICT = "conflict"


@dataclass
class EnhancedChunkingConfig(ChunkingConfig):
    dialogue_chunk_size: int = 800
    narrative_chunk_size: int = 1200
    action_chunk_size: int = 600
    description_chunk_size: int = 1000
    min_scene_size: int = 200
    max_scene_size: int = 3000
    preserve_dialogue_integrity: bool = True
    preserve_emotional_beats: bool = True
    preserve_action_sequences: bool = True
    scene_break_threshold: float = 0.6


@dataclass
class SceneMetadata:
    scene_type: str
    content_types: List[str]
    characters: List[str]
    locations: List[str]
    emotional_tone: str
    tension_level: float
    dialogue_ratio: float
    action_ratio: float
    description_ratio: float
    importance_score: float
    narrative_function: str
    temporal_markers: List[str]
    scene_length: int = 0
    estimated_reading_time: float = 0.0


class EnhancedSceneChunker:
    def __init__(self, config: Optional[EnhancedChunkingConfig] = None):
        self.config = config or EnhancedChunkingConfig()
        # placeholder for tokenizer/encoding if needed by future tests
        self.encoding = object()

    # --- Analysis helpers ---
    def _classify_content_types(self, text: str) -> List[str]:
        types: List[str] = []
        if re.search(r"\".+?\"", text):
            types.append(ContentType.DIALOGUE)
        if re.search(r"\b(explosion|blast|gun|ran|sprint|attack|fight|shouted)\b", text, re.I):
            types.append(ContentType.ACTION)
        if re.search(r"\b(ancient|old|dust|smell|leather|windows|architecture)\b", text, re.I):
            types.append(ContentType.DESCRIPTION)
        if not types:
            types.append(ContentType.NARRATIVE)
        return list(set(types))

    def _calculate_dialogue_ratio(self, text: str) -> float:
        quotes = re.findall(r"\".+?\"", text)
        return min(1.0, sum(len(q) for q in quotes) / max(1, len(text)))

    def _calculate_action_ratio(self, text: str) -> float:
        actions = len(re.findall(r"\b(explosion|blast|gun|ran|sprint|attack|fight|shouted)\b", text, re.I))
        return min(1.0, actions / max(1, len(text.split()) / 5))

    def _extract_characters(self, text: str) -> List[str]:
        # naive: capitalized words not at sentence start; also common names
        names = set(re.findall(r"\b([A-Z][a-z]+)\b", text))
        return sorted(list(names))

    def _analyze_emotional_tone(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["joy", "delighted", "happy", "happiness"]):
            return "joy"
        if any(w in lower for w in ["sorrow", "melancholy", "grief", "sad"]):
            return "sadness"
        if any(w in lower for w in ["fear", "anxious", "terror", "worry"]):
            return "fear"
        if any(w in lower for w in ["anger", "rage", "furious"]):
            return "anger"
        return "neutral"

    def _calculate_tension_level(self, text: str) -> float:
        level = 0.0
        level += self._calculate_action_ratio(text) * 0.6
        level += min(1.0, text.count("!") / 5) * 0.4
        return min(1.0, level)

    def _classify_scene_type(self, text: str, content_types: List[str], dialogue_ratio: float, action_ratio: float) -> str:
        if ContentType.ACTION in content_types or action_ratio > 0.3:
            return SceneType.ACTION_SEQUENCE
        if ContentType.DIALOGUE in content_types and dialogue_ratio > 0.3:
            return SceneType.DIALOGUE_SCENE
        return SceneType.EXPLORATION

    def _calculate_importance_score(self, scene_type: str, content_types: List[str], tension_level: float, key_character_count: int) -> float:
        base = 0.3
        if scene_type == SceneType.CLIMAX:
            base = 0.9
        elif scene_type == SceneType.CONFLICT:
            base = 0.7
        elif scene_type == SceneType.DIALOGUE_SCENE:
            base = 0.5
        base += min(0.4, tension_level * 0.4)
        base += min(0.2, key_character_count * 0.05)
        return max(0.0, min(1.0, base))

    def _determine_narrative_function(self, scene_type: str, content_types: List[str]) -> str:
        if scene_type == SceneType.CHARACTER_INTRODUCTION:
            return "character_development"
        if scene_type == SceneType.CONFLICT:
            return "plot_advancement"
        if SceneType.DIALOGUE_SCENE == scene_type or ContentType.DIALOGUE in content_types:
            return "character_interaction"
        if scene_type == SceneType.TRANSITION:
            return "transition"
        return "world_building"

    def _detect_enhanced_scenes(self, text: str) -> List[str]:
        # split by explicit breaks or multiple newlines
        parts = re.split(r"\n\s*(?:\*\*\*|---)\s*\n|\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    def _detect_implicit_scene_breaks(self, text: str) -> List[str]:
        # naive: split on words like 'Later that day' or 'Meanwhile'
        parts = re.split(r"\b(Later that|Meanwhile|Back at|Hours later)\b", text)
        # Rebuild keeping markers as part of following text
        merged: List[str] = []
        buf = ""
        for i, seg in enumerate(parts):
            if i % 2 == 1:  # marker word
                if buf:
                    merged.append(buf.strip())
                buf = seg
            else:
                buf = (buf + " " + seg) if buf else seg
        if buf.strip():
            merged.append(buf.strip())
        return [m for m in merged if m]

    def _detect_pov_character(self, text: str, candidates: List[str]) -> str:
        lower = text.lower()
        if re.search(r"\bI\b", text):
            return "first_person_narrator"
        for name in candidates:
            if name.lower() in lower:
                return name
        return candidates[0] if candidates else "unknown"

    def _has_temporal_break(self, text: str) -> bool:
        return bool(re.search(r"\b(later|meanwhile|after|before|the next day)\b", text, re.I))

    def _has_internal_monologue(self, text: str) -> bool:
        return bool(re.search(r"\b(thought|wondered|considered|realized)\b", text, re.I))

    def _has_exposition(self, text: str) -> bool:
        return bool(re.search(r"\b(founded|history|origin|legend|was known for)\b", text, re.I))

    # --- Main analysis ---
    def _analyze_scene(self, text: str) -> SceneMetadata:
        content_types = self._classify_content_types(text)
        dialogue_ratio = self._calculate_dialogue_ratio(text)
        action_ratio = self._calculate_action_ratio(text)
        characters = self._extract_characters(text)
        emotional_tone = self._analyze_emotional_tone(text)
        tension = self._calculate_tension_level(text)
        scene_type = self._classify_scene_type(text, content_types, dialogue_ratio, action_ratio)
        importance = self._calculate_importance_score(scene_type, content_types, tension, len(characters))
        narrative_function = self._determine_narrative_function(scene_type, content_types)
        temporal_markers: List[str] = []
        if self._has_temporal_break(text):
            temporal_markers.append("temporal_break")
        if self._has_internal_monologue(text):
            temporal_markers.append("internal_monologue")
        if self._has_exposition(text):
            temporal_markers.append("exposition")
        meta = SceneMetadata(
            scene_type=scene_type,
            content_types=content_types,
            characters=characters,
            locations=[],
            emotional_tone=emotional_tone,
            tension_level=tension,
            dialogue_ratio=dialogue_ratio,
            action_ratio=action_ratio,
            description_ratio=max(0.0, 1.0 - dialogue_ratio - action_ratio),
            importance_score=importance,
            narrative_function=narrative_function,
            temporal_markers=temporal_markers,
        )
        meta.scene_length = len(text)
        meta.estimated_reading_time = max(0.1, len(text.split()) / 200.0)  # minutes
        return meta

    def _get_optimal_chunk_size(self, metadata: SceneMetadata) -> int:
        if metadata.scene_type == SceneType.DIALOGUE_SCENE:
            return self.config.dialogue_chunk_size
        if metadata.scene_type == SceneType.ACTION_SEQUENCE:
            return self.config.action_chunk_size
        if ContentType.DESCRIPTION in metadata.content_types:
            return self.config.description_chunk_size
        return self.config.narrative_chunk_size

    # --- Splitting methods ---
    def _split_by_dialogue_turns(
        self, text: str, base_index: int, base_start: int, scene_meta: SceneMetadata, base_metadata: Dict[str, Any], max_size: int
    ) -> List[DocumentChunk]:
        parts = re.split(r"(\n\s*\n|\".+?\")", text)
        content_parts = [p for p in parts if p and not p.isspace()]
        chunks: List[DocumentChunk] = []
        cursor = base_start
        idx = base_index
        buf = ""
        for seg in content_parts:
            next_buf = (buf + (" " if buf else "") + seg).strip()
            if len(next_buf) > max_size and buf:
                start = text.find(buf, cursor)
                end = start + len(buf)
                meta = dict(base_metadata)
                meta.update({
                    "chunk_method": "enhanced_scene_level",
                    "split_method": "dialogue_turn",
                    "scene_type": scene_meta.scene_type,
                    "content_types": scene_meta.content_types,
                    "importance_score": scene_meta.importance_score,
                })
                chunks.append(DocumentChunk(buf, idx, start, end, meta))
                idx += 1
                cursor = end
                buf = seg
            else:
                buf = next_buf
        if buf:
            start = text.find(buf, cursor)
            end = start + len(buf)
            meta = dict(base_metadata)
            meta.update({
                "chunk_method": "enhanced_scene_level",
                "split_method": "dialogue_turn",
                "scene_type": scene_meta.scene_type,
                "content_types": scene_meta.content_types,
                "importance_score": scene_meta.importance_score,
            })
            chunks.append(DocumentChunk(buf, idx, start, end, meta))
        return chunks

    def _split_by_action_beats(
        self, text: str, base_index: int, base_start: int, scene_meta: SceneMetadata, base_metadata: Dict[str, Any], max_size: int
    ) -> List[DocumentChunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[DocumentChunk] = []
        buf = ""
        idx = base_index
        cursor = base_start
        for s in sentences:
            next_buf = (buf + (" " if buf else "") + s).strip()
            if len(next_buf) > max_size and buf:
                start = text.find(buf, cursor)
                end = start + len(buf)
                meta = dict(base_metadata)
                meta.update({
                    "chunk_method": "enhanced_scene_level",
                    "split_method": "action_beat",
                    "scene_type": scene_meta.scene_type,
                    "content_types": scene_meta.content_types,
                    "importance_score": scene_meta.importance_score,
                })
                chunks.append(DocumentChunk(buf, idx, start, end, meta))
                idx += 1
                cursor = end
                buf = s
            else:
                buf = next_buf
        if buf:
            start = text.find(buf, cursor)
            end = start + len(buf)
            meta = dict(base_metadata)
            meta.update({
                "chunk_method": "enhanced_scene_level",
                "split_method": "action_beat",
                "scene_type": scene_meta.scene_type,
                "content_types": scene_meta.content_types,
                "importance_score": scene_meta.importance_score,
            })
            chunks.append(DocumentChunk(buf, idx, start, end, meta))
        return chunks

    def _merge_chunks(self, a: DocumentChunk, b: DocumentChunk) -> DocumentChunk:
        content = (a.content + "\n\n" + b.content).strip()
        meta = dict(a.metadata)
        meta.update(b.metadata)
        meta["chunk_type"] = "merged_scene"
        # merge characters/locations if present
        for key in ("characters", "locations"):
            va = set(meta.get(key, []))
            vb = set(b.metadata.get(key, []))
            meta[key] = sorted(list(va | vb))
        return DocumentChunk(
            content=content,
            index=min(a.index, b.index),
            start_char=min(a.start_char, b.start_char),
            end_char=max(a.end_char, b.end_char),
            metadata=meta,
        )

    def chunk_document(self, content: str, title: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        if not content or not content.strip():
            return []
        scenes = self._detect_enhanced_scenes(content)
        if len(scenes) == 1:
            # also try implicit
            scenes = self._detect_implicit_scene_breaks(content)
            if not scenes:
                scenes = [content]
        base_meta = {"title": title, "source": source, "chunk_method": "enhanced_scene_level"}
        if metadata:
            base_meta.update(metadata)
        chunks: List[DocumentChunk] = []
        idx = 0
        start_cursor = 0
        for scene in scenes:
            meta = self._analyze_scene(scene)
            max_size = self._get_optimal_chunk_size(meta)
            if meta.scene_type == SceneType.DIALOGUE_SCENE and self.config.preserve_dialogue_integrity:
                parts = self._split_by_dialogue_turns(scene, idx, start_cursor, meta, base_meta, max_size)
            elif meta.scene_type == SceneType.ACTION_SEQUENCE and self.config.preserve_action_sequences:
                parts = self._split_by_action_beats(scene, idx, start_cursor, meta, base_meta, max_size)
            else:
                # default paragraph/sentence based chunking
                parts = self._split_by_action_beats(scene, idx, start_cursor, meta, base_meta, self.config.chunk_size)
            chunks.extend(parts)
            idx = chunks[-1].index + 1 if chunks else idx
            # advance cursor by scene length
            scene_start = content.find(scene, start_cursor)
            start_cursor = scene_start + len(scene) if scene_start != -1 else start_cursor + len(scene)
        # set total_chunks
        total = len(chunks)
        for ch in chunks:
            ch.metadata["total_chunks"] = total
            ch.metadata.setdefault("scene_type", meta.scene_type)
            ch.metadata.setdefault("content_types", meta.content_types)
            ch.metadata.setdefault("importance_score", meta.importance_score)
        return chunks


def create_enhanced_chunker(config: Optional[EnhancedChunkingConfig] = None) -> EnhancedSceneChunker:
    return EnhancedSceneChunker(config)
