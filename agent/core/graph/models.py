"""Data models for graph operations."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class NovelEntityType(Enum):
    """Types of novel entities."""
    CHARACTER = "character"
    LOCATION = "location"
    PLOT_POINT = "plot_point"
    THEME = "theme"
    OBJECT = "object"
    CONCEPT = "concept"
    SCENE = "scene"
    CHAPTER = "chapter"


class RelationshipType(Enum):
    """Types of relationships in novels."""
    CHARACTER_RELATIONSHIP = "character_relationship"
    CHARACTER_LOCATION = "character_at_location"
    CHARACTER_PLOT = "character_in_plot"
    PLOT_SEQUENCE = "plot_sequence"
    THEME_MANIFESTATION = "theme_manifestation"
    EMOTIONAL_CONNECTION = "emotional_connection"
    TEMPORAL_SEQUENCE = "temporal_sequence"


@dataclass
class NovelEntity:
    """Represents a novel entity with metadata."""
    name: str
    entity_type: NovelEntityType
    description: str
    first_appearance: Optional[str] = None
    significance_score: float = 0.0
    emotional_associations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.emotional_associations is None:
            self.emotional_associations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CharacterProfile:
    """Detailed character profile."""
    name: str
    personality_traits: List[str]
    relationships: Dict[str, str]
    development_arc: List[str]
    emotional_states: Dict[str, float]
    first_appearance: str
    significance_score: float
    dialogue_patterns: List[str] = None
    
    def __post_init__(self):
        if self.dialogue_patterns is None:
            self.dialogue_patterns = []