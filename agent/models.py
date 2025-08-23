"""
Pydantic models for data validation and serialization.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchType(str, Enum):
    """Search type enumeration."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"


# Novel-specific enums
class EmotionalTone(str, Enum):
    """Emotional tones for novel content."""
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    TENSE = "tense"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    PEACEFUL = "peaceful"
    DRAMATIC = "dramatic"
    HUMOROUS = "humorous"
    ANGRY = "angry"
    FEARFUL = "fearful"
    HOPEFUL = "hopeful"
    NOSTALGIC = "nostalgic"


class ChunkType(str, Enum):
    """Types of narrative chunks."""
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    DESCRIPTION = "description"
    ACTION = "action"
    INTERNAL_MONOLOGUE = "internal_monologue"
    TRANSITION = "transition"
    SCENE_BREAK = "scene_break"


class CharacterRole(str, Enum):
    """Character roles in the story."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    SUPPORTING = "supporting"
    MINOR = "minor"
    NARRATOR = "narrator"


class PlotSignificance(str, Enum):
    """Plot significance levels."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    MODERATE = "moderate"
    MINOR = "minor"
    BACKGROUND = "background"


# Request Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    
    model_config = ConfigDict(use_enum_values=True)


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    
    model_config = ConfigDict(use_enum_values=True)


# Response Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class ChunkResult(BaseModel):
    """Chunk search result model."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        return max(0.0, min(1.0, v))


class GraphSearchResult(BaseModel):
    """Knowledge graph search result model."""
    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None


class EntityRelationship(BaseModel):
    """Entity relationship model."""
    from_entity: str
    to_entity: str
    relationship_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[ChunkResult] = Field(default_factory=list)
    graph_results: List[GraphSearchResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


class ToolCall(BaseModel):
    """Tool call information model."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    session_id: str
    sources: List[DocumentMetadata] = Field(default_factory=list)
    tools_used: List[ToolCall] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamDelta(BaseModel):
    """Streaming response delta."""
    content: str
    delta_type: Literal["text", "tool_call", "end"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Database Models
class Document(BaseModel):
    """Document model."""
    id: Optional[str] = None
    title: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Chunk(BaseModel):
    """Document chunk model."""
    id: Optional[str] = None
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimensions."""
        if v is not None and len(v) != 1536:  # OpenAI text-embedding-3-small
            raise ValueError(f"Embedding must have 1536 dimensions, got {len(v)}")
        return v


class Session(BaseModel):
    """Session model."""
    id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class Message(BaseModel):
    """Message model."""
    id: Optional[str] = None
    session_id: str
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    model_config = ConfigDict(use_enum_values=True)


# Agent Models
class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    session_id: str
    database_url: Optional[str] = None
    neo4j_uri: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)




class AgentContext(BaseModel):
    """Agent execution context."""
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    search_results: List[ChunkResult] = Field(default_factory=list)
    graph_results: List[GraphSearchResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Ingestion Models
class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True
    extract_entities: bool = True
    # New option for faster ingestion
    skip_graph_building: bool = Field(default=False, description="Skip knowledge graph building for faster ingestion")
    
    # Enhanced Scene Chunking Options
    use_enhanced_scene_chunking: bool = Field(default=False, description="Use enhanced scene-level chunking for narrative content")
    dialogue_chunk_size: int = Field(default=800, ge=200, le=2000, description="Optimal chunk size for dialogue-heavy content")
    narrative_chunk_size: int = Field(default=1200, ge=400, le=3000, description="Optimal chunk size for narrative content")
    action_chunk_size: int = Field(default=600, ge=200, le=1500, description="Optimal chunk size for action sequences")
    description_chunk_size: int = Field(default=1000, ge=300, le=2500, description="Optimal chunk size for descriptive content")
    min_scene_size: int = Field(default=200, ge=50, le=1000, description="Minimum size for a scene chunk")
    max_scene_size: int = Field(default=3000, ge=1000, le=10000, description="Maximum size for a scene chunk")
    preserve_dialogue_integrity: bool = Field(default=True, description="Preserve dialogue turn integrity when chunking")
    preserve_emotional_beats: bool = Field(default=True, description="Preserve emotional beats and transitions")
    preserve_action_sequences: bool = Field(default=True, description="Keep action sequences intact when possible")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


# Approval Workflow Models
class ProposalItem(BaseModel):
    """Individual item in a proposal."""
    type: Literal["character", "relationship", "location", "event"]
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    canonical_id: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    excerpt: Optional[str] = None  # Sample text where entity was mentioned


class ApprovalRequest(BaseModel):
    """Request to create a new proposal."""
    kind: Literal["character", "relationship", "location", "event", "mixed"]
    items: List[ProposalItem]
    source_doc: Optional[str] = None
    suggested_by: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """Result from a consistency validator."""
    validator_name: str
    score: float = Field(ge=0.0, le=1.0)
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ProposalResponse(BaseModel):
    """Response for proposal operations."""
    proposal_id: str
    status: Literal["pending", "approved", "rejected", "failed"]
    kind: str
    confidence: float
    created_at: datetime
    processed_at: Optional[datetime] = None
    processed_by: Optional[str] = None
    neo4j_tx: Optional[Dict[str, Any]] = None
    validation_results: List[ValidationResult] = Field(default_factory=list)
    risk_level: Optional[Literal["low_risk", "medium_risk", "high_risk"]] = None


class ApprovalDecision(BaseModel):
    """Decision to approve/reject a proposal."""
    action: Literal["approve", "reject"]
    processed_by: str
    rejection_reason: Optional[str] = None
    selected_items: Optional[List[int]] = None  # Indices of items to approve (for partial approval)


class Neo4jPushResult(BaseModel):
    """Result of pushing data to Neo4j."""
    transaction_id: Optional[str] = None
    nodes_created: int = 0
    relationships_created: int = 0
    nodes_updated: int = 0
    relationships_updated: int = 0
    errors: List[str] = Field(default_factory=list)


# Health Check Models
class HealthStatus(BaseModel):
    """Health check status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    database: bool
    graph_database: bool
    llm_connection: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


# Novel-specific Models
class Character(BaseModel):
    """Character model for novels."""
    id: Optional[str] = None
    name: str
    personality_traits: List[str] = Field(default_factory=list)
    background: str = ""
    motivations: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)  # character_id -> relationship_type
    emotional_state: Optional[Dict[str, float]] = None  # emotion -> intensity
    development_arc: Optional[str] = None
    role: CharacterRole = CharacterRole.MINOR
    first_appearance: Optional[str] = None
    dialogue_patterns: List[str] = Field(default_factory=list)
    physical_description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Location(BaseModel):
    """Location/setting model for novels."""
    id: Optional[str] = None
    name: str
    description: str
    location_type: str = "general"  # e.g., "city", "building", "room", "landscape"
    atmosphere: Optional[str] = None
    significance: PlotSignificance = PlotSignificance.BACKGROUND
    first_appearance: Optional[str] = None
    associated_characters: List[str] = Field(default_factory=list)
    associated_events: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Scene(BaseModel):
    """Scene model for novels."""
    id: Optional[str] = None
    chapter_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    setting: Optional[str] = None
    characters_present: List[str] = Field(default_factory=list)
    plot_points: List[str] = Field(default_factory=list)
    emotional_tone: Optional[EmotionalTone] = None
    conflict_level: float = Field(default=0.5, ge=0.0, le=1.0)
    purpose: str = ""  # E.g., character development, plot advancement
    chunk_type: ChunkType = ChunkType.NARRATION
    significance: PlotSignificance = PlotSignificance.MODERATE
    word_count: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Chapter(BaseModel):
    """Chapter model for novels."""
    id: Optional[str] = None
    novel_id: str
    chapter_number: int
    title: Optional[str] = None
    summary: Optional[str] = None
    word_count: Optional[int] = None
    scenes: List[str] = Field(default_factory=list)  # scene IDs
    main_characters: List[str] = Field(default_factory=list)
    plot_threads: List[str] = Field(default_factory=list)
    emotional_arc: Optional[Dict[str, Any]] = None
    significance: PlotSignificance = PlotSignificance.MODERATE
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Novel(BaseModel):
    """Novel model."""
    id: Optional[str] = None
    title: str
    author: str
    genre: str = "general"
    summary: Optional[str] = None
    total_word_count: Optional[int] = None
    chapter_count: Optional[int] = None
    main_characters: List[str] = Field(default_factory=list)
    main_themes: List[str] = Field(default_factory=list)
    setting_overview: Optional[str] = None
    target_audience: Optional[str] = None
    completion_status: str = "in_progress"  # "planning", "in_progress", "completed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PlotThread(BaseModel):
    """Plot thread/storyline model."""
    id: Optional[str] = None
    novel_id: str
    name: str
    description: str
    status: str = "active"  # "active", "resolved", "abandoned"
    significance: PlotSignificance = PlotSignificance.MODERATE
    involved_characters: List[str] = Field(default_factory=list)
    key_events: List[str] = Field(default_factory=list)
    resolution: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class EmotionalArc(BaseModel):
    """Emotional arc tracking model."""
    id: Optional[str] = None
    entity_id: str  # character_id, scene_id, or chapter_id
    entity_type: str  # "character", "scene", "chapter"
    emotional_progression: List[Dict[str, Any]] = Field(default_factory=list)
    dominant_emotions: List[str] = Field(default_factory=list)
    emotional_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    turning_points: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Novel-specific Request Models
class NovelGenerationRequest(ChatRequest):
    """Extended request for novel generation."""
    novel_id: Optional[str] = None
    chapter_number: Optional[int] = None
    scene_id: Optional[str] = None
    genre: str = "fantasy"
    tone: str = "serious"
    target_emotional_arc: Optional[Dict[str, Any]] = None
    character_states: Optional[Dict[str, Dict[str, Any]]] = None
    target_word_count: int = 500
    generation_type: str = "continuation"  # "continuation", "dialogue", "description", etc.
    constraints: Optional[Dict[str, Any]] = None


class CharacterAnalysisRequest(BaseModel):
    """Request for character analysis."""
    character_name: str
    novel_id: Optional[str] = None
    analysis_type: str = "development"  # "development", "relationships", "consistency"
    from_chapter: int = 1
    to_chapter: Optional[int] = None


class EmotionalAnalysisRequest(BaseModel):
    """Request for emotional analysis."""
    content: str
    context: Optional[Dict[str, Any]] = None
    analysis_depth: str = "basic"  # "basic", "detailed", "comprehensive"


class PlotAnalysisRequest(BaseModel):
    """Request for plot analysis."""
    novel_id: str
    analysis_type: str = "structure"  # "structure", "consistency", "pacing"
    focus_elements: List[str] = Field(default_factory=list)


# Novel-specific Response Models
class CharacterAnalysisResponse(BaseModel):
    """Response for character analysis."""
    character_name: str
    development_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    development_arc: List[Dict[str, Any]] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    dialogue_patterns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class EmotionalAnalysisResponse(BaseModel):
    """Response for emotional analysis."""
    dominant_emotions: List[str] = Field(default_factory=list)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    emotional_progression: List[Dict[str, Any]] = Field(default_factory=list)
    consistency_score: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)


class PlotAnalysisResponse(BaseModel):
    """Response for plot analysis."""
    structure_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    pacing_score: float = Field(ge=0.0, le=1.0)
    plot_threads: List[Dict[str, Any]] = Field(default_factory=list)
    plot_holes: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class NovelConsistencyReport(BaseModel):
    """Comprehensive consistency report for a novel."""
    novel_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    character_consistency: float = Field(ge=0.0, le=1.0)
    plot_consistency: float = Field(ge=0.0, le=1.0)
    emotional_consistency: float = Field(ge=0.0, le=1.0)
    style_consistency: float = Field(ge=0.0, le=1.0)
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# Enhanced Chunk Model for Novels
class NovelChunk(Chunk):
    """Enhanced chunk model for novel content."""
    chunk_type: ChunkType = ChunkType.NARRATION
    emotional_tone: Optional[EmotionalTone] = None
    characters_present: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    plot_significance: PlotSignificance = PlotSignificance.BACKGROUND
    dialogue_count: int = 0
    action_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    narrative_purpose: Optional[str] = None
