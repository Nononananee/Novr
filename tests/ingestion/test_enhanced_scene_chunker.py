"""
Tests for Enhanced Scene-Level Chunker
"""

import pytest
from typing import List, Dict, Any

from ingestion.enhanced_scene_chunker import (
    EnhancedSceneChunker,
    EnhancedChunkingConfig,
    ContentType,
    SceneType,
    SceneMetadata,
    create_enhanced_chunker
)
from ingestion.chunker import DocumentChunk


class TestEnhancedSceneChunker:
    """Test cases for Enhanced Scene Chunker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EnhancedChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            dialogue_chunk_size=800,
            narrative_chunk_size=1200,
            action_chunk_size=600,
            description_chunk_size=1000,
            min_scene_size=200,
            max_scene_size=3000
        )
    
    @pytest.fixture
    def chunker(self, config):
        """Create test chunker."""
        return EnhancedSceneChunker(config)
    
    @pytest.fixture
    def sample_dialogue_scene(self):
        """Sample dialogue-heavy scene."""
        return '''
        "Hello, John," Mary said as she entered the room.
        
        "Mary! I wasn't expecting you," John replied, standing up from his chair.
        
        "I needed to talk to you about what happened yesterday," she continued, her voice trembling slightly.
        
        John's face grew serious. "I was hoping we could discuss that. Please, sit down."
        
        Mary took a seat across from him, her hands folded in her lap. "I don't know where to begin."
        
        "Start wherever feels right," John said gently.
        '''
    
    @pytest.fixture
    def sample_action_scene(self):
        """Sample action-heavy scene."""
        return '''
        The explosion rocked the building, sending debris flying in all directions. Sarah dove behind the concrete barrier just as another blast shook the ground beneath her feet.
        
        "Move! Move!" she shouted to her team, gesturing frantically toward the exit.
        
        Bullets whizzed past her head as she sprinted across the open courtyard. Her heart pounded as she leaped over fallen rubble, her training taking over.
        
        Behind her, she could hear the enemy forces advancing. There was no time to think, only to act.
        '''
    
    @pytest.fixture
    def sample_mixed_scene(self):
        """Sample scene with mixed content types."""
        return '''
        The old library stood majestically against the evening sky, its Gothic architecture casting long shadows across the cobblestone courtyard. Emma approached the heavy wooden doors, her footsteps echoing in the silence.
        
        "Professor Williams?" she called out as she entered the dimly lit interior.
        
        The smell of old books and leather filled her nostrils. Dust motes danced in the shafts of light streaming through the tall windows. She had always loved this place, but tonight it felt different—ominous somehow.
        
        "Emma, you came," a voice said from the shadows.
        
        She turned to see Professor Williams emerging from between the towering bookshelves, his face grave with concern.
        '''
    
    def test_chunker_initialization(self, config):
        """Test chunker initialization."""
        chunker = EnhancedSceneChunker(config)
        assert chunker.config == config
        assert chunker.encoding is not None
    
    def test_create_enhanced_chunker(self):
        """Test factory function."""
        chunker = create_enhanced_chunker()
        assert isinstance(chunker, EnhancedSceneChunker)
        assert isinstance(chunker.config, EnhancedChunkingConfig)
    
    def test_content_type_classification(self, chunker, sample_dialogue_scene, sample_action_scene):
        """Test content type classification."""
        # Test dialogue classification
        dialogue_types = chunker._classify_content_types(sample_dialogue_scene)
        assert ContentType.DIALOGUE in dialogue_types
        
        # Test action classification
        action_types = chunker._classify_content_types(sample_action_scene)
        assert ContentType.ACTION in action_types
    
    def test_dialogue_ratio_calculation(self, chunker, sample_dialogue_scene):
        """Test dialogue ratio calculation."""
        ratio = chunker._calculate_dialogue_ratio(sample_dialogue_scene)
        assert ratio > 0.1  # Should detect significant dialogue
    
    def test_action_ratio_calculation(self, chunker, sample_action_scene):
        """Test action ratio calculation."""
        ratio = chunker._calculate_action_ratio(sample_action_scene)
        assert ratio > 0.1  # Should detect action content
    
    def test_character_extraction(self, chunker, sample_dialogue_scene):
        """Test character name extraction."""
        characters = chunker._extract_characters(sample_dialogue_scene)
        assert "John" in characters
        assert "Mary" in characters
    
    def test_emotional_tone_analysis(self, chunker):
        """Test emotional tone analysis."""
        happy_text = "She was delighted and joyful, her face beaming with happiness."
        sad_text = "He felt sorrowful and melancholy, grief overwhelming his heart."
        
        happy_tone = chunker._analyze_emotional_tone(happy_text)
        sad_tone = chunker._analyze_emotional_tone(sad_text)
        
        assert happy_tone == "joy"
        assert sad_tone == "sadness"
    
    def test_tension_level_calculation(self, chunker, sample_action_scene):
        """Test tension level calculation."""
        tension = chunker._calculate_tension_level(sample_action_scene)
        assert tension > 0.3  # Action scene should have higher tension
    
    def test_scene_type_classification(self, chunker, sample_dialogue_scene, sample_action_scene):
        """Test scene type classification."""
        # Analyze dialogue scene
        dialogue_types = chunker._classify_content_types(sample_dialogue_scene)
        dialogue_ratio = chunker._calculate_dialogue_ratio(sample_dialogue_scene)
        action_ratio = chunker._calculate_action_ratio(sample_dialogue_scene)
        
        dialogue_scene_type = chunker._classify_scene_type(
            sample_dialogue_scene, dialogue_types, dialogue_ratio, action_ratio
        )
        assert dialogue_scene_type == SceneType.DIALOGUE_SCENE
        
        # Analyze action scene
        action_types = chunker._classify_content_types(sample_action_scene)
        action_dialogue_ratio = chunker._calculate_dialogue_ratio(sample_action_scene)
        action_action_ratio = chunker._calculate_action_ratio(sample_action_scene)
        
        action_scene_type = chunker._classify_scene_type(
            sample_action_scene, action_types, action_dialogue_ratio, action_action_ratio
        )
        assert action_scene_type == SceneType.ACTION_SEQUENCE
    
    def test_scene_analysis(self, chunker, sample_mixed_scene):
        """Test complete scene analysis."""
        metadata = chunker._analyze_scene(sample_mixed_scene)
        
        assert isinstance(metadata, SceneMetadata)
        assert metadata.scene_length > 0
        assert metadata.estimated_reading_time > 0
        assert len(metadata.characters) > 0
        assert metadata.importance_score >= 0.0
        assert metadata.importance_score <= 1.0
    
    def test_optimal_chunk_size_calculation(self, chunker):
        """Test optimal chunk size calculation."""
        # Create metadata for different scene types
        dialogue_metadata = SceneMetadata(
            scene_type=SceneType.DIALOGUE_SCENE,
            content_types=[ContentType.DIALOGUE],
            characters=["John", "Mary"],
            locations=["room"],
            emotional_tone="neutral",
            tension_level=0.3,
            dialogue_ratio=0.8,
            action_ratio=0.1,
            description_ratio=0.1,
            importance_score=0.6,
            narrative_function="character_interaction",
            temporal_markers=[]
        )
        
        action_metadata = SceneMetadata(
            scene_type=SceneType.ACTION_SEQUENCE,
            content_types=[ContentType.ACTION],
            characters=["Sarah"],
            locations=["building"],
            emotional_tone="fear",
            tension_level=0.9,
            dialogue_ratio=0.1,
            action_ratio=0.8,
            description_ratio=0.1,
            importance_score=0.8,
            narrative_function="tension_building",
            temporal_markers=[]
        )
        
        dialogue_size = chunker._get_optimal_chunk_size(dialogue_metadata)
        action_size = chunker._get_optimal_chunk_size(action_metadata)
        
        # Dialogue chunks should be smaller than narrative chunks
        assert dialogue_size <= chunker.config.dialogue_chunk_size * 1.2
        # Action chunks should be smaller than dialogue chunks
        assert action_size <= chunker.config.action_chunk_size * 1.2
    
    def test_scene_detection(self, chunker):
        """Test scene detection."""
        multi_scene_text = '''
        Scene one content here with some dialogue.
        "Hello," she said.
        
        ***
        
        Scene two begins after the break.
        This is different content with new characters.
        
        ---
        
        Scene three has another type of break.
        More content follows.
        '''
        
        scenes = chunker._detect_enhanced_scenes(multi_scene_text)
        assert len(scenes) >= 3  # Should detect multiple scenes
    
    def test_implicit_scene_break_detection(self, chunker):
        """Test implicit scene break detection."""
        text_with_implicit_breaks = '''
        John was in the kitchen making breakfast.
        "Good morning," he said to Mary.
        
        Later that day, Sarah arrived at the office.
        She had never been there before.
        The building was impressive.
        
        Meanwhile, back at home, John was worried.
        He hadn't heard from Sarah all day.
        '''
        
        scenes = chunker._detect_implicit_scene_breaks(text_with_implicit_breaks)
        assert len(scenes) > 1  # Should detect character and location changes
    
    def test_dialogue_turn_splitting(self, chunker, sample_dialogue_scene):
        """Test splitting by dialogue turns."""
        # Create a long dialogue scene that needs splitting
        long_dialogue = sample_dialogue_scene * 5  # Repeat to make it long
        
        scene_metadata = chunker._analyze_scene(long_dialogue)
        base_metadata = {"title": "Test", "source": "test"}
        
        chunks = chunker._split_by_dialogue_turns(
            long_dialogue, 0, 0, scene_metadata, base_metadata, 500
        )
        
        assert len(chunks) > 1  # Should split into multiple chunks
        for chunk in chunks:
            assert chunk.metadata["split_method"] == "dialogue_turn"
    
    def test_action_beat_splitting(self, chunker, sample_action_scene):
        """Test splitting by action beats."""
        # Create a long action scene
        long_action = sample_action_scene * 3
        
        scene_metadata = chunker._analyze_scene(long_action)
        base_metadata = {"title": "Test", "source": "test"}
        
        chunks = chunker._split_by_action_beats(
            long_action, 0, 0, scene_metadata, base_metadata, 400
        )
        
        assert len(chunks) > 1  # Should split into multiple chunks
        for chunk in chunks:
            assert chunk.metadata["split_method"] == "action_beat"
    
    def test_document_chunking(self, chunker, sample_mixed_scene):
        """Test complete document chunking."""
        chunks = chunker.chunk_document(
            content=sample_mixed_scene,
            title="Test Novel",
            source="test_file.md"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # Check metadata
        for chunk in chunks:
            assert chunk.metadata["title"] == "Test Novel"
            assert chunk.metadata["source"] == "test_file.md"
            assert chunk.metadata["chunk_method"] == "enhanced_scene_level"
            assert "scene_type" in chunk.metadata
            assert "content_types" in chunk.metadata
            assert "importance_score" in chunk.metadata
    
    def test_chunk_merging(self, chunker):
        """Test chunk merging for small chunks."""
        # Create two small chunks that should be merged
        small_chunk1 = DocumentChunk(
            content="Short content.",
            index=0,
            start_char=0,
            end_char=14,
            metadata={
                "importance_score": 0.2,
                "characters": ["John"],
                "locations": ["room"]
            },
            token_count=3
        )
        
        small_chunk2 = DocumentChunk(
            content="Another short content.",
            index=1,
            start_char=14,
            end_char=36,
            metadata={
                "importance_score": 0.3,
                "characters": ["Mary"],
                "locations": ["kitchen"]
            },
            token_count=4
        )
        
        merged = chunker._merge_chunks(small_chunk1, small_chunk2)
        
        assert "Short content." in merged.content
        assert "Another short content." in merged.content
        assert merged.metadata["chunk_type"] == "merged_scene"
        assert "John" in merged.metadata["characters"]
        assert "Mary" in merged.metadata["characters"]
    
    def test_pov_character_detection(self, chunker):
        """Test POV character detection."""
        first_person_text = "I walked into the room and saw John sitting there."
        third_person_text = "Sarah thought about the conversation. She wondered what John meant."
        
        first_pov = chunker._detect_pov_character(first_person_text, ["John"])
        third_pov = chunker._detect_pov_character(third_person_text, ["Sarah", "John"])
        
        assert first_pov == "first_person_narrator"
        assert third_pov == "Sarah"
    
    def test_temporal_break_detection(self, chunker):
        """Test temporal break detection."""
        temporal_text = "Later that day, she arrived at the office."
        non_temporal_text = "She walked into the room."
        
        assert chunker._has_temporal_break(temporal_text) == True
        assert chunker._has_temporal_break(non_temporal_text) == False
    
    def test_internal_monologue_detection(self, chunker):
        """Test internal monologue detection."""
        monologue_text = "She thought about what he had said and wondered if it was true."
        dialogue_text = '"What do you think?" she asked.'
        
        assert chunker._has_internal_monologue(monologue_text) == True
        assert chunker._has_internal_monologue(dialogue_text) == False
    
    def test_exposition_detection(self, chunker):
        """Test exposition detection."""
        exposition_text = "The city had been founded years ago by settlers from the east."
        action_text = "She ran quickly down the street."
        
        assert chunker._has_exposition(exposition_text) == True
        assert chunker._has_exposition(action_text) == False
    
    def test_empty_content_handling(self, chunker):
        """Test handling of empty content."""
        chunks = chunker.chunk_document("", "Empty", "test.md")
        assert chunks == []
        
        chunks = chunker.chunk_document("   ", "Whitespace", "test.md")
        assert chunks == []
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = EnhancedChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            min_scene_size=100,
            max_scene_size=2000
        )
        assert config.chunk_size == 1000
        
        # Invalid config should raise error
        with pytest.raises(ValueError):
            EnhancedChunkingConfig(
                chunk_size=1000,
                chunk_overlap=1200  # Overlap larger than chunk size
            )
    
    def test_importance_score_calculation(self, chunker):
        """Test importance score calculation."""
        # High importance scene (climax with high tension)
        high_score = chunker._calculate_importance_score(
            SceneType.CLIMAX, [ContentType.ACTION, ContentType.DIALOGUE], 0.9, 3
        )
        
        # Low importance scene (transition with low tension)
        low_score = chunker._calculate_importance_score(
            SceneType.TRANSITION, [ContentType.NARRATIVE], 0.1, 1
        )
        
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
    
    def test_narrative_function_determination(self, chunker):
        """Test narrative function determination."""
        char_intro_function = chunker._determine_narrative_function(
            SceneType.CHARACTER_INTRODUCTION, [ContentType.DIALOGUE]
        )
        
        conflict_function = chunker._determine_narrative_function(
            SceneType.CONFLICT, [ContentType.ACTION]
        )
        
        assert char_intro_function == "character_development"
        assert conflict_function == "plot_advancement"


class TestEnhancedChunkingConfig:
    """Test cases for Enhanced Chunking Configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancedChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.dialogue_chunk_size == 800
        assert config.action_chunk_size == 600
        assert config.min_scene_size == 200
        assert config.preserve_dialogue_integrity == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EnhancedChunkingConfig(
            dialogue_chunk_size=1000,
            action_chunk_size=500,
            scene_break_threshold=0.8
        )
        
        assert config.dialogue_chunk_size == 1000
        assert config.action_chunk_size == 500
        assert config.scene_break_threshold == 0.8


class TestSceneMetadata:
    """Test cases for Scene Metadata."""
    
    def test_scene_metadata_creation(self):
        """Test scene metadata creation."""
        metadata = SceneMetadata(
            scene_type=SceneType.DIALOGUE_SCENE,
            content_types=[ContentType.DIALOGUE],
            characters=["John", "Mary"],
            locations=["kitchen"],
            emotional_tone="neutral",
            tension_level=0.5,
            dialogue_ratio=0.8,
            action_ratio=0.1,
            description_ratio=0.1,
            importance_score=0.6,
            narrative_function="character_interaction",
            temporal_markers=[]
        )
        
        assert metadata.scene_type == SceneType.DIALOGUE_SCENE
        assert ContentType.DIALOGUE in metadata.content_types
        assert "John" in metadata.characters
        assert metadata.tension_level == 0.5


if __name__ == "__main__":
    pytest.main([__file__])